package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"syscall"
	"time"
	"unicode"
)

type Config struct {
	LlamaBinary   string   `json:"llama_binary"`
	Workdir       string   `json:"workdir"`
	LlamaHost     string   `json:"llama_host"`
	LlamaPort     int      `json:"llama_port"`
	ListenAddr    string   `json:"listen_addr"`
	AllowedFlags  []string `json:"allowed_flags"`
	AllowedEnv    []string `json:"allowed_env"`
	LoadTimeout   int      `json:"load_timeout"`
	LogBufferSize int      `json:"log_buffer_size"`
}

type ModelStatus struct {
	HFRepo    string    `json:"hf_repo"`
	StartedAt time.Time `json:"started_at"`
}

type StatusResponse struct {
	State string       `json:"state"`
	PID   int          `json:"pid"`
	Model *ModelStatus `json:"model,omitempty"`
}

type LoadRequest struct {
	HFRepo         string                 `json:"hf_repo"`
	Context        int                    `json:"context"`
	GPULayers      int                    `json:"gpu_layers"`
	FlashAttention bool                   `json:"flash_attention"`
	Parallel       int                    `json:"parallel"`
	Env            map[string]string      `json:"env"`
	Flags          map[string]interface{} `json:"flags"`
}

type LogEntry struct {
	Type string `json:"type"`
	Line string `json:"line"`
	ID   int64  `json:"id"`
}

type ErrConflict struct {
	msg string
}

func (e ErrConflict) Error() string { return e.msg }

func requestsEqual(a, b LoadRequest) bool {
	return a.HFRepo == b.HFRepo &&
		a.Context == b.Context &&
		a.GPULayers == b.GPULayers &&
		a.FlashAttention == b.FlashAttention &&
		a.Parallel == b.Parallel &&
		reflect.DeepEqual(a.Env, b.Env) &&
		reflect.DeepEqual(a.Flags, b.Flags)
}

type LogBuffer struct {
	mu      sync.RWMutex
	entries []LogEntry
	size    int
	nextID  int64
}

func NewLogBuffer(size int) *LogBuffer {
	return &LogBuffer{
		entries: make([]LogEntry, 0, size),
		size:    size,
		nextID:  1,
	}
}

func (lb *LogBuffer) Add(logType, line string) LogEntry {
	lb.mu.Lock()
	defer lb.mu.Unlock()

	entry := LogEntry{
		Type: logType,
		Line: strings.TrimSpace(line),
		ID:   lb.nextID,
	}
	lb.nextID++

	if len(lb.entries) >= lb.size {
		lb.entries = lb.entries[1:]
	}
	lb.entries = append(lb.entries, entry)

	return entry
}

func (lb *LogBuffer) GetEntries() []LogEntry {
	lb.mu.RLock()
	defer lb.mu.RUnlock()

	res := make([]LogEntry, len(lb.entries))
	copy(res, lb.entries)
	return res
}

type Jukebox struct {
	config Config
	mu     sync.Mutex

	cmd        *exec.Cmd
	currentMod *ModelStatus
	currentReq LoadRequest
	loading    bool
	stopping   bool

	logs        *LogBuffer
	subscribers []chan LogEntry
	subMu       sync.Mutex

	offloadSig  chan struct{}
	processDone chan struct{}
}

func NewJukebox(cfg Config) *Jukebox {
	return &Jukebox{
		config: cfg,
		logs:   NewLogBuffer(cfg.LogBufferSize),
	}
}

var unsafeRunes = []rune{'`', '$', '|', '&', ';', '<', '>', '(', ')', '[', ']', '{', '}', '!'}

func isSafe(s string) bool {
	for _, r := range s {
		if unicode.IsSpace(r) {
			return false
		}
		for _, u := range unsafeRunes {
			if r == u {
				return false
			}
		}
	}
	return true
}

func (j *Jukebox) broadcastLog(entry LogEntry) {
	j.subMu.Lock()
	defer j.subMu.Unlock()

	for _, sub := range j.subscribers {
		select {
		case sub <- entry:
		default:
		}
	}
}

func (j *Jukebox) Status() StatusResponse {
	j.mu.Lock()
	defer j.mu.Unlock()

	res := StatusResponse{State: "idle"}
	if j.cmd != nil && j.cmd.Process != nil {
		res.State = "running"
		res.PID = j.cmd.Process.Pid
		if j.currentMod != nil {
			res.Model = j.currentMod
		}
	}
	return res
}

func (j *Jukebox) monitorProcess(cmd *exec.Cmd, done chan struct{}) {
	_ = cmd.Wait()
	close(done)

	j.mu.Lock()
	defer j.mu.Unlock()

	if j.cmd == cmd {
		j.cmd = nil
		j.currentMod = nil
		j.currentReq = LoadRequest{}
		j.offloadSig = nil
		j.processDone = nil
	}
}

func (j *Jukebox) Offload() error {
	j.mu.Lock()
	if j.cmd == nil || j.cmd.Process == nil {
		j.mu.Unlock()
		return nil
	}

	cmd := j.cmd
	offloadCh := j.offloadSig
	doneCh := j.processDone
	alreadyStopping := j.stopping
	if !alreadyStopping {
		j.stopping = true
		j.loading = false
		j.offloadSig = nil
	}
	j.mu.Unlock()

	if !alreadyStopping && offloadCh != nil {
		close(offloadCh)
	}
	if !alreadyStopping {
		_ = cmd.Process.Signal(syscall.SIGTERM)
	}
	if doneCh == nil {
		j.mu.Lock()
		j.stopping = false
		j.mu.Unlock()
		return nil
	}

	timer := time.NewTimer(5 * time.Second)
	defer timer.Stop()

	select {
	case <-doneCh:
	case <-timer.C:
		_ = cmd.Process.Signal(syscall.SIGKILL)
		<-doneCh
	}

	j.mu.Lock()
	j.stopping = false
	j.mu.Unlock()
	return nil
}

func (j *Jukebox) Load(ctx context.Context, req LoadRequest) error {
	if err := j.validateRequest(req); err != nil {
		return err
	}

	j.mu.Lock()
	defer j.mu.Unlock()

	if j.loading || j.stopping {
		return ErrConflict{"a model is already running; offload it first"}
	}
	if j.cmd != nil && j.cmd.Process != nil && requestsEqual(req, j.currentReq) {
		return nil
	}

	j.loading = true
	defer func() { j.loading = false }()

	// Offload existing process, if any
	if j.cmd != nil && j.cmd.Process != nil {
		oldCmd := j.cmd
		oldOffloadCh := j.offloadSig

		j.cmd = nil
		j.currentMod = nil
		j.currentReq = LoadRequest{}
		j.offloadSig = nil
		j.processDone = nil

		if oldOffloadCh != nil {
			close(oldOffloadCh)
		}
		_ = oldCmd.Process.Signal(syscall.SIGTERM)

		done := make(chan struct{})
		go func() {
			_ = oldCmd.Wait()
			close(done)
		}()

		timer := time.NewTimer(5 * time.Second)
		select {
		case <-done:
			timer.Stop()
		case <-timer.C:
			_ = oldCmd.Process.Signal(syscall.SIGKILL)
			<-done
		}
	}

	args := []string{"-hf", req.HFRepo}
	args = append(args, "--host", j.config.LlamaHost, "--port", fmt.Sprintf("%d", j.config.LlamaPort))

	if req.Context > 0 {
		args = append(args, "-c", fmt.Sprintf("%d", req.Context))
	}
	if req.GPULayers > 0 {
		args = append(args, "-ngl", fmt.Sprintf("%d", req.GPULayers))
	}
	if req.FlashAttention {
		args = append(args, "-fa", "on")
	}
	if req.Parallel > 0 {
		args = append(args, "-np", fmt.Sprintf("%d", req.Parallel))
	}

	for k, v := range req.Flags {
		switch val := v.(type) {
		case float64:
			args = append(args, "--"+k, fmt.Sprintf("%g", val))
		case string:
			args = append(args, "--"+k, val)
		case bool:
			if val {
				args = append(args, "--"+k)
			}
		default:
			return fmt.Errorf("unsupported flag type for %s", k)
		}
	}

	cmd := exec.Command(j.config.LlamaBinary, args...)
	cmd.Dir = j.config.Workdir
	cmd.Env = os.Environ()
	for k, v := range req.Env {
		cmd.Env = append(cmd.Env, fmt.Sprintf("%s=%s", k, v))
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return err
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start llama-server: %v", err)
	}

	offloadCh := make(chan struct{})
	doneCh := make(chan struct{})
	j.offloadSig = offloadCh
	j.processDone = doneCh
	j.cmd = cmd
	j.currentReq = req
	j.currentMod = &ModelStatus{HFRepo: req.HFRepo, StartedAt: time.Now()}

	go j.monitorProcess(cmd, doneCh)
	go j.streamLogs(stdout, "stdout")
	go j.streamLogs(stderr, "stderr")

	j.mu.Unlock()
	err = j.waitForReady(ctx, offloadCh, doneCh)
	if err != nil {
		_ = j.Offload()
	}
	j.mu.Lock()

	if err != nil {
		return fmt.Errorf("model failed to reach ready state: %v", err)
	}

	return nil
}

func (j *Jukebox) isAllowedFlag(flag string) bool {
	for _, allowed := range j.config.AllowedFlags {
		if allowed == flag {
			return true
		}
	}
	return false
}

func (j *Jukebox) validateRequest(req LoadRequest) error {
	if req.HFRepo == "" {
		return fmt.Errorf("hf_repo is required")
	}
	if !isSafe(req.HFRepo) {
		return fmt.Errorf("invalid HF_REPO: contains unsafe characters")
	}
	if req.Context > 0 && !j.isAllowedFlag("c") {
		return fmt.Errorf("flag c is not whitelisted")
	}
	if req.GPULayers > 0 && !j.isAllowedFlag("ngl") {
		return fmt.Errorf("flag ngl is not whitelisted")
	}
	if req.FlashAttention && !j.isAllowedFlag("fa") {
		return fmt.Errorf("flag fa is not whitelisted")
	}
	if req.Parallel > 0 && !j.isAllowedFlag("np") {
		return fmt.Errorf("flag np is not whitelisted")
	}

	for k, v := range req.Env {
		allowed := false
		for _, a := range j.config.AllowedEnv {
			if a == k {
				allowed = true
				break
			}
		}
		if !allowed {
			return fmt.Errorf("environment variable %s is not whitelisted", k)
		}
		if !isSafe(v) {
			return fmt.Errorf("unsafe value for env %s", k)
		}
	}

	for k, v := range req.Flags {
		allowed := false
		for _, a := range j.config.AllowedFlags {
			if a == k {
				allowed = true
				break
			}
		}
		if !allowed {
			return fmt.Errorf("flag %s is not whitelisted", k)
		}
		if strVal, ok := v.(string); ok && !isSafe(strVal) {
			return fmt.Errorf("unsafe value for flag %s", k)
		}
	}

	return nil
}

func (j *Jukebox) streamLogs(r io.Reader, logType string) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		entry := j.logs.Add(logType, scanner.Text())
		j.broadcastLog(entry)
	}
	if err := scanner.Err(); err != nil {
		log.Printf("streamLogs(%s): %v", logType, err)
	}
}

func (j *Jukebox) waitForReady(ctx context.Context, offloadCh chan struct{}, doneCh <-chan struct{}) error {
	client := &http.Client{Timeout: 2 * time.Second}
	url := fmt.Sprintf("http://%s:%d/", j.config.LlamaHost, j.config.LlamaPort)

	timeout := time.Duration(j.config.LoadTimeout) * time.Second
	if timeout == 0 {
		timeout = 15 * time.Minute
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-offloadCh:
			return fmt.Errorf("model was offloaded during loading")
		case <-doneCh:
			return fmt.Errorf("llama-server exited before becoming ready")
		case <-timer.C:
			return fmt.Errorf("timeout waiting for llama-server to become ready")
		case <-ticker.C:
			req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
			resp, err := client.Do(req)
			if err == nil {
				if resp.StatusCode == http.StatusOK {
					resp.Body.Close()
					return nil
				}
				resp.Body.Close()
			}
		}
	}
}

func writeJSON(w http.ResponseWriter, status int, v interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeLogEvent(w io.Writer, entry LogEntry) {
	fmt.Fprintf(w, "event: log\ndata: %s\n\n", mustMarshal(entry))
}

func streamLiveLogs(ctx context.Context, w http.ResponseWriter, sub <-chan LogEntry, lastSentID int64) {
	for {
		select {
		case <-ctx.Done():
			return
		case entry, ok := <-sub:
			if !ok {
				return
			}
			if entry.ID <= lastSentID {
				continue
			}
			writeLogEvent(w, entry)
			lastSentID = entry.ID
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}
}

func (j *Jukebox) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "Method not allowed"})
		return
	}
	writeJSON(w, http.StatusOK, j.Status())
}

func (j *Jukebox) handleLoad(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "Method not allowed"})
		return
	}

	var req LoadRequest
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		if err == nil {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": "request body must contain exactly one JSON object"})
			return
		}
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}

	if err := j.Load(r.Context(), req); err != nil {
		if _, ok := err.(ErrConflict); ok {
			writeJSON(w, http.StatusConflict, map[string]string{"error": err.Error()})
			return
		}

		msg := err.Error()
		if strings.Contains(msg, "offloaded during loading") {
			writeJSON(w, http.StatusBadRequest, map[string]string{"error": msg})
			return
		}
		if strings.Contains(msg, "failed to start") || strings.Contains(msg, "timeout") || strings.Contains(msg, "failed to reach ready") {
			writeJSON(w, http.StatusInternalServerError, map[string]string{"error": msg})
			return
		}
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": msg})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "success"})
}

func (j *Jukebox) handleOffload(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "Method not allowed"})
		return
	}
	if err := j.Offload(); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "offloaded"})
}

func (j *Jukebox) handleLogs(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"error": "Method not allowed"})
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	sub := make(chan LogEntry, 100)
	j.subMu.Lock()
	j.subscribers = append(j.subscribers, sub)
	j.subMu.Unlock()

	defer func() {
		j.subMu.Lock()
		for i, s := range j.subscribers {
			if s == sub {
				j.subscribers = append(j.subscribers[:i], j.subscribers[i+1:]...)
				break
			}
		}
		j.subMu.Unlock()
		close(sub)
	}()

	history := j.logs.GetEntries()
	var lastSentID int64
	for _, entry := range history {
		writeLogEvent(w, entry)
		lastSentID = entry.ID
	}

	if f, ok := w.(http.Flusher); ok {
		f.Flush()
	}

	streamLiveLogs(r.Context(), w, sub, lastSentID)
}

func mustMarshal(v interface{}) string {
	b, _ := json.Marshal(v)
	return string(b)
}

func validateConfig(cfg *Config) error {
	if cfg.LlamaBinary == "" {
		return fmt.Errorf("llama_binary is required")
	}
	if cfg.Workdir == "" {
		return fmt.Errorf("workdir is required")
	}
	if cfg.LlamaHost == "" {
		return fmt.Errorf("llama_host is required")
	}
	if cfg.LlamaPort == 0 {
		return fmt.Errorf("llama_port is required")
	}
	if cfg.ListenAddr == "" {
		cfg.ListenAddr = ":4468"
	}
	if cfg.LogBufferSize <= 0 {
		cfg.LogBufferSize = 5000
	}
	if cfg.LoadTimeout <= 0 {
		cfg.LoadTimeout = 900
	}
	return nil
}

func main() {
	configFile := os.Getenv("LLAMA_JUKEBOX_CONFIG")
	if configFile == "" {
		log.Fatal("LLAMA_JUKEBOX_CONFIG is required (set it to your config.json path). Run with 'init' subcommand to generate a template.")
	}

	if len(os.Args) > 1 && os.Args[1] == "init" {
		if err := writeExampleConfig(configFile); err != nil {
			log.Fatalf("Failed to write config to %s: %v", configFile, err)
		}
		log.Printf("Wrote example config to %s. Edit it, then run without 'init' to start the server.", configFile)
		return
	}

	data, err := os.ReadFile(configFile)
	if err != nil {
		log.Fatalf("Failed to read config from %s: %v", configFile, err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		log.Fatalf("Failed to parse config: %v", err)
	}
	if err := validateConfig(&cfg); err != nil {
		log.Fatalf("Invalid config: %v", err)
	}

	daemon := NewJukebox(cfg)
	mux := http.NewServeMux()
	mux.HandleFunc("/status", daemon.handleStatus)
	mux.HandleFunc("/load", daemon.handleLoad)
	mux.HandleFunc("/offload", daemon.handleOffload)
	mux.HandleFunc("/logs", daemon.handleLogs)

	server := &http.Server{
		Addr:    cfg.ListenAddr,
		Handler: mux,
	}

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)

	go func() {
		log.Printf("llama-jukebox starting on %s...", cfg.ListenAddr)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("ListenAndServe error: %v", err)
		}
	}()

	<-stop
	log.Println("Shutting down llama-jukebox...")
	_ = daemon.Offload()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server Shutdown Failed:%+v", err)
	}
	log.Println("llama-jukebox exited")
}

var exampleConfig = []byte(`{
  "llama_binary": "/path/to/your/llama-server",
  "workdir": "/home/user",
  "llama_host": "127.0.0.1",
  "llama_port": 11434,
  "listen_addr": ":4468",
  "allowed_flags": [
    "spec-type",
    "spec-draft-n-max",
    "temp",
    "top-p",
    "top-k",
    "min-p",
    "repeat-penalty",
    "seed",
    "threads",
    "batch",
    "ubatch",
    "ngl",
    "c",
    "fa",
    "np"
  ],
  "allowed_env": [
    "LLAMA_CACHE",
    "HUGGING_FACE_HUB_TOKEN"
  ],
  "load_timeout": 900,
  "log_buffer_size": 5000
}
`)

func writeExampleConfig(path string) error {
	dir := filepath.Dir(path)
	if dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("create config directory: %w", err)
		}
	}
	return os.WriteFile(path, exampleConfig, 0644)
}
