package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestNewLogBuffer(t *testing.T) {
	lb := NewLogBuffer(100)
	if lb == nil {
		t.Fatal("expected non-nil LogBuffer")
	}
	if lb.size != 100 {
		t.Fatalf("expected size 100, got %d", lb.size)
	}
	if lb.nextID != 1 {
		t.Fatalf("expected nextID 1, got %d", lb.nextID)
	}
	if len(lb.entries) != 0 {
		t.Fatalf("expected 0 entries, got %d", len(lb.entries))
	}
}

func TestLogBufferAdd(t *testing.T) {
	lb := NewLogBuffer(10)

	lb.Add("stdout", "hello world")

	entries := lb.GetEntries()
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}

	if entries[0].Type != "stdout" {
		t.Errorf("expected type stdout, got %s", entries[0].Type)
	}
	if entries[0].Line != "hello world" {
		t.Errorf("expected line 'hello world', got %q", entries[0].Line)
	}
	if entries[0].ID != 1 {
		t.Errorf("expected id 1, got %d", entries[0].ID)
	}
}

func TestLogBufferTrimSpace(t *testing.T) {
	lb := NewLogBuffer(10)

	lb.Add("stdout", "  hello with spaces  ")

	entries := lb.GetEntries()
	if entries[0].Line != "hello with spaces" {
		t.Errorf("expected trimmed line, got %q", entries[0].Line)
	}
}

func TestLogBufferRingEviction(t *testing.T) {
	lb := NewLogBuffer(3)

	lb.Add("stdout", "line1")
	lb.Add("stdout", "line2")
	lb.Add("stdout", "line3")
	lb.Add("stdout", "line4")

	entries := lb.GetEntries()
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries after eviction, got %d", len(entries))
	}
	if entries[0].Line != "line2" {
		t.Errorf("expected first entry to be 'line2', got %q", entries[0].Line)
	}
	if entries[2].Line != "line4" {
		t.Errorf("expected last entry to be 'line4', got %q", entries[2].Line)
	}
}

func TestLogBufferSequentialIDs(t *testing.T) {
	lb := NewLogBuffer(100)

	for i := 0; i < 10; i++ {
		lb.Add("stdout", "line")
	}

	entries := lb.GetEntries()
	for i, entry := range entries {
		expected := int64(i + 1)
		if entry.ID != expected {
			t.Errorf("entry[%d]: expected ID %d, got %d", i, expected, entry.ID)
		}
	}
}

func TestLogBufferGetEntriesReturnsCopy(t *testing.T) {
	lb := NewLogBuffer(10)
	lb.Add("stdout", "line1")

	entries1 := lb.GetEntries()
	entries1[0].Line = "modified"

	entries2 := lb.GetEntries()
	if entries2[0].Line != "line1" {
		t.Errorf("expected 'line1' (copy should be independent), got %q", entries2[0].Line)
	}
}

func TestLogBufferMultipleTypes(t *testing.T) {
	lb := NewLogBuffer(10)

	lb.Add("stdout", "out line")
	lb.Add("stderr", "err line")

	entries := lb.GetEntries()
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
	if entries[0].Type != "stdout" || entries[0].Line != "out line" {
		t.Errorf("first entry mismatch: %+v", entries[0])
	}
	if entries[1].Type != "stderr" || entries[1].Line != "err line" {
		t.Errorf("second entry mismatch: %+v", entries[1])
	}
}

func TestIsSafe(t *testing.T) {
	safe := []string{
		"hello",
		"hello-world",
		"hello_world",
		"hello.world",
		"hello:world",
		"hello/world",
		"",
		"123",
		"Qwen3.6-27B-MTP-GGUF:UD-Q5_K_XL",
	}

	unsafe := []string{
		"hello world",
		"hello\tworld",
		"hello\nworld",
		"hello`",
		"hello$",
		"hello|",
		"hello&",
		"hello;",
		"hello<",
		"hello>",
		"hello(",
		"hello)",
		"hello[",
		"hello]",
		"hello{",
		"hello}",
		"hello!",
	}

	for _, s := range safe {
		if !isSafe(s) {
			t.Errorf("isSafe(%q) = false, want true", s)
		}
	}

	for _, s := range unsafe {
		if isSafe(s) {
			t.Errorf("isSafe(%q) = true, want false", s)
		}
	}
}

func TestValidateConfig(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
		errMsg  string
	}{
		{
			name: "valid config",
			cfg: Config{
				LlamaBinary: "/usr/bin/llama-server",
				Workdir:     "/tmp",
				LlamaHost:   "localhost",
				LlamaPort:   8080,
			},
			wantErr: false,
		},
		{
			name: "missing llama binary",
			cfg: Config{
				Workdir:   "/tmp",
				LlamaHost: "localhost",
				LlamaPort: 8080,
			},
			wantErr: true,
			errMsg:  "llama_binary is required",
		},
		{
			name: "missing workdir",
			cfg: Config{
				LlamaBinary: "/usr/bin/llama-server",
				LlamaHost:   "localhost",
				LlamaPort:   8080,
			},
			wantErr: true,
			errMsg:  "workdir is required",
		},
		{
			name: "missing llama host",
			cfg: Config{
				LlamaBinary: "/usr/bin/llama-server",
				Workdir:     "/tmp",
				LlamaPort:   8080,
			},
			wantErr: true,
			errMsg:  "llama_host is required",
		},
		{
			name: "missing llama port",
			cfg: Config{
				LlamaBinary: "/usr/bin/llama-server",
				Workdir:     "/tmp",
				LlamaHost:   "localhost",
			},
			wantErr: true,
			errMsg:  "llama_port is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := validateConfig(&tt.cfg)
			if tt.wantErr {
				if err == nil {
					t.Error("expected error, got nil")
					return
				}
				if tt.errMsg != "" && err.Error() != tt.errMsg {
					t.Errorf("expected error %q, got %q", tt.errMsg, err.Error())
				}
			} else if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestValidateConfigDefaults(t *testing.T) {
	cfg := Config{
		LlamaBinary: "/usr/bin/llama-server",
		Workdir:     "/tmp",
		LlamaHost:   "localhost",
		LlamaPort:   8080,
	}

	if err := validateConfig(&cfg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if cfg.ListenAddr != ":4468" {
		t.Errorf("expected default listen addr :4468, got %q", cfg.ListenAddr)
	}
	if cfg.LogBufferSize != 5000 {
		t.Errorf("expected default log buffer size 5000, got %d", cfg.LogBufferSize)
	}
	if cfg.LoadTimeout != 900 {
		t.Errorf("expected default load timeout 900, got %d", cfg.LoadTimeout)
	}
}

func TestValidateRequestMissingHFRepo(t *testing.T) {
	cfg := Config{
		LlamaBinary: "/usr/bin/llama-server",
		Workdir:     "/tmp",
		LlamaHost:   "localhost",
		LlamaPort:   8080,
	}
	j := NewJukebox(cfg)

	req := LoadRequest{}
	err := j.validateRequest(req)
	if err == nil {
		t.Fatal("expected error for missing hf_repo, got nil")
	}
}

func TestValidateRequestUnsafeHFRepo(t *testing.T) {
	cfg := Config{
		LlamaBinary: "/usr/bin/llama-server",
		Workdir:     "/tmp",
		LlamaHost:   "localhost",
		LlamaPort:   8080,
	}
	j := NewJukebox(cfg)

	req := LoadRequest{HFRepo: "unsloth/Qwen3.6-27B"}
	if err := j.validateRequest(req); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	req = LoadRequest{HFRepo: "bad;repo"}
	if err := j.validateRequest(req); err == nil {
		t.Fatal("expected error for unsafe hf_repo, got nil")
	}
}

func TestValidateRequestEnvWhitelist(t *testing.T) {
	cfg := Config{
		LlamaBinary: "/usr/bin/llama-server",
		Workdir:     "/tmp",
		LlamaHost:   "localhost",
		LlamaPort:   8080,
		AllowedEnv:  []string{"LLAMA_CACHE"},
	}
	j := NewJukebox(cfg)

	req := LoadRequest{
		HFRepo: "unsloth/Qwen3.6-27B",
		Env:    map[string]string{"LLAMA_CACHE": "/cache"},
	}
	if err := j.validateRequest(req); err != nil {
		t.Fatalf("expected success, got error: %v", err)
	}

	req.Env["UNAUTHORIZED"] = "value"
	if err := j.validateRequest(req); err == nil {
		t.Fatal("expected error for unauthorized env var, got nil")
	}
}

func TestValidateRequestFlagsWhitelist(t *testing.T) {
	cfg := Config{
		LlamaBinary:  "/usr/bin/llama-server",
		Workdir:      "/tmp",
		LlamaHost:    "localhost",
		LlamaPort:    8080,
		AllowedFlags: []string{"temp"},
	}
	j := NewJukebox(cfg)

	req := LoadRequest{
		HFRepo: "unsloth/Qwen3.6-27B",
		Flags:  map[string]interface{}{"temp": 0.8},
	}
	if err := j.validateRequest(req); err != nil {
		t.Fatalf("expected success, got error: %v", err)
	}

	req.Flags["unauthorized"] = "value"
	if err := j.validateRequest(req); err == nil {
		t.Fatal("expected error for unauthorized flag, got nil")
	}
}

func TestValidateRequestEnvUnsafeValue(t *testing.T) {
	cfg := Config{
		LlamaBinary: "/usr/bin/llama-server",
		Workdir:     "/tmp",
		LlamaHost:   "localhost",
		LlamaPort:   8080,
		AllowedEnv:  []string{"LLAMA_CACHE"},
	}
	j := NewJukebox(cfg)

	req := LoadRequest{
		HFRepo: "unsloth/Qwen3.6-27B",
		Env:    map[string]string{"LLAMA_CACHE": "bad;value"},
	}
	if err := j.validateRequest(req); err == nil {
		t.Fatal("expected error for unsafe env value, got nil")
	}
}

func TestValidateRequestFlagUnsafeValue(t *testing.T) {
	cfg := Config{
		LlamaBinary:  "/usr/bin/llama-server",
		Workdir:      "/tmp",
		LlamaHost:    "localhost",
		LlamaPort:    8080,
		AllowedFlags: []string{"model"},
	}
	j := NewJukebox(cfg)

	req := LoadRequest{
		HFRepo: "unsloth/Qwen3.6-27B",
		Flags:  map[string]interface{}{"model": "bad;value"},
	}
	if err := j.validateRequest(req); err == nil {
		t.Fatal("expected error for unsafe flag value, got nil")
	}
}

func TestValidateRequestStructuredFlagsRequireWhitelist(t *testing.T) {
	tests := []struct {
		name   string
		req    LoadRequest
		errMsg string
	}{
		{
			name:   "context requires c whitelist",
			req:    LoadRequest{HFRepo: "repo/model", Context: 4096},
			errMsg: "flag c is not whitelisted",
		},
		{
			name:   "gpu layers require ngl whitelist",
			req:    LoadRequest{HFRepo: "repo/model", GPULayers: 99},
			errMsg: "flag ngl is not whitelisted",
		},
		{
			name:   "flash attention requires fa whitelist",
			req:    LoadRequest{HFRepo: "repo/model", FlashAttention: true},
			errMsg: "flag fa is not whitelisted",
		},
		{
			name:   "parallel requires np whitelist",
			req:    LoadRequest{HFRepo: "repo/model", Parallel: 2},
			errMsg: "flag np is not whitelisted",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			j := NewJukebox(Config{
				LlamaBinary: "/usr/bin/llama-server",
				Workdir:     "/tmp",
				LlamaHost:   "localhost",
				LlamaPort:   8080,
			})

			err := j.validateRequest(tt.req)
			if err == nil || err.Error() != tt.errMsg {
				t.Fatalf("expected error %q, got %v", tt.errMsg, err)
			}
		})
	}
}

func TestValidateRequestStructuredFlagsAllowed(t *testing.T) {
	j := NewJukebox(Config{
		LlamaBinary:  "/usr/bin/llama-server",
		Workdir:      "/tmp",
		LlamaHost:    "localhost",
		LlamaPort:    8080,
		AllowedFlags: []string{"c", "ngl", "fa", "np", "temp"},
	})

	req := LoadRequest{
		HFRepo:         "repo/model",
		Context:        4096,
		GPULayers:      99,
		FlashAttention: true,
		Parallel:       2,
		Flags:          map[string]interface{}{"temp": 0.8},
	}

	if err := j.validateRequest(req); err != nil {
		t.Fatalf("expected success, got error: %v", err)
	}
}

func TestErrConflict(t *testing.T) {
	e := ErrConflict{"a model is already running"}
	if e.Error() != "a model is already running" {
		t.Errorf("unexpected error message: %s", e.Error())
	}
}

// --- HTTP Handler Tests ---

func newTestJukebox() *Jukebox {
	cfg := Config{
		LlamaBinary:   "/usr/bin/llama-server",
		Workdir:       "/tmp",
		LlamaHost:     "localhost",
		LlamaPort:     8080,
		ListenAddr:    ":4468",
		AllowedFlags:  []string{"temp", "top-p", "model"},
		AllowedEnv:    []string{"LLAMA_CACHE"},
		LogBufferSize: 500,
	}
	return NewJukebox(cfg)
}

func writeBlockingProcessScript(t *testing.T) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "fake-llama.sh")
	script := "#!/bin/sh\ntrap 'exit 0' TERM INT\nwhile :; do\n  sleep 1\ndone\n"
	if err := os.WriteFile(path, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write helper script: %v", err)
	}
	return path
}

func writeExitProcessScript(t *testing.T, exitCode int) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "exit-llama.sh")
	script := "#!/bin/sh\nexit " + strconv.Itoa(exitCode) + "\n"
	if err := os.WriteFile(path, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write helper script: %v", err)
	}
	return path
}

func writeSlowExitProcessScript(t *testing.T, delay time.Duration) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "slow-exit-llama.sh")
	script := "#!/bin/sh\ntrap 'sleep " + strconv.Itoa(int(delay/time.Second)) + "; exit 0' TERM INT\nwhile :; do\n  sleep 1\ndone\n"
	if err := os.WriteFile(path, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write helper script: %v", err)
	}
	return path
}

func readyEndpoint(t *testing.T) (string, int) {
	t.Helper()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(server.Close)

	host, portStr, err := net.SplitHostPort(server.Listener.Addr().String())
	if err != nil {
		t.Fatalf("failed to parse readiness address: %v", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		t.Fatalf("failed to parse readiness port: %v", err)
	}
	return host, port
}

func newProcessTestJukebox(t *testing.T) *Jukebox {
	t.Helper()

	host, port := readyEndpoint(t)
	cfg := Config{
		LlamaBinary:   writeBlockingProcessScript(t),
		Workdir:       t.TempDir(),
		LlamaHost:     host,
		LlamaPort:     port,
		ListenAddr:    ":4468",
		LoadTimeout:   5,
		LogBufferSize: 500,
	}
	return NewJukebox(cfg)
}

func newSlowOffloadTestJukebox(t *testing.T, delay time.Duration) *Jukebox {
	t.Helper()

	host, port := readyEndpoint(t)
	cfg := Config{
		LlamaBinary:   writeSlowExitProcessScript(t, delay),
		Workdir:       t.TempDir(),
		LlamaHost:     host,
		LlamaPort:     port,
		ListenAddr:    ":4468",
		LoadTimeout:   5,
		LogBufferSize: 500,
	}
	return NewJukebox(cfg)
}

func newExitBeforeReadyTestJukebox(t *testing.T) *Jukebox {
	t.Helper()

	host, port := readyEndpoint(t)
	cfg := Config{
		LlamaBinary:   writeExitProcessScript(t, 1),
		Workdir:       t.TempDir(),
		LlamaHost:     host,
		LlamaPort:     port,
		ListenAddr:    ":4468",
		LoadTimeout:   30,
		LogBufferSize: 500,
	}
	return NewJukebox(cfg)
}

func writeFailingProcessScript(t *testing.T, stderrLine string, exitCode int) string {
	t.Helper()

	path := filepath.Join(t.TempDir(), "failing-llama.sh")
	script := "#!/bin/sh\necho '" + stderrLine + "' 1>&2\nexit " + strconv.Itoa(exitCode) + "\n"
	if err := os.WriteFile(path, []byte(script), 0755); err != nil {
		t.Fatalf("failed to write helper script: %v", err)
	}
	return path
}

func newFailingExitTestJukebox(t *testing.T, stderrLine string, exitCode int) *Jukebox {
	t.Helper()

	host, port := readyEndpoint(t)
	cfg := Config{
		LlamaBinary:   writeFailingProcessScript(t, stderrLine, exitCode),
		Workdir:       t.TempDir(),
		LlamaHost:     host,
		LlamaPort:     port,
		ListenAddr:    ":4468",
		LoadTimeout:   30,
		LogBufferSize: 500,
	}
	return NewJukebox(cfg)
}

func waitForState(t *testing.T, j *Jukebox, want string) {
	t.Helper()

	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		if got := j.Status().State; got == want {
			return
		}
		time.Sleep(25 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for state %q; last state %q", want, j.Status().State)
}

func TestHandleStatusIdle(t *testing.T) {
	j := newTestJukebox()

	req := httptest.NewRequest(http.MethodGet, "/status", nil)
	w := httptest.NewRecorder()

	j.handleStatus(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp StatusResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.State != "idle" {
		t.Errorf("expected state 'idle', got %q", resp.State)
	}
}

func TestHandleStatusMethodNotAllowed(t *testing.T) {
	j := newTestJukebox()

	req := httptest.NewRequest(http.MethodPost, "/status", nil)
	w := httptest.NewRecorder()

	j.handleStatus(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected status 405, got %d", w.Code)
	}
}

func TestHandleLoadInvalidJSON(t *testing.T) {
	j := newTestJukebox()

	req := httptest.NewRequest(http.MethodPost, "/load", bytes.NewBufferString("{invalid"))
	w := httptest.NewRecorder()

	j.handleLoad(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", w.Code)
	}
}

func TestHandleLoadRejectsUnknownFields(t *testing.T) {
	j := newTestJukebox()

	body := []byte(`{"hf_repo":"repo/model","gpu_layer":1}`)
	req := httptest.NewRequest(http.MethodPost, "/load", bytes.NewBuffer(body))
	w := httptest.NewRecorder()

	j.handleLoad(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", w.Code)
	}
	if !strings.Contains(w.Body.String(), "unknown field") {
		t.Fatalf("expected unknown field error, got %q", w.Body.String())
	}
}

func TestHandleLoadRejectsTrailingJSON(t *testing.T) {
	j := newTestJukebox()

	body := []byte(`{"hf_repo":"repo/model"} {}`)
	req := httptest.NewRequest(http.MethodPost, "/load", bytes.NewBuffer(body))
	w := httptest.NewRecorder()

	j.handleLoad(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", w.Code)
	}
}

func TestLoadPassesConfiguredHostAndPort(t *testing.T) {
	tmpDir := t.TempDir()
	argsFile := filepath.Join(tmpDir, "args.txt")
	script := filepath.Join(tmpDir, "capture-args.sh")
	contents := "#!/bin/sh\nprintf '%s\\n' \"$@\" >\"$LLAMA_ARGS_FILE\"\ntrap 'exit 0' TERM INT\nwhile :; do\n  sleep 1\ndone\n"
	if err := os.WriteFile(script, []byte(contents), 0755); err != nil {
		t.Fatalf("failed to write helper script: %v", err)
	}

	host, port := readyEndpoint(t)
	j := NewJukebox(Config{
		LlamaBinary:   script,
		Workdir:       tmpDir,
		LlamaHost:     host,
		LlamaPort:     port,
		LoadTimeout:   5,
		LogBufferSize: 500,
	})

	originalEnv := os.Getenv("LLAMA_ARGS_FILE")
	if err := os.Setenv("LLAMA_ARGS_FILE", argsFile); err != nil {
		t.Fatalf("failed to set LLAMA_ARGS_FILE: %v", err)
	}
	defer func() {
		if originalEnv == "" {
			_ = os.Unsetenv("LLAMA_ARGS_FILE")
			return
		}
		_ = os.Setenv("LLAMA_ARGS_FILE", originalEnv)
	}()

	if err := j.Load(context.Background(), LoadRequest{HFRepo: "repo/model"}); err != nil {
		t.Fatalf("load failed: %v", err)
	}
	defer func() {
		if err := j.Offload(); err != nil {
			t.Fatalf("offload failed: %v", err)
		}
	}()

	data, err := os.ReadFile(argsFile)
	if err != nil {
		t.Fatalf("failed to read captured args: %v", err)
	}
	args := strings.Fields(string(data))

	expected := []string{"-hf", "repo/model", "--host", host, "--port", strconv.Itoa(port)}
	for _, arg := range expected {
		if !containsArg(args, arg) {
			t.Fatalf("expected args %v to contain %q", args, arg)
		}
	}
}

func TestHandleLoadMissingHFRepo(t *testing.T) {
	j := newTestJukebox()

	body := []byte(`{}`)
	req := httptest.NewRequest(http.MethodPost, "/load", bytes.NewBuffer(body))
	w := httptest.NewRecorder()

	j.handleLoad(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", w.Code)
	}
}

func TestHandleLoadUnsafeHFRepo(t *testing.T) {
	j := newTestJukebox()

	body := []byte(`{"hf_repo": "bad;repo"}`)
	req := httptest.NewRequest(http.MethodPost, "/load", bytes.NewBuffer(body))
	w := httptest.NewRecorder()

	j.handleLoad(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected status 400, got %d", w.Code)
	}
}

func TestHandleLoadMethodNotAllowed(t *testing.T) {
	j := newTestJukebox()

	req := httptest.NewRequest(http.MethodGet, "/load", nil)
	w := httptest.NewRecorder()

	j.handleLoad(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected status 405, got %d", w.Code)
	}
}

func TestLoadKeepsProcessRunningAfterContextCancel(t *testing.T) {
	j := newProcessTestJukebox(t)
	req := LoadRequest{HFRepo: "repo/model"}
	ctx, cancel := context.WithCancel(context.Background())
	defer func() {
		if err := j.Offload(); err != nil {
			t.Fatalf("offload failed: %v", err)
		}
	}()

	if err := j.Load(ctx, req); err != nil {
		t.Fatalf("load failed: %v", err)
	}
	waitForState(t, j, "running")

	cancel()
	time.Sleep(200 * time.Millisecond)

	if got := j.Status().State; got != "running" {
		t.Fatalf("expected process to outlive request context cancel, got %q", got)
	}
}

func TestLoadRejectsConcurrentStarts(t *testing.T) {
	j := newProcessTestJukebox(t)
	req := LoadRequest{HFRepo: "repo/model"}
	start := make(chan struct{})
	results := make(chan error, 2)

	for i := 0; i < 2; i++ {
		go func() {
			<-start
			results <- j.Load(context.Background(), req)
		}()
	}

	close(start)
	err1 := <-results
	err2 := <-results

	successes := 0
	conflicts := 0
	for _, err := range []error{err1, err2} {
		switch {
		case err == nil:
			successes++
		case errors.As(err, new(ErrConflict)):
			conflicts++
		default:
			t.Fatalf("unexpected load result: %v", err)
		}
	}

	if successes != 1 || conflicts != 1 {
		t.Fatalf("expected one success and one conflict, got %d successes and %d conflicts", successes, conflicts)
	}

	if err := j.Offload(); err != nil {
		t.Fatalf("offload failed: %v", err)
	}
	waitForState(t, j, "idle")
}

func TestLoadRejectsStartWhileOffloadInProgress(t *testing.T) {
	j := newSlowOffloadTestJukebox(t, 2*time.Second)
	req := LoadRequest{HFRepo: "repo/model"}

	if err := j.Load(context.Background(), req); err != nil {
		t.Fatalf("load failed: %v", err)
	}
	waitForState(t, j, "running")

	offloadDone := make(chan error, 1)
	go func() {
		offloadDone <- j.Offload()
	}()

	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		j.mu.Lock()
		stopping := j.stopping
		j.mu.Unlock()
		if stopping {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}

	err := j.Load(context.Background(), req)
	if !errors.As(err, new(ErrConflict)) {
		t.Fatalf("expected ErrConflict while offload is in progress, got %v", err)
	}

	if err := <-offloadDone; err != nil {
		t.Fatalf("offload failed: %v", err)
	}
	waitForState(t, j, "idle")

	if err := j.Load(context.Background(), req); err != nil {
		t.Fatalf("expected load to succeed after offload completes, got %v", err)
	}
	if err := j.Offload(); err != nil {
		t.Fatalf("cleanup offload failed: %v", err)
	}
}

func TestLoadFailsFastWhenProcessExitsBeforeReady(t *testing.T) {
	j := newExitBeforeReadyTestJukebox(t)
	start := time.Now()

	err := j.Load(context.Background(), LoadRequest{HFRepo: "repo/model"})
	if err == nil {
		t.Fatal("expected load to fail when process exits before readiness")
	}
	if !strings.Contains(err.Error(), "llama-server exited before becoming ready") {
		t.Fatalf("expected early-exit error, got %v", err)
	}
	if elapsed := time.Since(start); elapsed >= 2*time.Second {
		t.Fatalf("expected early-exit failure before readiness polling timeout, got %v", elapsed)
	}

	waitForState(t, j, "idle")
}

func TestLoadSurfacesExitCodeAndOutputWhenProcessExitsBeforeReady(t *testing.T) {
	j := newFailingExitTestJukebox(t, "fatal: could not load model", 3)

	err := j.Load(context.Background(), LoadRequest{HFRepo: "repo/model"})
	if err == nil {
		t.Fatal("expected load to fail when process exits before readiness")
	}
	msg := err.Error()
	if !strings.Contains(msg, "exit code 3") {
		t.Fatalf("expected exit code in error, got %v", err)
	}
	if !strings.Contains(msg, "fatal: could not load model") {
		t.Fatalf("expected captured stderr in error, got %v", err)
	}

	waitForState(t, j, "idle")
}

func TestMonitorProcessClosesCapturedDoneChannel(t *testing.T) {
	script := filepath.Join(t.TempDir(), "exit-immediately.sh")
	contents := "#!/bin/sh\nexit 0\n"
	if err := os.WriteFile(script, []byte(contents), 0755); err != nil {
		t.Fatalf("failed to write helper script: %v", err)
	}

	cmd := execCommandContextless(t, script)
	if err := cmd.Start(); err != nil {
		t.Fatalf("failed to start helper process: %v", err)
	}

	j := newTestJukebox()
	done := make(chan struct{})
	j.cmd = cmd
	j.processDone = done

	go j.monitorProcess(cmd, done)

	j.mu.Lock()
	j.processDone = make(chan struct{})
	j.mu.Unlock()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("monitorProcess did not close the captured done channel")
	}
}

func TestHandleOffloadMethodNotAllowed(t *testing.T) {
	j := newTestJukebox()

	req := httptest.NewRequest(http.MethodGet, "/offload", nil)
	w := httptest.NewRecorder()

	j.handleOffload(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected status 405, got %d", w.Code)
	}
}

func TestHandleOffloadSuccess(t *testing.T) {
	j := newTestJukebox()

	req := httptest.NewRequest(http.MethodPost, "/offload", nil)
	w := httptest.NewRecorder()

	j.handleOffload(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}
}

func TestHandleLogsMethodNotAllowed(t *testing.T) {
	j := newTestJukebox()

	req := httptest.NewRequest(http.MethodPost, "/logs", nil)
	w := httptest.NewRecorder()

	j.handleLogs(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Fatalf("expected status 405, got %d", w.Code)
	}
}

func TestHandleLogsReturnsHistory(t *testing.T) {
	j := newTestJukebox()

	// Add some log entries
	j.logs.Add("stdout", "loading model...")
	j.logs.Add("stderr", "warning: something")

	req := httptest.NewRequest(http.MethodGet, "/logs", nil)
	w := httptest.NewRecorder()

	// Use a custom response recorder that supports flushing
	fw := &flushRecorder{ResponseRecorder: w}

	// Start the handler with a cancelled context so it returns immediately
	done := make(chan struct{})
	go func() {
		defer close(done)
		j.handleLogs(fw, req.WithContext(newCancellableContext()))
	}()

	// Wait for handler to complete
	<-done

	// Verify headers were set
	if w.Header().Get("Content-Type") != "text/event-stream" {
		t.Errorf("expected content-type text/event-stream, got %q", w.Header().Get("Content-Type"))
	}
}

func TestStreamLogsBroadcastsStoredEntry(t *testing.T) {
	j := newTestJukebox()
	sub := make(chan LogEntry, 1)

	j.subMu.Lock()
	j.subscribers = append(j.subscribers, sub)
	j.subMu.Unlock()
	defer func() {
		j.subMu.Lock()
		j.subscribers = nil
		j.subMu.Unlock()
	}()

	r, w := io.Pipe()
	done := make(chan struct{})
	go func() {
		defer close(done)
		j.streamLogs(r, "stdout")
	}()

	if _, err := w.Write([]byte("hello world\n")); err != nil {
		t.Fatalf("failed to write log line: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("failed to close log writer: %v", err)
	}
	<-done

	entries := j.logs.GetEntries()
	if len(entries) != 1 {
		t.Fatalf("expected 1 stored entry, got %d", len(entries))
	}

	select {
	case entry := <-sub:
		if entry != entries[0] {
			t.Fatalf("expected broadcast entry %+v to match stored entry %+v", entry, entries[0])
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for broadcast log entry")
	}
}

func TestStreamLiveLogsSkipsAlreadySentEntries(t *testing.T) {
	sub := make(chan LogEntry, 3)
	sub <- LogEntry{Type: "stdout", Line: "history", ID: 2}
	sub <- LogEntry{Type: "stdout", Line: "live", ID: 3}
	close(sub)

	w := &flushRecorder{ResponseRecorder: httptest.NewRecorder()}
	streamLiveLogs(context.Background(), w, sub, 2)

	body := w.Body.String()
	if strings.Contains(body, "history") {
		t.Fatalf("expected already-sent history entry to be skipped, got body %q", body)
	}
	if !strings.Contains(body, "live") {
		t.Fatalf("expected live entry to be written, got body %q", body)
	}
}

type flushRecorder struct {
	*httptest.ResponseRecorder
}

func (f *flushRecorder) Flush() {}

func newCancellableContext() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}

func TestWriteJSON(t *testing.T) {
	w := httptest.NewRecorder()
	writeJSON(w, http.StatusOK, map[string]string{"key": "value"})

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	ct := w.Header().Get("Content-Type")
	if ct != "application/json" {
		t.Errorf("expected content-type application/json, got %q", ct)
	}

	var result map[string]string
	if err := json.NewDecoder(w.Body).Decode(&result); err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	if result["key"] != "value" {
		t.Errorf("expected key=value, got %v", result)
	}
}

func TestMustMarshal(t *testing.T) {
	entry := LogEntry{Type: "stdout", Line: "hello", ID: 1}
	s := mustMarshal(entry)

	var decoded LogEntry
	if err := json.Unmarshal([]byte(s), &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if decoded.Type != "stdout" || decoded.Line != "hello" || decoded.ID != 1 {
		t.Errorf("marshaled/unmarshaled mismatch: %+v", decoded)
	}
}

func TestWriteExampleConfig(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "config.json")

	if err := writeExampleConfig(path); err != nil {
		t.Fatalf("writeExampleConfig failed: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read written config: %v", err)
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("written config is not valid JSON: %v", err)
	}

	if cfg.LlamaBinary != "/path/to/your/llama-server" {
		t.Errorf("unexpected llama_binary: %q", cfg.LlamaBinary)
	}
	if cfg.ListenAddr != ":4468" {
		t.Errorf("unexpected listen_addr: %q", cfg.ListenAddr)
	}
}

func TestWriteExampleConfigCreatesParentDir(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "nested", "dir", "config.json")

	if err := writeExampleConfig(path); err != nil {
		t.Fatalf("writeExampleConfig failed: %v", err)
	}

	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Fatal("config file was not created")
	}
}

func execCommandContextless(t *testing.T, script string) *exec.Cmd {
	t.Helper()
	return exec.Command(script)
}

func containsArg(args []string, want string) bool {
	for _, arg := range args {
		if arg == want {
			return true
		}
	}
	return false
}

func TestRequestsEqual(t *testing.T) {
	a := LoadRequest{
		HFRepo:         "repo/model",
		Context:        4096,
		GPULayers:      32,
		FlashAttention: true,
		Parallel:       2,
		Env:            map[string]string{"LLAMA_CACHE": "/tmp"},
		Flags:          map[string]interface{}{"temp": 0.7},
	}
	b := LoadRequest{
		HFRepo:         "repo/model",
		Context:        4096,
		GPULayers:      32,
		FlashAttention: true,
		Parallel:       2,
		Env:            map[string]string{"LLAMA_CACHE": "/tmp"},
		Flags:          map[string]interface{}{"temp": 0.7},
	}

	if !requestsEqual(a, b) {
		t.Fatal("expected identical requests to be equal")
	}

	b.HFRepo = "other/model"
	if requestsEqual(a, b) {
		t.Fatal("expected different HFRepo to not be equal")
	}

	b.HFRepo = "repo/model"
	b.Context = 2048
	if requestsEqual(a, b) {
		t.Fatal("expected different Context to not be equal")
	}

	b.Context = 4096
	b.FlashAttention = false
	if requestsEqual(a, b) {
		t.Fatal("expected different FlashAttention to not be equal")
	}

	b.FlashAttention = true
	b.Env = map[string]string{"LLAMA_CACHE": "/other"}
	if requestsEqual(a, b) {
		t.Fatal("expected different Env to not be equal")
	}
}

func TestLoadSameModelNoOp(t *testing.T) {
	j := newProcessTestJukebox(t)
	req := LoadRequest{HFRepo: "repo/model"}

	if err := j.Load(context.Background(), req); err != nil {
		t.Fatalf("first load failed: %v", err)
	}
	waitForState(t, j, "running")

	pidBefore := j.Status().PID

	if err := j.Load(context.Background(), req); err != nil {
		t.Fatalf("second load with same model should succeed, got: %v", err)
	}

	pidAfter := j.Status().PID
	if pidBefore != pidAfter {
		t.Fatalf("expected same process PID, got %d before and %d after", pidBefore, pidAfter)
	}

	if err := j.Offload(); err != nil {
		t.Fatalf("offload failed: %v", err)
	}
	waitForState(t, j, "idle")
}

func TestLoadDifferentModelSwaps(t *testing.T) {
	tmpDir := t.TempDir()
	loadOrderFile := filepath.Join(tmpDir, "load-order.txt")
	script := filepath.Join(tmpDir, "fake-llama.sh")
	contents := "#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"" + loadOrderFile + "\"\ntrap 'exit 0' TERM INT\nwhile :; do\n  sleep 1\ndone\n"
	if err := os.WriteFile(script, []byte(contents), 0755); err != nil {
		t.Fatalf("failed to write helper script: %v", err)
	}

	host, port := readyEndpoint(t)
	j := NewJukebox(Config{
		LlamaBinary:   script,
		Workdir:       tmpDir,
		LlamaHost:     host,
		LlamaPort:     port,
		LoadTimeout:   5,
		LogBufferSize: 500,
	})

	if err := j.Load(context.Background(), LoadRequest{HFRepo: "repo/model1"}); err != nil {
		t.Fatalf("first load failed: %v", err)
	}
	waitForState(t, j, "running")

	if err := j.Load(context.Background(), LoadRequest{HFRepo: "repo/model2"}); err != nil {
		t.Fatalf("second load with different model failed: %v", err)
	}
	waitForState(t, j, "running")

	time.Sleep(100 * time.Millisecond)

	data, err := os.ReadFile(loadOrderFile)
	if err != nil {
		t.Fatalf("failed to read load order file: %v", err)
	}

	lines := strings.TrimSpace(string(data))
	lineSlice := strings.Split(lines, "\n")

	var hfRepos []string
	for i := 0; i < len(lineSlice)-1; i++ {
		if lineSlice[i] == "-hf" {
			hfRepos = append(hfRepos, lineSlice[i+1])
		}
	}

	if len(hfRepos) < 2 {
		t.Fatalf("expected at least 2 model loads, got %d: %v", len(hfRepos), hfRepos)
	}

	if hfRepos[0] != "repo/model1" {
		t.Errorf("expected first load to be repo/model1, got %q", hfRepos[0])
	}
	if hfRepos[len(hfRepos)-1] != "repo/model2" {
		t.Errorf("expected last load to be repo/model2, got %q", hfRepos[len(hfRepos)-1])
	}

	if err := j.Offload(); err != nil {
		t.Fatalf("offload failed: %v", err)
	}
	waitForState(t, j, "idle")
}

func TestLoadDifferentParamsReloads(t *testing.T) {
	tmpDir := t.TempDir()
	script := writeBlockingProcessScript(t)

	host, port := readyEndpoint(t)
	j := NewJukebox(Config{
		LlamaBinary:   script,
		Workdir:       tmpDir,
		LlamaHost:     host,
		LlamaPort:     port,
		LoadTimeout:   5,
		AllowedFlags:  []string{"c"},
		LogBufferSize: 500,
	})

	req := LoadRequest{HFRepo: "repo/model", Context: 2048}

	if err := j.Load(context.Background(), req); err != nil {
		t.Fatalf("first load failed: %v", err)
	}
	waitForState(t, j, "running")

	pidBefore := j.Status().PID

	req2 := LoadRequest{HFRepo: "repo/model", Context: 4096}
	if err := j.Load(context.Background(), req2); err != nil {
		t.Fatalf("second load with different params failed: %v", err)
	}
	waitForState(t, j, "running")

	pidAfter := j.Status().PID
	if pidBefore == pidAfter {
		t.Fatal("expected different process PID after param change, got same PID")
	}

	if j.Status().Model.HFRepo != "repo/model" {
		t.Fatalf("expected model to be repo/model, got %q", j.Status().Model.HFRepo)
	}

	if err := j.Offload(); err != nil {
		t.Fatalf("offload failed: %v", err)
	}
	waitForState(t, j, "idle")
}

func TestHandleLoadSameModelRespondsSuccess(t *testing.T) {
	j := newProcessTestJukebox(t)
	req := LoadRequest{HFRepo: "repo/model"}

	if err := j.Load(context.Background(), req); err != nil {
		t.Fatalf("first load failed: %v", err)
	}
	waitForState(t, j, "running")

	body := []byte(`{"hf_repo":"repo/model"}`)
	httpReq := httptest.NewRequest(http.MethodPost, "/load", bytes.NewBuffer(body))
	w := httptest.NewRecorder()

	j.handleLoad(w, httpReq)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]string
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp["status"] != "success" {
		t.Fatalf("expected status success, got %q", resp["status"])
	}

	if err := j.Offload(); err != nil {
		t.Fatalf("offload failed: %v", err)
	}
	waitForState(t, j, "idle")
}
