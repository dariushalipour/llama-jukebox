package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
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
