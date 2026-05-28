# llama-jukebox

A highly secure, lightweight, and specialized Go-based daemon designed to manage `llama-server` (llama.cpp) instances. It provides a robust remote control interface via a REST API and real-time log streaming, specifically optimized for switching between large-scale LLM models on dedicated hardware.

## 1. Problem Statement

Managing high-performance LLM servers like `llama-server` presents several operational challenges:

1.  **Complexity of Command Lines**: Launching models requires long, complex shell commands with numerous flags (`-ngl`, `-c`, `--spec-type`, etc.) and environment variables (`LLAMA_CACHE`).
2.  **Process Management Friction**: Stopping a running server to switch models requires manual process killing (`pkill`), which is imprecise and can lead to orphaned processes or port conflicts.
3.  **Observability Gaps**: Monitoring the loading process (VRAM allocation, model weights loading) is difficult without constantly tailing terminal outputs.
4.  **Synchronization Issues**: Standard process managers return immediately after the process starts. This creates a race condition where clients attempt to send inference requests before the model is actually loaded into memory.
5.  **Remote Orchestration Complexity**: When the client machine is separate from the inference server, automating model switches via SSH/scripts becomes fragile, cumbersome, and difficult to synchronize with application-level logic.
6.  **Security Risks**: Exposing a server that executes shell commands is extremely dangerous. A naive API could easily be exploited for remote command execution.

## 2. The Solution

**llama-jukebox** acts as an intelligent supervisor daemon that sits between the user (or an orchestration layer) and the `llama-server` process.

### Key Features
*   **Atomic Model Switching**: A single `/load` call manages the entire lifecycle: validation, (optional) cleanup of old processes, starting the new process, and waiting for readiness.
*   **Blocking Readiness Guarantee**: The `/load` endpoint is synchronous. It does not return a success response until the `llama-server` is fully loaded and capable of serving requests.
*   **Strict Security Sandbox**: It uses a "Default Deny" security model. No arbitrary commands are executed; only whitelisted flags and environment variables are permitted.
*   **Real-time Observability**: Provides a dedicated Server-Sent Events (SSE) endpoint to stream `stdout` and `stderr`, allowing users to watch the model loading progress in real-time.
*   **Zero Dependencies**: Written in pure Go using only the standard library for maximum portability and minimal attack surface.

---

## 3. Getting Started

### Prerequisites
* Go 1.21+
* A compiled `llama-server` binary (from [llama.cpp](https://github.com/ggml-org/llama.cpp))

### Installation

Install the latest release directly into your `$GOPATH/bin`:
```bash
go install github.com/dariushalipour/llama-jukebox@latest
```

Or build from source:
```bash
go build -o llama-jukebox .
```

### Configuration

Set the `LLAMA_JUKEBOX_CONFIG` environment variable to tell the daemon where to find its config file. Use the `init` subcommand to generate a template:

```bash
LLAMA_JUKEBOX_CONFIG=/etc/llama-jukebox/config.json ./llama-jukebox init
```

This creates the config file (and any parent directories) with sensible defaults. Edit it with your `llama-server` binary path, network settings, and allowed flags.

### Run
```bash
# Config path is required via LLAMA_JUKEBOX_CONFIG:
LLAMA_JUKEBOX_CONFIG=/etc/llama-jukebox/config.json ./llama-jukebox
```

### Quick API Walkthrough

**Check status:**
```bash
curl http://localhost:4468/status
```

**Load a model (blocks until ready):**

The typical workflow is to offload any running model, then load the new one. The `/load` endpoint blocks until the model is fully loaded and ready to serve.

```bash
# Offload the current model (if any)
curl -X POST http://localhost:4468/offload

# Watch logs in real-time (new terminal)
curl -N http://localhost:4468/logs

# Load a new model (blocks until ready)
curl -X POST http://localhost:4468/load \
  -H "Content-Type: application/json" \
  -d '{...}'
```

**Example: Switch to a different model:**
```bash
curl -X POST http://localhost:4468/offload && curl -X POST http://localhost:4468/load \
  -H "Content-Type: application/json" \
  -d '{...}'
```

**Example models:**

```bash
# unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL
curl -X POST http://localhost:4468/load \
  -H "Content-Type: application/json" \
  -d '{
    "hf_repo": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL",
    "context": 260000,
    "gpu_layers": 99,
    "flash_attention": true,
    "parallel": 1,
    "env": { "LLAMA_CACHE": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF" },
    "flags": { "spec-type": "draft-mtp", "spec-draft-n-max": 2 }
  }'

# unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q5_K_XL
curl -X POST http://localhost:4468/load \
  -H "Content-Type: application/json" \
  -d '{
    "hf_repo": "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q5_K_XL",
    "context": 128000,
    "gpu_layers": 99,
    "flash_attention": true,
    "parallel": 1,
    "env": { "LLAMA_CACHE": "unsloth/Qwen3.6-27B-MTP-GGUF" },
    "flags": { "spec-type": "draft-mtp", "spec-draft-n-max": 2 }
  }'

# unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL
curl -X POST http://localhost:4468/load \
  -H "Content-Type: application/json" \
  -d '{
    "hf_repo": "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q6_K_XL",
    "context": 260000,
    "gpu_layers": 99,
    "flash_attention": true,
    "parallel": 1,
    "env": { "LLAMA_CACHE": "unsloth/gemma-4-26B-A4B-it-GGUF" },
    "flags": { "temp": 1.0, "top-p": 0.95, "top-k": 64 }
  }'

# unsloth/gemma-4-31B-it-GGUF:UD-Q5_K_XL
curl -X POST http://localhost:4468/load \
  -H "Content-Type: application/json" \
  -d '{
    "hf_repo": "unsloth/gemma-4-31B-it-GGUF:UD-Q5_K_XL",
    "context": 131072,
    "gpu_layers": 99,
    "flash_attention": true,
    "parallel": 1,
    "env": { "LLAMA_CACHE": "unsloth/gemma-4-31B-it-GGUF" },
    "flags": { "temp": 1.0, "top-p": 0.95, "top-k": 64 }
  }'
```

---

## 4. Technical Specifications

### 4.1 Configuration (`config.json`)

The `config.json` file is the single source of truth for the daemon's security boundaries and operational defaults. **It cannot be overridden by API requests.**

#### Schema & Fields

| Field | Type | Description |
| :--- | :--- | :--- |
| `llama_binary` | `string` | **Mandatory.** Absolute path to the `llama-server` executable. |
| `workdir` | `string` | **Mandatory.** The directory where the process will be executed. |
| `llama_host` | `string` | **Mandatory.** The IP/Hostname the `llama-server` will bind to. |
| `llama_port` | `int` | **Mandatory.** The port the `llama-server` will listen on. |
| `listen_addr` | `string` | (Optional, default `:4468`). The address the daemon listens on. |
| `allowed_flags` | `[]string` | A list of permitted command-line flags (e.g., `["temp", "top-p", "spec-type"]`). |
| `allowed_env` | `[]string` | A list of permitted environment variable keys (e.g., `["LLAMA_CACHE"]`). |
| `load_timeout` | `int` | (Optional) Max seconds to wait for the model to load before failing. |
| `log_buffer_size`| `int` | (Optional) Number of log lines to keep in the in-memory ring buffer. |

### 4.2 Security Architecture

The daemon implements a multi-layered defense strategy:

1.  **No Shell Execution**: The daemon uses `os/exec.Command` to invoke the binary directly. It never invokes `/bin/sh` or `/bin/bash`, rendering shell metacharacter injection (like `; rm -rf /`) ineffective at the process level.
2.  **Parameter Whitelisting**: 
    *   **Flags**: Only keys present in `allowed_flags` are accepted in the `/load` body.
    *   **Environment**: Only keys present in `allowed_env` are accepted.
3.  **Strict String Validation**: Every string input (model names, paths, flag values) is scanned for shell-sensitive characters: `` ` $ ( ) [ ] { } | ; & < > ! `` and whitespace. Any such character triggers an immediate `400 Bad Request`.
4.  **Fixed Infrastructure**: The network binding (`host`/`port`) and the executable path are locked in `config.json`. An attacker cannot use the API to redirect the server to a different port or execute a different binary.

### 4.3 Process Lifecycle & Readiness Detection

#### The `/load` Workflow
1.  **Request Received**: Client sends a `POST /load` with the desired model configuration.
2.  **Validation**: The daemon verifies all flags and env vars against `config.json` and checks string safety. Empty `hf_repo` is rejected.
3.  **Conflict Check**: If a `llama-server` is already running, the daemon returns `409 Conflict`.
4.  **Execution**: `llama-server` is launched as a child process using the fixed `llama_binary` and `workdir`.
5.  **Process Monitor**: A dedicated goroutine calls `cmd.Wait()` to detect if the process exits on its own (OOM, crash, segfault). If it does, the zombie is reaped and state is automatically reset to `idle`.
6.  **Active Polling (The "Ready" Signal)**:
    *   The daemon enters a loop.
    *   It attempts to perform a lightweight HTTP request (e.g., `GET /`) to the `llama-server`'s internal `host:port`.
    *   **Success**: If the server returns `200 OK`, the model is considered "fully loaded and idle."
    *   **Failure**: If the connection is refused or returns a non-200 status (indicating the server is still initializing), the daemon waits and retries.
    *   **External Offload**: If `/offload` is called during this polling, the request is aborted immediately via a signaling channel.
    *   **Client Cancel**: If the HTTP client disconnects, the context cancels, triggering the `CommandContext` to terminate the child process.
7.  **Completion**: Once ready, the `/load` request returns `{"status": "success"}`.

#### The `/offload` Workflow
1.  **Signal**: Sends `SIGTERM` to the running child process.
2.  **Wait**: Monitors the process exit.
3.  **Escalation**: If the process does not exit within a 5-second grace period, it sends `SIGKILL`.

### 4.4 HTTP Method Enforcement

All endpoints enforce strict HTTP method requirements:
- `/status`: `GET` only
- `/load`: `POST` only
- `/offload`: `POST` only
- `/logs`: `GET` only

Any other method receives a `405 Method Not Allowed` JSON response.

---

## 5. API Reference

### `GET /status`
Returns the current operational state of the daemon.
**Response (JSON):**
```json
{
  "state": "idle | running",
  "pid": 1234,
  "model": {
    "hf_repo": "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q5_K_XL",
    "started_at": "2026-05-29T10:00:00Z"
  }
}
```

### `POST /load`
Triggers the loading of a model. **This call is blocking.**
**Request Body (JSON):**
```json
{
  "hf_repo": "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q5_K_XL",
  "context": 128000,
  "gpu_layers": 99,
  "flash_attention": true,
  "parallel": 1,
  "env": {
    "LLAMA_CACHE": "unsloth/Qwen3.6-27B-MTP-GGUF"
  },
  "flags": {
    "spec-type": "draft-mtp",
    "spec-draft-n-max": 2
  }
}
```
**Success Response**: `200 OK` with `{"status": "success"}`.
**Error Responses**:
- `400 Bad Request` — invalid JSON, unsafe characters, non-whitelisted flags/env vars, or model was offloaded during loading.
- `409 Conflict` — a model is already running.
- `500 Internal Server Error` — timeout waiting for readiness, failed to start process, or model crashed during loading.

### `POST /offload`
Stops the current model.
**Success Response**: `200 OK`.

### `GET /logs`
A streaming endpoint for real-time observation.
**Protocol**: Server-Sent Events (SSE).
**Format**:
```text
event: log
data: {"type": "stdout", "line": "llama_server: loading model...", "id": 101}

event: log
data: {"type": "stderr", "line": "error: failed to allocate VRAM", "id": 102}
```
*On connect, clients immediately receive the most recent log history from the in-memory buffer, followed by a continuous stream of live log entries.*

**Usage with cURL:**
```bash
# Raw SSE stream (stays open until Ctrl+C)
curl -N http://localhost:4468/logs

# Parse JSON data lines only
curl -N -s http://localhost:4468/logs | grep '^data: ' | sed 's/^data: //'

# Pretty-print with jq
curl -N -s http://localhost:4468/logs | grep '^data: ' | sed 's/^data: //' | jq .

# Filter stderr only
curl -N -s http://localhost:4468/logs | grep '^data: ' | sed 's/^data: //' | jq -r 'select(.type=="stderr") | .line'
```
*The `-N` flag disables cURL buffering for real-time output. Press Ctrl+C to disconnect.*

---

## 6. Implementation Guidelines for Developers

1.  **Concurrency**: Use `context.Context` throughout to ensure that if the daemon is shut down, the `llama-server` child process is immediately and gracefully terminated.
2.  **Single Wait Contract**: Only `monitorProcess` calls `cmd.Wait()`. `Offload` signals termination via `SIGTERM`/`SIGKILL` and waits on a `processExited` channel that `monitorProcess` writes to. This prevents double-wait races and zombie processes.
3.  **Atomic State Transitions**: All mutations to `j.cmd`, `j.currentMod`, `j.offloadSig`, and `j.processExited` happen under `j.mu` to prevent races between `Load` and `Offload`.
4.  **Log Buffering**: Implement a thread-safe ring buffer for logs. Use a `sync.RWMutex` to allow multiple concurrent SSE clients to read the history without blocking the main log-writing loop.
5.  **Error Codes**: Client validation errors return `400 Bad Request`. Server-side failures (timeout, crash) return `500 Internal Server Error`.
6.  **No External Dependencies**: Strictly use the Go Standard Library to ensure the binary remains small and highly auditable.
