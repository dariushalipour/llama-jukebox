//go:build !windows

package main

import (
	"os"
	"syscall"
)

func terminateProcess(p *os.Process) error {
	return p.Signal(syscall.SIGTERM)
}

func killProcess(p *os.Process) error {
	return p.Signal(syscall.SIGKILL)
}

func shutdownSignals() []os.Signal {
	return []os.Signal{os.Interrupt, syscall.SIGTERM}
}
