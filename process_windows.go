//go:build windows

package main

import "os"

func terminateProcess(p *os.Process) error {
	return p.Kill()
}

func killProcess(p *os.Process) error {
	return p.Kill()
}

func shutdownSignals() []os.Signal {
	return []os.Signal{os.Interrupt}
}
