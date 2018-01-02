// +build !linux,!windows

package main

import (
	"os"
	"os/signal"

	"google.golang.org/grpc"

	"github.com/containerd/containerd/reaper"
	runc "github.com/containerd/go-runc"
)

// setupSignals creates a new signal handler for all signals and sets the shim as a
// sub-reaper so that the container processes are reparented
func setupSignals() (chan os.Signal, error) {
	signals := make(chan os.Signal, 2048)
	signal.Notify(signals)
	// make sure runc is setup to use the monitor
	// for waiting on processes
	runc.Monitor = reaper.Default
	return signals, nil
}

func newServer() *grpc.Server {
	return grpc.NewServer()
}
