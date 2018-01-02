package server

import "context"

const (
	// DefaultAddress is the default unix socket address
	DefaultAddress = "/var/run/containerd/containerd.sock"
	// DefaultDebugAddress is the default unix socket address for pprof data
	DefaultDebugAddress = "/var/run/containerd/debug.sock"
)

func apply(_ context.Context, _ *Config) error {
	return nil
}
