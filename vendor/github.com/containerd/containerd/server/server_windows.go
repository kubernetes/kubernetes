// +build windows

package server

import (
	"context"
	"os"
	"path/filepath"
)

var (
	// DefaultRootDir is the default location used by containerd to store
	// persistent data
	DefaultRootDir = filepath.Join(os.Getenv("programfiles"), "containerd", "root")
	// DefaultStateDir is the default location used by containerd to store
	// transient data
	DefaultStateDir = filepath.Join(os.Getenv("programfiles"), "containerd", "state")
)

const (
	// DefaultAddress is the default winpipe address
	DefaultAddress = `\\.\pipe\containerd-containerd`
	// DefaultDebugAddress is the default winpipe address for pprof data
	DefaultDebugAddress = `\\.\pipe\containerd-debug`
)

func apply(_ context.Context, _ *Config) error {
	return nil
}
