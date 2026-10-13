//go:build darwin
// +build darwin

package socket

const (
	// These operating systems do not support CLOEXEC and NONBLOCK socket
	// options.
	flagCLOEXEC = false
	socketFlags = 0
)
