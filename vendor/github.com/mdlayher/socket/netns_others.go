//go:build !linux
// +build !linux

package socket

import (
	"fmt"
	"runtime"
)

// withNetNS returns an error on non-Linux systems.
func withNetNS(_ int, _ func() (*Conn, error)) (*Conn, error) {
	return nil, fmt.Errorf("socket: Linux network namespace support is not available on %s", runtime.GOOS)
}
