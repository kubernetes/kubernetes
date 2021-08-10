//go:build windows
// +build windows

package transport

import (
	"fmt"
	"syscall"
)

func setReusePort(network, address string, c syscall.RawConn) error {
	return fmt.Errorf("port reuse is not supported on Windows")
}

// Windows supports SO_REUSEADDR, but it may cause undefined behavior, as
// there is no protection against port hijacking.
func setReuseAddress(network, addr string, conn syscall.RawConn) error {
	return fmt.Errorf("address reuse is not supported on Windows")
}
