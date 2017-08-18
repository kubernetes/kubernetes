// Package tcpfailfast allows control over the TCP "user timeout" behavior.
package tcpfailfast

import (
	"errors"
	"net"
	"time"
)

// ErrUnsupported is returned when there is no implementation of the timeout
// on the current platform.
var ErrUnsupported = errors.New("tcp-failfast is unsupported on this platform")

// FailFastTCP instructs the kernel to forcefully close the connection when
// transmitted data remains unacknowledged after the given timeout. This is the
// "user timeout" specified in RFC 793, but bear in mind that not all platforms
// support this option.
func FailFastTCP(tcp *net.TCPConn, timeout time.Duration) error {
	if timeout <= 0 {
		return errors.New("timeout must be > 0")
	}
	return ff(tcp, timeout)
}
