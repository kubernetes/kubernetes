// +build !linux,!darwin

package tcpfailfast

import (
	"net"
	"time"
)

func ff(tcp *net.TCPConn, timeout time.Duration) error {
	return ErrUnsupported
}
