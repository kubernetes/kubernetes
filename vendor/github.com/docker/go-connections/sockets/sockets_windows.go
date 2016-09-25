package sockets

import (
	"net"
	"time"

	"github.com/Microsoft/go-winio"
)

// DialPipe connects to a Windows named pipe.
func DialPipe(addr string, timeout time.Duration) (net.Conn, error) {
	return winio.DialPipe(addr, &timeout)
}
