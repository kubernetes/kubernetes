// +build linux

package tcpfailfast

import (
	"net"
	"time"

	"golang.org/x/sys/unix"
)

func ff(tcp *net.TCPConn, timeout time.Duration) error {
	fd, err := tcp.File()
	if err != nil {
		return err
	}
	const tcpUserTimeout = 0x12
	return unix.SetsockoptInt(int(fd.Fd()), unix.IPPROTO_TCP, tcpUserTimeout, int(timeout/time.Millisecond))
}
