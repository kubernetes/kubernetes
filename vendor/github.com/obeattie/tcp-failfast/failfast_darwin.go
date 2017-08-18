// +build darwin

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
	return unix.SetsockoptInt(int(fd.Fd()), unix.IPPROTO_TCP, unix.TCP_RXT_CONNDROPTIME, int(timeout/time.Second))
}
