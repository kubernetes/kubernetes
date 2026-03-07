//go:build linux
// +build linux

package socket

import "golang.org/x/sys/unix"

// setReadBuffer wraps the SO_RCVBUF{,FORCE} setsockopt(2) options.
func (c *Conn) setReadBuffer(bytes int) error {
	err := c.SetsockoptInt(unix.SOL_SOCKET, unix.SO_RCVBUFFORCE, bytes)
	if err != nil {
		err = c.SetsockoptInt(unix.SOL_SOCKET, unix.SO_RCVBUF, bytes)
	}
	return err
}

// setWriteBuffer wraps the SO_SNDBUF{,FORCE} setsockopt(2) options.
func (c *Conn) setWriteBuffer(bytes int) error {
	err := c.SetsockoptInt(unix.SOL_SOCKET, unix.SO_SNDBUFFORCE, bytes)
	if err != nil {
		err = c.SetsockoptInt(unix.SOL_SOCKET, unix.SO_SNDBUF, bytes)
	}
	return err
}
