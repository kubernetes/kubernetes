//go:build !linux
// +build !linux

package socket

import "golang.org/x/sys/unix"

// setReadBuffer wraps the SO_RCVBUF setsockopt(2) option.
func (c *Conn) setReadBuffer(bytes int) error {
	return c.SetsockoptInt(unix.SOL_SOCKET, unix.SO_RCVBUF, bytes)
}

// setWriteBuffer wraps the SO_SNDBUF setsockopt(2) option.
func (c *Conn) setWriteBuffer(bytes int) error {
	return c.SetsockoptInt(unix.SOL_SOCKET, unix.SO_SNDBUF, bytes)
}
