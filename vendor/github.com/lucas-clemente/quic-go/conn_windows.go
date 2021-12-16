//go:build windows
// +build windows

package quic

import (
	"errors"
	"fmt"
	"net"
	"syscall"

	"golang.org/x/sys/windows"
)

const (
	disablePathMTUDiscovery = true
	IP_DONTFRAGMENT         = 14
)

func newConn(c OOBCapablePacketConn) (connection, error) {
	rawConn, err := c.SyscallConn()
	if err != nil {
		return nil, fmt.Errorf("couldn't get syscall.RawConn: %w", err)
	}
	if err := rawConn.Control(func(fd uintptr) {
		// This should succeed if the connection is a IPv4 or a dual-stack connection.
		// It will fail for IPv6 connections.
		// TODO: properly handle error.
		_ = windows.SetsockoptInt(windows.Handle(fd), windows.IPPROTO_IP, IP_DONTFRAGMENT, 1)
	}); err != nil {
		return nil, err
	}
	return &basicConn{PacketConn: c}, nil
}

func inspectReadBuffer(c net.PacketConn) (int, error) {
	conn, ok := c.(interface {
		SyscallConn() (syscall.RawConn, error)
	})
	if !ok {
		return 0, errors.New("doesn't have a SyscallConn")
	}
	rawConn, err := conn.SyscallConn()
	if err != nil {
		return 0, fmt.Errorf("couldn't get syscall.RawConn: %w", err)
	}
	var size int
	var serr error
	if err := rawConn.Control(func(fd uintptr) {
		size, serr = windows.GetsockoptInt(windows.Handle(fd), windows.SOL_SOCKET, windows.SO_RCVBUF)
	}); err != nil {
		return 0, err
	}
	return size, serr
}

func (i *packetInfo) OOB() []byte { return nil }
