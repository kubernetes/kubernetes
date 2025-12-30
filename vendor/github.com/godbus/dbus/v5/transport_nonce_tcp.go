//go:build !windows
// +build !windows

package dbus

import (
	"errors"
	"net"
	"os"
)

func init() {
	transports["nonce-tcp"] = newNonceTcpTransport
}

func newNonceTcpTransport(keys string) (transport, error) {
	host := getKey(keys, "host")
	port := getKey(keys, "port")
	noncefile := getKey(keys, "noncefile")
	if host == "" || port == "" || noncefile == "" {
		return nil, errors.New("dbus: unsupported address (must set host, port and noncefile)")
	}
	protocol, err := tcpFamily(keys)
	if err != nil {
		return nil, err
	}
	socket, err := net.Dial(protocol, net.JoinHostPort(host, port))
	if err != nil {
		return nil, err
	}
	b, err := os.ReadFile(noncefile)
	if err != nil {
		socket.Close()
		return nil, err
	}
	_, err = socket.Write(b)
	if err != nil {
		socket.Close()
		return nil, err
	}
	return NewConn(socket)
}
