//+build !windows

package dbus

import (
	"errors"
	"io/ioutil"
	"net"
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
	b, err := ioutil.ReadFile(noncefile)
	if err != nil {
		return nil, err
	}
	_, err = socket.Write(b)
	if err != nil {
		return nil, err
	}
	return NewConn(socket)
}
