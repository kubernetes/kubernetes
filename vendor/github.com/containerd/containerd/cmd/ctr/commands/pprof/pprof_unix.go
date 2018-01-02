// +build !windows

package pprof

import "net"

func (d *pprofDialer) pprofDial(proto, addr string) (conn net.Conn, err error) {
	return net.Dial(d.proto, d.addr)
}

func getPProfDialer(addr string) *pprofDialer {
	return &pprofDialer{"unix", addr}
}
