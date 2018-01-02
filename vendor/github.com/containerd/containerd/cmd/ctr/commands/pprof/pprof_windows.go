package pprof

import (
	"net"

	winio "github.com/Microsoft/go-winio"
)

func (d *pprofDialer) pprofDial(proto, addr string) (conn net.Conn, err error) {
	return winio.DialPipe(d.addr, nil)
}

func getPProfDialer(addr string) *pprofDialer {
	return &pprofDialer{"winpipe", addr}
}
