package subscriber

import (
	"net"

	"github.com/influxdata/influxdb/coordinator"
)

// UDP supports writing points over UDP using the line protocol.
type UDP struct {
	addr string
}

// NewUDP returns a new UDP listener with default options.
func NewUDP(addr string) *UDP {
	return &UDP{addr: addr}
}

// WritePoints writes points over UDP transport.
func (u *UDP) WritePoints(p *coordinator.WritePointsRequest) (err error) {
	var addr *net.UDPAddr
	var con *net.UDPConn
	addr, err = net.ResolveUDPAddr("udp", u.addr)
	if err != nil {
		return
	}

	con, err = net.DialUDP("udp", nil, addr)
	if err != nil {
		return
	}
	defer con.Close()

	for _, p := range p.Points {
		_, err = con.Write([]byte(p.String()))
		if err != nil {
			return
		}

	}
	return
}
