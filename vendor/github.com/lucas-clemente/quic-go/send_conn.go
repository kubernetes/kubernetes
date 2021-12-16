package quic

import (
	"net"
)

// A sendConn allows sending using a simple Write() on a non-connected packet conn.
type sendConn interface {
	Write([]byte) error
	Close() error
	LocalAddr() net.Addr
	RemoteAddr() net.Addr
}

type sconn struct {
	connection

	remoteAddr net.Addr
	info       *packetInfo
	oob        []byte
}

var _ sendConn = &sconn{}

func newSendConn(c connection, remote net.Addr, info *packetInfo) sendConn {
	return &sconn{
		connection: c,
		remoteAddr: remote,
		info:       info,
		oob:        info.OOB(),
	}
}

func (c *sconn) Write(p []byte) error {
	_, err := c.WritePacket(p, c.remoteAddr, c.oob)
	return err
}

func (c *sconn) RemoteAddr() net.Addr {
	return c.remoteAddr
}

func (c *sconn) LocalAddr() net.Addr {
	addr := c.connection.LocalAddr()
	if c.info != nil {
		if udpAddr, ok := addr.(*net.UDPAddr); ok {
			addrCopy := *udpAddr
			addrCopy.IP = c.info.addr
			addr = &addrCopy
		}
	}
	return addr
}

type spconn struct {
	net.PacketConn

	remoteAddr net.Addr
}

var _ sendConn = &spconn{}

func newSendPconn(c net.PacketConn, remote net.Addr) sendConn {
	return &spconn{PacketConn: c, remoteAddr: remote}
}

func (c *spconn) Write(p []byte) error {
	_, err := c.WriteTo(p, c.remoteAddr)
	return err
}

func (c *spconn) RemoteAddr() net.Addr {
	return c.remoteAddr
}
