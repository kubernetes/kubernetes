// +build windows

package dns

import "net"

// SessionUDP holds the remote address
type SessionUDP struct {
	raddr *net.UDPAddr
}

// RemoteAddr returns the remote network address.
func (s *SessionUDP) RemoteAddr() net.Addr { return s.raddr }

// ReadFromSessionUDP acts just like net.UDPConn.ReadFrom(), but returns a session object instead of a
// net.UDPAddr.
// TODO(fastest963): Once go1.10 is released, use ReadMsgUDP.
func ReadFromSessionUDP(conn *net.UDPConn, b []byte) (int, *SessionUDP, error) {
	n, raddr, err := conn.ReadFrom(b)
	if err != nil {
		return n, nil, err
	}
	return n, &SessionUDP{raddr.(*net.UDPAddr)}, err
}

// WriteToSessionUDP acts just like net.UDPConn.WriteTo(), but uses a *SessionUDP instead of a net.Addr.
// TODO(fastest963): Once go1.10 is released, use WriteMsgUDP.
func WriteToSessionUDP(conn *net.UDPConn, b []byte, session *SessionUDP) (int, error) {
	return conn.WriteTo(b, session.raddr)
}

// TODO(fastest963): Once go1.10 is released and we can use *MsgUDP methods
// use the standard method in udp.go for these.
func setUDPSocketOptions(*net.UDPConn) error { return nil }
func parseDstFromOOB([]byte, net.IP) net.IP  { return nil }
