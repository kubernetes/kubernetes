// +build windows

package dns

import "net"

type sessionUDP struct {
	raddr *net.UDPAddr
}

// readFromSessionUDP acts just like net.UDPConn.ReadFrom(), but returns a session object instead of a
// net.UDPAddr.
func readFromSessionUDP(conn *net.UDPConn, b []byte) (int, *sessionUDP, error) {
	n, raddr, err := conn.ReadFrom(b)
	if err != nil {
		return n, nil, err
	}
	session := &sessionUDP{raddr.(*net.UDPAddr)}
	return n, session, err
}

// writeToSessionUDP acts just like net.UDPConn.WritetTo(), but uses a *sessionUDP instead of a net.Addr.
func writeToSessionUDP(conn *net.UDPConn, b []byte, session *sessionUDP) (int, error) {
	n, err := conn.WriteTo(b, session.raddr)
	return n, err
}

func (s *sessionUDP) RemoteAddr() net.Addr { return s.raddr }

// setUDPSocketOptions sets the UDP socket options.
// This function is implemented on a per platform basis. See udp_*.go for more details
func setUDPSocketOptions(conn *net.UDPConn) error {
	return nil
}
