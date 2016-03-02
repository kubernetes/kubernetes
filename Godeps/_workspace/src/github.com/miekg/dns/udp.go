// +build !windows

package dns

import (
	"net"
	"syscall"
)

type sessionUDP struct {
	raddr   *net.UDPAddr
	context []byte
}

func (s *sessionUDP) RemoteAddr() net.Addr { return s.raddr }

// setUDPSocketOptions sets the UDP socket options.
// This function is implemented on a per platform basis. See udp_*.go for more details
func setUDPSocketOptions(conn *net.UDPConn) error {
	sa, err := getUDPSocketName(conn)
	if err != nil {
		return err
	}
	switch sa.(type) {
	case *syscall.SockaddrInet6:
		v6only, err := getUDPSocketOptions6Only(conn)
		if err != nil {
			return err
		}
		setUDPSocketOptions6(conn)
		if !v6only {
			setUDPSocketOptions4(conn)
		}
	case *syscall.SockaddrInet4:
		setUDPSocketOptions4(conn)
	}
	return nil
}

// readFromSessionUDP acts just like net.UDPConn.ReadFrom(), but returns a session object instead of a
// net.UDPAddr.
func readFromSessionUDP(conn *net.UDPConn, b []byte) (int, *sessionUDP, error) {
	oob := make([]byte, 40)
	n, oobn, _, raddr, err := conn.ReadMsgUDP(b, oob)
	if err != nil {
		return n, nil, err
	}
	return n, &sessionUDP{raddr, oob[:oobn]}, err
}

// writeToSessionUDP acts just like net.UDPConn.WritetTo(), but uses a *sessionUDP instead of a net.Addr.
func writeToSessionUDP(conn *net.UDPConn, b []byte, session *sessionUDP) (int, error) {
	n, _, err := conn.WriteMsgUDP(b, session.context, session.raddr)
	return n, err
}
