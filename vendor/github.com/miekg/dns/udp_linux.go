// +build linux

package dns

// See:
// * http://stackoverflow.com/questions/3062205/setting-the-source-ip-for-a-udp-socket and
// * http://blog.powerdns.com/2012/10/08/on-binding-datagram-udp-sockets-to-the-any-addresses/
//
// Why do we need this: When listening on 0.0.0.0 with UDP so kernel decides what is the outgoing
// interface, this might not always be the correct one. This code will make sure the egress
// packet's interface matched the ingress' one.

import (
	"net"
	"syscall"
)

// setUDPSocketOptions4 prepares the v4 socket for sessions.
func setUDPSocketOptions4(conn *net.UDPConn) error {
	file, err := conn.File()
	if err != nil {
		return err
	}
	if err := syscall.SetsockoptInt(int(file.Fd()), syscall.IPPROTO_IP, syscall.IP_PKTINFO, 1); err != nil {
		return err
	}
	// Calling File() above results in the connection becoming blocking, we must fix that.
	// See https://github.com/miekg/dns/issues/279
	err = syscall.SetNonblock(int(file.Fd()), true)
	if err != nil {
		return err
	}
	return nil
}

// setUDPSocketOptions6 prepares the v6 socket for sessions.
func setUDPSocketOptions6(conn *net.UDPConn) error {
	file, err := conn.File()
	if err != nil {
		return err
	}
	if err := syscall.SetsockoptInt(int(file.Fd()), syscall.IPPROTO_IPV6, syscall.IPV6_RECVPKTINFO, 1); err != nil {
		return err
	}
	err = syscall.SetNonblock(int(file.Fd()), true)
	if err != nil {
		return err
	}
	return nil
}

// getUDPSocketOption6Only return true if the socket is v6 only and false when it is v4/v6 combined
// (dualstack).
func getUDPSocketOptions6Only(conn *net.UDPConn) (bool, error) {
	file, err := conn.File()
	if err != nil {
		return false, err
	}
	// dual stack. See http://stackoverflow.com/questions/1618240/how-to-support-both-ipv4-and-ipv6-connections
	v6only, err := syscall.GetsockoptInt(int(file.Fd()), syscall.IPPROTO_IPV6, syscall.IPV6_V6ONLY)
	if err != nil {
		return false, err
	}
	return v6only == 1, nil
}

func getUDPSocketName(conn *net.UDPConn) (syscall.Sockaddr, error) {
	file, err := conn.File()
	if err != nil {
		return nil, err
	}
	return syscall.Getsockname(int(file.Fd()))
}
