package websocket

import "net"

func (nc *netConn) RemoteAddr() net.Addr {
	return websocketAddr{}
}

func (nc *netConn) LocalAddr() net.Addr {
	return websocketAddr{}
}
