package swag

import (
	"net"
	"strconv"
)

// SplitHostPort splits a network address into a host and a port.
// The port is -1 when there is no port to be found
func SplitHostPort(addr string) (host string, port int, err error) {
	h, p, err := net.SplitHostPort(addr)
	if err != nil {
		return "", -1, err
	}
	if p == "" {
		return "", -1, &net.AddrError{Err: "missing port in address", Addr: addr}
	}

	pi, err := strconv.Atoi(p)
	if err != nil {
		return "", -1, err
	}
	return h, pi, nil
}
