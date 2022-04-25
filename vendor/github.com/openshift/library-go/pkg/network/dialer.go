package network

import (
	"context"
	"net"
)

type DialContext func(ctx context.Context, network, address string) (net.Conn, error)

// DefaultDialContext returns a DialContext function from a network dialer with default options sets.
func DefaultClientDialContext() DialContext {
	return dialerWithDefaultOptions()
}
