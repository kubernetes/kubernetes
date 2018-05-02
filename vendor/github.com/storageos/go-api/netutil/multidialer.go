package netutil

import (
	"context"
	"math/rand"
	"net"
	"time"
)

var DefaultDialPort = "5705"

func init() {
	rand.Seed(time.Now().UnixNano())
}

// Dialer is an interface that matches *net.Dialer. The intention is to allow either the stdlib
// dialer or a custom implementation to be passed to the MultiDialer constructor. This also makes
// the component easier to test.
type Dialer interface {
	DialContext(context.Context, string, string) (net.Conn, error)
}

// MultiDialer is a custom net Dialer (to be used in a net.Transport field) that attemps to dial
// out to any (potentialy many) of a set of pre-defined addresses. The intended use of this
// function is to extend the functionality of the stdlib http.Client to transparently support
// requests to any member of a given storageos cluster.
type MultiDialer struct {
	Addresses []string
	Dialer    *net.Dialer
}

// NewMultiDialer returns a new MultiDialer instance, configured to dial out to the given set of
// nodes. Nodes can be provided using a URL format (e.g. http://google.com:80), or a host-port pair
// (e.g. localhost:4567).
//
// If a port number is omitted, the value of DefaultDialPort is used.
// Given hostnames are resolved to IP addresses, and IP addresses are used verbatim.
//
// If called with a non-nil dialer, the MultiDialer instance will use this for internall dial
// requests. If this value is nil, the function will initialise one with sane defaults.
func NewMultiDialer(nodes []string, dialer *net.Dialer) (*MultiDialer, error) {
	// If a dialer is not provided, initialise one with sane defaults
	if dialer == nil {
		dialer = &net.Dialer{
			Timeout:   5 * time.Second,
			KeepAlive: 5 * time.Second,
		}
	}

	addrs, err := addrsFromNodes(nodes)
	if err != nil {
		return nil, err
	}

	return &MultiDialer{
		Addresses: addrs,
		Dialer:    dialer,
	}, nil
}

// DialContext will dial each of the MultiDialer's internal addresses in a random order until one
// successfully returns a connection, it has run out of addresses (returning ErrAllFailed), or the
// given context has been closed.
//
// Due to the intrinsic behaviour of this function, any address passed to this function will be
// ignored.
func (m *MultiDialer) DialContext(ctx context.Context, network, ignoredAddress string) (net.Conn, error) {
	if len(m.Addresses) == 0 {
		return nil, newInvalidNodeError(errNoAddresses)
	}

	// Shuffle a copy of the addresses (for even load balancing)
	addrs := make([]string, len(m.Addresses))
	copy(addrs, m.Addresses)

	// Fisher–Yates shuffle algorithm
	for i := len(addrs) - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		addrs[i], addrs[j] = addrs[j], addrs[i]
	}

	// Try to dial each of these addresses in turn, or return on closed context
	for _, addr := range addrs {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()

		default:
			// Create new child context for a single dial
			dctx, cancel := context.WithTimeout(ctx, time.Second)
			defer cancel()

			conn, err := m.Dialer.DialContext(dctx, network, addr)
			if err != nil {
				continue
			}

			return conn, nil
		}
	}

	// We failed to dail all of the addresses we have
	return nil, errAllFailed(m.Addresses)
}

// Dial returns the result of a call to m.DialContext passing in the background context
func (m *MultiDialer) Dial(network, addr string) (net.Conn, error) {
	return m.DialContext(context.Background(), network, addr)
}
