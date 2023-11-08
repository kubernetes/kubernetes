package transport

import (
	"context"
	"crypto/tls"
	"net"
	"strings"

	utilnet "k8s.io/apimachinery/pkg/util/net"
)

type tlsDialer struct {
	dial      utilnet.DialFunc
	tlsConfig *tls.Config
}

// TODO: pass a function that will return the updated RootCAs
func newTLSDialerFor(dial utilnet.DialFunc, tlsConfig *tls.Config) *tlsDialer {
	return &tlsDialer{dial: dial, tlsConfig: tlsConfig}
}

func (d *tlsDialer) DialContext(ctx context.Context, network, addr string) (net.Conn, error) {
	rawConn, err := d.dial(ctx, network, addr)
	if err != nil {
		return nil, err
	}

	// TODO: swap the tlsConfig.RootCAs on dial if required
	tlsConfigCopy := d.tlsConfig.Clone()

	colonPos := strings.LastIndex(addr, ":")
	if colonPos == -1 {
		colonPos = len(addr)
	}
	hostname := addr[:colonPos]

	// this seems to be mandatory
	// if no ServerName is set, infer the ServerName
	// from the hostname we're connecting to.
	if tlsConfigCopy.ServerName == "" {
		tlsConfigCopy.ServerName = hostname
	}

	// TODO: do we have to set some custom timeouts ?
	tlsConn := tls.Client(rawConn, tlsConfigCopy)
	if err := tlsConn.HandshakeContext(ctx); err != nil {
		rawConn.Close()
		return nil, err
	}

	return tlsConn, nil
}
