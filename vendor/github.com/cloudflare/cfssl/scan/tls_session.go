package scan

import "github.com/cloudflare/cfssl/scan/crypto/tls"

// TLSSession contains tests of host TLS Session Resumption via
// Session Tickets and Session IDs
var TLSSession = &Family{
	Description: "Scans host's implementation of TLS session resumption using session tickets/session IDs",
	Scanners: map[string]*Scanner{
		"SessionResume": {
			"Host is able to resume sessions across all addresses",
			sessionResumeScan,
		},
	},
}

// SessionResumeScan tests that host is able to resume sessions across all addresses.
func sessionResumeScan(addr, hostname string) (grade Grade, output Output, err error) {
	config := defaultTLSConfig(hostname)
	config.ClientSessionCache = tls.NewLRUClientSessionCache(1)

	conn, err := tls.DialWithDialer(Dialer, Network, addr, config)
	if err != nil {
		return
	}
	if err = conn.Close(); err != nil {
		return
	}

	return multiscan(addr, func(addrport string) (g Grade, o Output, e error) {
		var conn *tls.Conn
		if conn, e = tls.DialWithDialer(Dialer, Network, addrport, config); e != nil {
			return
		}
		conn.Close()

		if o = conn.ConnectionState().DidResume; o.(bool) {
			g = Good
		}
		return
	})
}
