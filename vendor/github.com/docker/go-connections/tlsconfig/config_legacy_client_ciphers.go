// +build !go1.5

// Package tlsconfig provides primitives to retrieve secure-enough TLS configurations for both clients and servers.
//
package tlsconfig

import (
	"crypto/tls"
)

// Client TLS cipher suites (dropping CBC ciphers for client preferred suite set)
var clientCipherSuites = []uint16{
	tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,
	tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
}
