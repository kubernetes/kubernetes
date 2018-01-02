// Package core contains core definitions for the transport package,
// the most salient of which is likely the Identity type. This type is
// used to build a Transport instance.
//
// The TLS configurations provided here are designed for three
// scenarios: mutual authentication for a clients, mutual
// authentication for servers, and a general-purpose server
// configuration applicable where mutual authentication is not
// appropriate.
//
package core

import (
	"time"

	"github.com/cloudflare/cfssl/csr"
)

// A Root stores information about a trusted root.
type Root struct {
	// Type should contain a string identifier for the type.
	Type string `json:"type"`

	// Metadata contains the information needed to load the
	// root(s).
	Metadata map[string]string `json:"metadata"`
}

// Identity is used to store information about a particular transport.
type Identity struct {
	// Request contains metadata for constructing certificate requests.
	Request *csr.CertificateRequest `json:"request"`

	// Roots contains a list of sources for trusted roots.
	Roots []*Root `json:"roots"`

	// ClientRoots contains a list of sources for trusted client
	// certificates.
	ClientRoots []*Root `json:"client_roots"`

	// Profiles contains a dictionary of names to dictionaries;
	// this is intended to allow flexibility in supporting
	// multiple configurations.
	Profiles map[string]map[string]string `json:"profiles"`
}

// DefaultBefore is a sensible default; attempt to regenerate certificates the
// day before they expire.
var DefaultBefore = 24 * time.Hour

// CipherSuites are the TLS cipher suites that should be used by CloudFlare programs.
var CipherSuites = []uint16{
	// These are manually specified because the SHA384 suites are
	// not specified in Go 1.4; in Go 1.4, they won't actually
	// be sent.
	0xc030, // TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384
	0xc02c, // TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
	0xc02f, // TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256
	0xc02b, // TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256
}
