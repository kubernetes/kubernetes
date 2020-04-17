/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package transport

import (
	"context"
	"crypto/tls"
	"net"
	"net/http"
)

// Config holds various options for establishing a transport.
type Config struct {
	// UserAgent is an optional field that specifies the caller of this
	// request.
	UserAgent string

	// The base TLS configuration for this transport.
	TLS TLSConfig

	// Username and password for basic authentication
	Username string
	Password string

	// Bearer token for authentication
	BearerToken string

	// Path to a file containing a BearerToken.
	// If set, the contents are periodically read.
	// The last successfully read value takes precedence over BearerToken.
	BearerTokenFile string

	// Impersonate is the config that this Config will impersonate using
	Impersonate ImpersonationConfig

	// DisableCompression bypasses automatic GZip compression requests to the
	// server.
	DisableCompression bool

	// Transport may be used for custom HTTP behavior. This attribute may
	// not be specified with the TLS client certificate options. Use
	// WrapTransport for most client level operations.
	Transport http.RoundTripper

	// WrapTransport will be invoked for custom HTTP behavior after the
	// underlying transport is initialized (either the transport created
	// from TLSClientConfig, Transport, or http.DefaultTransport). The
	// config may layer other RoundTrippers on top of the returned
	// RoundTripper.
	//
	// A future release will change this field to an array. Use config.Wrap()
	// instead of setting this value directly.
	WrapTransport WrapperFunc

	// Dial specifies the dial function for creating unencrypted TCP connections.
	Dial func(ctx context.Context, network, address string) (net.Conn, error)
}

// ImpersonationConfig has all the available impersonation options
type ImpersonationConfig struct {
	// UserName matches user.Info.GetName()
	UserName string
	// Groups matches user.Info.GetGroups()
	Groups []string
	// Extra matches user.Info.GetExtra()
	Extra map[string][]string
}

// HasCA returns whether the configuration has a certificate authority or not.
func (c *Config) HasCA() bool {
	return len(c.TLS.CAData) > 0 || len(c.TLS.CAFile) > 0
}

// HasBasicAuth returns whether the configuration has basic authentication or not.
func (c *Config) HasBasicAuth() bool {
	return len(c.Username) != 0
}

// HasTokenAuth returns whether the configuration has token authentication or not.
func (c *Config) HasTokenAuth() bool {
	return len(c.BearerToken) != 0 || len(c.BearerTokenFile) != 0
}

// HasCertAuth returns whether the configuration has certificate authentication or not.
func (c *Config) HasCertAuth() bool {
	return (len(c.TLS.CertData) != 0 || len(c.TLS.CertFile) != 0) && (len(c.TLS.KeyData) != 0 || len(c.TLS.KeyFile) != 0)
}

// HasCertCallbacks returns whether the configuration has certificate callback or not.
func (c *Config) HasCertCallback() bool {
	return c.TLS.GetCert != nil
}

// Wrap adds a transport middleware function that will give the caller
// an opportunity to wrap the underlying http.RoundTripper prior to the
// first API call being made. The provided function is invoked after any
// existing transport wrappers are invoked.
func (c *Config) Wrap(fn WrapperFunc) {
	c.WrapTransport = Wrappers(c.WrapTransport, fn)
}

// TLSConfig holds the information needed to set up a TLS transport.
type TLSConfig struct {
	CAFile         string // Path of the PEM-encoded server trusted root certificates.
	CertFile       string // Path of the PEM-encoded client certificate.
	KeyFile        string // Path of the PEM-encoded client key.
	ReloadTLSFiles bool   // Set to indicate that the original config provided files, and that they should be reloaded

	Insecure   bool   // Server should be accessed without verifying the certificate. For testing only.
	ServerName string // Override for the server name passed to the server for SNI and used to verify certificates.

	CAData   []byte // Bytes of the PEM-encoded server trusted root certificates. Supercedes CAFile.
	CertData []byte // Bytes of the PEM-encoded client certificate. Supercedes CertFile.
	KeyData  []byte // Bytes of the PEM-encoded client key. Supercedes KeyFile.

	// NextProtos is a list of supported application level protocols, in order of preference.
	// Used to populate tls.Config.NextProtos.
	// To indicate to the server http/1.1 is preferred over http/2, set to ["http/1.1", "h2"] (though the server is free to ignore that preference).
	// To use only http/1.1, set to ["http/1.1"].
	NextProtos []string

	GetCert func() (*tls.Certificate, error) // Callback that returns a TLS client certificate. CertData, CertFile, KeyData and KeyFile supercede this field.
}
