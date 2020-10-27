/*
 *
 * Copyright 2014 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package credentials

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net"

	"google.golang.org/grpc/credentials/internal"
)

// TLSInfo contains the auth information for a TLS authenticated connection.
// It implements the AuthInfo interface.
type TLSInfo struct {
	State tls.ConnectionState
	CommonAuthInfo
}

// AuthType returns the type of TLSInfo as a string.
func (t TLSInfo) AuthType() string {
	return "tls"
}

// GetSecurityValue returns security info requested by channelz.
func (t TLSInfo) GetSecurityValue() ChannelzSecurityValue {
	v := &TLSChannelzSecurityValue{
		StandardName: cipherSuiteLookup[t.State.CipherSuite],
	}
	// Currently there's no way to get LocalCertificate info from tls package.
	if len(t.State.PeerCertificates) > 0 {
		v.RemoteCertificate = t.State.PeerCertificates[0].Raw
	}
	return v
}

// tlsCreds is the credentials required for authenticating a connection using TLS.
type tlsCreds struct {
	// TLS configuration
	config *tls.Config
}

func (c tlsCreds) Info() ProtocolInfo {
	return ProtocolInfo{
		SecurityProtocol: "tls",
		SecurityVersion:  "1.2",
		ServerName:       c.config.ServerName,
	}
}

func (c *tlsCreds) ClientHandshake(ctx context.Context, authority string, rawConn net.Conn) (_ net.Conn, _ AuthInfo, err error) {
	// use local cfg to avoid clobbering ServerName if using multiple endpoints
	cfg := cloneTLSConfig(c.config)
	if cfg.ServerName == "" {
		serverName, _, err := net.SplitHostPort(authority)
		if err != nil {
			// If the authority had no host port or if the authority cannot be parsed, use it as-is.
			serverName = authority
		}
		cfg.ServerName = serverName
	}
	conn := tls.Client(rawConn, cfg)
	errChannel := make(chan error, 1)
	go func() {
		errChannel <- conn.Handshake()
		close(errChannel)
	}()
	select {
	case err := <-errChannel:
		if err != nil {
			conn.Close()
			return nil, nil, err
		}
	case <-ctx.Done():
		conn.Close()
		return nil, nil, ctx.Err()
	}
	return internal.WrapSyscallConn(rawConn, conn), TLSInfo{conn.ConnectionState(), CommonAuthInfo{PrivacyAndIntegrity}}, nil
}

func (c *tlsCreds) ServerHandshake(rawConn net.Conn) (net.Conn, AuthInfo, error) {
	conn := tls.Server(rawConn, c.config)
	if err := conn.Handshake(); err != nil {
		conn.Close()
		return nil, nil, err
	}
	return internal.WrapSyscallConn(rawConn, conn), TLSInfo{conn.ConnectionState(), CommonAuthInfo{PrivacyAndIntegrity}}, nil
}

func (c *tlsCreds) Clone() TransportCredentials {
	return NewTLS(c.config)
}

func (c *tlsCreds) OverrideServerName(serverNameOverride string) error {
	c.config.ServerName = serverNameOverride
	return nil
}

const alpnProtoStrH2 = "h2"

func appendH2ToNextProtos(ps []string) []string {
	for _, p := range ps {
		if p == alpnProtoStrH2 {
			return ps
		}
	}
	ret := make([]string, 0, len(ps)+1)
	ret = append(ret, ps...)
	return append(ret, alpnProtoStrH2)
}

// NewTLS uses c to construct a TransportCredentials based on TLS.
func NewTLS(c *tls.Config) TransportCredentials {
	tc := &tlsCreds{cloneTLSConfig(c)}
	tc.config.NextProtos = appendH2ToNextProtos(tc.config.NextProtos)
	return tc
}

// NewClientTLSFromCert constructs TLS credentials from the input certificate for client.
// serverNameOverride is for testing only. If set to a non empty string,
// it will override the virtual host name of authority (e.g. :authority header field) in requests.
func NewClientTLSFromCert(cp *x509.CertPool, serverNameOverride string) TransportCredentials {
	return NewTLS(&tls.Config{ServerName: serverNameOverride, RootCAs: cp})
}

// NewClientTLSFromFile constructs TLS credentials from the input certificate file for client.
// serverNameOverride is for testing only. If set to a non empty string,
// it will override the virtual host name of authority (e.g. :authority header field) in requests.
func NewClientTLSFromFile(certFile, serverNameOverride string) (TransportCredentials, error) {
	b, err := ioutil.ReadFile(certFile)
	if err != nil {
		return nil, err
	}
	cp := x509.NewCertPool()
	if !cp.AppendCertsFromPEM(b) {
		return nil, fmt.Errorf("credentials: failed to append certificates")
	}
	return NewTLS(&tls.Config{ServerName: serverNameOverride, RootCAs: cp}), nil
}

// NewServerTLSFromCert constructs TLS credentials from the input certificate for server.
func NewServerTLSFromCert(cert *tls.Certificate) TransportCredentials {
	return NewTLS(&tls.Config{Certificates: []tls.Certificate{*cert}})
}

// NewServerTLSFromFile constructs TLS credentials from the input certificate file and key
// file for server.
func NewServerTLSFromFile(certFile, keyFile string) (TransportCredentials, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, err
	}
	return NewTLS(&tls.Config{Certificates: []tls.Certificate{cert}}), nil
}

// TLSChannelzSecurityValue defines the struct that TLS protocol should return
// from GetSecurityValue(), containing security info like cipher and certificate used.
//
// This API is EXPERIMENTAL.
type TLSChannelzSecurityValue struct {
	ChannelzSecurityValue
	StandardName      string
	LocalCertificate  []byte
	RemoteCertificate []byte
}

var cipherSuiteLookup = map[uint16]string{
	tls.TLS_RSA_WITH_RC4_128_SHA:                "TLS_RSA_WITH_RC4_128_SHA",
	tls.TLS_RSA_WITH_3DES_EDE_CBC_SHA:           "TLS_RSA_WITH_3DES_EDE_CBC_SHA",
	tls.TLS_RSA_WITH_AES_128_CBC_SHA:            "TLS_RSA_WITH_AES_128_CBC_SHA",
	tls.TLS_RSA_WITH_AES_256_CBC_SHA:            "TLS_RSA_WITH_AES_256_CBC_SHA",
	tls.TLS_RSA_WITH_AES_128_GCM_SHA256:         "TLS_RSA_WITH_AES_128_GCM_SHA256",
	tls.TLS_RSA_WITH_AES_256_GCM_SHA384:         "TLS_RSA_WITH_AES_256_GCM_SHA384",
	tls.TLS_ECDHE_ECDSA_WITH_RC4_128_SHA:        "TLS_ECDHE_ECDSA_WITH_RC4_128_SHA",
	tls.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA:    "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA",
	tls.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA:    "TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA",
	tls.TLS_ECDHE_RSA_WITH_RC4_128_SHA:          "TLS_ECDHE_RSA_WITH_RC4_128_SHA",
	tls.TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA:     "TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA",
	tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA:      "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
	tls.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA:      "TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA",
	tls.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256:   "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256",
	tls.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256: "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256",
	tls.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384:   "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384",
	tls.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384: "TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384",
	tls.TLS_FALLBACK_SCSV:                       "TLS_FALLBACK_SCSV",
	tls.TLS_RSA_WITH_AES_128_CBC_SHA256:         "TLS_RSA_WITH_AES_128_CBC_SHA256",
	tls.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256: "TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256",
	tls.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256:   "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256",
	tls.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305:    "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305",
	tls.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305:  "TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305",
}

// cloneTLSConfig returns a shallow clone of the exported
// fields of cfg, ignoring the unexported sync.Once, which
// contains a mutex and must not be copied.
//
// If cfg is nil, a new zero tls.Config is returned.
//
// TODO: inline this function if possible.
func cloneTLSConfig(cfg *tls.Config) *tls.Config {
	if cfg == nil {
		return &tls.Config{}
	}

	return cfg.Clone()
}
