/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

// Package credentials implements various credentials supported by gRPC library,
// which encapsulate all the state needed by a client to authenticate with a
// server and make various assertions, e.g., about the client's identity, role,
// or whether it is authorized to make a particular call.
package credentials

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net"
	"strings"
	"time"

	"github.com/coreos/etcd/Godeps/_workspace/src/golang.org/x/net/context"
	"github.com/coreos/etcd/Godeps/_workspace/src/golang.org/x/oauth2"
	"github.com/coreos/etcd/Godeps/_workspace/src/golang.org/x/oauth2/google"
	"github.com/coreos/etcd/Godeps/_workspace/src/golang.org/x/oauth2/jwt"
)

var (
	// alpnProtoStr are the specified application level protocols for gRPC.
	alpnProtoStr = []string{"h2-14", "h2-15", "h2-16"}
)

// Credentials defines the common interface all supported credentials must
// implement.
type Credentials interface {
	// GetRequestMetadata gets the current request metadata, refreshing
	// tokens if required. This should be called by the transport layer on
	// each request, and the data should be populated in headers or other
	// context. When supported by the underlying implementation, ctx can
	// be used for timeout and cancellation.
	// TODO(zhaoq): Define the set of the qualified keys instead of leaving
	// it as an arbitrary string.
	GetRequestMetadata(ctx context.Context) (map[string]string, error)
}

// ProtocolInfo provides information regarding the gRPC wire protocol version,
// security protocol, security protocol version in use, etc.
type ProtocolInfo struct {
	// ProtocolVersion is the gRPC wire protocol version.
	ProtocolVersion string
	// SecurityProtocol is the security protocol in use.
	SecurityProtocol string
	// SecurityVersion is the security protocol version.
	SecurityVersion string
}

// TransportAuthenticator defines the common interface for all the live gRPC wire
// protocols and supported transport security protocols (e.g., TLS, SSL).
type TransportAuthenticator interface {
	// ClientHandshake does the authentication handshake specified by the corresponding
	// authentication protocol on rawConn for clients.
	ClientHandshake(addr string, rawConn net.Conn, timeout time.Duration) (net.Conn, error)
	// ServerHandshake does the authentication handshake for servers.
	ServerHandshake(rawConn net.Conn) (net.Conn, error)
	// Info provides the ProtocolInfo of this TransportAuthenticator.
	Info() ProtocolInfo
	Credentials
}

// tlsCreds is the credentials required for authenticating a connection using TLS.
type tlsCreds struct {
	// TLS configuration
	config tls.Config
}

func (c *tlsCreds) Info() ProtocolInfo {
	return ProtocolInfo{
		SecurityProtocol: "tls",
		SecurityVersion:  "1.2",
	}
}

// GetRequestMetadata returns nil, nil since TLS credentials does not have
// metadata.
func (c *tlsCreds) GetRequestMetadata(ctx context.Context) (map[string]string, error) {
	return nil, nil
}

type timeoutError struct{}

func (timeoutError) Error() string   { return "credentials: Dial timed out" }
func (timeoutError) Timeout() bool   { return true }
func (timeoutError) Temporary() bool { return true }

func (c *tlsCreds) ClientHandshake(addr string, rawConn net.Conn, timeout time.Duration) (_ net.Conn, err error) {
	// borrow some code from tls.DialWithDialer
	var errChannel chan error
	if timeout != 0 {
		errChannel = make(chan error, 2)
		time.AfterFunc(timeout, func() {
			errChannel <- timeoutError{}
		})
	}
	if c.config.ServerName == "" {
		colonPos := strings.LastIndex(addr, ":")
		if colonPos == -1 {
			colonPos = len(addr)
		}
		c.config.ServerName = addr[:colonPos]
	}
	conn := tls.Client(rawConn, &c.config)
	if timeout == 0 {
		err = conn.Handshake()
	} else {
		go func() {
			errChannel <- conn.Handshake()
		}()
		err = <-errChannel
	}
	if err != nil {
		rawConn.Close()
		return nil, err
	}
	return conn, nil
}

func (c *tlsCreds) ServerHandshake(rawConn net.Conn) (net.Conn, error) {
	conn := tls.Server(rawConn, &c.config)
	if err := conn.Handshake(); err != nil {
		rawConn.Close()
		return nil, err
	}
	return conn, nil
}

// NewTLS uses c to construct a TransportAuthenticator based on TLS.
func NewTLS(c *tls.Config) TransportAuthenticator {
	tc := &tlsCreds{*c}
	tc.config.NextProtos = alpnProtoStr
	return tc
}

// NewClientTLSFromCert constructs a TLS from the input certificate for client.
func NewClientTLSFromCert(cp *x509.CertPool, serverName string) TransportAuthenticator {
	return NewTLS(&tls.Config{ServerName: serverName, RootCAs: cp})
}

// NewClientTLSFromFile constructs a TLS from the input certificate file for client.
func NewClientTLSFromFile(certFile, serverName string) (TransportAuthenticator, error) {
	b, err := ioutil.ReadFile(certFile)
	if err != nil {
		return nil, err
	}
	cp := x509.NewCertPool()
	if !cp.AppendCertsFromPEM(b) {
		return nil, fmt.Errorf("credentials: failed to append certificates")
	}
	return NewTLS(&tls.Config{ServerName: serverName, RootCAs: cp}), nil
}

// NewServerTLSFromCert constructs a TLS from the input certificate for server.
func NewServerTLSFromCert(cert *tls.Certificate) TransportAuthenticator {
	return NewTLS(&tls.Config{Certificates: []tls.Certificate{*cert}})
}

// NewServerTLSFromFile constructs a TLS from the input certificate file and key
// file for server.
func NewServerTLSFromFile(certFile, keyFile string) (TransportAuthenticator, error) {
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, err
	}
	return NewTLS(&tls.Config{Certificates: []tls.Certificate{cert}}), nil
}

// TokenSource supplies credentials from an oauth2.TokenSource.
type TokenSource struct {
	oauth2.TokenSource
}

// GetRequestMetadata gets the request metadata as a map from a TokenSource.
func (ts TokenSource) GetRequestMetadata(ctx context.Context) (map[string]string, error) {
	token, err := ts.Token()
	if err != nil {
		return nil, err
	}
	return map[string]string{
		"authorization": token.TokenType + " " + token.AccessToken,
	}, nil
}

// NewComputeEngine constructs the credentials that fetches access tokens from
// Google Compute Engine (GCE)'s metadata server. It is only valid to use this
// if your program is running on a GCE instance.
// TODO(dsymonds): Deprecate and remove this.
func NewComputeEngine() Credentials {
	return TokenSource{google.ComputeTokenSource("")}
}

// serviceAccount represents credentials via JWT signing key.
type serviceAccount struct {
	config *jwt.Config
}

func (s serviceAccount) GetRequestMetadata(ctx context.Context) (map[string]string, error) {
	token, err := s.config.TokenSource(ctx).Token()
	if err != nil {
		return nil, err
	}
	return map[string]string{
		"authorization": token.TokenType + " " + token.AccessToken,
	}, nil
}

// NewServiceAccountFromKey constructs the credentials using the JSON key slice
// from a Google Developers service account.
func NewServiceAccountFromKey(jsonKey []byte, scope ...string) (Credentials, error) {
	config, err := google.JWTConfigFromJSON(jsonKey, scope...)
	if err != nil {
		return nil, err
	}
	return serviceAccount{config: config}, nil
}

// NewServiceAccountFromFile constructs the credentials using the JSON key file
// of a Google Developers service account.
func NewServiceAccountFromFile(keyFile string, scope ...string) (Credentials, error) {
	jsonKey, err := ioutil.ReadFile(keyFile)
	if err != nil {
		return nil, fmt.Errorf("credentials: failed to read the service account key file: %v", err)
	}
	return NewServiceAccountFromKey(jsonKey, scope...)
}

// NewApplicationDefault returns "Application Default Credentials". For more
// detail, see https://developers.google.com/accounts/docs/application-default-credentials.
func NewApplicationDefault(ctx context.Context, scope ...string) (Credentials, error) {
	t, err := google.DefaultTokenSource(ctx, scope...)
	if err != nil {
		return nil, err
	}
	return TokenSource{t}, nil
}
