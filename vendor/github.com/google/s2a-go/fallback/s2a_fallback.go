/*
 *
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package fallback provides default implementations of fallback options when S2A fails.
package fallback

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"

	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
)

const (
	alpnProtoStrH2   = "h2"
	alpnProtoStrHTTP = "http/1.1"
	defaultHTTPSPort = "443"
)

// FallbackTLSConfigGRPC is a tls.Config used by the DefaultFallbackClientHandshakeFunc function.
// It supports GRPC use case, thus the alpn is set to 'h2'.
var FallbackTLSConfigGRPC = tls.Config{
	MinVersion:         tls.VersionTLS13,
	ClientSessionCache: nil,
	NextProtos:         []string{alpnProtoStrH2},
}

// FallbackTLSConfigHTTP is a tls.Config used by the DefaultFallbackDialerAndAddress func.
// It supports the HTTP use case and the alpn is set to both 'http/1.1' and 'h2'.
var FallbackTLSConfigHTTP = tls.Config{
	MinVersion:         tls.VersionTLS13,
	ClientSessionCache: nil,
	NextProtos:         []string{alpnProtoStrH2, alpnProtoStrHTTP},
}

// ClientHandshake establishes a TLS connection and returns it, plus its auth info.
// Inputs:
//
//	targetServer: the server attempted with S2A.
//	conn: the tcp connection to the server at address targetServer that was passed into S2A's ClientHandshake func.
//	            If fallback is successful, the `conn` should be closed.
//	err: the error encountered when performing the client-side TLS handshake with S2A.
type ClientHandshake func(ctx context.Context, targetServer string, conn net.Conn, err error) (net.Conn, credentials.AuthInfo, error)

// DefaultFallbackClientHandshakeFunc returns a ClientHandshake function,
// which establishes a TLS connection to the provided fallbackAddr, returns the new connection and its auth info.
// Example use:
//
//	transportCreds, _ = s2a.NewClientCreds(&s2a.ClientOptions{
//		S2AAddress: s2aAddress,
//		FallbackOpts: &s2a.FallbackOptions{ // optional
//			FallbackClientHandshakeFunc: fallback.DefaultFallbackClientHandshakeFunc(fallbackAddr),
//		},
//	})
//
// The fallback server's certificate must be verifiable using OS root store.
// The fallbackAddr is expected to be a network address, e.g. example.com:port. If port is not specified,
// it uses default port 443.
// In the returned function's TLS config, ClientSessionCache is explicitly set to nil to disable TLS resumption,
// and min TLS version is set to 1.3.
func DefaultFallbackClientHandshakeFunc(fallbackAddr string) (ClientHandshake, error) {
	var fallbackDialer = tls.Dialer{Config: &FallbackTLSConfigGRPC}
	return defaultFallbackClientHandshakeFuncInternal(fallbackAddr, fallbackDialer.DialContext)
}

func defaultFallbackClientHandshakeFuncInternal(fallbackAddr string, dialContextFunc func(context.Context, string, string) (net.Conn, error)) (ClientHandshake, error) {
	fallbackServerAddr, err := processFallbackAddr(fallbackAddr)
	if err != nil {
		if grpclog.V(1) {
			grpclog.Infof("error processing fallback address [%s]: %v", fallbackAddr, err)
		}
		return nil, err
	}
	return func(ctx context.Context, targetServer string, conn net.Conn, s2aErr error) (net.Conn, credentials.AuthInfo, error) {
		fbConn, fbErr := dialContextFunc(ctx, "tcp", fallbackServerAddr)
		if fbErr != nil {
			grpclog.Infof("dialing to fallback server %s failed: %v", fallbackServerAddr, fbErr)
			return nil, nil, fmt.Errorf("dialing to fallback server %s failed: %v; S2A client handshake with %s error: %w", fallbackServerAddr, fbErr, targetServer, s2aErr)
		}

		tc, success := fbConn.(*tls.Conn)
		if !success {
			grpclog.Infof("the connection with fallback server is expected to be tls but isn't")
			return nil, nil, fmt.Errorf("the connection with fallback server is expected to be tls but isn't; S2A client handshake with %s error: %w", targetServer, s2aErr)
		}

		tlsInfo := credentials.TLSInfo{
			State: tc.ConnectionState(),
			CommonAuthInfo: credentials.CommonAuthInfo{
				SecurityLevel: credentials.PrivacyAndIntegrity,
			},
		}
		if grpclog.V(1) {
			grpclog.Infof("ConnectionState.NegotiatedProtocol: %v", tc.ConnectionState().NegotiatedProtocol)
			grpclog.Infof("ConnectionState.HandshakeComplete: %v", tc.ConnectionState().HandshakeComplete)
			grpclog.Infof("ConnectionState.ServerName: %v", tc.ConnectionState().ServerName)
		}
		conn.Close()
		return fbConn, tlsInfo, nil
	}, nil
}

// DefaultFallbackDialerAndAddress returns a TLS dialer and the network address to dial.
// Example use:
//
//	    fallbackDialer, fallbackServerAddr := fallback.DefaultFallbackDialerAndAddress(fallbackAddr)
//		dialTLSContext := s2a.NewS2aDialTLSContextFunc(&s2a.ClientOptions{
//			S2AAddress:         s2aAddress, // required
//			FallbackOpts: &s2a.FallbackOptions{
//				FallbackDialer: &s2a.FallbackDialer{
//					Dialer:     fallbackDialer,
//					ServerAddr: fallbackServerAddr,
//				},
//			},
//	})
//
// The fallback server's certificate should be verifiable using OS root store.
// The fallbackAddr is expected to be a network address, e.g. example.com:port. If port is not specified,
// it uses default port 443.
// In the returned function's TLS config, ClientSessionCache is explicitly set to nil to disable TLS resumption,
// and min TLS version is set to 1.3.
func DefaultFallbackDialerAndAddress(fallbackAddr string) (*tls.Dialer, string, error) {
	fallbackServerAddr, err := processFallbackAddr(fallbackAddr)
	if err != nil {
		if grpclog.V(1) {
			grpclog.Infof("error processing fallback address [%s]: %v", fallbackAddr, err)
		}
		return nil, "", err
	}
	return &tls.Dialer{Config: &FallbackTLSConfigHTTP}, fallbackServerAddr, nil
}

func processFallbackAddr(fallbackAddr string) (string, error) {
	var fallbackServerAddr string
	var err error

	if fallbackAddr == "" {
		return "", fmt.Errorf("empty fallback address")
	}
	_, _, err = net.SplitHostPort(fallbackAddr)
	if err != nil {
		// fallbackAddr does not have port suffix
		fallbackServerAddr = net.JoinHostPort(fallbackAddr, defaultHTTPSPort)
	} else {
		// FallbackServerAddr already has port suffix
		fallbackServerAddr = fallbackAddr
	}
	return fallbackServerAddr, nil
}
