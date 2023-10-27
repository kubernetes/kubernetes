/*
 *
 * Copyright 2021 Google LLC
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

// Package s2a provides the S2A transport credentials used by a gRPC
// application.
package s2a

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"sync"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/google/s2a-go/fallback"
	"github.com/google/s2a-go/internal/handshaker"
	"github.com/google/s2a-go/internal/handshaker/service"
	"github.com/google/s2a-go/internal/tokenmanager"
	"github.com/google/s2a-go/internal/v2"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"

	commonpb "github.com/google/s2a-go/internal/proto/common_go_proto"
	s2av2pb "github.com/google/s2a-go/internal/proto/v2/s2a_go_proto"
)

const (
	s2aSecurityProtocol = "tls"
	// defaultTimeout specifies the default server handshake timeout.
	defaultTimeout = 30.0 * time.Second
)

// s2aTransportCreds are the transport credentials required for establishing
// a secure connection using the S2A. They implement the
// credentials.TransportCredentials interface.
type s2aTransportCreds struct {
	info          *credentials.ProtocolInfo
	minTLSVersion commonpb.TLSVersion
	maxTLSVersion commonpb.TLSVersion
	// tlsCiphersuites contains the ciphersuites used in the S2A connection.
	// Note that these are currently unconfigurable.
	tlsCiphersuites []commonpb.Ciphersuite
	// localIdentity should only be used by the client.
	localIdentity *commonpb.Identity
	// localIdentities should only be used by the server.
	localIdentities []*commonpb.Identity
	// targetIdentities should only be used by the client.
	targetIdentities            []*commonpb.Identity
	isClient                    bool
	s2aAddr                     string
	ensureProcessSessionTickets *sync.WaitGroup
}

// NewClientCreds returns a client-side transport credentials object that uses
// the S2A to establish a secure connection with a server.
func NewClientCreds(opts *ClientOptions) (credentials.TransportCredentials, error) {
	if opts == nil {
		return nil, errors.New("nil client options")
	}
	var targetIdentities []*commonpb.Identity
	for _, targetIdentity := range opts.TargetIdentities {
		protoTargetIdentity, err := toProtoIdentity(targetIdentity)
		if err != nil {
			return nil, err
		}
		targetIdentities = append(targetIdentities, protoTargetIdentity)
	}
	localIdentity, err := toProtoIdentity(opts.LocalIdentity)
	if err != nil {
		return nil, err
	}
	if opts.EnableLegacyMode {
		return &s2aTransportCreds{
			info: &credentials.ProtocolInfo{
				SecurityProtocol: s2aSecurityProtocol,
			},
			minTLSVersion: commonpb.TLSVersion_TLS1_3,
			maxTLSVersion: commonpb.TLSVersion_TLS1_3,
			tlsCiphersuites: []commonpb.Ciphersuite{
				commonpb.Ciphersuite_AES_128_GCM_SHA256,
				commonpb.Ciphersuite_AES_256_GCM_SHA384,
				commonpb.Ciphersuite_CHACHA20_POLY1305_SHA256,
			},
			localIdentity:               localIdentity,
			targetIdentities:            targetIdentities,
			isClient:                    true,
			s2aAddr:                     opts.S2AAddress,
			ensureProcessSessionTickets: opts.EnsureProcessSessionTickets,
		}, nil
	}
	verificationMode := getVerificationMode(opts.VerificationMode)
	var fallbackFunc fallback.ClientHandshake
	if opts.FallbackOpts != nil && opts.FallbackOpts.FallbackClientHandshakeFunc != nil {
		fallbackFunc = opts.FallbackOpts.FallbackClientHandshakeFunc
	}
	return v2.NewClientCreds(opts.S2AAddress, localIdentity, verificationMode, fallbackFunc, opts.getS2AStream, opts.serverAuthorizationPolicy)
}

// NewServerCreds returns a server-side transport credentials object that uses
// the S2A to establish a secure connection with a client.
func NewServerCreds(opts *ServerOptions) (credentials.TransportCredentials, error) {
	if opts == nil {
		return nil, errors.New("nil server options")
	}
	var localIdentities []*commonpb.Identity
	for _, localIdentity := range opts.LocalIdentities {
		protoLocalIdentity, err := toProtoIdentity(localIdentity)
		if err != nil {
			return nil, err
		}
		localIdentities = append(localIdentities, protoLocalIdentity)
	}
	if opts.EnableLegacyMode {
		return &s2aTransportCreds{
			info: &credentials.ProtocolInfo{
				SecurityProtocol: s2aSecurityProtocol,
			},
			minTLSVersion: commonpb.TLSVersion_TLS1_3,
			maxTLSVersion: commonpb.TLSVersion_TLS1_3,
			tlsCiphersuites: []commonpb.Ciphersuite{
				commonpb.Ciphersuite_AES_128_GCM_SHA256,
				commonpb.Ciphersuite_AES_256_GCM_SHA384,
				commonpb.Ciphersuite_CHACHA20_POLY1305_SHA256,
			},
			localIdentities: localIdentities,
			isClient:        false,
			s2aAddr:         opts.S2AAddress,
		}, nil
	}
	verificationMode := getVerificationMode(opts.VerificationMode)
	return v2.NewServerCreds(opts.S2AAddress, localIdentities, verificationMode, opts.getS2AStream)
}

// ClientHandshake initiates a client-side TLS handshake using the S2A.
func (c *s2aTransportCreds) ClientHandshake(ctx context.Context, serverAuthority string, rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	if !c.isClient {
		return nil, nil, errors.New("client handshake called using server transport credentials")
	}

	// Connect to the S2A.
	hsConn, err := service.Dial(c.s2aAddr)
	if err != nil {
		grpclog.Infof("Failed to connect to S2A: %v", err)
		return nil, nil, err
	}

	var cancel context.CancelFunc
	ctx, cancel = context.WithCancel(ctx)
	defer cancel()

	opts := &handshaker.ClientHandshakerOptions{
		MinTLSVersion:               c.minTLSVersion,
		MaxTLSVersion:               c.maxTLSVersion,
		TLSCiphersuites:             c.tlsCiphersuites,
		TargetIdentities:            c.targetIdentities,
		LocalIdentity:               c.localIdentity,
		TargetName:                  serverAuthority,
		EnsureProcessSessionTickets: c.ensureProcessSessionTickets,
	}
	chs, err := handshaker.NewClientHandshaker(ctx, hsConn, rawConn, c.s2aAddr, opts)
	if err != nil {
		grpclog.Infof("Call to handshaker.NewClientHandshaker failed: %v", err)
		return nil, nil, err
	}
	defer func() {
		if err != nil {
			if closeErr := chs.Close(); closeErr != nil {
				grpclog.Infof("Close failed unexpectedly: %v", err)
				err = fmt.Errorf("%v: close unexpectedly failed: %v", err, closeErr)
			}
		}
	}()

	secConn, authInfo, err := chs.ClientHandshake(context.Background())
	if err != nil {
		grpclog.Infof("Handshake failed: %v", err)
		return nil, nil, err
	}
	return secConn, authInfo, nil
}

// ServerHandshake initiates a server-side TLS handshake using the S2A.
func (c *s2aTransportCreds) ServerHandshake(rawConn net.Conn) (net.Conn, credentials.AuthInfo, error) {
	if c.isClient {
		return nil, nil, errors.New("server handshake called using client transport credentials")
	}

	// Connect to the S2A.
	hsConn, err := service.Dial(c.s2aAddr)
	if err != nil {
		grpclog.Infof("Failed to connect to S2A: %v", err)
		return nil, nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), defaultTimeout)
	defer cancel()

	opts := &handshaker.ServerHandshakerOptions{
		MinTLSVersion:   c.minTLSVersion,
		MaxTLSVersion:   c.maxTLSVersion,
		TLSCiphersuites: c.tlsCiphersuites,
		LocalIdentities: c.localIdentities,
	}
	shs, err := handshaker.NewServerHandshaker(ctx, hsConn, rawConn, c.s2aAddr, opts)
	if err != nil {
		grpclog.Infof("Call to handshaker.NewServerHandshaker failed: %v", err)
		return nil, nil, err
	}
	defer func() {
		if err != nil {
			if closeErr := shs.Close(); closeErr != nil {
				grpclog.Infof("Close failed unexpectedly: %v", err)
				err = fmt.Errorf("%v: close unexpectedly failed: %v", err, closeErr)
			}
		}
	}()

	secConn, authInfo, err := shs.ServerHandshake(context.Background())
	if err != nil {
		grpclog.Infof("Handshake failed: %v", err)
		return nil, nil, err
	}
	return secConn, authInfo, nil
}

func (c *s2aTransportCreds) Info() credentials.ProtocolInfo {
	return *c.info
}

func (c *s2aTransportCreds) Clone() credentials.TransportCredentials {
	info := *c.info
	var localIdentity *commonpb.Identity
	if c.localIdentity != nil {
		localIdentity = proto.Clone(c.localIdentity).(*commonpb.Identity)
	}
	var localIdentities []*commonpb.Identity
	if c.localIdentities != nil {
		localIdentities = make([]*commonpb.Identity, len(c.localIdentities))
		for i, localIdentity := range c.localIdentities {
			localIdentities[i] = proto.Clone(localIdentity).(*commonpb.Identity)
		}
	}
	var targetIdentities []*commonpb.Identity
	if c.targetIdentities != nil {
		targetIdentities = make([]*commonpb.Identity, len(c.targetIdentities))
		for i, targetIdentity := range c.targetIdentities {
			targetIdentities[i] = proto.Clone(targetIdentity).(*commonpb.Identity)
		}
	}
	return &s2aTransportCreds{
		info:             &info,
		minTLSVersion:    c.minTLSVersion,
		maxTLSVersion:    c.maxTLSVersion,
		tlsCiphersuites:  c.tlsCiphersuites,
		localIdentity:    localIdentity,
		localIdentities:  localIdentities,
		targetIdentities: targetIdentities,
		isClient:         c.isClient,
		s2aAddr:          c.s2aAddr,
	}
}

func (c *s2aTransportCreds) OverrideServerName(serverNameOverride string) error {
	c.info.ServerName = serverNameOverride
	return nil
}

// TLSClientConfigOptions specifies parameters for creating client TLS config.
type TLSClientConfigOptions struct {
	// ServerName is required by s2a as the expected name when verifying the hostname found in server's certificate.
	// 		tlsConfig, _ := factory.Build(ctx, &s2a.TLSClientConfigOptions{
	//			ServerName: "example.com",
	//		})
	ServerName string
}

// TLSClientConfigFactory defines the interface for a client TLS config factory.
type TLSClientConfigFactory interface {
	Build(ctx context.Context, opts *TLSClientConfigOptions) (*tls.Config, error)
}

// NewTLSClientConfigFactory returns an instance of s2aTLSClientConfigFactory.
func NewTLSClientConfigFactory(opts *ClientOptions) (TLSClientConfigFactory, error) {
	if opts == nil {
		return nil, fmt.Errorf("opts must be non-nil")
	}
	if opts.EnableLegacyMode {
		return nil, fmt.Errorf("NewTLSClientConfigFactory only supports S2Av2")
	}
	tokenManager, err := tokenmanager.NewSingleTokenAccessTokenManager()
	if err != nil {
		// The only possible error is: access token not set in the environment,
		// which is okay in environments other than serverless.
		grpclog.Infof("Access token manager not initialized: %v", err)
		return &s2aTLSClientConfigFactory{
			s2av2Address:              opts.S2AAddress,
			tokenManager:              nil,
			verificationMode:          getVerificationMode(opts.VerificationMode),
			serverAuthorizationPolicy: opts.serverAuthorizationPolicy,
		}, nil
	}
	return &s2aTLSClientConfigFactory{
		s2av2Address:              opts.S2AAddress,
		tokenManager:              tokenManager,
		verificationMode:          getVerificationMode(opts.VerificationMode),
		serverAuthorizationPolicy: opts.serverAuthorizationPolicy,
	}, nil
}

type s2aTLSClientConfigFactory struct {
	s2av2Address              string
	tokenManager              tokenmanager.AccessTokenManager
	verificationMode          s2av2pb.ValidatePeerCertificateChainReq_VerificationMode
	serverAuthorizationPolicy []byte
}

func (f *s2aTLSClientConfigFactory) Build(
	ctx context.Context, opts *TLSClientConfigOptions) (*tls.Config, error) {
	serverName := ""
	if opts != nil && opts.ServerName != "" {
		serverName = opts.ServerName
	}
	return v2.NewClientTLSConfig(ctx, f.s2av2Address, f.tokenManager, f.verificationMode, serverName, f.serverAuthorizationPolicy)
}

func getVerificationMode(verificationMode VerificationModeType) s2av2pb.ValidatePeerCertificateChainReq_VerificationMode {
	switch verificationMode {
	case ConnectToGoogle:
		return s2av2pb.ValidatePeerCertificateChainReq_CONNECT_TO_GOOGLE
	case Spiffe:
		return s2av2pb.ValidatePeerCertificateChainReq_SPIFFE
	default:
		return s2av2pb.ValidatePeerCertificateChainReq_UNSPECIFIED
	}
}

// NewS2ADialTLSContextFunc returns a dialer which establishes an MTLS connection using S2A.
// Example use with http.RoundTripper:
//
//		dialTLSContext := s2a.NewS2aDialTLSContextFunc(&s2a.ClientOptions{
//			S2AAddress:         s2aAddress, // required
//		})
//	 	transport := http.DefaultTransport
//	 	transport.DialTLSContext = dialTLSContext
func NewS2ADialTLSContextFunc(opts *ClientOptions) func(ctx context.Context, network, addr string) (net.Conn, error) {

	return func(ctx context.Context, network, addr string) (net.Conn, error) {

		fallback := func(err error) (net.Conn, error) {
			if opts.FallbackOpts != nil && opts.FallbackOpts.FallbackDialer != nil &&
				opts.FallbackOpts.FallbackDialer.Dialer != nil && opts.FallbackOpts.FallbackDialer.ServerAddr != "" {
				fbDialer := opts.FallbackOpts.FallbackDialer
				grpclog.Infof("fall back to dial: %s", fbDialer.ServerAddr)
				fbConn, fbErr := fbDialer.Dialer.DialContext(ctx, network, fbDialer.ServerAddr)
				if fbErr != nil {
					return nil, fmt.Errorf("error fallback to %s: %v; S2A error: %w", fbDialer.ServerAddr, fbErr, err)
				}
				return fbConn, nil
			}
			return nil, err
		}

		factory, err := NewTLSClientConfigFactory(opts)
		if err != nil {
			grpclog.Infof("error creating S2A client config factory: %v", err)
			return fallback(err)
		}

		serverName, _, err := net.SplitHostPort(addr)
		if err != nil {
			serverName = addr
		}
		timeoutCtx, cancel := context.WithTimeout(ctx, v2.GetS2ATimeout())
		defer cancel()
		s2aTLSConfig, err := factory.Build(timeoutCtx, &TLSClientConfigOptions{
			ServerName: serverName,
		})
		if err != nil {
			grpclog.Infof("error building S2A TLS config: %v", err)
			return fallback(err)
		}

		s2aDialer := &tls.Dialer{
			Config: s2aTLSConfig,
		}
		c, err := s2aDialer.DialContext(ctx, network, addr)
		if err != nil {
			grpclog.Infof("error dialing with S2A to %s: %v", addr, err)
			return fallback(err)
		}
		grpclog.Infof("success dialing MTLS to %s with S2A", addr)
		return c, nil
	}
}
