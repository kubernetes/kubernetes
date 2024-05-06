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

// Package handshaker communicates with the S2A handshaker service.
package handshaker

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"sync"

	"github.com/google/s2a-go/internal/authinfo"
	commonpb "github.com/google/s2a-go/internal/proto/common_go_proto"
	s2apb "github.com/google/s2a-go/internal/proto/s2a_go_proto"
	"github.com/google/s2a-go/internal/record"
	"github.com/google/s2a-go/internal/tokenmanager"
	grpc "google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/grpclog"
)

var (
	// appProtocol contains the application protocol accepted by the handshaker.
	appProtocol = "grpc"
	// frameLimit is the maximum size of a frame in bytes.
	frameLimit = 1024 * 64
	// peerNotRespondingError is the error thrown when the peer doesn't respond.
	errPeerNotResponding = errors.New("peer is not responding and re-connection should be attempted")
)

// Handshaker defines a handshaker interface.
type Handshaker interface {
	// ClientHandshake starts and completes a TLS handshake from the client side,
	// and returns a secure connection along with additional auth information.
	ClientHandshake(ctx context.Context) (net.Conn, credentials.AuthInfo, error)
	// ServerHandshake starts and completes a TLS handshake from the server side,
	// and returns a secure connection along with additional auth information.
	ServerHandshake(ctx context.Context) (net.Conn, credentials.AuthInfo, error)
	// Close terminates the Handshaker. It should be called when the handshake
	// is complete.
	Close() error
}

// ClientHandshakerOptions contains the options needed to configure the S2A
// handshaker service on the client-side.
type ClientHandshakerOptions struct {
	// MinTLSVersion specifies the min TLS version supported by the client.
	MinTLSVersion commonpb.TLSVersion
	// MaxTLSVersion specifies the max TLS version supported by the client.
	MaxTLSVersion commonpb.TLSVersion
	// TLSCiphersuites is the ordered list of ciphersuites supported by the
	// client.
	TLSCiphersuites []commonpb.Ciphersuite
	// TargetIdentities contains a list of allowed server identities. One of the
	// target identities should match the peer identity in the handshake
	// result; otherwise, the handshake fails.
	TargetIdentities []*commonpb.Identity
	// LocalIdentity is the local identity of the client application. If none is
	// provided, then the S2A will choose the default identity.
	LocalIdentity *commonpb.Identity
	// TargetName is the allowed server name, which may be used for server
	// authorization check by the S2A if it is provided.
	TargetName string
	// EnsureProcessSessionTickets allows users to wait and ensure that all
	// available session tickets are sent to S2A before a process completes.
	EnsureProcessSessionTickets *sync.WaitGroup
}

// ServerHandshakerOptions contains the options needed to configure the S2A
// handshaker service on the server-side.
type ServerHandshakerOptions struct {
	// MinTLSVersion specifies the min TLS version supported by the server.
	MinTLSVersion commonpb.TLSVersion
	// MaxTLSVersion specifies the max TLS version supported by the server.
	MaxTLSVersion commonpb.TLSVersion
	// TLSCiphersuites is the ordered list of ciphersuites supported by the
	// server.
	TLSCiphersuites []commonpb.Ciphersuite
	// LocalIdentities is the list of local identities that may be assumed by
	// the server. If no local identity is specified, then the S2A chooses a
	// default local identity.
	LocalIdentities []*commonpb.Identity
}

// s2aHandshaker performs a TLS handshake using the S2A handshaker service.
type s2aHandshaker struct {
	// stream is used to communicate with the S2A handshaker service.
	stream s2apb.S2AService_SetUpSessionClient
	// conn is the connection to the peer.
	conn net.Conn
	// clientOpts should be non-nil iff the handshaker is client-side.
	clientOpts *ClientHandshakerOptions
	// serverOpts should be non-nil iff the handshaker is server-side.
	serverOpts *ServerHandshakerOptions
	// isClient determines if the handshaker is client or server side.
	isClient bool
	// hsAddr stores the address of the S2A handshaker service.
	hsAddr string
	// tokenManager manages access tokens for authenticating to S2A.
	tokenManager tokenmanager.AccessTokenManager
	// localIdentities is the set of local identities for whom the
	// tokenManager should fetch a token when preparing a request to be
	// sent to S2A.
	localIdentities []*commonpb.Identity
}

// NewClientHandshaker creates an s2aHandshaker instance that performs a
// client-side TLS handshake using the S2A handshaker service.
func NewClientHandshaker(ctx context.Context, conn *grpc.ClientConn, c net.Conn, hsAddr string, opts *ClientHandshakerOptions) (Handshaker, error) {
	stream, err := s2apb.NewS2AServiceClient(conn).SetUpSession(ctx, grpc.WaitForReady(true))
	if err != nil {
		return nil, err
	}
	tokenManager, err := tokenmanager.NewSingleTokenAccessTokenManager()
	if err != nil {
		grpclog.Infof("failed to create single token access token manager: %v", err)
	}
	return newClientHandshaker(stream, c, hsAddr, opts, tokenManager), nil
}

func newClientHandshaker(stream s2apb.S2AService_SetUpSessionClient, c net.Conn, hsAddr string, opts *ClientHandshakerOptions, tokenManager tokenmanager.AccessTokenManager) *s2aHandshaker {
	var localIdentities []*commonpb.Identity
	if opts != nil {
		localIdentities = []*commonpb.Identity{opts.LocalIdentity}
	}
	return &s2aHandshaker{
		stream:          stream,
		conn:            c,
		clientOpts:      opts,
		isClient:        true,
		hsAddr:          hsAddr,
		tokenManager:    tokenManager,
		localIdentities: localIdentities,
	}
}

// NewServerHandshaker creates an s2aHandshaker instance that performs a
// server-side TLS handshake using the S2A handshaker service.
func NewServerHandshaker(ctx context.Context, conn *grpc.ClientConn, c net.Conn, hsAddr string, opts *ServerHandshakerOptions) (Handshaker, error) {
	stream, err := s2apb.NewS2AServiceClient(conn).SetUpSession(ctx, grpc.WaitForReady(true))
	if err != nil {
		return nil, err
	}
	tokenManager, err := tokenmanager.NewSingleTokenAccessTokenManager()
	if err != nil {
		grpclog.Infof("failed to create single token access token manager: %v", err)
	}
	return newServerHandshaker(stream, c, hsAddr, opts, tokenManager), nil
}

func newServerHandshaker(stream s2apb.S2AService_SetUpSessionClient, c net.Conn, hsAddr string, opts *ServerHandshakerOptions, tokenManager tokenmanager.AccessTokenManager) *s2aHandshaker {
	var localIdentities []*commonpb.Identity
	if opts != nil {
		localIdentities = opts.LocalIdentities
	}
	return &s2aHandshaker{
		stream:          stream,
		conn:            c,
		serverOpts:      opts,
		isClient:        false,
		hsAddr:          hsAddr,
		tokenManager:    tokenManager,
		localIdentities: localIdentities,
	}
}

// ClientHandshake performs a client-side TLS handshake using the S2A handshaker
// service. When complete, returns a TLS connection.
func (h *s2aHandshaker) ClientHandshake(_ context.Context) (net.Conn, credentials.AuthInfo, error) {
	if !h.isClient {
		return nil, nil, errors.New("only handshakers created using NewClientHandshaker can perform a client-side handshake")
	}
	// Extract the hostname from the target name. The target name is assumed to be an authority.
	hostname, _, err := net.SplitHostPort(h.clientOpts.TargetName)
	if err != nil {
		// If the target name had no host port or could not be parsed, use it as is.
		hostname = h.clientOpts.TargetName
	}

	// Prepare a client start message to send to the S2A handshaker service.
	req := &s2apb.SessionReq{
		ReqOneof: &s2apb.SessionReq_ClientStart{
			ClientStart: &s2apb.ClientSessionStartReq{
				ApplicationProtocols: []string{appProtocol},
				MinTlsVersion:        h.clientOpts.MinTLSVersion,
				MaxTlsVersion:        h.clientOpts.MaxTLSVersion,
				TlsCiphersuites:      h.clientOpts.TLSCiphersuites,
				TargetIdentities:     h.clientOpts.TargetIdentities,
				LocalIdentity:        h.clientOpts.LocalIdentity,
				TargetName:           hostname,
			},
		},
		AuthMechanisms: h.getAuthMechanisms(),
	}
	conn, result, err := h.setUpSession(req)
	if err != nil {
		return nil, nil, err
	}
	authInfo, err := authinfo.NewS2AAuthInfo(result)
	if err != nil {
		return nil, nil, err
	}
	return conn, authInfo, nil
}

// ServerHandshake performs a server-side TLS handshake using the S2A handshaker
// service. When complete, returns a TLS connection.
func (h *s2aHandshaker) ServerHandshake(_ context.Context) (net.Conn, credentials.AuthInfo, error) {
	if h.isClient {
		return nil, nil, errors.New("only handshakers created using NewServerHandshaker can perform a server-side handshake")
	}
	p := make([]byte, frameLimit)
	n, err := h.conn.Read(p)
	if err != nil {
		return nil, nil, err
	}
	// Prepare a server start message to send to the S2A handshaker service.
	req := &s2apb.SessionReq{
		ReqOneof: &s2apb.SessionReq_ServerStart{
			ServerStart: &s2apb.ServerSessionStartReq{
				ApplicationProtocols: []string{appProtocol},
				MinTlsVersion:        h.serverOpts.MinTLSVersion,
				MaxTlsVersion:        h.serverOpts.MaxTLSVersion,
				TlsCiphersuites:      h.serverOpts.TLSCiphersuites,
				LocalIdentities:      h.serverOpts.LocalIdentities,
				InBytes:              p[:n],
			},
		},
		AuthMechanisms: h.getAuthMechanisms(),
	}
	conn, result, err := h.setUpSession(req)
	if err != nil {
		return nil, nil, err
	}
	authInfo, err := authinfo.NewS2AAuthInfo(result)
	if err != nil {
		return nil, nil, err
	}
	return conn, authInfo, nil
}

// setUpSession proxies messages between the peer and the S2A handshaker
// service.
func (h *s2aHandshaker) setUpSession(req *s2apb.SessionReq) (net.Conn, *s2apb.SessionResult, error) {
	resp, err := h.accessHandshakerService(req)
	if err != nil {
		return nil, nil, err
	}
	// Check if the returned status is an error.
	if resp.GetStatus() != nil {
		if got, want := resp.GetStatus().Code, uint32(codes.OK); got != want {
			return nil, nil, fmt.Errorf("%v", resp.GetStatus().Details)
		}
	}
	// Calculate the extra unread bytes from the Session. Attempting to consume
	// more than the bytes sent will throw an error.
	var extra []byte
	if req.GetServerStart() != nil {
		if resp.GetBytesConsumed() > uint32(len(req.GetServerStart().GetInBytes())) {
			return nil, nil, errors.New("handshaker service consumed bytes value is out-of-bounds")
		}
		extra = req.GetServerStart().GetInBytes()[resp.GetBytesConsumed():]
	}
	result, extra, err := h.processUntilDone(resp, extra)
	if err != nil {
		return nil, nil, err
	}
	if result.GetLocalIdentity() == nil {
		return nil, nil, errors.New("local identity must be populated in session result")
	}

	// Create a new TLS record protocol using the Session Result.
	newConn, err := record.NewConn(&record.ConnParameters{
		NetConn:                     h.conn,
		Ciphersuite:                 result.GetState().GetTlsCiphersuite(),
		TLSVersion:                  result.GetState().GetTlsVersion(),
		InTrafficSecret:             result.GetState().GetInKey(),
		OutTrafficSecret:            result.GetState().GetOutKey(),
		UnusedBuf:                   extra,
		InSequence:                  result.GetState().GetInSequence(),
		OutSequence:                 result.GetState().GetOutSequence(),
		HSAddr:                      h.hsAddr,
		ConnectionID:                result.GetState().GetConnectionId(),
		LocalIdentity:               result.GetLocalIdentity(),
		EnsureProcessSessionTickets: h.ensureProcessSessionTickets(),
	})
	if err != nil {
		return nil, nil, err
	}
	return newConn, result, nil
}

func (h *s2aHandshaker) ensureProcessSessionTickets() *sync.WaitGroup {
	if h.clientOpts == nil {
		return nil
	}
	return h.clientOpts.EnsureProcessSessionTickets
}

// accessHandshakerService sends the session request to the S2A handshaker
// service and returns the session response.
func (h *s2aHandshaker) accessHandshakerService(req *s2apb.SessionReq) (*s2apb.SessionResp, error) {
	if err := h.stream.Send(req); err != nil {
		return nil, err
	}
	resp, err := h.stream.Recv()
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// processUntilDone continues proxying messages between the peer and the S2A
// handshaker service until the handshaker service returns the SessionResult at
// the end of the handshake or an error occurs.
func (h *s2aHandshaker) processUntilDone(resp *s2apb.SessionResp, unusedBytes []byte) (*s2apb.SessionResult, []byte, error) {
	for {
		if len(resp.OutFrames) > 0 {
			if _, err := h.conn.Write(resp.OutFrames); err != nil {
				return nil, nil, err
			}
		}
		if resp.Result != nil {
			return resp.Result, unusedBytes, nil
		}
		buf := make([]byte, frameLimit)
		n, err := h.conn.Read(buf)
		if err != nil && err != io.EOF {
			return nil, nil, err
		}
		// If there is nothing to send to the handshaker service and nothing is
		// received from the peer, then we are stuck. This covers the case when
		// the peer is not responding. Note that handshaker service connection
		// issues are caught in accessHandshakerService before we even get
		// here.
		if len(resp.OutFrames) == 0 && n == 0 {
			return nil, nil, errPeerNotResponding
		}
		// Append extra bytes from the previous interaction with the handshaker
		// service with the current buffer read from conn.
		p := append(unusedBytes, buf[:n]...)
		// From here on, p and unusedBytes point to the same slice.
		resp, err = h.accessHandshakerService(&s2apb.SessionReq{
			ReqOneof: &s2apb.SessionReq_Next{
				Next: &s2apb.SessionNextReq{
					InBytes: p,
				},
			},
			AuthMechanisms: h.getAuthMechanisms(),
		})
		if err != nil {
			return nil, nil, err
		}

		// Cache the local identity returned by S2A, if it is populated. This
		// overwrites any existing local identities. This is done because, once the
		// S2A has selected a local identity, then only that local identity should
		// be asserted in future requests until the end of the current handshake.
		if resp.GetLocalIdentity() != nil {
			h.localIdentities = []*commonpb.Identity{resp.GetLocalIdentity()}
		}

		// Set unusedBytes based on the handshaker service response.
		if resp.GetBytesConsumed() > uint32(len(p)) {
			return nil, nil, errors.New("handshaker service consumed bytes value is out-of-bounds")
		}
		unusedBytes = p[resp.GetBytesConsumed():]
	}
}

// Close shuts down the handshaker and the stream to the S2A handshaker service
// when the handshake is complete. It should be called when the caller obtains
// the secure connection at the end of the handshake.
func (h *s2aHandshaker) Close() error {
	return h.stream.CloseSend()
}

func (h *s2aHandshaker) getAuthMechanisms() []*s2apb.AuthenticationMechanism {
	if h.tokenManager == nil {
		return nil
	}
	// First handle the special case when no local identities have been provided
	// by the application. In this case, an AuthenticationMechanism with no local
	// identity will be sent.
	if len(h.localIdentities) == 0 {
		token, err := h.tokenManager.DefaultToken()
		if err != nil {
			grpclog.Infof("unable to get token for empty local identity: %v", err)
			return nil
		}
		return []*s2apb.AuthenticationMechanism{
			{
				MechanismOneof: &s2apb.AuthenticationMechanism_Token{
					Token: token,
				},
			},
		}
	}

	// Next, handle the case where the application (or the S2A) has provided
	// one or more local identities.
	var authMechanisms []*s2apb.AuthenticationMechanism
	for _, localIdentity := range h.localIdentities {
		token, err := h.tokenManager.Token(localIdentity)
		if err != nil {
			grpclog.Infof("unable to get token for local identity %v: %v", localIdentity, err)
			continue
		}

		authMechanism := &s2apb.AuthenticationMechanism{
			Identity: localIdentity,
			MechanismOneof: &s2apb.AuthenticationMechanism_Token{
				Token: token,
			},
		}
		authMechanisms = append(authMechanisms, authMechanism)
	}
	return authMechanisms
}
