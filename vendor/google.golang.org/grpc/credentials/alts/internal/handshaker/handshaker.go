/*
 *
 * Copyright 2018 gRPC authors.
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

// Package handshaker provides ALTS handshaking functionality for GCP.
package handshaker

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"time"

	"golang.org/x/sync/semaphore"
	grpc "google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	core "google.golang.org/grpc/credentials/alts/internal"
	"google.golang.org/grpc/credentials/alts/internal/authinfo"
	"google.golang.org/grpc/credentials/alts/internal/conn"
	altsgrpc "google.golang.org/grpc/credentials/alts/internal/proto/grpc_gcp"
	altspb "google.golang.org/grpc/credentials/alts/internal/proto/grpc_gcp"
	"google.golang.org/grpc/internal/envconfig"
)

const (
	// The maximum byte size of receive frames.
	frameLimit              = 64 * 1024 // 64 KB
	rekeyRecordProtocolName = "ALTSRP_GCM_AES128_REKEY"
)

var (
	hsProtocol      = altspb.HandshakeProtocol_ALTS
	appProtocols    = []string{"grpc"}
	recordProtocols = []string{rekeyRecordProtocolName}
	keyLength       = map[string]int{
		rekeyRecordProtocolName: 44,
	}
	altsRecordFuncs = map[string]conn.ALTSRecordFunc{
		// ALTS handshaker protocols.
		rekeyRecordProtocolName: func(s core.Side, keyData []byte) (conn.ALTSRecordCrypto, error) {
			return conn.NewAES128GCMRekey(s, keyData)
		},
	}
	// control number of concurrent created (but not closed) handshakes.
	clientHandshakes = semaphore.NewWeighted(int64(envconfig.ALTSMaxConcurrentHandshakes))
	serverHandshakes = semaphore.NewWeighted(int64(envconfig.ALTSMaxConcurrentHandshakes))
	// errOutOfBound occurs when the handshake service returns a consumed
	// bytes value larger than the buffer that was passed to it originally.
	errOutOfBound = errors.New("handshaker service consumed bytes value is out-of-bound")
)

func init() {
	for protocol, f := range altsRecordFuncs {
		if err := conn.RegisterProtocol(protocol, f); err != nil {
			panic(err)
		}
	}
}

// ClientHandshakerOptions contains the client handshaker options that can
// provided by the caller.
type ClientHandshakerOptions struct {
	// ClientIdentity is the handshaker client local identity.
	ClientIdentity *altspb.Identity
	// TargetName is the server service account name for secure name
	// checking.
	TargetName string
	// TargetServiceAccounts contains a list of expected target service
	// accounts. One of these accounts should match one of the accounts in
	// the handshaker results. Otherwise, the handshake fails.
	TargetServiceAccounts []string
	// RPCVersions specifies the gRPC versions accepted by the client.
	RPCVersions *altspb.RpcProtocolVersions
	// BoundAccessToken is a bound access token to be sent to the server for authentication.
	BoundAccessToken string
}

// ServerHandshakerOptions contains the server handshaker options that can
// provided by the caller.
type ServerHandshakerOptions struct {
	// RPCVersions specifies the gRPC versions accepted by the server.
	RPCVersions *altspb.RpcProtocolVersions
}

// DefaultClientHandshakerOptions returns the default client handshaker options.
func DefaultClientHandshakerOptions() *ClientHandshakerOptions {
	return &ClientHandshakerOptions{}
}

// DefaultServerHandshakerOptions returns the default client handshaker options.
func DefaultServerHandshakerOptions() *ServerHandshakerOptions {
	return &ServerHandshakerOptions{}
}

// altsHandshaker is used to complete an ALTS handshake between client and
// server. This handshaker talks to the ALTS handshaker service in the metadata
// server.
type altsHandshaker struct {
	// RPC stream used to access the ALTS Handshaker service.
	stream altsgrpc.HandshakerService_DoHandshakeClient
	// the connection to the peer.
	conn net.Conn
	// a virtual connection to the ALTS handshaker service.
	clientConn *grpc.ClientConn
	// client handshake options.
	clientOpts *ClientHandshakerOptions
	// server handshake options.
	serverOpts *ServerHandshakerOptions
	// defines the side doing the handshake, client or server.
	side core.Side
}

// NewClientHandshaker creates a core.Handshaker that performs a client-side
// ALTS handshake by acting as a proxy between the peer and the ALTS handshaker
// service in the metadata server.
func NewClientHandshaker(_ context.Context, conn *grpc.ClientConn, c net.Conn, opts *ClientHandshakerOptions) (core.Handshaker, error) {
	return &altsHandshaker{
		stream:     nil,
		conn:       c,
		clientConn: conn,
		clientOpts: opts,
		side:       core.ClientSide,
	}, nil
}

// NewServerHandshaker creates a core.Handshaker that performs a server-side
// ALTS handshake by acting as a proxy between the peer and the ALTS handshaker
// service in the metadata server.
func NewServerHandshaker(_ context.Context, conn *grpc.ClientConn, c net.Conn, opts *ServerHandshakerOptions) (core.Handshaker, error) {
	return &altsHandshaker{
		stream:     nil,
		conn:       c,
		clientConn: conn,
		serverOpts: opts,
		side:       core.ServerSide,
	}, nil
}

// ClientHandshake starts and completes a client ALTS handshake for GCP. Once
// done, ClientHandshake returns a secure connection.
func (h *altsHandshaker) ClientHandshake(ctx context.Context) (net.Conn, credentials.AuthInfo, error) {
	if err := clientHandshakes.Acquire(ctx, 1); err != nil {
		return nil, nil, err
	}
	defer clientHandshakes.Release(1)

	if h.side != core.ClientSide {
		return nil, nil, errors.New("only handshakers created using NewClientHandshaker can perform a client handshaker")
	}

	// TODO(matthewstevenson88): Change unit tests to use public APIs so
	// that h.stream can unconditionally be set based on h.clientConn.
	if h.stream == nil {
		stream, err := altsgrpc.NewHandshakerServiceClient(h.clientConn).DoHandshake(ctx)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to establish stream to ALTS handshaker service: %v", err)
		}
		h.stream = stream
	}

	// Create target identities from service account list.
	targetIdentities := make([]*altspb.Identity, 0, len(h.clientOpts.TargetServiceAccounts))
	for _, account := range h.clientOpts.TargetServiceAccounts {
		targetIdentities = append(targetIdentities, &altspb.Identity{
			IdentityOneof: &altspb.Identity_ServiceAccount{
				ServiceAccount: account,
			},
		})
	}
	req := &altspb.HandshakerReq{
		ReqOneof: &altspb.HandshakerReq_ClientStart{
			ClientStart: &altspb.StartClientHandshakeReq{
				HandshakeSecurityProtocol: hsProtocol,
				ApplicationProtocols:      appProtocols,
				RecordProtocols:           recordProtocols,
				TargetIdentities:          targetIdentities,
				LocalIdentity:             h.clientOpts.ClientIdentity,
				TargetName:                h.clientOpts.TargetName,
				RpcVersions:               h.clientOpts.RPCVersions,
			},
		},
	}
	if h.clientOpts.BoundAccessToken != "" {
		req.GetClientStart().AccessToken = h.clientOpts.BoundAccessToken
	}
	conn, result, err := h.doHandshake(req)
	if err != nil {
		return nil, nil, err
	}
	authInfo := authinfo.New(result)
	return conn, authInfo, nil
}

// ServerHandshake starts and completes a server ALTS handshake for GCP. Once
// done, ServerHandshake returns a secure connection.
func (h *altsHandshaker) ServerHandshake(ctx context.Context) (net.Conn, credentials.AuthInfo, error) {
	if err := serverHandshakes.Acquire(ctx, 1); err != nil {
		return nil, nil, err
	}
	defer serverHandshakes.Release(1)

	if h.side != core.ServerSide {
		return nil, nil, errors.New("only handshakers created using NewServerHandshaker can perform a server handshaker")
	}

	// TODO(matthewstevenson88): Change unit tests to use public APIs so
	// that h.stream can unconditionally be set based on h.clientConn.
	if h.stream == nil {
		stream, err := altsgrpc.NewHandshakerServiceClient(h.clientConn).DoHandshake(ctx)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to establish stream to ALTS handshaker service: %v", err)
		}
		h.stream = stream
	}

	p := make([]byte, frameLimit)
	n, err := h.conn.Read(p)
	if err != nil {
		return nil, nil, err
	}

	// Prepare server parameters.
	params := make(map[int32]*altspb.ServerHandshakeParameters)
	params[int32(altspb.HandshakeProtocol_ALTS)] = &altspb.ServerHandshakeParameters{
		RecordProtocols: recordProtocols,
	}
	req := &altspb.HandshakerReq{
		ReqOneof: &altspb.HandshakerReq_ServerStart{
			ServerStart: &altspb.StartServerHandshakeReq{
				ApplicationProtocols: appProtocols,
				HandshakeParameters:  params,
				InBytes:              p[:n],
				RpcVersions:          h.serverOpts.RPCVersions,
			},
		},
	}

	conn, result, err := h.doHandshake(req)
	if err != nil {
		return nil, nil, err
	}
	authInfo := authinfo.New(result)
	return conn, authInfo, nil
}

func (h *altsHandshaker) doHandshake(req *altspb.HandshakerReq) (net.Conn, *altspb.HandshakerResult, error) {
	resp, err := h.accessHandshakerService(req)
	if err != nil {
		return nil, nil, err
	}
	// Check of the returned status is an error.
	if resp.GetStatus() != nil {
		if got, want := resp.GetStatus().Code, uint32(codes.OK); got != want {
			return nil, nil, fmt.Errorf("%v", resp.GetStatus().Details)
		}
	}

	var extra []byte
	if req.GetServerStart() != nil {
		if resp.GetBytesConsumed() > uint32(len(req.GetServerStart().GetInBytes())) {
			return nil, nil, errOutOfBound
		}
		extra = req.GetServerStart().GetInBytes()[resp.GetBytesConsumed():]
	}
	result, extra, err := h.processUntilDone(resp, extra)
	if err != nil {
		return nil, nil, err
	}
	// The handshaker returns a 128 bytes key. It should be truncated based
	// on the returned record protocol.
	keyLen, ok := keyLength[result.RecordProtocol]
	if !ok {
		return nil, nil, fmt.Errorf("unknown resulted record protocol %v", result.RecordProtocol)
	}
	sc, err := conn.NewConn(h.conn, h.side, result.GetRecordProtocol(), result.KeyData[:keyLen], extra)
	if err != nil {
		return nil, nil, err
	}
	return sc, result, nil
}

func (h *altsHandshaker) accessHandshakerService(req *altspb.HandshakerReq) (*altspb.HandshakerResp, error) {
	if err := h.stream.Send(req); err != nil {
		return nil, fmt.Errorf("failed to send ALTS handshaker request: %w", err)
	}
	resp, err := h.stream.Recv()
	if err != nil {
		return nil, fmt.Errorf("failed to receive ALTS handshaker response: %w", err)
	}
	return resp, nil
}

// processUntilDone processes the handshake until the handshaker service returns
// the results. Handshaker service takes care of frame parsing, so we read
// whatever received from the network and send it to the handshaker service.
func (h *altsHandshaker) processUntilDone(resp *altspb.HandshakerResp, extra []byte) (*altspb.HandshakerResult, []byte, error) {
	var lastWriteTime time.Time
	buf := make([]byte, frameLimit)
	for {
		if len(resp.OutFrames) > 0 {
			lastWriteTime = time.Now()
			if _, err := h.conn.Write(resp.OutFrames); err != nil {
				return nil, nil, err
			}
		}
		if resp.Result != nil {
			return resp.Result, extra, nil
		}
		n, err := h.conn.Read(buf)
		if err != nil && err != io.EOF {
			return nil, nil, err
		}
		// If there is nothing to send to the handshaker service, and
		// nothing is received from the peer, then we are stuck.
		// This covers the case when the peer is not responding. Note
		// that handshaker service connection issues are caught in
		// accessHandshakerService before we even get here.
		if len(resp.OutFrames) == 0 && n == 0 {
			return nil, nil, core.PeerNotRespondingError
		}
		// Append extra bytes from the previous interaction with the
		// handshaker service with the current buffer read from conn.
		p := append(extra, buf[:n]...)
		// Compute the time elapsed since the last write to the peer.
		timeElapsed := time.Since(lastWriteTime)
		timeElapsedMs := uint32(timeElapsed.Milliseconds())
		// From here on, p and extra point to the same slice.
		resp, err = h.accessHandshakerService(&altspb.HandshakerReq{
			ReqOneof: &altspb.HandshakerReq_Next{
				Next: &altspb.NextHandshakeMessageReq{
					InBytes:          p,
					NetworkLatencyMs: timeElapsedMs,
				},
			},
		})
		if err != nil {
			return nil, nil, err
		}
		// Set extra based on handshaker service response.
		if resp.GetBytesConsumed() > uint32(len(p)) {
			return nil, nil, errOutOfBound
		}
		extra = p[resp.GetBytesConsumed():]
	}
}

// Close terminates the Handshaker. It should be called when the caller obtains
// the secure connection.
func (h *altsHandshaker) Close() {
	if h.stream != nil {
		h.stream.CloseSend()
	}
}

// ResetConcurrentHandshakeSemaphoreForTesting resets the handshake semaphores
// to allow numberOfAllowedHandshakes concurrent handshakes each.
func ResetConcurrentHandshakeSemaphoreForTesting(numberOfAllowedHandshakes int64) {
	clientHandshakes = semaphore.NewWeighted(numberOfAllowedHandshakes)
	serverHandshakes = semaphore.NewWeighted(numberOfAllowedHandshakes)
}
