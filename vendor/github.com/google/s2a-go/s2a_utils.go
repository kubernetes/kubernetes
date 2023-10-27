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

package s2a

import (
	"context"
	"errors"

	commonpb "github.com/google/s2a-go/internal/proto/common_go_proto"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/peer"
)

// AuthInfo exposes security information from the S2A to the application.
type AuthInfo interface {
	// AuthType returns the authentication type.
	AuthType() string
	// ApplicationProtocol returns the application protocol, e.g. "grpc".
	ApplicationProtocol() string
	// TLSVersion returns the TLS version negotiated during the handshake.
	TLSVersion() commonpb.TLSVersion
	// Ciphersuite returns the ciphersuite negotiated during the handshake.
	Ciphersuite() commonpb.Ciphersuite
	// PeerIdentity returns the authenticated identity of the peer.
	PeerIdentity() *commonpb.Identity
	// LocalIdentity returns the local identity of the application used during
	// session setup.
	LocalIdentity() *commonpb.Identity
	// PeerCertFingerprint returns the SHA256 hash of the peer certificate used in
	// the S2A handshake.
	PeerCertFingerprint() []byte
	// LocalCertFingerprint returns the SHA256 hash of the local certificate used
	// in the S2A handshake.
	LocalCertFingerprint() []byte
	// IsHandshakeResumed returns true if a cached session was used to resume
	// the handshake.
	IsHandshakeResumed() bool
	// SecurityLevel returns the security level of the connection.
	SecurityLevel() credentials.SecurityLevel
}

// AuthInfoFromPeer extracts the authinfo.S2AAuthInfo object from the given
// peer, if it exists. This API should be used by gRPC clients after
// obtaining a peer object using the grpc.Peer() CallOption.
func AuthInfoFromPeer(p *peer.Peer) (AuthInfo, error) {
	s2aAuthInfo, ok := p.AuthInfo.(AuthInfo)
	if !ok {
		return nil, errors.New("no S2AAuthInfo found in Peer")
	}
	return s2aAuthInfo, nil
}

// AuthInfoFromContext extracts the authinfo.S2AAuthInfo object from the given
// context, if it exists. This API should be used by gRPC server RPC handlers
// to get information about the peer. On the client-side, use the grpc.Peer()
// CallOption and the AuthInfoFromPeer function.
func AuthInfoFromContext(ctx context.Context) (AuthInfo, error) {
	p, ok := peer.FromContext(ctx)
	if !ok {
		return nil, errors.New("no Peer found in Context")
	}
	return AuthInfoFromPeer(p)
}
