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

// Package authinfo provides authentication and authorization information that
// results from the TLS handshake.
package authinfo

import (
	"errors"

	commonpb "github.com/google/s2a-go/internal/proto/common_go_proto"
	contextpb "github.com/google/s2a-go/internal/proto/s2a_context_go_proto"
	grpcpb "github.com/google/s2a-go/internal/proto/s2a_go_proto"
	"google.golang.org/grpc/credentials"
)

var _ credentials.AuthInfo = (*S2AAuthInfo)(nil)

const s2aAuthType = "s2a"

// S2AAuthInfo exposes authentication and authorization information from the
// S2A session result to the gRPC stack.
type S2AAuthInfo struct {
	s2aContext     *contextpb.S2AContext
	commonAuthInfo credentials.CommonAuthInfo
}

// NewS2AAuthInfo returns a new S2AAuthInfo object from the S2A session result.
func NewS2AAuthInfo(result *grpcpb.SessionResult) (credentials.AuthInfo, error) {
	return newS2AAuthInfo(result)
}

func newS2AAuthInfo(result *grpcpb.SessionResult) (*S2AAuthInfo, error) {
	if result == nil {
		return nil, errors.New("NewS2aAuthInfo given nil session result")
	}
	return &S2AAuthInfo{
		s2aContext: &contextpb.S2AContext{
			ApplicationProtocol:  result.GetApplicationProtocol(),
			TlsVersion:           result.GetState().GetTlsVersion(),
			Ciphersuite:          result.GetState().GetTlsCiphersuite(),
			PeerIdentity:         result.GetPeerIdentity(),
			LocalIdentity:        result.GetLocalIdentity(),
			PeerCertFingerprint:  result.GetPeerCertFingerprint(),
			LocalCertFingerprint: result.GetLocalCertFingerprint(),
			IsHandshakeResumed:   result.GetState().GetIsHandshakeResumed(),
		},
		commonAuthInfo: credentials.CommonAuthInfo{SecurityLevel: credentials.PrivacyAndIntegrity},
	}, nil
}

// AuthType returns the authentication type.
func (s *S2AAuthInfo) AuthType() string {
	return s2aAuthType
}

// ApplicationProtocol returns the application protocol, e.g. "grpc".
func (s *S2AAuthInfo) ApplicationProtocol() string {
	return s.s2aContext.GetApplicationProtocol()
}

// TLSVersion returns the TLS version negotiated during the handshake.
func (s *S2AAuthInfo) TLSVersion() commonpb.TLSVersion {
	return s.s2aContext.GetTlsVersion()
}

// Ciphersuite returns the ciphersuite negotiated during the handshake.
func (s *S2AAuthInfo) Ciphersuite() commonpb.Ciphersuite {
	return s.s2aContext.GetCiphersuite()
}

// PeerIdentity returns the authenticated identity of the peer.
func (s *S2AAuthInfo) PeerIdentity() *commonpb.Identity {
	return s.s2aContext.GetPeerIdentity()
}

// LocalIdentity returns the local identity of the application used during
// session setup.
func (s *S2AAuthInfo) LocalIdentity() *commonpb.Identity {
	return s.s2aContext.GetLocalIdentity()
}

// PeerCertFingerprint returns the SHA256 hash of the peer certificate used in
// the S2A handshake.
func (s *S2AAuthInfo) PeerCertFingerprint() []byte {
	return s.s2aContext.GetPeerCertFingerprint()
}

// LocalCertFingerprint returns the SHA256 hash of the local certificate used
// in the S2A handshake.
func (s *S2AAuthInfo) LocalCertFingerprint() []byte {
	return s.s2aContext.GetLocalCertFingerprint()
}

// IsHandshakeResumed returns true if a cached session was used to resume
// the handshake.
func (s *S2AAuthInfo) IsHandshakeResumed() bool {
	return s.s2aContext.GetIsHandshakeResumed()
}

// SecurityLevel returns the security level of the connection.
func (s *S2AAuthInfo) SecurityLevel() credentials.SecurityLevel {
	return s.commonAuthInfo.SecurityLevel
}
