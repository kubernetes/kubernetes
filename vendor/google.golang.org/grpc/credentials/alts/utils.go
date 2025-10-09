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

package alts

import (
	"context"
	"errors"
	"strings"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"
)

// AuthInfoFromContext extracts the alts.AuthInfo object from the given context,
// if it exists. This API should be used by gRPC server RPC handlers to get
// information about the communicating peer. For client-side, use grpc.Peer()
// CallOption.
func AuthInfoFromContext(ctx context.Context) (AuthInfo, error) {
	p, ok := peer.FromContext(ctx)
	if !ok {
		return nil, errors.New("no Peer found in Context")
	}
	return AuthInfoFromPeer(p)
}

// AuthInfoFromPeer extracts the alts.AuthInfo object from the given peer, if it
// exists. This API should be used by gRPC clients after obtaining a peer object
// using the grpc.Peer() CallOption.
func AuthInfoFromPeer(p *peer.Peer) (AuthInfo, error) {
	altsAuthInfo, ok := p.AuthInfo.(AuthInfo)
	if !ok {
		return nil, errors.New("no alts.AuthInfo found in Peer")
	}
	return altsAuthInfo, nil
}

// ClientAuthorizationCheck checks whether the client is authorized to access
// the requested resources based on the given expected client service accounts.
// This API should be used by gRPC server RPC handlers. This API should not be
// used by clients.
func ClientAuthorizationCheck(ctx context.Context, expectedServiceAccounts []string) error {
	authInfo, err := AuthInfoFromContext(ctx)
	if err != nil {
		return status.Errorf(codes.PermissionDenied, "The context is not an ALTS-compatible context: %v", err)
	}
	peer := authInfo.PeerServiceAccount()
	for _, sa := range expectedServiceAccounts {
		if strings.EqualFold(peer, sa) {
			return nil
		}
	}
	return status.Errorf(codes.PermissionDenied, "Client %v is not authorized", peer)
}
