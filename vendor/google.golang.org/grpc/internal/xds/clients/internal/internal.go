/*
 *
 * Copyright 2025 gRPC authors.
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

// Package internal contains helpers for xDS and LRS clients.
package internal

import (
	"google.golang.org/grpc/internal/xds/clients"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/structpb"

	v3corepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
)

// NodeProto returns a protobuf representation of clients.Node n.
//
// This function is intended to be used by the client implementation to convert
// the user-provided Node configuration to its protobuf representation.
func NodeProto(n clients.Node) *v3corepb.Node {
	return &v3corepb.Node{
		Id:      n.ID,
		Cluster: n.Cluster,
		Locality: func() *v3corepb.Locality {
			if isLocalityEmpty(n.Locality) {
				return nil
			}
			return &v3corepb.Locality{
				Region:  n.Locality.Region,
				Zone:    n.Locality.Zone,
				SubZone: n.Locality.SubZone,
			}
		}(),
		Metadata: func() *structpb.Struct {
			if n.Metadata == nil {
				return nil
			}
			if md, ok := n.Metadata.(*structpb.Struct); ok {
				return proto.Clone(md).(*structpb.Struct)
			}
			return nil
		}(),
		UserAgentName:        n.UserAgentName,
		UserAgentVersionType: &v3corepb.Node_UserAgentVersion{UserAgentVersion: n.UserAgentVersion},
	}
}

// isLocalityEqual reports whether clients.Locality l is considered empty.
func isLocalityEmpty(l clients.Locality) bool {
	return isLocalityEqual(l, clients.Locality{})
}

// isLocalityEqual returns true if clients.Locality l1 and l2 are considered
// equal.
func isLocalityEqual(l1, l2 clients.Locality) bool {
	return l1.Region == l2.Region && l1.Zone == l2.Zone && l1.SubZone == l2.SubZone
}
