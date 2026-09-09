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

// Package weight contains utilities to manage endpoint weights.
// Weights may be used by LB policies to distribute load across
// multiple endpoints.
//
// # Experimental
//
// Notice: All APIs in this package are EXPERIMENTAL and may be changed
// or removed in a later release.
package weight

import "google.golang.org/grpc/resolver"

// attributeKey is the type used as the key to store EndpointInfo in the
// Attributes field of resolver.Endpoint.
type attributeKey struct{}

// EndpointInfo will be stored in the Attributes field of Endpoints.
type EndpointInfo struct {
	Weight uint32
}

// Equal allows the values to be compared by Attributes.Equal.
func (a EndpointInfo) Equal(o any) bool {
	oa, ok := o.(EndpointInfo)
	return ok && oa.Weight == a.Weight
}

// Set returns a copy of endpoint in which the Attributes field is
// updated with EndpointInfo.
func Set(endpoint resolver.Endpoint, epInfo EndpointInfo) resolver.Endpoint {
	endpoint.Attributes = endpoint.Attributes.WithValue(attributeKey{}, epInfo)
	return endpoint
}

// FromEndpoint returns the EndpointInfo stored in the Attributes
// field of an endpoint. It returns an empty EndpointInfo if attribute
// is not found.
func FromEndpoint(endpoint resolver.Endpoint) EndpointInfo {
	v := endpoint.Attributes.Value(attributeKey{})
	ei, _ := v.(EndpointInfo)
	return ei
}
