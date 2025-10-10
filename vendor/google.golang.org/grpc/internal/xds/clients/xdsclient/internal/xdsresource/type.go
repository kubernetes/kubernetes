/*
 *
 * Copyright 2021 gRPC authors.
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
 */

package xdsresource

import (
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"

	v3discoverypb "github.com/envoyproxy/go-control-plane/envoy/service/discovery/v3"
)

// UpdateMetadata contains the metadata for each update, including timestamp,
// raw message, and so on.
type UpdateMetadata struct {
	// Status is the status of this resource, e.g. ACKed, NACKed, or
	// Not_exist(removed).
	Status ServiceStatus
	// Version is the version of the xds response. Note that this is the version
	// of the resource in use (previous ACKed). If a response is NACKed, the
	// NACKed version is in ErrState.
	Version string
	// Timestamp is when the response is received.
	Timestamp time.Time
	// ErrState is set when the update is NACKed.
	ErrState *UpdateErrorMetadata
}

// IsListenerResource returns true if the provider URL corresponds to an xDS
// Listener resource.
func IsListenerResource(url string) bool {
	return url == V3ListenerURL
}

// IsHTTPConnManagerResource returns true if the provider URL corresponds to an xDS
// HTTPConnManager resource.
func IsHTTPConnManagerResource(url string) bool {
	return url == V3HTTPConnManagerURL
}

// UnwrapResource unwraps and returns the inner resource if it's in a resource
// wrapper. The original resource is returned if it's not wrapped.
func UnwrapResource(r *anypb.Any) (*anypb.Any, error) {
	url := r.GetTypeUrl()
	if url != V3ResourceWrapperURL {
		// Not wrapped.
		return r, nil
	}
	inner := &v3discoverypb.Resource{}
	if err := proto.Unmarshal(r.GetValue(), inner); err != nil {
		return nil, err
	}
	return inner.Resource, nil
}

// ServiceStatus is the status of the update.
type ServiceStatus int

const (
	// ServiceStatusUnknown is the default state, before a watch is started for
	// the resource.
	ServiceStatusUnknown ServiceStatus = iota
	// ServiceStatusRequested is when the watch is started, but before and
	// response is received.
	ServiceStatusRequested
	// ServiceStatusNotExist is when the resource doesn't exist in
	// state-of-the-world responses (e.g. LDS and CDS), which means the resource
	// is removed by the management server.
	ServiceStatusNotExist // Resource is removed in the server, in LDS/CDS.
	// ServiceStatusACKed is when the resource is ACKed.
	ServiceStatusACKed
	// ServiceStatusNACKed is when the resource is NACKed.
	ServiceStatusNACKed
)

// UpdateErrorMetadata is part of UpdateMetadata. It contains the error state
// when a response is NACKed.
type UpdateErrorMetadata struct {
	// Version is the version of the NACKed response.
	Version string
	// Err contains why the response was NACKed.
	Err error
	// Timestamp is when the NACKed response was received.
	Timestamp time.Time
}
