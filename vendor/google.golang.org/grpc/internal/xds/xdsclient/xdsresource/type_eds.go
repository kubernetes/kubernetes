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
	"google.golang.org/grpc/internal/xds/clients"
	"google.golang.org/protobuf/types/known/anypb"
)

// OverloadDropConfig contains the config to drop overloads.
type OverloadDropConfig struct {
	Category    string
	Numerator   uint32
	Denominator uint32
}

// EndpointHealthStatus represents the health status of an endpoint.
type EndpointHealthStatus int32

const (
	// EndpointHealthStatusUnknown represents HealthStatus UNKNOWN.
	EndpointHealthStatusUnknown EndpointHealthStatus = iota
	// EndpointHealthStatusHealthy represents HealthStatus HEALTHY.
	EndpointHealthStatusHealthy
	// EndpointHealthStatusUnhealthy represents HealthStatus UNHEALTHY.
	EndpointHealthStatusUnhealthy
	// EndpointHealthStatusDraining represents HealthStatus DRAINING.
	EndpointHealthStatusDraining
	// EndpointHealthStatusTimeout represents HealthStatus TIMEOUT.
	EndpointHealthStatusTimeout
	// EndpointHealthStatusDegraded represents HealthStatus DEGRADED.
	EndpointHealthStatusDegraded
)

// Endpoint contains information of an endpoint.
type Endpoint struct {
	Addresses    []string
	HealthStatus EndpointHealthStatus
	Weight       uint32
	HashKey      string
	Metadata     map[string]any
}

// Locality contains information of a locality.
type Locality struct {
	Endpoints []Endpoint
	ID        clients.Locality
	Priority  uint32
	Weight    uint32
	Metadata  map[string]any
}

// EndpointsUpdate contains an EDS update.
type EndpointsUpdate struct {
	Drops []OverloadDropConfig
	// Localities in the EDS response with `load_balancing_weight` field not set
	// or explicitly set to 0 are ignored while parsing the resource, and
	// therefore do not show up here.
	Localities []Locality

	// Raw is the resource from the xds response.
	Raw *anypb.Any
}
