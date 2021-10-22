/*
 *
 * Copyright 2019 gRPC authors.
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

package client

import (
	"fmt"
	"net"
	"strconv"

	xdspb "github.com/envoyproxy/go-control-plane/envoy/api/v2"
	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
	endpointpb "github.com/envoyproxy/go-control-plane/envoy/api/v2/endpoint"
	typepb "github.com/envoyproxy/go-control-plane/envoy/type"
	"github.com/golang/protobuf/ptypes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/xds/internal"
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
	Address      string
	HealthStatus EndpointHealthStatus
	Weight       uint32
}

// Locality contains information of a locality.
type Locality struct {
	Endpoints []Endpoint
	ID        internal.Locality
	Priority  uint32
	Weight    uint32
}

// EDSUpdate contains an EDS update.
type EDSUpdate struct {
	Drops      []OverloadDropConfig
	Localities []Locality
}

func parseAddress(socketAddress *corepb.SocketAddress) string {
	return net.JoinHostPort(socketAddress.GetAddress(), strconv.Itoa(int(socketAddress.GetPortValue())))
}

func parseDropPolicy(dropPolicy *xdspb.ClusterLoadAssignment_Policy_DropOverload) OverloadDropConfig {
	percentage := dropPolicy.GetDropPercentage()
	var (
		numerator   = percentage.GetNumerator()
		denominator uint32
	)
	switch percentage.GetDenominator() {
	case typepb.FractionalPercent_HUNDRED:
		denominator = 100
	case typepb.FractionalPercent_TEN_THOUSAND:
		denominator = 10000
	case typepb.FractionalPercent_MILLION:
		denominator = 1000000
	}
	return OverloadDropConfig{
		Category:    dropPolicy.GetCategory(),
		Numerator:   numerator,
		Denominator: denominator,
	}
}

func parseEndpoints(lbEndpoints []*endpointpb.LbEndpoint) []Endpoint {
	endpoints := make([]Endpoint, 0, len(lbEndpoints))
	for _, lbEndpoint := range lbEndpoints {
		endpoints = append(endpoints, Endpoint{
			HealthStatus: EndpointHealthStatus(lbEndpoint.GetHealthStatus()),
			Address:      parseAddress(lbEndpoint.GetEndpoint().GetAddress().GetSocketAddress()),
			Weight:       lbEndpoint.GetLoadBalancingWeight().GetValue(),
		})
	}
	return endpoints
}

// ParseEDSRespProto turns EDS response proto message to EDSUpdate.
//
// This is temporarily exported to be used in eds balancer, before it switches
// to use xds client. TODO: unexport.
func ParseEDSRespProto(m *xdspb.ClusterLoadAssignment) (*EDSUpdate, error) {
	ret := &EDSUpdate{}
	for _, dropPolicy := range m.GetPolicy().GetDropOverloads() {
		ret.Drops = append(ret.Drops, parseDropPolicy(dropPolicy))
	}
	priorities := make(map[uint32]struct{})
	for _, locality := range m.Endpoints {
		l := locality.GetLocality()
		if l == nil {
			return nil, fmt.Errorf("EDS response contains a locality without ID, locality: %+v", locality)
		}
		lid := internal.Locality{
			Region:  l.Region,
			Zone:    l.Zone,
			SubZone: l.SubZone,
		}
		priority := locality.GetPriority()
		priorities[priority] = struct{}{}
		ret.Localities = append(ret.Localities, Locality{
			ID:        lid,
			Endpoints: parseEndpoints(locality.GetLbEndpoints()),
			Weight:    locality.GetLoadBalancingWeight().GetValue(),
			Priority:  priority,
		})
	}
	for i := 0; i < len(priorities); i++ {
		if _, ok := priorities[uint32(i)]; !ok {
			return nil, fmt.Errorf("priority %v missing (with different priorities %v received)", i, priorities)
		}
	}
	return ret, nil
}

// ParseEDSRespProtoForTesting parses EDS response, and panic if parsing fails.
// This is used by EDS balancer tests.
//
// TODO: delete this. The EDS balancer should build an EDSUpdate directly,
//  instead of building and parsing a proto message.
func ParseEDSRespProtoForTesting(m *xdspb.ClusterLoadAssignment) *EDSUpdate {
	u, err := ParseEDSRespProto(m)
	if err != nil {
		panic(err.Error())
	}
	return u
}

func (v2c *v2Client) handleEDSResponse(resp *xdspb.DiscoveryResponse) error {
	v2c.mu.Lock()
	defer v2c.mu.Unlock()

	wi := v2c.watchMap[edsURL]
	if wi == nil {
		return fmt.Errorf("xds: no EDS watcher found when handling EDS response: %+v", resp)
	}

	var returnUpdate *EDSUpdate
	for _, r := range resp.GetResources() {
		var resource ptypes.DynamicAny
		if err := ptypes.UnmarshalAny(r, &resource); err != nil {
			return fmt.Errorf("xds: failed to unmarshal resource in EDS response: %v", err)
		}
		cla, ok := resource.Message.(*xdspb.ClusterLoadAssignment)
		if !ok {
			return fmt.Errorf("xds: unexpected resource type: %T in EDS response", resource.Message)
		}

		if cla.GetClusterName() != wi.target[0] {
			grpclog.Warningf("xds: got uninteresting EDS resource, got %s, want %s", cla.GetClusterName(), wi.target[0])
			// We won't validate the remaining resources. If one of the
			// uninteresting ones is invalid, we will still ACK the response.
			continue
		}

		u, err := ParseEDSRespProto(cla)
		if err != nil {
			return err
		}

		returnUpdate = u
		// Break from the loop because the request resource is found. But
		// this also means we won't validate the remaining resources. If one
		// of the uninteresting ones is invalid, we will still ACK the
		// response.
		break
	}

	if returnUpdate != nil {
		wi.stopTimer()
		wi.callback.(edsCallback)(returnUpdate, nil)
	}

	return nil
}
