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
	"fmt"
	"math"
	"net"
	"strconv"

	v3corepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	v3endpointpb "github.com/envoyproxy/go-control-plane/envoy/config/endpoint/v3"
	v3typepb "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc/internal/envconfig"
	"google.golang.org/grpc/internal/pretty"
	xdsinternal "google.golang.org/grpc/internal/xds"
	"google.golang.org/grpc/internal/xds/clients"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

func unmarshalEndpointsResource(r *anypb.Any) (string, EndpointsUpdate, error) {
	r, err := UnwrapResource(r)
	if err != nil {
		return "", EndpointsUpdate{}, fmt.Errorf("failed to unwrap resource: %v", err)
	}

	if !IsEndpointsResource(r.GetTypeUrl()) {
		return "", EndpointsUpdate{}, fmt.Errorf("unexpected resource type: %q ", r.GetTypeUrl())
	}

	cla := &v3endpointpb.ClusterLoadAssignment{}
	if err := proto.Unmarshal(r.GetValue(), cla); err != nil {
		return "", EndpointsUpdate{}, fmt.Errorf("failed to unmarshal resource: %v", err)
	}

	u, err := parseEDSRespProto(cla)
	if err != nil {
		return cla.GetClusterName(), EndpointsUpdate{}, err
	}
	u.Raw = r
	return cla.GetClusterName(), u, nil
}

func parseAddress(socketAddress *v3corepb.SocketAddress) string {
	return net.JoinHostPort(socketAddress.GetAddress(), strconv.Itoa(int(socketAddress.GetPortValue())))
}

func parseDropPolicy(dropPolicy *v3endpointpb.ClusterLoadAssignment_Policy_DropOverload) OverloadDropConfig {
	percentage := dropPolicy.GetDropPercentage()
	var (
		numerator   = percentage.GetNumerator()
		denominator uint32
	)
	switch percentage.GetDenominator() {
	case v3typepb.FractionalPercent_HUNDRED:
		denominator = 100
	case v3typepb.FractionalPercent_TEN_THOUSAND:
		denominator = 10000
	case v3typepb.FractionalPercent_MILLION:
		denominator = 1000000
	}
	return OverloadDropConfig{
		Category:    dropPolicy.GetCategory(),
		Numerator:   numerator,
		Denominator: denominator,
	}
}

func parseEndpoints(lbEndpoints []*v3endpointpb.LbEndpoint, uniqueEndpointAddrs map[string]bool) ([]Endpoint, error) {
	endpoints := make([]Endpoint, 0, len(lbEndpoints))
	for _, lbEndpoint := range lbEndpoints {
		// If the load_balancing_weight field is specified, it must be set to a
		// value of at least 1.  If unspecified, each host is presumed to have
		// equal weight in a locality.
		weight := uint32(1)
		if w := lbEndpoint.GetLoadBalancingWeight(); w != nil {
			if w.GetValue() == 0 {
				return nil, fmt.Errorf("EDS response contains an endpoint with zero weight: %+v", lbEndpoint)
			}
			weight = w.GetValue()
		}
		addrs := []string{parseAddress(lbEndpoint.GetEndpoint().GetAddress().GetSocketAddress())}
		if envconfig.XDSDualstackEndpointsEnabled {
			for _, sa := range lbEndpoint.GetEndpoint().GetAdditionalAddresses() {
				addrs = append(addrs, parseAddress(sa.GetAddress().GetSocketAddress()))
			}
		}

		for _, a := range addrs {
			if uniqueEndpointAddrs[a] {
				return nil, fmt.Errorf("duplicate endpoint with the same address %s", a)
			}
			uniqueEndpointAddrs[a] = true
		}

		var endpointMetadata map[string]any
		var hashKey string
		if envconfig.XDSHTTPConnectEnabled || !envconfig.XDSEndpointHashKeyBackwardCompat {
			var err error
			endpointMetadata, err = validateAndConstructMetadata(lbEndpoint.GetMetadata())
			if err != nil {
				return nil, err
			}

			// "The xDS resolver, described in A74, will be changed to set the hash_key
			// endpoint attribute to the value of LbEndpoint.Metadata envoy.lb hash_key
			// field, as described in Envoy's documentation for the ring hash load
			// balancer." - A76
			if !envconfig.XDSEndpointHashKeyBackwardCompat {
				hashKey = hashKeyFromMetadata(endpointMetadata)
			}
		}
		endpoints = append(endpoints, Endpoint{
			HealthStatus: EndpointHealthStatus(lbEndpoint.GetHealthStatus()),
			Addresses:    addrs,
			Weight:       weight,
			HashKey:      hashKey,
			Metadata:     endpointMetadata,
		})
	}
	return endpoints, nil
}

// hashKey extracts and returns the hash key from the given endpoint metadata.
// If no hash key is found, it returns an empty string.
func hashKeyFromMetadata(metadata map[string]any) string {
	envoyLB, ok := metadata["envoy.lb"].(StructMetadataValue)
	if ok {
		if h, ok := envoyLB.Data["hash_key"].(string); ok {
			return h
		}
	}
	return ""
}

func parseEDSRespProto(m *v3endpointpb.ClusterLoadAssignment) (EndpointsUpdate, error) {
	ret := EndpointsUpdate{}
	for _, dropPolicy := range m.GetPolicy().GetDropOverloads() {
		ret.Drops = append(ret.Drops, parseDropPolicy(dropPolicy))
	}
	priorities := make(map[uint32]map[string]bool)
	sumOfWeights := make(map[uint32]uint64)
	uniqueEndpointAddrs := make(map[string]bool)
	for _, locality := range m.Endpoints {
		l := locality.GetLocality()
		if l == nil {
			return EndpointsUpdate{}, fmt.Errorf("EDS response contains a locality without ID, locality: %+v", locality)
		}
		weight := locality.GetLoadBalancingWeight().GetValue()
		if weight == 0 {
			logger.Warningf("Ignoring locality %s with weight 0", pretty.ToJSON(l))
			continue
		}
		priority := locality.GetPriority()
		sumOfWeights[priority] += uint64(weight)
		if sumOfWeights[priority] > math.MaxUint32 {
			return EndpointsUpdate{}, fmt.Errorf("sum of weights of localities at the same priority %d exceeded maximal value", priority)
		}
		localitiesWithPriority := priorities[priority]
		if localitiesWithPriority == nil {
			localitiesWithPriority = make(map[string]bool)
			priorities[priority] = localitiesWithPriority
		}
		lid := clients.Locality{
			Region:  l.Region,
			Zone:    l.Zone,
			SubZone: l.SubZone,
		}
		lidStr := xdsinternal.LocalityString(lid)

		// "Since an xDS configuration can place a given locality under multiple
		// priorities, it is possible to see locality weight attributes with
		// different values for the same locality." - A52
		//
		// This is handled in the client by emitting the locality weight
		// specified for the priority it is specified in. If the same locality
		// has a different weight in two priorities, each priority will specify
		// a locality with the locality weight specified for that priority, and
		// thus the subsequent tree of balancers linked to that priority will
		// use that locality weight as well.
		if localitiesWithPriority[lidStr] {
			return EndpointsUpdate{}, fmt.Errorf("duplicate locality %s with the same priority %v", lidStr, priority)
		}
		localitiesWithPriority[lidStr] = true
		endpoints, err := parseEndpoints(locality.GetLbEndpoints(), uniqueEndpointAddrs)
		if err != nil {
			return EndpointsUpdate{}, err
		}
		var localityMetadata map[string]any
		if envconfig.XDSHTTPConnectEnabled {
			var err error
			localityMetadata, err = validateAndConstructMetadata(locality.GetMetadata())
			if err != nil {
				return EndpointsUpdate{}, err
			}
		}

		ret.Localities = append(ret.Localities, Locality{
			ID:        lid,
			Endpoints: endpoints,
			Weight:    weight,
			Priority:  priority,
			Metadata:  localityMetadata,
		})
	}
	for i := 0; i < len(priorities); i++ {
		if _, ok := priorities[uint32(i)]; !ok {
			return EndpointsUpdate{}, fmt.Errorf("priority %v missing (with different priorities %v received)", i, priorities)
		}
	}
	return ret, nil
}

func validateAndConstructMetadata(metadataProto *v3corepb.Metadata) (map[string]any, error) {
	if metadataProto == nil {
		return nil, nil
	}
	metadata := make(map[string]any)
	// First go through TypedFilterMetadata.
	for key, anyProto := range metadataProto.GetTypedFilterMetadata() {
		converter := metadataConverterForType(anyProto.GetTypeUrl())
		// Ignore types we don't have a converter for.
		if converter == nil {
			continue
		}
		val, err := converter.convert(anyProto)
		if err != nil {
			// If the converter fails, nack the whole resource.
			return nil, fmt.Errorf("metadata conversion for key %q and type %q failed: %v", key, anyProto.GetTypeUrl(), err)
		}
		metadata[key] = val
	}

	// Process FilterMetadata for any keys not already handled.
	for key, structProto := range metadataProto.GetFilterMetadata() {
		// Skip keys already added from TyperFilterMetadata.
		if metadata[key] != nil {
			continue
		}
		metadata[key] = StructMetadataValue{Data: structProto.AsMap()}
	}
	return metadata, nil
}
