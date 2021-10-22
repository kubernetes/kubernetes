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

// All structs/functions in this file should be unexported. They are used in EDS
// balancer tests now, to generate test inputs. Eventually, EDS balancer tests
// should generate EDSUpdate directly, instead of generating and parsing the
// proto message.
// TODO: unexported everything in this file.

package client

import (
	"fmt"
	"net"
	"strconv"

	xdspb "github.com/envoyproxy/go-control-plane/envoy/api/v2"
	corepb "github.com/envoyproxy/go-control-plane/envoy/api/v2/core"
	endpointpb "github.com/envoyproxy/go-control-plane/envoy/api/v2/endpoint"
	typepb "github.com/envoyproxy/go-control-plane/envoy/type"
	wrapperspb "github.com/golang/protobuf/ptypes/wrappers"
)

// ClusterLoadAssignmentBuilder builds a ClusterLoadAssignment, aka EDS
// response.
type ClusterLoadAssignmentBuilder struct {
	v *xdspb.ClusterLoadAssignment
}

// NewClusterLoadAssignmentBuilder creates a ClusterLoadAssignmentBuilder.
func NewClusterLoadAssignmentBuilder(clusterName string, dropPercents []uint32) *ClusterLoadAssignmentBuilder {
	var drops []*xdspb.ClusterLoadAssignment_Policy_DropOverload
	for i, d := range dropPercents {
		drops = append(drops, &xdspb.ClusterLoadAssignment_Policy_DropOverload{
			Category: fmt.Sprintf("test-drop-%d", i),
			DropPercentage: &typepb.FractionalPercent{
				Numerator:   d,
				Denominator: typepb.FractionalPercent_HUNDRED,
			},
		})
	}

	return &ClusterLoadAssignmentBuilder{
		v: &xdspb.ClusterLoadAssignment{
			ClusterName: clusterName,
			Policy: &xdspb.ClusterLoadAssignment_Policy{
				DropOverloads: drops,
			},
		},
	}
}

// AddLocalityOptions contains options when adding locality to the builder.
type AddLocalityOptions struct {
	Health []corepb.HealthStatus
	Weight []uint32
}

// AddLocality adds a locality to the builder.
func (clab *ClusterLoadAssignmentBuilder) AddLocality(subzone string, weight uint32, priority uint32, addrsWithPort []string, opts *AddLocalityOptions) {
	var lbEndPoints []*endpointpb.LbEndpoint
	for i, a := range addrsWithPort {
		host, portStr, err := net.SplitHostPort(a)
		if err != nil {
			panic("failed to split " + a)
		}
		port, err := strconv.Atoi(portStr)
		if err != nil {
			panic("failed to atoi " + portStr)
		}

		lbe := &endpointpb.LbEndpoint{
			HostIdentifier: &endpointpb.LbEndpoint_Endpoint{
				Endpoint: &endpointpb.Endpoint{
					Address: &corepb.Address{
						Address: &corepb.Address_SocketAddress{
							SocketAddress: &corepb.SocketAddress{
								Protocol: corepb.SocketAddress_TCP,
								Address:  host,
								PortSpecifier: &corepb.SocketAddress_PortValue{
									PortValue: uint32(port)}}}}}},
		}
		if opts != nil {
			if i < len(opts.Health) {
				lbe.HealthStatus = opts.Health[i]
			}
			if i < len(opts.Weight) {
				lbe.LoadBalancingWeight = &wrapperspb.UInt32Value{Value: opts.Weight[i]}
			}
		}
		lbEndPoints = append(lbEndPoints, lbe)
	}

	var localityID *corepb.Locality
	if subzone != "" {
		localityID = &corepb.Locality{
			Region:  "",
			Zone:    "",
			SubZone: subzone,
		}
	}

	clab.v.Endpoints = append(clab.v.Endpoints, &endpointpb.LocalityLbEndpoints{
		Locality:            localityID,
		LbEndpoints:         lbEndPoints,
		LoadBalancingWeight: &wrapperspb.UInt32Value{Value: weight},
		Priority:            priority,
	})
}

// Build builds ClusterLoadAssignment.
func (clab *ClusterLoadAssignmentBuilder) Build() *xdspb.ClusterLoadAssignment {
	return clab.v
}
