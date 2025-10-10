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
 *
 */

// Package rls implements the RLS cluster specifier plugin.
package rls

import (
	"encoding/json"
	"fmt"

	"google.golang.org/grpc/balancer"
	"google.golang.org/grpc/internal"
	rlspb "google.golang.org/grpc/internal/proto/grpc_lookup_v1"
	"google.golang.org/grpc/internal/xds/clusterspecifier"
	"google.golang.org/protobuf/encoding/protojson"
	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/anypb"
)

func init() {
	clusterspecifier.Register(rls{})
}

type rls struct{}

func (rls) TypeURLs() []string {
	return []string{"type.googleapis.com/grpc.lookup.v1.RouteLookupClusterSpecifier"}
}

// lbConfigJSON is the RLS LB Policies configuration in JSON format.
// RouteLookupConfig will be a raw JSON string from the passed in proto
// configuration, and the other fields will be hardcoded.
type lbConfigJSON struct {
	RouteLookupConfig                json.RawMessage              `json:"routeLookupConfig"`
	ChildPolicy                      []map[string]json.RawMessage `json:"childPolicy"`
	ChildPolicyConfigTargetFieldName string                       `json:"childPolicyConfigTargetFieldName"`
}

func (rls) ParseClusterSpecifierConfig(cfg proto.Message) (clusterspecifier.BalancerConfig, error) {
	if cfg == nil {
		return nil, fmt.Errorf("rls_csp: nil configuration message provided")
	}
	m, ok := cfg.(*anypb.Any)
	if !ok {
		return nil, fmt.Errorf("rls_csp: error parsing config %v: unknown type %T", cfg, cfg)
	}
	rlcs := new(rlspb.RouteLookupClusterSpecifier)

	if err := m.UnmarshalTo(rlcs); err != nil {
		return nil, fmt.Errorf("rls_csp: error parsing config %v: %v", cfg, err)
	}
	rlcJSON, err := protojson.Marshal(rlcs.GetRouteLookupConfig())
	if err != nil {
		return nil, fmt.Errorf("rls_csp: error marshaling route lookup config: %v: %v", rlcs.GetRouteLookupConfig(), err)
	}
	lbCfgJSON := &lbConfigJSON{
		RouteLookupConfig: rlcJSON, // "JSON form of RouteLookupClusterSpecifier.config" - RLS in xDS Design Doc
		ChildPolicy: []map[string]json.RawMessage{
			{
				"cds_experimental": json.RawMessage("{}"),
			},
		},
		ChildPolicyConfigTargetFieldName: "cluster",
	}

	rawJSON, err := json.Marshal(lbCfgJSON)
	if err != nil {
		return nil, fmt.Errorf("rls_csp: error marshaling load balancing config %v: %v", lbCfgJSON, err)
	}

	rlsBB := balancer.Get(internal.RLSLoadBalancingPolicyName)
	if rlsBB == nil {
		return nil, fmt.Errorf("RLS LB policy not registered")
	}
	if _, err = rlsBB.(balancer.ConfigParser).ParseConfig(rawJSON); err != nil {
		return nil, fmt.Errorf("rls_csp: validation error from rls lb policy parsing: %v", err)
	}

	return clusterspecifier.BalancerConfig{{internal.RLSLoadBalancingPolicyName: lbCfgJSON}}, nil
}
