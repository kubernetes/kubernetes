/*
 *
 * Copyright 2023 gRPC authors.
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

// Package xdslbregistry provides a registry of converters that convert proto
// from load balancing configuration, defined by the xDS API spec, to JSON load
// balancing configuration.
package xdslbregistry

import (
	"encoding/json"
	"fmt"

	v3clusterpb "github.com/envoyproxy/go-control-plane/envoy/config/cluster/v3"
)

var (
	// m is a map from proto type to Converter.
	m = make(map[string]Converter)
)

// Register registers the converter to the map keyed on a proto type. Must be
// called at init time. Not thread safe.
func Register(protoType string, c Converter) {
	m[protoType] = c
}

// SetRegistry sets the xDS LB registry. Must be called at init time. Not thread
// safe.
func SetRegistry(registry map[string]Converter) {
	m = registry
}

// Converter converts raw proto bytes into the internal Go JSON representation
// of the proto passed. Returns the json message,  and an error. If both
// returned are nil, it represents continuing to the next proto.
type Converter func([]byte, int) (json.RawMessage, error)

// ConvertToServiceConfig converts a proto Load Balancing Policy configuration
// into a json string. Returns an error if:
//   - no supported policy found
//   - there is more than 16 layers of recursion in the configuration
//   - a failure occurs when converting the policy
func ConvertToServiceConfig(lbPolicy *v3clusterpb.LoadBalancingPolicy, depth int) (json.RawMessage, error) {
	// "Configurations that require more than 16 levels of recursion are
	// considered invalid and should result in a NACK response." - A51
	if depth > 15 {
		return nil, fmt.Errorf("lb policy %v exceeds max depth supported: 16 layers", lbPolicy)
	}

	// "This function iterate over the list of policy messages in
	// LoadBalancingPolicy, attempting to convert each one to gRPC form,
	// stopping at the first supported policy." - A52
	for _, policy := range lbPolicy.GetPolicies() {
		converter := m[policy.GetTypedExtensionConfig().GetTypedConfig().GetTypeUrl()]
		// "Any entry not in the above list is unsupported and will be skipped."
		// - A52
		if converter == nil {
			continue
		}
		json, err := converter(policy.GetTypedExtensionConfig().GetTypedConfig().GetValue(), depth)
		if json == nil && err == nil {
			continue
		}
		return json, err
	}
	return nil, fmt.Errorf("no supported policy found in policy list +%v", lbPolicy)
}
