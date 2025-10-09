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

// Package clusterspecifier contains the ClusterSpecifier interface and a registry for
// storing and retrieving their implementations.
package clusterspecifier

import (
	"google.golang.org/protobuf/proto"
)

// BalancerConfig is the Go Native JSON representation of a balancer
// configuration.
type BalancerConfig []map[string]any

// ClusterSpecifier defines the parsing functionality of a Cluster Specifier.
type ClusterSpecifier interface {
	// TypeURLs are the proto message types supported by this
	// ClusterSpecifierPlugin. A ClusterSpecifierPlugin will be registered by
	// each of its supported message types.
	TypeURLs() []string
	// ParseClusterSpecifierConfig parses the provided configuration
	// proto.Message from the top level RDS configuration. The resulting
	// BalancerConfig will be used as configuration for a child LB Policy of the
	// Cluster Manager LB Policy. A nil BalancerConfig is invalid.
	ParseClusterSpecifierConfig(proto.Message) (BalancerConfig, error)
}

var (
	// m is a map from scheme to filter.
	m = make(map[string]ClusterSpecifier)
)

// Register registers the ClusterSpecifierPlugin to the ClusterSpecifier map.
// cs.TypeURLs() will be used as the types for this ClusterSpecifierPlugin.
//
// NOTE: this function must only be called during initialization time (i.e. in
// an init() function), and is not thread-safe. If multiple cluster specifier
// plugins are registered with the same type URL, the one registered last will
// take effect.
func Register(cs ClusterSpecifier) {
	for _, u := range cs.TypeURLs() {
		m[u] = cs
	}
}

// Get returns the ClusterSpecifier registered with typeURL.
//
// If no cluster specifier is registered with typeURL, nil will be returned.
func Get(typeURL string) ClusterSpecifier {
	return m[typeURL]
}

// UnregisterForTesting unregisters the ClusterSpecifier for testing purposes.
func UnregisterForTesting(typeURL string) {
	delete(m, typeURL)
}
