/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gce

import (
	"fmt"
)

const (
	// AlphaFeatureNetworkTiers allows Services backed by a GCP load balancer to choose
	// what network tier to use. Currently supports "Standard" and "Premium" (default).
	//
	// alpha: v1.8 (for Services)
	AlphaFeatureNetworkTiers = "NetworkTiers"
)

// AlphaFeatureGate contains a mapping of alpha features to whether they are enabled
type AlphaFeatureGate struct {
	features map[string]bool
}

// Enabled returns true if the provided alpha feature is enabled
func (af *AlphaFeatureGate) Enabled(key string) bool {
	return af.features[key]
}

// NewAlphaFeatureGate marks the provided alpha features as enabled
func NewAlphaFeatureGate(features []string) *AlphaFeatureGate {
	featureMap := make(map[string]bool)
	for _, name := range features {
		featureMap[name] = true
	}
	return &AlphaFeatureGate{featureMap}
}

func (g *Cloud) alphaFeatureEnabled(feature string) error {
	if !g.AlphaFeatureGate.Enabled(feature) {
		return fmt.Errorf("alpha feature %q is not enabled", feature)
	}
	return nil
}
