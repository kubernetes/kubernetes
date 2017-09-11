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

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
)

const (
	// alpha: v1.8 (for Services)
	//
	// Allows Services backed by a GCP load balancer to choose what network
	// tier to use. Currently supports "Standard" and "Premium" (default).
	AlphaFeatureNetworkTiers = "NetworkTiers"

	GCEDiskAlphaFeatureGate = "DiskAlphaAPI"
)

// All known alpha features
var knownAlphaFeatures = map[string]bool{
	AlphaFeatureNetworkTiers: true,
	GCEDiskAlphaFeatureGate:  true,
}

type AlphaFeatureGate struct {
	features map[string]bool
}

func (af *AlphaFeatureGate) Enabled(key string) bool {
	return af.features[key]
}

func NewAlphaFeatureGate(features []string) (*AlphaFeatureGate, error) {
	errList := []error{}
	featureMap := make(map[string]bool)
	for _, name := range features {
		if _, ok := knownAlphaFeatures[name]; !ok {
			errList = append(errList, fmt.Errorf("alpha feature %q is not supported.", name))
		} else {
			featureMap[name] = true
		}
	}
	return &AlphaFeatureGate{featureMap}, utilerrors.NewAggregate(errList)
}
