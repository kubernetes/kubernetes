/*
Copyright 2025 The Kubernetes Authors.

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

package kubelet

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate" // Import for the type conversion
	"k8s.io/component-helpers/nodedeclaredfeatures"
)

// FeatureGateAdapter adapts a component-base FeatureGate to the nodedeclaredfeatures FeatureGate interface.
// This is needed because we cannot use featuregate.Feature in the library code (forbidden import).
type FeatureGateAdapter struct {
	featuregate.FeatureGate
}

// Enabled implements the nodedeclaredfeatures.FeatureGate interface
func (a FeatureGateAdapter) Enabled(key string) bool {
	// Convert the string key to featuregate.Feature
	return a.FeatureGate.Enabled(featuregate.Feature(key))
}

// discoverNodeDeclaredFeatures determines the final set of node features to be declared by using the discovery library.
func (kl *Kubelet) discoverNodeDeclaredFeatures() []string {
	staticConfig := nodedeclaredfeatures.StaticConfiguration{
		CPUManagerPolicy: kl.containerManager.GetNodeConfig().CPUManagerPolicy,
	}

	adaptedFG := FeatureGateAdapter{FeatureGate: utilfeature.DefaultFeatureGate}
	cfg := &nodedeclaredfeatures.NodeConfiguration{
		FeatureGates: adaptedFG,
		StaticConfig: staticConfig,
		Version:      kl.version,
	}
	return kl.nodeDeclaredFeaturesFramework.DiscoverNodeFeatures(cfg)
}
