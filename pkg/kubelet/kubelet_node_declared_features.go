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
	"k8s.io/component-helpers/nodedeclaredfeatures"
)

// discoverNodeDeclaredFeatures determines the final set of node features to be declared by using the discovery library.
func (kl *Kubelet) discoverNodeDeclaredFeatures() ([]string, error) {
	// Fill the necessary feature gates and kubelet config.
	featureGates := map[string]bool{}
	kubeletConfigMap := map[string]string{}
	cfg := &nodedeclaredfeatures.NodeConfiguration{
		FeatureGates:  featureGates,
		KubeletConfig: kubeletConfigMap,
	}
	return kl.nodeDeclaredFeaturesHelper.DiscoverNodeFeatures(cfg)
}
