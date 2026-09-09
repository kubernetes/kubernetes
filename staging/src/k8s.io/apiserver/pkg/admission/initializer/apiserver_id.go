/*
Copyright The Kubernetes Authors.

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

package initializer

import "k8s.io/apiserver/pkg/admission"

// apiServerIDInitializer injects the API server identity into admission plugins
// that implement WantsAPIServerID. This is used for metrics labeling in HA setups
// where multiple API servers may be running with different manifest configurations.
type apiServerIDInitializer struct {
	apiServerID string
}

var _ admission.PluginInitializer = &apiServerIDInitializer{}

// NewAPIServerIDInitializer returns a PluginInitializer that injects the given
// API server ID into admission plugins that implement WantsAPIServerID.
// This initializer must be placed before the generic initializer in the chain
// so that the API server ID is available when SetExternalKubeInformerFactory is called.
func NewAPIServerIDInitializer(apiServerID string) admission.PluginInitializer {
	return &apiServerIDInitializer{apiServerID: apiServerID}
}

// Initialize checks whether the plugin implements WantsAPIServerID and injects the ID.
func (i *apiServerIDInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsAPIServerID); ok {
		wants.SetAPIServerID(i.apiServerID)
	}
}
