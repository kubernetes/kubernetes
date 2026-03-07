/*
Copyright 2016 The Kubernetes Authors.

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

package admission

import (
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	policyloader "k8s.io/kubernetes/pkg/admission/plugin/policy/manifest/loader"
	webhookloader "k8s.io/kubernetes/pkg/admission/plugin/webhook/manifest/loader"
)

// TODO add a `WantsToRun` which takes a stopCh.  Might make it generic.

// PluginInitializer is used for initialization of the Kubernetes specific admission plugins.
type PluginInitializer struct {
	loaders *initializer.ManifestLoaders
}

var _ admission.PluginInitializer = &PluginInitializer{}

// NewPluginInitializer constructs new instance of PluginInitializer
func NewPluginInitializer() *PluginInitializer {
	return &PluginInitializer{
		loaders: newManifestLoaders(),
	}
}

// Initialize checks the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *PluginInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(initializer.WantsManifestLoaders); ok {
		wants.SetManifestLoaders(i.loaders)
	}
}

func newManifestLoaders() *initializer.ManifestLoaders {
	if !utilfeature.DefaultFeatureGate.Enabled(features.ManifestBasedAdmissionControlConfig) {
		return &initializer.ManifestLoaders{}
	}
	return &initializer.ManifestLoaders{
		LoadValidatingWebhookManifests: func(dir string) ([]*admissionregistrationv1.ValidatingWebhookConfiguration, string, error) {
			result, err := webhookloader.LoadValidatingManifests(dir)
			if err != nil {
				return nil, "", err
			}
			return result.Configurations, result.Hash, nil
		},
		LoadMutatingWebhookManifests: func(dir string) ([]*admissionregistrationv1.MutatingWebhookConfiguration, string, error) {
			result, err := webhookloader.LoadMutatingManifests(dir)
			if err != nil {
				return nil, "", err
			}
			return result.Configurations, result.Hash, nil
		},
		LoadValidatingPolicyManifests: func(dir string) ([]*admissionregistrationv1.ValidatingAdmissionPolicy, []*admissionregistrationv1.ValidatingAdmissionPolicyBinding, string, error) {
			manifests, err := policyloader.LoadValidatingManifestsFromDirectory(dir)
			if err != nil {
				return nil, nil, "", err
			}
			return manifests.Policies, manifests.Bindings, manifests.Hash, nil
		},
		LoadMutatingPolicyManifests: func(dir string) ([]*admissionregistrationv1.MutatingAdmissionPolicy, []*admissionregistrationv1.MutatingAdmissionPolicyBinding, string, error) {
			manifests, err := policyloader.LoadMutatingManifestsFromDirectory(dir)
			if err != nil {
				return nil, nil, "", err
			}
			return manifests.Policies, manifests.Bindings, manifests.Hash, nil
		},
	}
}
