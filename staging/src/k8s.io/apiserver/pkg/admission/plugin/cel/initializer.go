/*
Copyright 2022 The Kubernetes Authors.

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

package cel

import (
	"context"

	"k8s.io/apiserver/pkg/admission"
)

type CELPolicyEvaluator interface {
	Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error
	HasSynced() bool
}

// NewPluginInitializer creates a plugin initializer which dependency injects a
// singleton cel admission controller into the plugins which desire it
func NewPluginInitializer(validator CELPolicyEvaluator) *PluginInitializer {
	return &PluginInitializer{validator: validator}
}

// WantsCELPolicyEvaluator gives the ability to have the shared
// CEL Admission Controller dependency injected at initialization-time.
type WantsCELPolicyEvaluator interface {
	SetCELPolicyEvaluator(CELPolicyEvaluator)
}

// PluginInitializer is used for initialization of the webhook admission plugin.
type PluginInitializer struct {
	validator CELPolicyEvaluator
}

var _ admission.PluginInitializer = &PluginInitializer{}

// Initialize checks the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *PluginInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsCELPolicyEvaluator); ok {
		wants.SetCELPolicyEvaluator(i.validator)
	}
}
