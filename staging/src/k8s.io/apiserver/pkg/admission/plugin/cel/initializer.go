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
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apiserver/pkg/admission"
)

type CELPolicyEvaluator interface {
	admission.InitializationValidator

	Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error
	HasSynced() bool
	Run(stopCh <-chan struct{})
}

// NewPluginInitializer creates a plugin initializer which dependency injects a
// singleton cel admission controller into the plugins which desire it
func NewPluginInitializer(restMapper meta.RESTMapper) admission.PluginInitializer {
	return &pluginInitializer{
		restMapper: restMapper,
	}
}

// PluginInitializer is used for initialization of the webhook admission plugin.
type pluginInitializer struct {
	restMapper meta.RESTMapper
}

var _ admission.PluginInitializer = &pluginInitializer{}

type WantsRESTMapper interface {
	SetRESTMapper(meta.RESTMapper)
}

// Initialize checks the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *pluginInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsRESTMapper); ok {
		wants.SetRESTMapper(i.restMapper)
	}
}
