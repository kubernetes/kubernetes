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

package wardleinitializer

import (
	"k8s.io/apiserver/pkg/admission"
	informers "k8s.io/sample-apiserver/pkg/generated/informers/externalversions"
)

type pluginInitializer struct {
	informers informers.SharedInformerFactory
}

var _ admission.PluginInitializer = pluginInitializer{}

// New creates an instance of wardle admission plugins initializer.
func New(informers informers.SharedInformerFactory) pluginInitializer {
	return pluginInitializer{
		informers: informers,
	}
}

// Initialize checks the initialization interfaces implemented by a plugin
// and provide the appropriate initialization data
func (i pluginInitializer) Initialize(plugin admission.Interface) {
	if wants, ok := plugin.(WantsInternalWardleInformerFactory); ok {
		wants.SetInternalWardleInformerFactory(i.informers)
	}
}
