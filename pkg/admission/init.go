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
	"k8s.io/kubernetes/pkg/auth/authorizer"
	"k8s.io/kubernetes/pkg/controller/framework/informers"
)

// PluginInitializer is used for initialization of shareable resources between admission plugins.
// After initialization the resources have to be set separately
type PluginInitializer interface {
	Initialize(plugins []Interface)
}

type pluginInitializer struct {
	informers  informers.SharedInformerFactory
	authorizer authorizer.Authorizer
}

// NewPluginInitializer constructs new instance of PluginInitializer
func NewPluginInitializer(sharedInformers informers.SharedInformerFactory, authz authorizer.Authorizer) PluginInitializer {
	plugInit := &pluginInitializer{
		informers:  sharedInformers,
		authorizer: authz,
	}
	return plugInit
}

// Initialize checks the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *pluginInitializer) Initialize(plugins []Interface) {
	for _, plugin := range plugins {
		if wantsInformerFactory, ok := plugin.(WantsInformerFactory); ok {
			wantsInformerFactory.SetInformerFactory(i.informers)
		}

		if wantsAuthorizer, ok := plugin.(WantsAuthorizer); ok {
			wantsAuthorizer.SetAuthorizer(i.authorizer)
		}
	}
}

// Validate will call the Validate function in each plugin if they implement
// the Validator interface.
func Validate(plugins []Interface) error {
	for _, plugin := range plugins {
		if validater, ok := plugin.(Validator); ok {
			err := validater.Validate()
			if err != nil {
				return err
			}
		}
	}
	return nil
}
