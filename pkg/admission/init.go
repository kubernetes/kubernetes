/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/controller/framework"
	"reflect"
)

// PluginInitializer is used for Initialization of shareable resources between admission plugins
// After Initialization the resources have to be set separately
type PluginInitializer interface {
	Initialize(plugins []Interface)
	SetNamespaceInformer(namespaceInformer framework.SharedIndexInformer)
}

type pluginInitializer struct {
	informers map[reflect.Type]framework.SharedIndexInformer
}

// NewPluginInitializer constructs new instance of PluginInitializer
func NewPluginInitializer() PluginInitializer {
	plugInit := &pluginInitializer{
		informers: make(map[reflect.Type]framework.SharedIndexInformer),
	}
	return plugInit
}

// SetNamespaceInformer sets unique namespaceInformer for instance of PluginInitializer
func (i *pluginInitializer) SetNamespaceInformer(namespaceInformer framework.SharedIndexInformer) {
	i.informers[reflect.TypeOf(&api.Namespace{})] = namespaceInformer
}

// Initialize will check the initialization interfaces implemented by each plugin
// and provide the appropriate initialization data
func (i *pluginInitializer) Initialize(plugins []Interface) {
	for _, plugin := range plugins {
		if wantsNamespaceInformer, ok := plugin.(WantsNamespaceInformer); ok {
			wantsNamespaceInformer.SetNamespaceInformer(i.informers[reflect.TypeOf(&api.Namespace{})])
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
