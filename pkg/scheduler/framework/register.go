/*
Copyright 2019 The Kubernetes Authors.

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

package framework

import (
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// NewRegistry builds a default registry with all the default plugins.
// This is the registry that Kubernetes default scheduler uses. A scheduler that
// runs custom plugins, can pass a different Registry and when initializing the
// scheduler.
func NewRegistry() v1alpha1.Registry {
	return v1alpha1.Registry{
		// FactoryMap:
		// New plugins are registered here.
		// example:
		// {
		//  stateful_plugin.Name: stateful.NewStatefulMultipointExample,
		//  fooplugin.Name: fooplugin.New,
		// }
		queuesort.Name: queuesort.New,
	}
}
