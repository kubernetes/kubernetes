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

package plugins

import (
	noop "k8s.io/kubernetes/pkg/scheduler/framework/plugins/noop"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
)

// NewDefaultRegistry builds a default registry with all the default plugins.
// This is the registry that Kubernetes default scheduler uses. A scheduler that
// runs custom plugins, can pass a different Registry when initializing the scheduler.
func NewDefaultRegistry() framework.Registry {
	return framework.Registry{
		// This is just a test plugin to showcase the setup, it should be deleted once
		// we have at least one legitimate plugin here.
		noop.Name: noop.New,
	}
}
