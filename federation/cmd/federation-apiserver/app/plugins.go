/*
Copyright 2014 The Kubernetes Authors.

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

package app

// This file exists to force the desired plugin implementations to be linked.
// This should probably be part of some configuration fed into the build for a
// given binary target.
import (
	// Cloud providers
	_ "k8s.io/kubernetes/pkg/cloudprovider/providers"

	// Admission policies
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/federation/plugin/pkg/admission/schedulingpolicy"
	"k8s.io/kubernetes/plugin/pkg/admission/admit"
	"k8s.io/kubernetes/plugin/pkg/admission/deny"
	"k8s.io/kubernetes/plugin/pkg/admission/gc"
	"k8s.io/kubernetes/plugin/pkg/admission/initialization"
)

// registerAllAdmissionPlugins registers all admission plugins
func registerAllAdmissionPlugins(plugins *admission.Plugins) {
	admit.Register(plugins)
	deny.Register(plugins)
	gc.Register(plugins)
	initialization.Register(plugins)
	schedulingpolicy.Register(plugins)
}
