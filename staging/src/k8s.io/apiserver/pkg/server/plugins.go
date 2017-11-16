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

package server

// This file exists to force the desired plugin implementations to be linked into genericapi pkg.
import (
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/initialization"
	"k8s.io/apiserver/pkg/admission/plugin/namespace/lifecycle"
	validatingwebhook "k8s.io/apiserver/pkg/admission/plugin/webhook/validating"
)

// RegisterAllAdmissionPlugins registers all admission plugins
func RegisterAllAdmissionPlugins(plugins *admission.Plugins) {
	lifecycle.Register(plugins)
	initialization.Register(plugins)
	validatingwebhook.Register(plugins)
}
