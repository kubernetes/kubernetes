/*
Copyright 2021 The Kubernetes Authors.

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

package v1

import (
	"k8s.io/apimachinery/pkg/runtime"
)

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_ExecConfig(exec *ExecConfig) {
	if len(exec.InteractiveMode) == 0 {
		switch exec.APIVersion {
		case "client.authentication.k8s.io/v1beta1", "client.authentication.k8s.io/v1alpha1":
			// default to IfAvailableExecInteractiveMode for backwards compatibility
			exec.InteractiveMode = IfAvailableExecInteractiveMode
		default:
			// require other versions to explicitly declare whether they want stdin or not
		}
	}
}
