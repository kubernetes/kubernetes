/*
Copyright 2018 The Kubernetes Authors.

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

package componentconfigs

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// NoValidator returns a dummy validator function when no validation method is available for the component
func NoValidator(component string) func(*kubeadmapi.ClusterConfiguration, *field.Path) field.ErrorList {
	return func(_ *kubeadmapi.ClusterConfiguration, _ *field.Path) field.ErrorList {
		klog.Warningf("Cannot validate %s config - no validator is available", component)
		return field.ErrorList{}
	}
}
