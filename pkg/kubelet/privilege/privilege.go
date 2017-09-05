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

package privilege

import (
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

type privilegeAdmitHandler struct{}

var _ lifecycle.PodAdmitHandler = &privilegeAdmitHandler{}

func NewPrivilegeAdmitHandler() *privilegeAdmitHandler {
	return &privilegeAdmitHandler{}
}

// Admit checks whether the privilege related requests satisfied
func (w *privilegeAdmitHandler) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	if err := kubelet.CanRunPod(attrs.Pod); err != nil {
		return lifecycle.PodAdmitResult{
			Admit:   false,
			Reason:  "Forbidden",
			Message: err.Error(),
		}
	}

	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}
