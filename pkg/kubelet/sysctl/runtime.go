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

package sysctl

import (
	"fmt"

	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

const (
	UnsupportedReason = "SysctlUnsupported"
	// CRI uses semver-compatible API version, while docker does not
	// (e.g., 1.24). Append the version with a ".0".
	dockerMinimumAPIVersion = "1.24.0"

	dockerTypeName = "docker"
)

// TODO: The admission logic in this file is runtime-dependent. It should be
// changed to be generic and CRI-compatible.

type runtimeAdmitHandler struct {
	result lifecycle.PodAdmitResult
}

var _ lifecycle.PodAdmitHandler = &runtimeAdmitHandler{}

// NewRuntimeAdmitHandler returns a sysctlRuntimeAdmitHandler which checks whether
// the given runtime support sysctls.
func NewRuntimeAdmitHandler(runtime container.Runtime) (*runtimeAdmitHandler, error) {
	switch runtime.Type() {
	case dockerTypeName:
		v, err := runtime.APIVersion()
		if err != nil {
			return nil, fmt.Errorf("failed to get runtime version: %v", err)
		}

		// only Docker API version >= 1.24 supports sysctls
		c, err := v.Compare(dockerMinimumAPIVersion)
		if err != nil {
			return nil, fmt.Errorf("failed to compare Docker version for sysctl support: %v", err)
		}
		if c >= 0 {
			return &runtimeAdmitHandler{
				result: lifecycle.PodAdmitResult{
					Admit: true,
				},
			}, nil
		}
		return &runtimeAdmitHandler{
			result: lifecycle.PodAdmitResult{
				Admit:   false,
				Reason:  UnsupportedReason,
				Message: "Docker API version before 1.24 does not support sysctls",
			},
		}, nil
	default:
		// Return admit for other runtimes.
		return &runtimeAdmitHandler{
			result: lifecycle.PodAdmitResult{
				Admit: true,
			},
		}, nil
	}
}

// Admit checks whether the runtime supports sysctls.
func (w *runtimeAdmitHandler) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	if attrs.Pod.Spec.SecurityContext != nil {

		if len(attrs.Pod.Spec.SecurityContext.Sysctls) > 0 {
			return w.result
		}
	}

	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}
