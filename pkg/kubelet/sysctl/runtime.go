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

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
)

const (
	UnsupportedReason = "SysctlUnsupported"
	// CRI uses semver-compatible API version, while docker does not
	// (e.g., 1.24). Append the version with a ".0" so that it works
	// with both the CRI and dockertools comparison logic.
	dockerMinimumAPIVersion = "1.24.0"
)

type runtimeAdmitHandler struct {
	result lifecycle.PodAdmitResult
}

var _ lifecycle.PodAdmitHandler = &runtimeAdmitHandler{}

// NewRuntimeAdmitHandler returns a sysctlRuntimeAdmitHandler which checks whether
// the given runtime support sysctls.
func NewRuntimeAdmitHandler(runtime container.Runtime) (*runtimeAdmitHandler, error) {
	if runtime.Type() == dockertools.DockerType {
		v, err := runtime.APIVersion()
		if err != nil {
			return nil, fmt.Errorf("failed to get runtime version: %v", err)
		}

		// only Docker >= 1.12 supports sysctls
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
				Message: "Docker before 1.12 does not support sysctls",
			},
		}, nil
	}

	// for other runtimes like rkt sysctls are not supported
	return &runtimeAdmitHandler{
		result: lifecycle.PodAdmitResult{
			Admit:   false,
			Reason:  UnsupportedReason,
			Message: fmt.Sprintf("runtime %v does not support sysctls", runtime.Type()),
		},
	}, nil
}

// Admit checks whether the runtime supports sysctls.
func (w *runtimeAdmitHandler) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	sysctls, unsafeSysctls, err := v1.SysctlsFromPodAnnotations(attrs.Pod.Annotations)
	if err != nil {
		return lifecycle.PodAdmitResult{
			Admit:   false,
			Reason:  AnnotationInvalidReason,
			Message: fmt.Sprintf("invalid sysctl annotation: %v", err),
		}
	}

	if len(sysctls)+len(unsafeSysctls) > 0 {
		return w.result
	}

	return lifecycle.PodAdmitResult{
		Admit: true,
	}
}
