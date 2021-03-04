/*
Copyright 2020 The Kubernetes Authors.

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

package node

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	imageutils "k8s.io/kubernetes/test/utils/image"
	utilpointer "k8s.io/utils/pointer"
)

// PreconfiguredRuntimeClassHandler returns configured runtime handler.
func PreconfiguredRuntimeClassHandler(handler string) string {
	if handler == "docker" {
		return handler
	}

	// test-handler is the name of the runtime handler that is expected to be
	// preconfigured in the test environment.
	return "test-handler"
}

// NewRuntimeClassPod returns a test pod with the given runtimeClassName
func NewRuntimeClassPod(runtimeClassName string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("test-runtimeclass-%s-", runtimeClassName),
		},
		Spec: v1.PodSpec{
			RuntimeClassName: &runtimeClassName,
			Containers: []v1.Container{{
				Name:    "test",
				Image:   imageutils.GetE2EImage(imageutils.BusyBox),
				Command: []string{"true"},
			}},
			RestartPolicy:                v1.RestartPolicyNever,
			AutomountServiceAccountToken: utilpointer.BoolPtr(false),
		},
	}
}
