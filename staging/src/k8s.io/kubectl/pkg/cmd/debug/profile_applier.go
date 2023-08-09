/*
Copyright 2022 The Kubernetes Authors.

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

package debug

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// ProfileLegacy represents the legacy debugging profile which is backwards-compatible with 1.23 behavior.
const ProfileLegacy = "legacy"

type ProfileApplier interface {
	// Apply applies the profile to the given container in the pod.
	Apply(pod *corev1.Pod, containerName string, target runtime.Object) error
}

// NewProfileApplier returns a new Options for the given profile name.
func NewProfileApplier(profile string) (ProfileApplier, error) {
	switch profile {
	case ProfileLegacy:
		return applierFunc(profileLegacy), nil
	}

	return nil, fmt.Errorf("unknown profile: %s", profile)
}

// applierFunc is a function that applies a profile to a container in the pod.
type applierFunc func(pod *corev1.Pod, containerName string, target runtime.Object) error

func (f applierFunc) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	return f(pod, containerName, target)
}
