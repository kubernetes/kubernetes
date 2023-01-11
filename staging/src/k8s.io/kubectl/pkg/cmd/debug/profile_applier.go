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

const (
	// ProfileLegacy represents the legacy debugging profile which is backwards-compatible with 1.23 behavior.
	ProfileLegacy = "legacy"
	// ProfileGeneral contains a reasonable set of defaults tailored for each debugging journey.
	ProfileGeneral = "general"
	// ProfileBaseline is identical to "general" but eliminates privileges that are disallowed under
	// the baseline security profile, such as host namespaces, host volume, mounts and SYS_PTRACE.
	ProfileBaseline = "baseline"
	// ProfileRestricted is identical to "baseline" but adds configuration that's required
	// under the restricted security profile, such as requiring a non-root user and dropping all capabilities.
	ProfileRestricted = "restricted"
)

// ProfileApplierFunc is an adapter to allow the use of ordinary functions
// as ProfileApplier. If f is a function with the appropriate signature,
// ProfileApplierFunc(f) is an ProfileApplier that calls f.
type ProfileApplierFunc func(pod *corev1.Pod, containerName string, target runtime.Object) error

func (p ProfileApplierFunc) Apply(pod *corev1.Pod, containerName string, target runtime.Object) error {
	return p(pod, containerName, target)
}

type ProfileApplier interface {
	// Apply applies the profile to the given container in the pod.
	Apply(pod *corev1.Pod, containerName string, target runtime.Object) error
}

// NewProfileApplier returns a new Options for the given profile name.
func NewProfileApplier(profile string) (ProfileApplier, error) {
	switch profile {
	case ProfileLegacy:
		return &legacyProfile{}, nil
	case ProfileGeneral:
		return &generalProfile{}, nil
	case ProfileBaseline:
		return &baselineProfile{}, nil
	case ProfileRestricted:
		return &restrictedProfile{}, nil
	}

	return nil, fmt.Errorf("unknown profile: %s", profile)
}
