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

package test

import (
	corev1 "k8s.io/api/core/v1"
)

const (
	annotationKeyPod             = "seccomp.security.alpha.kubernetes.io/pod"
	annotationKeyContainerPrefix = "container.seccomp.security.alpha.kubernetes.io/"
)

var (
	// the RuntimeDefault seccomp profile
	seccompProfileRuntimeDefault *corev1.SeccompProfile = &corev1.SeccompProfile{
		Type: corev1.SeccompProfileTypeRuntimeDefault,
	}

	// the Unconfined seccomp profile
	seccompProfileUnconfined *corev1.SeccompProfile = &corev1.SeccompProfile{
		Type: corev1.SeccompProfileTypeUnconfined,
	}
)

// the Localhost seccomp profile
func seccompProfileLocalhost(profile string) *corev1.SeccompProfile {
	return &corev1.SeccompProfile{
		Type:             corev1.SeccompProfileTypeLocalhost,
		LocalhostProfile: &profile,
	}
}

// annotationKeyContainer builds the annotation key for a specific container
func annotationKeyContainer(c corev1.Container) string {
	return annotationKeyContainerPrefix + c.Name
}
