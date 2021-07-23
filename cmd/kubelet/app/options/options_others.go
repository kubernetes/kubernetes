// +build !linux,!windows

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

// Package options contains all of the primary arguments for a kubelet.
package options

import (
	"path/filepath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// NewKubeletFlags will create a new KubeletFlags with default values
func NewKubeletFlags() *KubeletFlags {
	remoteRuntimeEndpoint := ""

	return &KubeletFlags{
		ContainerRuntimeOptions: *NewContainerRuntimeOptions(),
		CertDirectory:           "/var/lib/kubelet/pki",
		RootDirectory:           defaultRootDir,
		MasterServiceNamespace:  metav1.NamespaceDefault,
		MaxContainerCount:       -1,
		MaxPerPodContainerCount: 1,
		MinimumGCAge:            metav1.Duration{Duration: 0},
		NonMasqueradeCIDR:       "10.0.0.0/8",
		RegisterSchedulable:     true,
		RemoteRuntimeEndpoint:   remoteRuntimeEndpoint,
		NodeLabels:              make(map[string]string),
		RegisterNode:            true,
		SeccompProfileRoot:      filepath.Join(defaultRootDir, "seccomp"),
	}
}
