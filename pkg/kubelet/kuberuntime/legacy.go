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

package kuberuntime

import (
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
)

// This file implements the functions that are needed for backward
// compatibility. Therefore, it imports various kubernetes packages
// directly.

const (
	// legacyContainerLogsDir is the legacy location of container logs. It is the same with
	// kubelet.containerLogsDir.
	legacyContainerLogsDir = "/var/log/containers"
	// legacyLogSuffix is the legacy log suffix.
	legacyLogSuffix = dockertools.LogSuffix
)

// legacyLogSymlink composes the legacy container log path. It is only used for legacy cluster
// logging support.
func legacyLogSymlink(containerID string, containerName, podName, podNamespace string) string {
	return dockertools.LogSymlink(legacyContainerLogsDir, kubecontainer.BuildPodFullName(podName, podNamespace),
		containerName, containerID)
}
