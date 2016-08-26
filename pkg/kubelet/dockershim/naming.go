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

package dockershim

import (
	"fmt"
	"math/rand"
	"strconv"
	"strings"

	"github.com/golang/glog"

	runtimeApi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
)

const (
	// kubePrefix is used to identify the containers/sandboxes on the node managed by kubelet
	kubePrefix = "k8s"
	// kubeSandboxNamePrefix is used to keep sandbox name consistent with old podInfraContainer name
	kubeSandboxNamePrefix = "POD"
)

// buildKubeGenericName creates a name which can be reversed to identify container/sandbox name.
// This function returns the unique name.
func buildKubeGenericName(sandboxConfig *runtimeApi.PodSandboxConfig, containerName string) string {
	stableName := fmt.Sprintf("%s_%s_%s_%s_%s",
		kubePrefix,
		containerName,
		sandboxConfig.Metadata.GetName(),
		sandboxConfig.Metadata.GetNamespace(),
		sandboxConfig.Metadata.GetUid(),
	)
	UID := fmt.Sprintf("%08x", rand.Uint32())
	return fmt.Sprintf("%s_%s", stableName, UID)
}

// buildSandboxName creates a name which can be reversed to identify sandbox full name.
func buildSandboxName(sandboxConfig *runtimeApi.PodSandboxConfig) string {
	sandboxName := fmt.Sprintf("%s.%d", kubeSandboxNamePrefix, sandboxConfig.Metadata.GetAttempt())
	return buildKubeGenericName(sandboxConfig, sandboxName)
}

// parseSandboxName unpacks a sandbox full name, returning the pod name, namespace, uid and attempt.
func parseSandboxName(name string) (string, string, string, uint32, error) {
	podName, podNamespace, podUID, _, attempt, err := parseContainerName(name)
	if err != nil {
		return "", "", "", 0, err
	}

	return podName, podNamespace, podUID, attempt, nil
}

// buildContainerName creates a name which can be reversed to identify container name.
// This function returns stable name, unique name and an unique id.
func buildContainerName(sandboxConfig *runtimeApi.PodSandboxConfig, containerConfig *runtimeApi.ContainerConfig) string {
	containerName := fmt.Sprintf("%s.%d", containerConfig.Metadata.GetName(), containerConfig.Metadata.GetAttempt())
	return buildKubeGenericName(sandboxConfig, containerName)
}

// parseContainerName unpacks a container name, returning the pod name, namespace, UID,
// container name and attempt.
func parseContainerName(name string) (podName, podNamespace, podUID, containerName string, attempt uint32, err error) {
	// Docker adds a "/" prefix to names. so trim it.
	name = strings.TrimPrefix(name, "/")

	parts := strings.Split(name, "_")
	if len(parts) == 0 || parts[0] != kubePrefix {
		err = fmt.Errorf("failed to parse container name %q into parts", name)
		return "", "", "", "", 0, err
	}
	if len(parts) < 6 {
		glog.Warningf("Found a container with the %q prefix, but too few fields (%d): %q", kubePrefix, len(parts), name)
		err = fmt.Errorf("container name %q has fewer parts than expected %v", name, parts)
		return "", "", "", "", 0, err
	}

	nameParts := strings.Split(parts[1], ".")
	containerName = nameParts[0]
	if len(nameParts) > 1 {
		attemptNumber, err := strconv.ParseUint(nameParts[1], 10, 32)
		if err != nil {
			glog.Warningf("invalid container attempt %q in container %q", nameParts[1], name)
		}

		attempt = uint32(attemptNumber)
	}

	return parts[2], parts[3], parts[4], containerName, attempt, nil
}
