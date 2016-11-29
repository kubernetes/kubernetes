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
	"strconv"
	"strings"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/leaky"
)

// Container "names" are implementation details that do not concern
// kubelet/CRI. This CRI shim uses names to fulfill the CRI requirement to
// make sandbox/container creation idempotent. CRI states that there can
// only exist one sandbox/container with the given metadata. To enforce this,
// this shim constructs a name using the fields in the metadata so that
// docker will reject the creation request if the name already exists.
//
// Note that changes to naming will likely break the backward compatibility.
// Code must be added to ensure the shim knows how to recognize and extract
// information the older containers.
//
// TODO: Add code to handle backward compatibility, i.e., making sure we can
// recognize older containers and extract information from their names if
// necessary.

const (
	// kubePrefix is used to identify the containers/sandboxes on the node managed by kubelet
	kubePrefix = "k8s"
	// sandboxContainerName is a string to include in the docker container so
	// that users can easily identify the sandboxes.
	sandboxContainerName = leaky.PodInfraContainerName
	// Delimiter used to construct docker container names.
	nameDelimiter = "_"
	// DockerImageIDPrefix is the prefix of image id in container status.
	DockerImageIDPrefix = dockertools.DockerPrefix
	// DockerPullableImageIDPrefix is the prefix of pullable image id in container status.
	DockerPullableImageIDPrefix = dockertools.DockerPullablePrefix
)

func makeSandboxName(s *runtimeapi.PodSandboxConfig) string {
	return strings.Join([]string{
		kubePrefix,                                 // 0
		sandboxContainerName,                       // 1
		s.Metadata.GetName(),                       // 2
		s.Metadata.GetNamespace(),                  // 3
		s.Metadata.GetUid(),                        // 4
		fmt.Sprintf("%d", s.Metadata.GetAttempt()), // 5
	}, nameDelimiter)
}

func makeContainerName(s *runtimeapi.PodSandboxConfig, c *runtimeapi.ContainerConfig) string {
	return strings.Join([]string{
		kubePrefix,                                 // 0
		c.Metadata.GetName(),                       // 1:
		s.Metadata.GetName(),                       // 2: sandbox name
		s.Metadata.GetNamespace(),                  // 3: sandbox namesapce
		s.Metadata.GetUid(),                        // 4  sandbox uid
		fmt.Sprintf("%d", c.Metadata.GetAttempt()), // 5
	}, nameDelimiter)

}

func parseUint32(s string) (uint32, error) {
	n, err := strconv.ParseUint(s, 10, 32)
	if err != nil {
		return 0, err
	}
	return uint32(n), nil
}

// TODO: Evaluate whether we should rely on labels completely.
func parseSandboxName(name string) (*runtimeapi.PodSandboxMetadata, error) {
	// Docker adds a "/" prefix to names. so trim it.
	name = strings.TrimPrefix(name, "/")

	parts := strings.Split(name, nameDelimiter)
	if len(parts) != 6 {
		return nil, fmt.Errorf("failed to parse the sandbox name: %q", name)
	}
	if parts[0] != kubePrefix {
		return nil, fmt.Errorf("container is not managed by kubernetes: %q", name)
	}

	attempt, err := parseUint32(parts[5])
	if err != nil {
		return nil, fmt.Errorf("failed to parse the sandbox name %q: %v", name, err)
	}

	return &runtimeapi.PodSandboxMetadata{
		Name:      &parts[2],
		Namespace: &parts[3],
		Uid:       &parts[4],
		Attempt:   &attempt,
	}, nil
}

// TODO: Evaluate whether we should rely on labels completely.
func parseContainerName(name string) (*runtimeapi.ContainerMetadata, error) {
	// Docker adds a "/" prefix to names. so trim it.
	name = strings.TrimPrefix(name, "/")

	parts := strings.Split(name, nameDelimiter)
	if len(parts) != 6 {
		return nil, fmt.Errorf("failed to parse the container name: %q", name)
	}
	if parts[0] != kubePrefix {
		return nil, fmt.Errorf("container is not managed by kubernetes: %q", name)
	}

	attempt, err := parseUint32(parts[5])
	if err != nil {
		return nil, fmt.Errorf("failed to parse the container name %q: %v", name, err)
	}

	return &runtimeapi.ContainerMetadata{
		Name:    &parts[1],
		Attempt: &attempt,
	}, nil
}
