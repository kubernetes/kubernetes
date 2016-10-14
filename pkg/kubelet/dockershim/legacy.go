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
	"io"

	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/term"
)

// This file implements the functions that are needed for backward
// compatibility. Therefore, it imports various kubernetes packages
// directly.

// TODO: implement the methods in this file.
func (ds *dockerService) AttachContainer(id kubecontainer.ContainerID, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) (err error) {
	return dockertools.AttachContainer(ds.client, id, stdin, stdout, stderr, tty, resize)
}

func (ds *dockerService) GetContainerLogs(pod *api.Pod, containerID kubecontainer.ContainerID, logOptions *api.PodLogOptions, stdout, stderr io.Writer) (err error) {
	return dockertools.GetContainerLogs(ds.client, pod, containerID, logOptions, stdout, stderr)
}

func (ds *dockerService) PortForward(sandboxID string, port uint16, stream io.ReadWriteCloser) error {
	return dockertools.PortForward(ds.client, sandboxID, port, stream)
}

func (ds *dockerService) ExecInContainer(containerID kubecontainer.ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) error {
	container, err := ds.client.InspectContainer(containerID.ID)
	if err != nil {
		return err
	}
	if !container.State.Running {
		return fmt.Errorf("container not running (%s)", container.ID)
	}

	handler := &dockertools.NativeExecHandler{}
	return handler.ExecInContainer(ds.client, container, cmd, stdin, stdout, stderr, tty, resize)
}

const (
	// legacyContainerLogDir is the legacy location of container logs. It is the same with
	// kubelet.containerLogsDir.
	legacyContainerLogDir = "/var/log/containers"
)

// logSymlink composes the container log path. It is only used for legacy cluster logging support.
// Notice that this function directly uses labels written by kubelet which is not recommended. This is
// only a temporary fix and will be removed after cluster logging supports the new container log path.
func logSymlink(labels map[string]string, containerID string) string {
	podName := labels[types.KubernetesPodNameLabel]
	podNamespace := labels[types.KubernetesPodNamespaceLabel]
	containerName := labels[types.KubernetesContainerNameLabel]
	return dockertools.LogSymlink(legacyContainerLogDir, kubecontainer.BuildPodFullName(podName, podNamespace),
		containerName, containerID)
}
