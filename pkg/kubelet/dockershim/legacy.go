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
	"io"

	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/util/term"
)

// This file implements the functions that are needed for backward
// compatibility. Therefore, it imports various kubernetes packages
// directly.

// TODO: implement the methods in this file.
func (ds *dockerService) LegacyAttach(id kubecontainer.ContainerID, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) (err error) {
	return ds.streamingRuntime.Attach(id.ID, stdin, stdout, stderr, resize)
}

func (ds *dockerService) GetContainerLogs(pod *api.Pod, containerID kubecontainer.ContainerID, logOptions *api.PodLogOptions, stdout, stderr io.Writer) (err error) {
	container, err := ds.client.InspectContainer(containerID.ID)
	if err != nil {
		return err
	}
	return dockertools.GetContainerLogs(ds.client, pod, containerID, logOptions, stdout, stderr, container.Config.Tty)
}

func (ds *dockerService) LegacyPortForward(sandboxID string, port uint16, stream io.ReadWriteCloser) error {
	return ds.streamingRuntime.PortForward(sandboxID, int32(port), stream)
}

func (ds *dockerService) LegacyExec(containerID kubecontainer.ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) error {
	return ds.streamingRuntime.Exec(containerID.ID, cmd, stdin, stdout, stderr, tty, resize)
}
