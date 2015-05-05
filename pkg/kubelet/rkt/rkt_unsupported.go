// +build !linux

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package rkt

import (
	"fmt"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
)

// rkt is unsupported in non-Linux builds.
type unsupportedRuntime struct {
}

var _ kubecontainer.Runtime = &unsupportedRuntime{}

var unsupportedError = fmt.Errorf("rkt runtime is unsupported in this platform")

func (ur *unsupportedRuntime) Version() (kubecontainer.Version, error) {
	return nil, unsupportedError
}

func (ur *unsupportedRuntime) GetPods(all bool) ([]*kubecontainer.Pod, error) {
	return []*kubecontainer.Pod{}, unsupportedError
}

func (ur *unsupportedRuntime) SyncPod(pod *api.Pod, runningPod kubecontainer.Pod, podStatus api.PodStatus) error {
	return unsupportedError
}

func (ur *unsupportedRuntime) KillPod(pod kubecontainer.Pod) error {
	return unsupportedError
}

func (ur *unsupportedRuntime) GetPodStatus(*api.Pod) (*api.PodStatus, error) {
	return nil, unsupportedError
}

func (ur *unsupportedRuntime) RunInContainer(containerID string, cmd []string) ([]byte, error) {
	return []byte{}, unsupportedError
}

func (ur *unsupportedRuntime) ExecInContainer(containerID string, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	return unsupportedError
}

func (ur *unsupportedRuntime) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	return unsupportedError
}

func (ur *unsupportedRuntime) PullImage(image string) error {
	return unsupportedError
}

func (ur *unsupportedRuntime) IsImagePresent(image string) (bool, error) {
	return false, unsupportedError
}

func (ur *unsupportedRuntime) ListImages() ([]kubecontainer.Image, error) {
	return []kubecontainer.Image{}, unsupportedError
}

func (ur *unsupportedRuntime) RemoveImage(image string) error {
	return unsupportedError
}

func (ur *unsupportedRuntime) GetContainerLogs(pod *api.Pod, containerID, tail string, follow bool, stdout, stderr io.Writer) error {
	return unsupportedError
}
