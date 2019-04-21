/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"context"
	"io"
	"time"

	"github.com/stretchr/testify/mock"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/remotecommand"
	"k8s.io/client-go/util/flowcontrol"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/volume"
)

// Mock is type for Mocking.
type Mock struct {
	mock.Mock
}

var _ container.Runtime = new(Mock)

// Start mocks starting the runtime.
func (r *Mock) Start() error {
	args := r.Called()
	return args.Error(0)
}

// Type mocks returning the runtime type.
func (r *Mock) Type() string {
	args := r.Called()
	return args.Get(0).(string)
}

// Version mocks returning the runtime version.
func (r *Mock) Version() (container.Version, error) {
	args := r.Called()
	return args.Get(0).(container.Version), args.Error(1)
}

// APIVersion mocks returning the runtime API version.
func (r *Mock) APIVersion() (container.Version, error) {
	args := r.Called()
	return args.Get(0).(container.Version), args.Error(1)
}

// Status mocks returning the runtime status.
func (r *Mock) Status() (*container.RuntimeStatus, error) {
	args := r.Called()
	return args.Get(0).(*container.RuntimeStatus), args.Error(0)
}

// GetPods mocks returning pods from the runtime.
func (r *Mock) GetPods(all bool) ([]*container.Pod, error) {
	args := r.Called(all)
	return args.Get(0).([]*container.Pod), args.Error(1)
}

// SyncPod mocks syncing Pods with the runtime.
func (r *Mock) SyncPod(pod *v1.Pod, status *container.PodStatus, secrets []v1.Secret, backOff *flowcontrol.Backoff) container.PodSyncResult {
	args := r.Called(pod, status, secrets, backOff)
	return args.Get(0).(container.PodSyncResult)
}

// KillPod mocks killing pods.
func (r *Mock) KillPod(pod *v1.Pod, runningPod container.Pod, gracePeriodOverride *int64) error {
	args := r.Called(pod, runningPod, gracePeriodOverride)
	return args.Error(0)
}

// RunContainerInPod mocks starting a contaienr in a pod.
func (r *Mock) RunContainerInPod(container v1.Container, pod *v1.Pod, volumeMap map[string]volume.VolumePlugin) error {
	args := r.Called(pod, pod, volumeMap)
	return args.Error(0)
}

// KillContainerInPod mocks killing a container in a pod.
func (r *Mock) KillContainerInPod(container v1.Container, pod *v1.Pod) error {
	args := r.Called(pod, pod)
	return args.Error(0)
}

// GetPodStatus mocks retrieving the PodStatus from the runtime.
func (r *Mock) GetPodStatus(uid types.UID, name, namespace string) (*container.PodStatus, error) {
	args := r.Called(uid, name, namespace)
	return args.Get(0).(*container.PodStatus), args.Error(1)
}

// ExecInContainer mockings running a command inside a container.
func (r *Mock) ExecInContainer(containerID container.ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize, timeout time.Duration) error {
	args := r.Called(containerID, cmd, stdin, stdout, stderr, tty)
	return args.Error(0)
}

// AttachContainer mocks attaching to a container.
func (r *Mock) AttachContainer(containerID container.ContainerID, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan remotecommand.TerminalSize) error {
	args := r.Called(containerID, stdin, stdout, stderr, tty)
	return args.Error(0)
}

// GetContainerLogs mocks retrieving container logs from the runtime.
func (r *Mock) GetContainerLogs(_ context.Context, pod *v1.Pod, containerID container.ContainerID, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) (err error) {
	args := r.Called(pod, containerID, logOptions, stdout, stderr)
	return args.Error(0)
}

// PullImage mocks pulling an image from the runtime.
func (r *Mock) PullImage(image container.ImageSpec, pullSecrets []v1.Secret, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	args := r.Called(image, pullSecrets)
	return image.Image, args.Error(0)
}

// GetImageRef mocks getting an imag ref from the runtime.
func (r *Mock) GetImageRef(image container.ImageSpec) (string, error) {
	args := r.Called(image)
	return args.Get(0).(string), args.Error(1)
}

// ListImages mocks listing images with the runtime.
func (r *Mock) ListImages() ([]container.Image, error) {
	args := r.Called()
	return args.Get(0).([]container.Image), args.Error(1)
}

// RemoveImage mocks removing an image with the runtime.
func (r *Mock) RemoveImage(image container.ImageSpec) error {
	args := r.Called(image)
	return args.Error(0)
}

// PortForward mocks port forwarding with a pod.
func (r *Mock) PortForward(pod *container.Pod, port uint16, stream io.ReadWriteCloser) error {
	args := r.Called(pod, port, stream)
	return args.Error(0)
}

// GarbageCollect mocks pod garbage collection.
func (r *Mock) GarbageCollect(gcPolicy container.GCPolicy, ready bool, evictNonDeletedPods bool) error {
	args := r.Called(gcPolicy, ready, evictNonDeletedPods)
	return args.Error(0)
}

// DeleteContainer mocks deleting a container.
func (r *Mock) DeleteContainer(containerID container.ContainerID) error {
	args := r.Called(containerID)
	return args.Error(0)
}

// ImageStats mocks retrieving image stats fr mthe runtime.
func (r *Mock) ImageStats() (*container.ImageStats, error) {
	args := r.Called()
	return args.Get(0).(*container.ImageStats), args.Error(1)
}

// UpdatePodCIDR fulfills the cri interface.
func (r *Mock) UpdatePodCIDR(c string) error {
	return nil
}
