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

package testing

import (
	"fmt"
	"io"
	"reflect"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	. "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/volume"
)

// FakeRuntime is a fake container runtime for testing.
type FakeRuntime struct {
	sync.Mutex
	CalledFunctions   []string
	PodList           []*Pod
	AllPodList        []*Pod
	ImageList         []Image
	APIPodStatus      api.PodStatus
	PodStatus         PodStatus
	StartedPods       []string
	KilledPods        []string
	StartedContainers []string
	KilledContainers  []string
	VersionInfo       string
	APIVersionInfo    string
	RuntimeType       string
	Err               error
	InspectErr        error
	StatusErr         error
}

// FakeRuntime should implement Runtime.
var _ Runtime = &FakeRuntime{}

type FakeVersion struct {
	Version string
}

func (fv *FakeVersion) String() string {
	return fv.Version
}

func (fv *FakeVersion) Compare(other string) (int, error) {
	result := 0
	if fv.Version > other {
		result = 1
	} else if fv.Version < other {
		result = -1
	}
	return result, nil
}

type podsGetter interface {
	GetPods(bool) ([]*Pod, error)
}

type FakeRuntimeCache struct {
	getter podsGetter
}

func NewFakeRuntimeCache(getter podsGetter) RuntimeCache {
	return &FakeRuntimeCache{getter}
}

func (f *FakeRuntimeCache) GetPods() ([]*Pod, error) {
	return f.getter.GetPods(false)
}

func (f *FakeRuntimeCache) ForceUpdateIfOlder(time.Time) error {
	return nil
}

// ClearCalls resets the FakeRuntime to the initial state.
func (f *FakeRuntime) ClearCalls() {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = []string{}
	f.PodList = []*Pod{}
	f.AllPodList = []*Pod{}
	f.APIPodStatus = api.PodStatus{}
	f.StartedPods = []string{}
	f.KilledPods = []string{}
	f.StartedContainers = []string{}
	f.KilledContainers = []string{}
	f.VersionInfo = ""
	f.RuntimeType = ""
	f.Err = nil
	f.InspectErr = nil
	f.StatusErr = nil
}

func (f *FakeRuntime) assertList(expect []string, test []string) error {
	if !reflect.DeepEqual(expect, test) {
		return fmt.Errorf("expected %#v, got %#v", expect, test)
	}
	return nil
}

// AssertCalls test if the invoked functions are as expected.
func (f *FakeRuntime) AssertCalls(calls []string) error {
	f.Lock()
	defer f.Unlock()
	return f.assertList(calls, f.CalledFunctions)
}

func (f *FakeRuntime) AssertStartedPods(pods []string) error {
	f.Lock()
	defer f.Unlock()
	return f.assertList(pods, f.StartedPods)
}

func (f *FakeRuntime) AssertKilledPods(pods []string) error {
	f.Lock()
	defer f.Unlock()
	return f.assertList(pods, f.KilledPods)
}

func (f *FakeRuntime) AssertStartedContainers(containers []string) error {
	f.Lock()
	defer f.Unlock()
	return f.assertList(containers, f.StartedContainers)
}

func (f *FakeRuntime) AssertKilledContainers(containers []string) error {
	f.Lock()
	defer f.Unlock()
	return f.assertList(containers, f.KilledContainers)
}

func (f *FakeRuntime) Type() string {
	return f.RuntimeType
}

func (f *FakeRuntime) Version() (Version, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "Version")
	return &FakeVersion{Version: f.VersionInfo}, f.Err
}

func (f *FakeRuntime) APIVersion() (Version, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "APIVersion")
	return &FakeVersion{Version: f.APIVersionInfo}, f.Err
}

func (f *FakeRuntime) Status() error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "Status")
	return f.StatusErr
}

func (f *FakeRuntime) GetPods(all bool) ([]*Pod, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetPods")
	if all {
		return f.AllPodList, f.Err
	}
	return f.PodList, f.Err
}

func (f *FakeRuntime) SyncPod(pod *api.Pod, _ api.PodStatus, _ *PodStatus, _ []api.Secret, backOff *flowcontrol.Backoff) (result PodSyncResult) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "SyncPod")
	f.StartedPods = append(f.StartedPods, string(pod.UID))
	for _, c := range pod.Spec.Containers {
		f.StartedContainers = append(f.StartedContainers, c.Name)
	}
	// TODO(random-liu): Add SyncResult for starting and killing containers
	if f.Err != nil {
		result.Fail(f.Err)
	}
	return
}

func (f *FakeRuntime) KillPod(pod *api.Pod, runningPod Pod, gracePeriodOverride *int64) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "KillPod")
	f.KilledPods = append(f.KilledPods, string(runningPod.ID))
	for _, c := range runningPod.Containers {
		f.KilledContainers = append(f.KilledContainers, c.Name)
	}
	return f.Err
}

func (f *FakeRuntime) RunContainerInPod(container api.Container, pod *api.Pod, volumeMap map[string]volume.VolumePlugin) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "RunContainerInPod")
	f.StartedContainers = append(f.StartedContainers, container.Name)

	pod.Spec.Containers = append(pod.Spec.Containers, container)
	for _, c := range pod.Spec.Containers {
		if c.Name == container.Name { // Container already in the pod.
			return f.Err
		}
	}
	pod.Spec.Containers = append(pod.Spec.Containers, container)
	return f.Err
}

func (f *FakeRuntime) KillContainerInPod(container api.Container, pod *api.Pod) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "KillContainerInPod")
	f.KilledContainers = append(f.KilledContainers, container.Name)

	var containers []api.Container
	for _, c := range pod.Spec.Containers {
		if c.Name == container.Name {
			continue
		}
		containers = append(containers, c)
	}
	return f.Err
}

func (f *FakeRuntime) GetPodStatus(uid types.UID, name, namespace string) (*PodStatus, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetPodStatus")
	status := f.PodStatus
	return &status, f.Err
}

func (f *FakeRuntime) ExecInContainer(containerID ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "ExecInContainer")
	return f.Err
}

func (f *FakeRuntime) AttachContainer(containerID ContainerID, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "AttachContainer")
	return f.Err
}

func (f *FakeRuntime) RunInContainer(containerID ContainerID, cmd []string) ([]byte, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "RunInContainer")
	return []byte{}, f.Err
}

func (f *FakeRuntime) GetContainerLogs(pod *api.Pod, containerID ContainerID, logOptions *api.PodLogOptions, stdout, stderr io.Writer) (err error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetContainerLogs")
	return f.Err
}

func (f *FakeRuntime) PullImage(image ImageSpec, pullSecrets []api.Secret) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "PullImage")
	return f.Err
}

func (f *FakeRuntime) IsImagePresent(image ImageSpec) (bool, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "IsImagePresent")
	for _, i := range f.ImageList {
		if i.ID == image.Image {
			return true, nil
		}
	}
	return false, f.InspectErr
}

func (f *FakeRuntime) ListImages() ([]Image, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "ListImages")
	return f.ImageList, f.Err
}

func (f *FakeRuntime) RemoveImage(image ImageSpec) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "RemoveImage")
	index := 0
	for i := range f.ImageList {
		if f.ImageList[i].ID == image.Image {
			index = i
			break
		}
	}
	f.ImageList = append(f.ImageList[:index], f.ImageList[index+1:]...)

	return f.Err
}

func (f *FakeRuntime) PortForward(pod *Pod, port uint16, stream io.ReadWriteCloser) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "PortForward")
	return f.Err
}

func (f *FakeRuntime) GarbageCollect(gcPolicy ContainerGCPolicy) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GarbageCollect")
	return f.Err
}

func (f *FakeRuntime) ImageStats() (*ImageStats, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "ImageStats")
	return nil, f.Err
}
