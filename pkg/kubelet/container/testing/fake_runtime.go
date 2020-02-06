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
	"fmt"
	"io"
	"net/url"
	"reflect"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/flowcontrol"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/volume"
)

type FakePod struct {
	Pod       *kubecontainer.Pod
	NetnsPath string
}

// FakeRuntime is a fake container runtime for testing.
type FakeRuntime struct {
	sync.Mutex
	CalledFunctions   []string
	PodList           []*FakePod
	AllPodList        []*FakePod
	ImageList         []kubecontainer.Image
	APIPodStatus      v1.PodStatus
	PodStatus         kubecontainer.PodStatus
	StartedPods       []string
	KilledPods        []string
	StartedContainers []string
	KilledContainers  []string
	RuntimeStatus     *kubecontainer.RuntimeStatus
	VersionInfo       string
	APIVersionInfo    string
	RuntimeType       string
	Err               error
	InspectErr        error
	StatusErr         error
}

const FakeHost = "localhost:12345"

type FakeStreamingRuntime struct {
	*FakeRuntime
}

var _ kubecontainer.StreamingRuntime = &FakeStreamingRuntime{}

// FakeRuntime should implement Runtime.
var _ kubecontainer.Runtime = &FakeRuntime{}

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
	GetPods(bool) ([]*kubecontainer.Pod, error)
}

type FakeRuntimeCache struct {
	getter podsGetter
}

func NewFakeRuntimeCache(getter podsGetter) kubecontainer.RuntimeCache {
	return &FakeRuntimeCache{getter}
}

func (f *FakeRuntimeCache) GetPods() ([]*kubecontainer.Pod, error) {
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
	f.PodList = []*FakePod{}
	f.AllPodList = []*FakePod{}
	f.APIPodStatus = v1.PodStatus{}
	f.StartedPods = []string{}
	f.KilledPods = []string{}
	f.StartedContainers = []string{}
	f.KilledContainers = []string{}
	f.RuntimeStatus = nil
	f.VersionInfo = ""
	f.RuntimeType = ""
	f.Err = nil
	f.InspectErr = nil
	f.StatusErr = nil
}

// UpdatePodCIDR fulfills the cri interface.
func (f *FakeRuntime) UpdatePodCIDR(c string) error {
	return nil
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

func (f *FakeRuntime) SupportsSingleFileMapping() bool {
	return true
}

func (f *FakeRuntime) Version() (kubecontainer.Version, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "Version")
	return &FakeVersion{Version: f.VersionInfo}, f.Err
}

func (f *FakeRuntime) APIVersion() (kubecontainer.Version, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "APIVersion")
	return &FakeVersion{Version: f.APIVersionInfo}, f.Err
}

func (f *FakeRuntime) Status() (*kubecontainer.RuntimeStatus, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "Status")
	return f.RuntimeStatus, f.StatusErr
}

func (f *FakeRuntime) GetPods(all bool) ([]*kubecontainer.Pod, error) {
	f.Lock()
	defer f.Unlock()

	var pods []*kubecontainer.Pod

	f.CalledFunctions = append(f.CalledFunctions, "GetPods")
	if all {
		for _, fakePod := range f.AllPodList {
			pods = append(pods, fakePod.Pod)
		}
	} else {
		for _, fakePod := range f.PodList {
			pods = append(pods, fakePod.Pod)
		}
	}
	return pods, f.Err
}

func (f *FakeRuntime) SyncPod(pod *v1.Pod, _ *kubecontainer.PodStatus, _ []v1.Secret, backOff *flowcontrol.Backoff) (result kubecontainer.PodSyncResult) {
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

func (f *FakeRuntime) KillPod(pod *v1.Pod, runningPod kubecontainer.Pod, gracePeriodOverride *int64) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "KillPod")
	f.KilledPods = append(f.KilledPods, string(runningPod.ID))
	for _, c := range runningPod.Containers {
		f.KilledContainers = append(f.KilledContainers, c.Name)
	}
	return f.Err
}

func (f *FakeRuntime) RunContainerInPod(container v1.Container, pod *v1.Pod, volumeMap map[string]volume.VolumePlugin) error {
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

func (f *FakeRuntime) KillContainerInPod(container v1.Container, pod *v1.Pod) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "KillContainerInPod")
	f.KilledContainers = append(f.KilledContainers, container.Name)
	return f.Err
}

func (f *FakeRuntime) GetPodStatus(uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetPodStatus")
	status := f.PodStatus
	return &status, f.Err
}

func (f *FakeRuntime) GetContainerLogs(_ context.Context, pod *v1.Pod, containerID kubecontainer.ContainerID, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) (err error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetContainerLogs")
	return f.Err
}

func (f *FakeRuntime) PullImage(image kubecontainer.ImageSpec, pullSecrets []v1.Secret, podSandboxConfig *runtimeapi.PodSandboxConfig) (string, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "PullImage")
	return image.Image, f.Err
}

func (f *FakeRuntime) GetImageRef(image kubecontainer.ImageSpec) (string, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetImageRef")
	for _, i := range f.ImageList {
		if i.ID == image.Image {
			return i.ID, nil
		}
	}
	return "", f.InspectErr
}

func (f *FakeRuntime) ListImages() ([]kubecontainer.Image, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "ListImages")
	return f.ImageList, f.Err
}

func (f *FakeRuntime) RemoveImage(image kubecontainer.ImageSpec) error {
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

func (f *FakeRuntime) GarbageCollect(gcPolicy kubecontainer.ContainerGCPolicy, ready bool, evictNonDeletedPods bool) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GarbageCollect")
	return f.Err
}

func (f *FakeRuntime) DeleteContainer(containerID kubecontainer.ContainerID) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "DeleteContainer")
	return f.Err
}

func (f *FakeRuntime) ImageStats() (*kubecontainer.ImageStats, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "ImageStats")
	return nil, f.Err
}

func (f *FakeStreamingRuntime) GetExec(id kubecontainer.ContainerID, cmd []string, stdin, stdout, stderr, tty bool) (*url.URL, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetExec")
	return &url.URL{Host: FakeHost}, f.Err
}

func (f *FakeStreamingRuntime) GetAttach(id kubecontainer.ContainerID, stdin, stdout, stderr, tty bool) (*url.URL, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetAttach")
	return &url.URL{Host: FakeHost}, f.Err
}

func (f *FakeStreamingRuntime) GetPortForward(podName, podNamespace string, podUID types.UID, ports []int32) (*url.URL, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetPortForward")
	return &url.URL{Host: FakeHost}, f.Err
}

type FakeContainerCommandRunner struct {
	// what to return
	Stdout string
	Err    error

	// actual values when invoked
	ContainerID kubecontainer.ContainerID
	Cmd         []string
}

var _ kubecontainer.ContainerCommandRunner = &FakeContainerCommandRunner{}

func (f *FakeContainerCommandRunner) RunInContainer(containerID kubecontainer.ContainerID, cmd []string, timeout time.Duration) ([]byte, error) {
	// record invoked values
	f.ContainerID = containerID
	f.Cmd = cmd

	return []byte(f.Stdout), f.Err
}
