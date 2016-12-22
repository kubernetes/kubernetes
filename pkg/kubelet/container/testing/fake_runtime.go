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
	"fmt"
	"io"
	"net/url"
	"reflect"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	. "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/term"
	"k8s.io/kubernetes/pkg/volume"
)

type FakePod struct {
	Pod       *Pod
	NetnsPath string
}

// FakeRuntime is a fake container runtime for testing.
type FakeRuntime struct {
	sync.Mutex
	CalledFunctions   []string
	PodList           []*FakePod
	AllPodList        []*FakePod
	ImageList         []Image
	APIPodStatus      v1.PodStatus
	PodStatus         PodStatus
	StartedPods       []string
	KilledPods        []string
	StartedContainers []string
	KilledContainers  []string
	RuntimeStatus     *RuntimeStatus
	VersionInfo       string
	APIVersionInfo    string
	RuntimeType       string
	Err               error
	InspectErr        error
	StatusErr         error
}

type FakeDirectStreamingRuntime struct {
	*FakeRuntime

	// Arguments to streaming method calls.
	Args struct {
		// Attach / Exec args
		ContainerID ContainerID
		Cmd         []string
		Stdin       io.Reader
		Stdout      io.WriteCloser
		Stderr      io.WriteCloser
		TTY         bool
		// Port-forward args
		Pod    *Pod
		Port   uint16
		Stream io.ReadWriteCloser
	}
}

var _ DirectStreamingRuntime = &FakeDirectStreamingRuntime{}

const FakeHost = "localhost:12345"

type FakeIndirectStreamingRuntime struct {
	*FakeRuntime
}

var _ IndirectStreamingRuntime = &FakeIndirectStreamingRuntime{}

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

func (f *FakeRuntime) Status() (*RuntimeStatus, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "Status")
	return f.RuntimeStatus, f.StatusErr
}

func (f *FakeRuntime) GetPods(all bool) ([]*Pod, error) {
	f.Lock()
	defer f.Unlock()

	var pods []*Pod

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

func (f *FakeRuntime) SyncPod(pod *v1.Pod, _ v1.PodStatus, _ *PodStatus, _ []v1.Secret, backOff *flowcontrol.Backoff) (result PodSyncResult) {
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

func (f *FakeRuntime) KillPod(pod *v1.Pod, runningPod Pod, gracePeriodOverride *int64) error {
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

	var containers []v1.Container
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

func (f *FakeDirectStreamingRuntime) ExecInContainer(containerID ContainerID, cmd []string, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size, timeout time.Duration) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "ExecInContainer")
	f.Args.ContainerID = containerID
	f.Args.Cmd = cmd
	f.Args.Stdin = stdin
	f.Args.Stdout = stdout
	f.Args.Stderr = stderr
	f.Args.TTY = tty

	return f.Err
}

func (f *FakeDirectStreamingRuntime) AttachContainer(containerID ContainerID, stdin io.Reader, stdout, stderr io.WriteCloser, tty bool, resize <-chan term.Size) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "AttachContainer")
	f.Args.ContainerID = containerID
	f.Args.Stdin = stdin
	f.Args.Stdout = stdout
	f.Args.Stderr = stderr
	f.Args.TTY = tty

	return f.Err
}

func (f *FakeRuntime) GetContainerLogs(pod *v1.Pod, containerID ContainerID, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) (err error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetContainerLogs")
	return f.Err
}

func (f *FakeRuntime) PullImage(image ImageSpec, pullSecrets []v1.Secret) error {
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

func (f *FakeDirectStreamingRuntime) PortForward(pod *Pod, port uint16, stream io.ReadWriteCloser) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "PortForward")
	f.Args.Pod = pod
	f.Args.Port = port
	f.Args.Stream = stream

	return f.Err
}

func (f *FakeRuntime) GetNetNS(containerID ContainerID) (string, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetNetNS")

	for _, fp := range f.AllPodList {
		for _, c := range fp.Pod.Containers {
			if c.ID == containerID {
				return fp.NetnsPath, nil
			}
		}
	}

	return "", f.Err
}

func (f *FakeRuntime) GetPodContainerID(pod *Pod) (ContainerID, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetPodContainerID")
	return ContainerID{}, f.Err
}

func (f *FakeRuntime) GarbageCollect(gcPolicy ContainerGCPolicy, ready bool) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GarbageCollect")
	return f.Err
}

func (f *FakeRuntime) DeleteContainer(containerID ContainerID) error {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "DeleteContainer")
	return f.Err
}

func (f *FakeRuntime) ImageStats() (*ImageStats, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "ImageStats")
	return nil, f.Err
}

func (f *FakeIndirectStreamingRuntime) GetExec(id ContainerID, cmd []string, stdin, stdout, stderr, tty bool) (*url.URL, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetExec")
	return &url.URL{Host: FakeHost}, f.Err
}

func (f *FakeIndirectStreamingRuntime) GetAttach(id ContainerID, stdin, stdout, stderr, tty bool) (*url.URL, error) {
	f.Lock()
	defer f.Unlock()

	f.CalledFunctions = append(f.CalledFunctions, "GetAttach")
	return &url.URL{Host: FakeHost}, f.Err
}

func (f *FakeIndirectStreamingRuntime) GetPortForward(podName, podNamespace string, podUID types.UID) (*url.URL, error) {
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
	ContainerID ContainerID
	Cmd         []string
}

var _ ContainerCommandRunner = &FakeContainerCommandRunner{}

func (f *FakeContainerCommandRunner) RunInContainer(containerID ContainerID, cmd []string, timeout time.Duration) ([]byte, error) {
	// record invoked values
	f.ContainerID = containerID
	f.Cmd = cmd

	return []byte(f.Stdout), f.Err
}
