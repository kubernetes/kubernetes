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
	"io/ioutil"
	"net/http"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/apimachinery/pkg/types"
	kubetypes "k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/credentialprovider"
	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/network"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

type fakeHTTP struct {
	url string
	err error
}

func (f *fakeHTTP) Get(url string) (*http.Response, error) {
	f.url = url
	return nil, f.err
}

// fakeRuntimeHelper implements kubecontainer.RuntimeHelper interfaces for testing purposes.
type fakeRuntimeHelper struct{}

func (f *fakeRuntimeHelper) GenerateRunContainerOptions(pod *v1.Pod, container *v1.Container, podIP string) (*kubecontainer.RunContainerOptions, error) {
	var opts kubecontainer.RunContainerOptions
	if len(container.TerminationMessagePath) != 0 {
		testPodContainerDir, err := ioutil.TempDir("", "fooPodContainerDir")
		if err != nil {
			return nil, err
		}
		opts.PodContainerDir = testPodContainerDir
	}
	return &opts, nil
}

func (f *fakeRuntimeHelper) GetClusterDNS(pod *v1.Pod) ([]string, []string, error) {
	return nil, nil, nil
}

// This is not used by docker runtime.
func (f *fakeRuntimeHelper) GeneratePodHostNameAndDomain(pod *v1.Pod) (string, string, error) {
	return "", "", nil
}

func (f *fakeRuntimeHelper) GetPodDir(kubetypes.UID) string {
	return ""
}

func (f *fakeRuntimeHelper) GetExtraSupplementalGroupsForPod(pod *v1.Pod) []int64 {
	return nil
}

type fakePodGetter struct {
	pods map[types.UID]*v1.Pod
}

func newFakePodGetter() *fakePodGetter {
	return &fakePodGetter{make(map[types.UID]*v1.Pod)}
}

func (f *fakePodGetter) GetPodByUID(uid types.UID) (*v1.Pod, bool) {
	pod, found := f.pods[uid]
	return pod, found
}

func NewFakeKubeRuntimeManager(runtimeService internalapi.RuntimeService, imageService internalapi.ImageManagerService, machineInfo *cadvisorapi.MachineInfo, networkPlugin network.NetworkPlugin, osInterface kubecontainer.OSInterface) (*kubeGenericRuntimeManager, error) {
	recorder := &record.FakeRecorder{}
	kubeRuntimeManager := &kubeGenericRuntimeManager{
		recorder:            recorder,
		cpuCFSQuota:         false,
		livenessManager:     proberesults.NewManager(),
		containerRefManager: kubecontainer.NewRefManager(),
		machineInfo:         machineInfo,
		osInterface:         osInterface,
		networkPlugin:       networkPlugin,
		runtimeHelper:       &fakeRuntimeHelper{},
		runtimeService:      runtimeService,
		imageService:        imageService,
		keyring:             credentialprovider.NewDockerKeyring(),
	}

	typedVersion, err := runtimeService.Version(kubeRuntimeAPIVersion)
	if err != nil {
		return nil, err
	}

	kubeRuntimeManager.containerGC = NewContainerGC(runtimeService, newFakePodGetter(), kubeRuntimeManager)
	kubeRuntimeManager.runtimeName = typedVersion.GetRuntimeName()
	kubeRuntimeManager.imagePuller = images.NewImageManager(
		kubecontainer.FilterEventRecorder(recorder),
		kubeRuntimeManager,
		flowcontrol.NewBackOff(time.Second, 300*time.Second),
		false,
		0, // Disable image pull throttling by setting QPS to 0,
		0,
	)
	kubeRuntimeManager.runner = lifecycle.NewHandlerRunner(
		&fakeHTTP{},
		kubeRuntimeManager,
		kubeRuntimeManager)

	return kubeRuntimeManager, nil
}
