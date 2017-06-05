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
	"net/http"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/credentialprovider"
	internalapi "k8s.io/kubernetes/pkg/kubelet/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/network"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
)

type fakeHTTP struct {
	url string
	err error
}

func (f *fakeHTTP) Get(url string) (*http.Response, error) {
	f.url = url
	return nil, f.err
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

func NewFakeKubeRuntimeManager(runtimeService internalapi.RuntimeService, imageService internalapi.ImageManagerService, machineInfo *cadvisorapi.MachineInfo, networkPlugin network.NetworkPlugin, osInterface kubecontainer.OSInterface, runtimeHelper kubecontainer.RuntimeHelper) (*kubeGenericRuntimeManager, error) {
	recorder := &record.FakeRecorder{}
	kubeRuntimeManager := &kubeGenericRuntimeManager{
		recorder:            recorder,
		cpuCFSQuota:         false,
		livenessManager:     proberesults.NewManager(),
		containerRefManager: kubecontainer.NewRefManager(),
		machineInfo:         machineInfo,
		osInterface:         osInterface,
		networkPlugin:       networkPlugin,
		runtimeHelper:       runtimeHelper,
		runtimeService:      runtimeService,
		imageService:        imageService,
		keyring:             credentialprovider.NewDockerKeyring(),
	}

	typedVersion, err := runtimeService.Version(kubeRuntimeAPIVersion)
	if err != nil {
		return nil, err
	}

	kubeRuntimeManager.containerGC = NewContainerGC(runtimeService, newFakePodGetter(), kubeRuntimeManager)
	kubeRuntimeManager.runtimeName = typedVersion.RuntimeName
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
