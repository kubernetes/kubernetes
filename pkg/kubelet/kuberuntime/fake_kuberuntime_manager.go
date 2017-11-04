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
	"k8s.io/kubernetes/pkg/credentialprovider"
	internalapi "k8s.io/kubernetes/pkg/kubelet/apis/cri"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
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

type fakePodStateProvider struct {
	existingPods map[types.UID]struct{}
	runningPods  map[types.UID]struct{}
}

func newFakePodStateProvider() *fakePodStateProvider {
	return &fakePodStateProvider{
		existingPods: make(map[types.UID]struct{}),
		runningPods:  make(map[types.UID]struct{}),
	}
}

func (f *fakePodStateProvider) IsPodDeleted(uid types.UID) bool {
	_, found := f.existingPods[uid]
	return !found
}

func (f *fakePodStateProvider) IsPodTerminated(uid types.UID) bool {
	_, found := f.runningPods[uid]
	return !found
}

func NewFakeKubeRuntimeManager(runtimeService internalapi.RuntimeService, imageService internalapi.ImageManagerService, machineInfo *cadvisorapi.MachineInfo, osInterface kubecontainer.OSInterface, runtimeHelper kubecontainer.RuntimeHelper, keyring credentialprovider.DockerKeyring) (*kubeGenericRuntimeManager, error) {
	recorder := &record.FakeRecorder{}
	kubeRuntimeManager := &kubeGenericRuntimeManager{
		recorder:            recorder,
		cpuCFSQuota:         false,
		livenessManager:     proberesults.NewManager(),
		containerRefManager: kubecontainer.NewRefManager(),
		machineInfo:         machineInfo,
		osInterface:         osInterface,
		runtimeHelper:       runtimeHelper,
		runtimeService:      runtimeService,
		imageService:        imageService,
		keyring:             keyring,
		internalLifecycle:   cm.NewFakeInternalContainerLifecycle(),
	}

	typedVersion, err := runtimeService.Version(kubeRuntimeAPIVersion)
	if err != nil {
		return nil, err
	}

	kubeRuntimeManager.containerGC = NewContainerGC(runtimeService, newFakePodStateProvider(), kubeRuntimeManager)
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
