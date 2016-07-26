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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/credentialprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/network"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	kubetypes "k8s.io/kubernetes/pkg/types"
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

// fakeRuntimeHelper implements kubecontainer.RuntimeHelper inter
// faces for testing purposes.
type fakeRuntimeHelper struct{}

func (f *fakeRuntimeHelper) GenerateRunContainerOptions(pod *api.Pod, container *api.Container, podIP string) (*kubecontainer.RunContainerOptions, error) {
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

func (f *fakeRuntimeHelper) GetClusterDNS(pod *api.Pod) ([]string, []string, error) {
	return nil, nil, nil
}

// This is not used by docker runtime.
func (f *fakeRuntimeHelper) GeneratePodHostNameAndDomain(pod *api.Pod) (string, string, error) {
	return "", "", nil
}

func (f *fakeRuntimeHelper) GetPodDir(kubetypes.UID) string {
	return ""
}

func (f *fakeRuntimeHelper) GetExtraSupplementalGroupsForPod(pod *api.Pod) []int64 {
	return nil
}

func NewFakeKubeRuntimeManager(runtime *fakeKubeRuntime, host network.Host, osInterface kubecontainer.OSInterface) *kubeGenericRuntimeManager {
	networkPlugin, _ := network.InitNetworkPlugin(
		[]network.NetworkPlugin{},
		"",
		host,
		componentconfig.HairpinNone,
		"10.0.0.0/8",
	)

	imageBackOff := flowcontrol.NewBackOff(time.Second, 300*time.Second)
	recorder := &record.FakeRecorder{}
	kubeRuntimeManager := &kubeGenericRuntimeManager{
		runtimeName:         fakeRuntimeName,
		recorder:            recorder,
		cpuCFSQuota:         false,
		livenessManager:     proberesults.NewManager(),
		containerRefManager: kubecontainer.NewRefManager(),
		osInterface:         osInterface,
		networkPlugin:       networkPlugin,
		runtimeHelper:       &fakeRuntimeHelper{},
		runtimeService:      runtime,
		imageService:        runtime,
		keyring:             credentialprovider.NewDockerKeyring(),
	}
	kubeRuntimeManager.imagePuller = images.NewImageManager(
		kubecontainer.FilterEventRecorder(recorder),
		kubeRuntimeManager,
		imageBackOff,
		false)
	kubeRuntimeManager.runner = lifecycle.NewHandlerRunner(&fakeHTTP{}, kubeRuntimeManager, kubeRuntimeManager)

	return kubeRuntimeManager
}
