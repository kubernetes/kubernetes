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
	internalApi "k8s.io/kubernetes/pkg/kubelet/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
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

// fakeRuntimeHelper implements kubecontainer.RuntimeHelper interfaces for testing purposes.
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

func NewFakeKubeRuntimeManager(runtimeService internalApi.RuntimeService, imageService internalApi.ImageManagerService) (*kubeGenericRuntimeManager, error) {
	networkPlugin, _ := network.InitNetworkPlugin(
		[]network.NetworkPlugin{},
		"",
		nettest.NewFakeHost(nil),
		componentconfig.HairpinNone,
		"10.0.0.0/8",
	)

	return NewKubeGenericRuntimeManager(
		&record.FakeRecorder{},
		proberesults.NewManager(),
		kubecontainer.NewRefManager(),
		&containertest.FakeOS{},
		networkPlugin,
		&fakeRuntimeHelper{},
		&fakeHTTP{},
		flowcontrol.NewBackOff(time.Second, 300*time.Second),
		false,
		false,
		runtimeService,
		imageService,
	)
}
