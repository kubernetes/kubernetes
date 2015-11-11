/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"fmt"
	"net"
	"net/url"
	"reflect"
	"sync"
	"testing"
	"time"

	cadvisorApi "github.com/google/cadvisor/info/v1"
	cadvisorApiv2 "github.com/google/cadvisor/info/v2"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/probe"
	httprobe "k8s.io/kubernetes/pkg/probe/http"
	"k8s.io/kubernetes/pkg/util"
)

// networkPluginMock returns a status update with both a podIP and localIP
type networkPluginMock struct {
	statusValues map[string]*network.PodNetworkStatus
}

func (m *networkPluginMock) Init(host network.Host) error {
	return nil
}

func (m *networkPluginMock) Name() string {
	return "networkPluginMock"
}

func (m *networkPluginMock) SetUpPod(namespace string, name string, podInfraContainerID kubetypes.DockerID) error {
	return nil
}

func (m *networkPluginMock) TearDownPod(namespace string, name string, podInfraContainerID kubetypes.DockerID) error {
	return nil
}

func (m *networkPluginMock) Status(namespace string, name string, podInfraContainerID kubetypes.DockerID) (*network.PodNetworkStatus, error) {
	podFullName := fmt.Sprintf("%s/%s", namespace, name)
	if status, ok := m.statusValues[podFullName]; ok {
		return status, nil
	}
	return nil, fmt.Errorf("unexpected status call for %s", podFullName)
}

// mockHttpProber records the host component of url requests.
type mockHttpProber struct {
	requestHosts []string
	waitGroup    *sync.WaitGroup
	mutex        sync.Mutex
}

func newMockHttpProber(wg *sync.WaitGroup) *mockHttpProber {
	mock := new(mockHttpProber)
	mock.requestHosts = make([]string, 0)
	mock.waitGroup = wg
	return mock
}

func (m *mockHttpProber) Probe(url *url.URL, timeout time.Duration) (probe.Result, string, error) {
	m.mutex.Lock()
	defer m.mutex.Unlock()
	m.requestHosts = append(m.requestHosts, url.Host)
	m.waitGroup.Done()
	return probe.Success, "", nil
}

type fakeOptionGenerator struct{}

func (*fakeOptionGenerator) GenerateRunContainerOptions(pod *api.Pod, container *api.Container) (*kubecontainer.RunContainerOptions, error) {
	opts := &kubecontainer.RunContainerOptions{}
	return opts, nil
}

func initFakeRuntime(kubelet *Kubelet, httpProber httprobe.HTTPProber) {
	fakeDocker := &dockertools.FakeDockerClient{}

	kubelet.probeManager = prober.NewTestManagerWithHttpProbe(
		120*time.Second,
		kubelet.statusManager,
		proberesults.NewManager(),
		kubelet.livenessManager,
		kubelet.runner,
		kubelet.containerRefManager,
		kubelet.recorder,
		httpProber)

	dockerManager := dockertools.NewFakeDockerManager(
		fakeDocker, kubelet.recorder, kubelet.livenessManager, kubelet.containerRefManager, &cadvisorApi.MachineInfo{},
		dockertools.PodInfraContainerImage, 0, 0, "", kubelet.os,
		kubelet.networkPlugin, &fakeOptionGenerator{}, nil, nil)

	kubelet.containerRuntime = dockerManager
	kubelet.runtimeCache = kubecontainer.NewFakeRuntimeCache(dockerManager)
}

func TestKubeletProbeAddress(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet

	kubelet.nodeLister = testNodeLister{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
			Status:     api.NodeStatus{Addresses: []api.NodeAddress{{api.NodeInternalIP, testKubeletHostname}}},
		},
	}}

	networkMock := &networkPluginMock{}
	networkMock.statusValues = make(map[string]*network.PodNetworkStatus)
	kubelet.networkPlugin, _ = network.InitNetworkPlugin(
		[]network.NetworkPlugin{networkMock}, "networkPluginMock", network.NewFakeHost(kubelet.kubeClient))

	wg := &sync.WaitGroup{}
	mockHttpProber := newMockHttpProber(wg)

	initFakeRuntime(kubelet, mockHttpProber)

	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)

	probe := &api.Probe{
		Handler: api.Handler{
			HTTPGet: &api.HTTPGetAction{
				Scheme: api.URISchemeHTTP,
				Path:   "healthz",
				Port:   util.NewIntOrStringFromInt(8080),
			},
		},
	}

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				Namespace:   "ns",
				Name:        "test1",
				UID:         "61a9c401da78",
				Annotations: map[string]string{},
				SelfLink:    testapi.Default.SelfLink("pod", ""),
			},
			Spec: api.PodSpec{Containers: []api.Container{{Name: "test1-c1", Image: "fake-image", LivenessProbe: probe}}},
		},
		{
			ObjectMeta: api.ObjectMeta{
				Namespace:   "ns",
				Name:        "test2",
				UID:         "e267f88312c1",
				Annotations: map[string]string{},
				SelfLink:    testapi.Default.SelfLink("pod", ""),
			},
			Spec: api.PodSpec{Containers: []api.Container{{Name: "test2-c1", Image: "fake-image", LivenessProbe: probe}}},
		},
	}

	testValues := []struct {
		podName  string
		status   *network.PodNetworkStatus
		expected string
	}{
		{
			"ns/test1",
			&network.PodNetworkStatus{IP: net.ParseIP("10.0.255.250"), LocalIP: net.ParseIP("169.254.0.2")},
			"169.254.0.2:8080",
		},
		{
			"ns/test2",
			&network.PodNetworkStatus{IP: net.ParseIP("10.0.255.251")},
			"10.0.255.251:8080",
		},
	}
	for _, test := range testValues {
		networkMock.statusValues[test.podName] = test.status
	}
	wg.Add(len(testValues))
	kubelet.HandlePodAdditions(pods)

	wg.Wait()

	for i, test := range testValues {
		if !reflect.DeepEqual(mockHttpProber.requestHosts[i], test.expected) {
			t.Errorf("expected %+v, got %+v", test.expected, mockHttpProber.requestHosts[i])
		}
	}
}
