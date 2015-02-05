/*
Copyright 2014 Google Inc. All rights reserved.

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

package master

import (
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/leaky"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type podInfoCall struct {
	host      string
	namespace string
	name      string
}

type podInfoResponse struct {
	useCount int
	data     api.PodStatusResult
	err      error
}

type podInfoCalls map[podInfoCall]*podInfoResponse

type FakePodInfoGetter struct {
	calls podInfoCalls
	lock  sync.Mutex

	// default data/error to return, or you can add
	// responses to specific calls-- that will take precedence.
	data api.PodStatusResult
	err  error
}

func (f *FakePodInfoGetter) GetPodStatus(host, namespace, name string) (api.PodStatusResult, error) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.calls == nil {
		f.calls = podInfoCalls{}
	}

	key := podInfoCall{host, namespace, name}
	call, ok := f.calls[key]
	if !ok {
		f.calls[key] = &podInfoResponse{
			0, f.data, f.err,
		}
		call = f.calls[key]
	}
	call.useCount++
	return call.data, call.err
}

func TestPodCacheGetDifferentNamespace(t *testing.T) {
	cache := NewPodCache(nil, nil, nil)

	expectedDefault := api.PodStatus{
		Info: api.PodInfo{
			"foo": api.ContainerStatus{},
		},
	}
	expectedOther := api.PodStatus{
		Info: api.PodInfo{
			"bar": api.ContainerStatus{},
		},
	}

	cache.podStatus[objKey{api.NamespaceDefault, "foo"}] = expectedDefault
	cache.podStatus[objKey{"other", "foo"}] = expectedOther

	info, err := cache.GetPodStatus(api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %+v", err)
	}
	if !reflect.DeepEqual(info, &expectedDefault) {
		t.Errorf("Unexpected mismatch. Expected: %+v, Got: %+v", &expectedOther, info)
	}

	info, err = cache.GetPodStatus("other", "foo")
	if err != nil {
		t.Errorf("Unexpected error: %+v", err)
	}
	if !reflect.DeepEqual(info, &expectedOther) {
		t.Errorf("Unexpected mismatch. Expected: %+v, Got: %+v", &expectedOther, info)
	}
}

func TestPodCacheGet(t *testing.T) {
	cache := NewPodCache(nil, nil, nil)

	expected := api.PodStatus{
		Info: api.PodInfo{
			"foo": api.ContainerStatus{},
		},
	}
	cache.podStatus[objKey{api.NamespaceDefault, "foo"}] = expected

	info, err := cache.GetPodStatus(api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %+v", err)
	}
	if !reflect.DeepEqual(info, &expected) {
		t.Errorf("Unexpected mismatch. Expected: %+v, Got: %+v", &expected, info)
	}
}

func TestPodCacheDelete(t *testing.T) {
	config := podCacheTestConfig{
		err: client.ErrPodInfoNotAvailable,
	}
	cache := config.Construct()

	expected := api.PodStatus{
		Info: api.PodInfo{
			"foo": api.ContainerStatus{},
		},
	}
	cache.podStatus[objKey{api.NamespaceDefault, "foo"}] = expected

	info, err := cache.GetPodStatus(api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %+v", err)
	}
	if !reflect.DeepEqual(info, &expected) {
		t.Errorf("Unexpected mismatch. Expected: %+v, Got: %+v", &expected, info)
	}

	cache.ClearPodStatus(api.NamespaceDefault, "foo")

	_, err = cache.GetPodStatus(api.NamespaceDefault, "foo")
	if err == nil {
		t.Errorf("Unexpected non-error after deleting")
	}
	if err != client.ErrPodInfoNotAvailable {
		t.Errorf("Unexpected error: %v, expecting: %v", err, client.ErrPodInfoNotAvailable)
	}
}

func TestPodCacheGetMissing(t *testing.T) {
	pod1 := makePod(api.NamespaceDefault, "foo", "machine", "bar")
	config := podCacheTestConfig{
		kubeletContainerInfo: api.PodStatus{
			Info: api.PodInfo{"bar": api.ContainerStatus{}}},
		nodes: []api.Node{*makeHealthyNode("machine", "1.2.3.5")},
		pod:   pod1,
	}
	cache := config.Construct()

	status, err := cache.GetPodStatus(api.NamespaceDefault, "foo")
	if err != nil {
		t.Errorf("Unexpected error: %+v", err)
	}
	if status == nil {
		t.Errorf("Unexpected non-status.")
	}
	expected := &api.PodStatus{
		Phase:  "Pending",
		Host:   "machine",
		HostIP: "1.2.3.5",
		Info: api.PodInfo{
			"bar": api.ContainerStatus{},
		},
	}
	if !reflect.DeepEqual(status, expected) {
		t.Errorf("expected:\n%#v\ngot:\n%#v\n", expected, status)
	}
}

type podCacheTestConfig struct {
	nodes                []api.Node
	pods                 []api.Pod
	pod                  *api.Pod
	err                  error
	kubeletContainerInfo api.PodStatus

	// Construct will fill in these fields
	fakePodInfo *FakePodInfoGetter
	fakeNodes   *client.Fake
	fakePods    *registrytest.PodRegistry
}

func (c *podCacheTestConfig) Construct() *PodCache {
	c.fakePodInfo = &FakePodInfoGetter{
		data: api.PodStatusResult{
			Status: c.kubeletContainerInfo,
		},
	}
	c.fakeNodes = &client.Fake{
		MinionsList: api.NodeList{
			Items: c.nodes,
		},
	}
	c.fakePods = registrytest.NewPodRegistry(&api.PodList{Items: c.pods})
	c.fakePods.Pod = c.pod
	c.fakePods.Err = c.err
	return NewPodCache(
		c.fakePodInfo,
		c.fakeNodes.Nodes(),
		c.fakePods,
	)
}

func makePod(namespace, name, host string, containers ...string) *api.Pod {
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{Namespace: namespace, Name: name},
		Status:     api.PodStatus{Host: host},
	}
	for _, c := range containers {
		pod.Spec.Containers = append(pod.Spec.Containers, api.Container{Name: c})
	}
	return pod
}

func makeHealthyNode(name string, ip string) *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{Name: name},
		Status: api.NodeStatus{
			HostIP: ip,
			Conditions: []api.NodeCondition{
				{Kind: api.NodeReady, Status: api.ConditionFull},
			},
		},
	}
}

func makeUnhealthyNode(name string) *api.Node {
	return &api.Node{
		ObjectMeta: api.ObjectMeta{Name: name},
		Status: api.NodeStatus{Conditions: []api.NodeCondition{
			{Kind: api.NodeReady, Status: api.ConditionNone},
		}},
	}
}

func TestPodUpdateAllContainersClearsNodeStatus(t *testing.T) {
	node := makeHealthyNode("machine", "1.2.3.5")
	pod1 := makePod(api.NamespaceDefault, "foo", "machine", "bar")
	pod2 := makePod(api.NamespaceDefault, "baz", "machine", "qux")
	config := podCacheTestConfig{
		kubeletContainerInfo: api.PodStatus{
			Info: api.PodInfo{"bar": api.ContainerStatus{}}},
		nodes: []api.Node{*node},
		pods:  []api.Pod{*pod1, *pod2},
	}
	cache := config.Construct()

	if len(cache.currentNodes) != 0 {
		t.Errorf("unexpected node cache: %v", cache.currentNodes)
	}
	key := objKey{"", "machine"}
	cache.currentNodes[key] = makeUnhealthyNode("machine").Status

	cache.UpdateAllContainers()

	if len(cache.currentNodes) != 1 {
		t.Errorf("unexpected empty node cache: %v", cache.currentNodes)
	}

	if !reflect.DeepEqual(cache.currentNodes[key], node.Status) {
		t.Errorf("unexpected status:\n%#v\nexpected:\n%#v\n", cache.currentNodes[key], node.Status)
	}
}

func TestPodUpdateAllContainers(t *testing.T) {
	pod1 := makePod(api.NamespaceDefault, "foo", "machine", "bar")
	pod2 := makePod(api.NamespaceDefault, "baz", "machine", "qux")
	config := podCacheTestConfig{
		kubeletContainerInfo: api.PodStatus{
			Info: api.PodInfo{"bar": api.ContainerStatus{}}},
		nodes: []api.Node{*makeHealthyNode("machine", "1.2.3.5")},
		pods:  []api.Pod{*pod1, *pod2},
	}
	cache := config.Construct()

	cache.UpdateAllContainers()

	call1 := config.fakePodInfo.calls[podInfoCall{"machine", api.NamespaceDefault, "foo"}]
	call2 := config.fakePodInfo.calls[podInfoCall{"machine", api.NamespaceDefault, "baz"}]
	if call1 == nil || call1.useCount != 1 {
		t.Errorf("Expected 1 call for 'foo': %+v", config.fakePodInfo.calls)
	}
	if call2 == nil || call2.useCount != 1 {
		t.Errorf("Expected 1 call for 'baz': %+v", config.fakePodInfo.calls)
	}
	if len(config.fakePodInfo.calls) != 2 {
		t.Errorf("Expected 2 calls: %+v", config.fakePodInfo.calls)
	}

	status, err := cache.GetPodStatus(api.NamespaceDefault, "foo")
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}
	if e, a := config.kubeletContainerInfo.Info, status.Info; !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected mismatch. Expected: %+v, Got: %+v", e, a)
	}
	if e, a := "1.2.3.5", status.HostIP; e != a {
		t.Errorf("Unexpected mismatch. Expected: %+v, Got: %+v", e, a)
	}

	if e, a := 1, len(config.fakeNodes.Actions); e != a {
		t.Errorf("Expected: %v, Got: %v; %+v", e, a, config.fakeNodes.Actions)
	} else {
		if e, a := "get-minion", config.fakeNodes.Actions[0].Action; e != a {
			t.Errorf("Expected: %v, Got: %v; %+v", e, a, config.fakeNodes.Actions)
		}
	}
}

func TestFillPodStatusNoHost(t *testing.T) {
	pod := makePod(api.NamespaceDefault, "foo", "", "bar")
	config := podCacheTestConfig{
		kubeletContainerInfo: api.PodStatus{},
		nodes:                []api.Node{*makeHealthyNode("machine", "")},
		pods:                 []api.Pod{*pod},
	}
	cache := config.Construct()
	err := cache.updatePodStatus(&config.pods[0])
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}

	status, err := cache.GetPodStatus(pod.Namespace, pod.Name)
	if e, a := api.PodPending, status.Phase; e != a {
		t.Errorf("Expected: %+v, Got %+v", e, a)
	}
}

func TestFillPodStatusMissingMachine(t *testing.T) {
	pod := makePod(api.NamespaceDefault, "foo", "machine", "bar")
	config := podCacheTestConfig{
		kubeletContainerInfo: api.PodStatus{},
		nodes:                []api.Node{},
		pods:                 []api.Pod{*pod},
	}
	cache := config.Construct()
	err := cache.updatePodStatus(&config.pods[0])
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}

	status, err := cache.GetPodStatus(pod.Namespace, pod.Name)
	if e, a := api.PodUnknown, status.Phase; e != a {
		t.Errorf("Expected: %+v, Got %+v", e, a)
	}
}

func TestFillPodStatus(t *testing.T) {
	pod := makePod(api.NamespaceDefault, "foo", "machine", "bar")
	expectedIP := "1.2.3.4"
	expectedTime, _ := time.Parse("2013-Feb-03", "2013-Feb-03")
	config := podCacheTestConfig{
		kubeletContainerInfo: api.PodStatus{
			Phase:  api.PodPending,
			Host:   "machine",
			HostIP: "ip of machine",
			PodIP:  expectedIP,
			Info: api.PodInfo{
				leaky.PodInfraContainerName: {
					State: api.ContainerState{
						Running: &api.ContainerStateRunning{
							StartedAt: util.NewTime(expectedTime),
						},
					},
					RestartCount: 1,
					PodIP:        expectedIP,
				},
			},
		},
		nodes: []api.Node{*makeHealthyNode("machine", "ip of machine")},
		pods:  []api.Pod{*pod},
	}
	cache := config.Construct()
	err := cache.updatePodStatus(&config.pods[0])
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}

	status, err := cache.GetPodStatus(pod.Namespace, pod.Name)
	if e, a := &config.kubeletContainerInfo, status; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected: %+v, Got %+v", e, a)
	}
}

func TestFillPodInfoNoData(t *testing.T) {
	pod := makePod(api.NamespaceDefault, "foo", "machine", "bar")
	expectedIP := ""
	config := podCacheTestConfig{
		kubeletContainerInfo: api.PodStatus{
			Phase:  api.PodPending,
			Host:   "machine",
			HostIP: "ip of machine",
			Info: api.PodInfo{
				leaky.PodInfraContainerName: {},
			},
		},
		nodes: []api.Node{*makeHealthyNode("machine", "ip of machine")},
		pods:  []api.Pod{*pod},
	}
	cache := config.Construct()
	err := cache.updatePodStatus(&config.pods[0])
	if err != nil {
		t.Fatalf("Unexpected error: %+v", err)
	}

	status, err := cache.GetPodStatus(pod.Namespace, pod.Name)
	if e, a := &config.kubeletContainerInfo, status; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected: %+v, Got %+v", e, a)
	}
	if status.PodIP != expectedIP {
		t.Errorf("Expected %s, Got %s", expectedIP, status.PodIP)
	}
}

func TestPodPhaseWithBadNode(t *testing.T) {
	desiredState := api.PodSpec{
		Containers: []api.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
	}
	runningState := api.ContainerStatus{
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
	stoppedState := api.ContainerStatus{
		State: api.ContainerState{
			Termination: &api.ContainerStateTerminated{},
		},
	}

	tests := []struct {
		pod    *api.Pod
		nodes  []api.Node
		status api.PodPhase
		test   string
	}{
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Host: "machine-two",
				},
			},
			[]api.Node{},
			api.PodUnknown,
			"no info, but bad machine",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": runningState,
					},
					Host: "machine-two",
				},
			},
			[]api.Node{},
			api.PodUnknown,
			"all running but minion is missing",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": stoppedState,
						"containerB": stoppedState,
					},
					Host: "machine-two",
				},
			},
			[]api.Node{},
			api.PodUnknown,
			"all stopped but minion missing",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": runningState,
					},
					Host: "machine-two",
				},
			},
			[]api.Node{*makeUnhealthyNode("machine-two")},
			api.PodUnknown,
			"all running but minion is unhealthy",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": stoppedState,
						"containerB": stoppedState,
					},
					Host: "machine-two",
				},
			},
			[]api.Node{*makeUnhealthyNode("machine-two")},
			api.PodUnknown,
			"all stopped but minion is unhealthy",
		},
	}
	for _, test := range tests {
		config := podCacheTestConfig{
			kubeletContainerInfo: test.pod.Status,
			nodes:                test.nodes,
			pods:                 []api.Pod{*test.pod},
		}
		cache := config.Construct()
		cache.UpdateAllContainers()
		status, err := cache.GetPodStatus(test.pod.Namespace, test.pod.Name)
		if err != nil {
			t.Errorf("%v: Unexpected error %v", test.test, err)
			continue
		}
		if e, a := test.status, status.Phase; e != a {
			t.Errorf("In test %s, expected %v, got %v", test.test, e, a)
		}
	}
}

func TestPodPhaseWithRestartAlways(t *testing.T) {
	desiredState := api.PodSpec{
		Containers: []api.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
	}
	currentState := api.PodStatus{
		Host: "machine",
	}
	runningState := api.ContainerStatus{
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
	stoppedState := api.ContainerStatus{
		State: api.ContainerState{
			Termination: &api.ContainerStateTerminated{},
		},
	}

	tests := []struct {
		pod    *api.Pod
		status api.PodPhase
		test   string
	}{
		{&api.Pod{Spec: desiredState, Status: currentState}, api.PodPending, "waiting"},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": runningState,
					},
					Host: "machine",
				},
			},
			api.PodRunning,
			"all running",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": stoppedState,
						"containerB": stoppedState,
					},
					Host: "machine",
				},
			},
			api.PodRunning,
			"all stopped with restart always",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": stoppedState,
					},
					Host: "machine",
				},
			},
			api.PodRunning,
			"mixed state #1 with restart always",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
					},
					Host: "machine",
				},
			},
			api.PodPending,
			"mixed state #2 with restart always",
		},
	}
	for _, test := range tests {
		if status := getPhase(&test.pod.Spec, test.pod.Status.Info); status != test.status {
			t.Errorf("In test %s, expected %v, got %v", test.test, test.status, status)
		}
	}
}

func TestPodPhaseWithRestartNever(t *testing.T) {
	desiredState := api.PodSpec{
		Containers: []api.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: api.RestartPolicy{Never: &api.RestartPolicyNever{}},
	}
	currentState := api.PodStatus{
		Host: "machine",
	}
	runningState := api.ContainerStatus{
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
	succeededState := api.ContainerStatus{
		State: api.ContainerState{
			Termination: &api.ContainerStateTerminated{
				ExitCode: 0,
			},
		},
	}
	failedState := api.ContainerStatus{
		State: api.ContainerState{
			Termination: &api.ContainerStateTerminated{
				ExitCode: -1,
			},
		},
	}

	tests := []struct {
		pod    *api.Pod
		status api.PodPhase
		test   string
	}{
		{&api.Pod{Spec: desiredState, Status: currentState}, api.PodPending, "waiting"},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": runningState,
					},
					Host: "machine",
				},
			},
			api.PodRunning,
			"all running with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": succeededState,
						"containerB": succeededState,
					},
					Host: "machine",
				},
			},
			api.PodSucceeded,
			"all succeeded with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": failedState,
						"containerB": failedState,
					},
					Host: "machine",
				},
			},
			api.PodFailed,
			"all failed with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": succeededState,
					},
					Host: "machine",
				},
			},
			api.PodRunning,
			"mixed state #1 with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
					},
					Host: "machine",
				},
			},
			api.PodPending,
			"mixed state #2 with restart never",
		},
	}
	for _, test := range tests {
		if status := getPhase(&test.pod.Spec, test.pod.Status.Info); status != test.status {
			t.Errorf("In test %s, expected %v, got %v", test.test, test.status, status)
		}
	}
}

func TestPodPhaseWithRestartOnFailure(t *testing.T) {
	desiredState := api.PodSpec{
		Containers: []api.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: api.RestartPolicy{OnFailure: &api.RestartPolicyOnFailure{}},
	}
	currentState := api.PodStatus{
		Host: "machine",
	}
	runningState := api.ContainerStatus{
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
	succeededState := api.ContainerStatus{
		State: api.ContainerState{
			Termination: &api.ContainerStateTerminated{
				ExitCode: 0,
			},
		},
	}
	failedState := api.ContainerStatus{
		State: api.ContainerState{
			Termination: &api.ContainerStateTerminated{
				ExitCode: -1,
			},
		},
	}

	tests := []struct {
		pod    *api.Pod
		status api.PodPhase
		test   string
	}{
		{&api.Pod{Spec: desiredState, Status: currentState}, api.PodPending, "waiting"},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": runningState,
					},
					Host: "machine",
				},
			},
			api.PodRunning,
			"all running with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": succeededState,
						"containerB": succeededState,
					},
					Host: "machine",
				},
			},
			api.PodSucceeded,
			"all succeeded with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": failedState,
						"containerB": failedState,
					},
					Host: "machine",
				},
			},
			api.PodRunning,
			"all failed with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
						"containerB": succeededState,
					},
					Host: "machine",
				},
			},
			api.PodRunning,
			"mixed state #1 with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					Info: map[string]api.ContainerStatus{
						"containerA": runningState,
					},
					Host: "machine",
				},
			},
			api.PodPending,
			"mixed state #2 with restart onfailure",
		},
	}
	for _, test := range tests {
		if status := getPhase(&test.pod.Spec, test.pod.Status.Info); status != test.status {
			t.Errorf("In test %s, expected %v, got %v", test.test, test.status, status)
		}
	}
}

func TestGarbageCollection(t *testing.T) {
	pod1 := makePod(api.NamespaceDefault, "foo", "machine", "bar")
	pod2 := makePod(api.NamespaceDefault, "baz", "machine", "qux")
	config := podCacheTestConfig{
		pods: []api.Pod{*pod1, *pod2},
	}
	cache := config.Construct()

	expected := api.PodStatus{
		Info: api.PodInfo{
			"extra": api.ContainerStatus{},
		},
	}
	cache.podStatus[objKey{api.NamespaceDefault, "extra"}] = expected

	cache.GarbageCollectPodStatus()

	status, found := cache.podStatus[objKey{api.NamespaceDefault, "extra"}]
	if found {
		t.Errorf("unexpectedly found: %v for key %v", status, objKey{api.NamespaceDefault, "extra"})
	}
}
