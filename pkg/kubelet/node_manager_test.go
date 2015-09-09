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
	"reflect"
	"testing"
	"time"

	cadvisorApi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/version"
)

type fakeInfoGetter struct {
	machineInfo       *cadvisorApi.MachineInfo
	versionInfo       *cadvisorApi.VersionInfo
	runtimeUp         bool
	networkConfigured bool
}

func (f *fakeInfoGetter) GetMachineInfo() (*cadvisorApi.MachineInfo, error) {
	return f.machineInfo, nil
}

func (f *fakeInfoGetter) GetVersionInfo() (*cadvisorApi.VersionInfo, error) {
	return f.versionInfo, nil
}

func (f *fakeInfoGetter) ContainerRuntimeUp() bool {
	return f.runtimeUp
}

func (f *fakeInfoGetter) NetworkConfigured() bool {
	return f.networkConfigured
}

var _ infoGetter = &fakeInfoGetter{}

type testNodeLister struct {
	nodes []api.Node
}

func (ls testNodeLister) GetNodeInfo(id string) (*api.Node, error) {
	for _, node := range ls.nodes {
		if node.Name == id {
			return &node, nil
		}
	}
	return nil, fmt.Errorf("Node with name: %s does not exist", id)
}

func (ls testNodeLister) List() (api.NodeList, error) {
	return api.NodeList{
		Items: ls.nodes,
	}, nil
}

type testNodeManager struct {
	fakeClient     *testclient.Fake
	fakeInfoGetter *fakeInfoGetter
	nodeManager    *realNodeManager
}

func newTestNodeManager() *testNodeManager {
	fakeRecorder := &record.FakeRecorder{}
	fakeClient := &testclient.Fake{}
	fakeInfoGetter := &fakeInfoGetter{}
	nodeManager := newRealNodeManager(fakeClient, nil, true, time.Second, fakeRecorder, testKubeletHostname,
		testKubeletHostname, "", 0, fakeInfoGetter, &api.NodeDaemonEndpoints{}, nil)
	nodeManager.nodeLister = &testNodeLister{}
	return &testNodeManager{fakeClient: fakeClient, fakeInfoGetter: fakeInfoGetter, nodeManager: nodeManager}
}

func TestUpdateNewNodeStatus(t *testing.T) {
	testNodeManager := newTestNodeManager()
	nodeManager := testNodeManager.nodeManager
	client := testNodeManager.fakeClient
	fakeInfoGetter := testNodeManager.fakeInfoGetter

	client.ReactionChain = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname}},
	}}).ReactionChain

	fakeInfoGetter.machineInfo = &cadvisorApi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	fakeInfoGetter.versionInfo = &cadvisorApi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}

	expectedNode := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
		Spec:       api.NodeSpec{},
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{
					Type:               api.NodeReady,
					Status:             api.ConditionTrue,
					Reason:             "KubeletReady",
					Message:            fmt.Sprintf("kubelet is posting ready status"),
					LastHeartbeatTime:  util.Time{},
					LastTransitionTime: util.Time{},
				},
			},
			NodeInfo: api.NodeSystemInfo{
				MachineID:               "123",
				SystemUUID:              "abc",
				BootID:                  "1b3",
				KernelVersion:           "3.16.0-0.bpo.4-amd64",
				OsImage:                 "Debian GNU/Linux 7 (wheezy)",
				ContainerRuntimeVersion: "docker://1.5.0",
				KubeletVersion:          version.Get().String(),
				KubeProxyVersion:        version.Get().String(),
			},
			Capacity: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(2000, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(1024, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
			},
			Addresses: []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: "127.0.0.1"},
				{Type: api.NodeInternalIP, Address: "127.0.0.1"},
			},
		},
	}
	fakeInfoGetter.runtimeUp = true
	fakeInfoGetter.networkConfigured = true

	if err := nodeManager.updateNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	actions := client.Actions()
	if len(actions) != 2 {
		t.Fatalf("unexpected actions: %v", actions)
	}
	if !actions[1].Matches("update", "nodes") || actions[1].GetSubresource() != "status" {
		t.Fatalf("unexpected actions: %v", actions)
	}
	updatedNode, ok := actions[1].(testclient.UpdateAction).GetObject().(*api.Node)
	if !ok {
		t.Errorf("unexpected object type")
	}
	if updatedNode.Status.Conditions[0].LastHeartbeatTime.IsZero() {
		t.Errorf("unexpected zero last probe timestamp")
	}
	if updatedNode.Status.Conditions[0].LastTransitionTime.IsZero() {
		t.Errorf("unexpected zero last transition timestamp")
	}
	updatedNode.Status.Conditions[0].LastHeartbeatTime = util.Time{}
	updatedNode.Status.Conditions[0].LastTransitionTime = util.Time{}
	if !reflect.DeepEqual(expectedNode, updatedNode) {
		t.Errorf("unexpected objects: %s", util.ObjectDiff(expectedNode, updatedNode))
	}
}

func TestUpdateExistingNodeStatus(t *testing.T) {
	testNodeManager := newTestNodeManager()
	nodeManager := testNodeManager.nodeManager
	client := testNodeManager.fakeClient
	fakeInfoGetter := testNodeManager.fakeInfoGetter

	client.ReactionChain = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
			Spec:       api.NodeSpec{},
			Status: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:               api.NodeReady,
						Status:             api.ConditionTrue,
						Reason:             "KubeletReady",
						Message:            fmt.Sprintf("kubelet is posting ready status"),
						LastHeartbeatTime:  util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						LastTransitionTime: util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					},
				},
				Capacity: api.ResourceList{
					api.ResourceCPU:    *resource.NewMilliQuantity(3000, resource.DecimalSI),
					api.ResourceMemory: *resource.NewQuantity(2048, resource.BinarySI),
					api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
				},
			},
		},
	}}).ReactionChain
	fakeInfoGetter.machineInfo = &cadvisorApi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	fakeInfoGetter.versionInfo = &cadvisorApi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}

	expectedNode := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
		Spec:       api.NodeSpec{},
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{
					Type:               api.NodeReady,
					Status:             api.ConditionTrue,
					Reason:             "KubeletReady",
					Message:            fmt.Sprintf("kubelet is posting ready status"),
					LastHeartbeatTime:  util.Time{}, // placeholder
					LastTransitionTime: util.Time{}, // placeholder
				},
			},
			NodeInfo: api.NodeSystemInfo{
				MachineID:               "123",
				SystemUUID:              "abc",
				BootID:                  "1b3",
				KernelVersion:           "3.16.0-0.bpo.4-amd64",
				OsImage:                 "Debian GNU/Linux 7 (wheezy)",
				ContainerRuntimeVersion: "docker://1.5.0",
				KubeletVersion:          version.Get().String(),
				KubeProxyVersion:        version.Get().String(),
			},
			Capacity: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(2000, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(1024, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
			},
			Addresses: []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: "127.0.0.1"},
				{Type: api.NodeInternalIP, Address: "127.0.0.1"},
			},
		},
	}
	fakeInfoGetter.runtimeUp = true
	fakeInfoGetter.networkConfigured = true

	if err := nodeManager.updateNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	actions := client.Actions()
	if len(actions) != 2 {
		t.Errorf("unexpected actions: %v", actions)
	}
	updateAction, ok := actions[1].(testclient.UpdateAction)
	if !ok {
		t.Errorf("unexpected action type.  expected UpdateAction, got %#v", actions[1])
	}
	updatedNode, ok := updateAction.GetObject().(*api.Node)
	if !ok {
		t.Errorf("unexpected object type")
	}
	// Expect LastProbeTime to be updated to Now, while LastTransitionTime to be the same.
	if reflect.DeepEqual(updatedNode.Status.Conditions[0].LastHeartbeatTime.Rfc3339Copy().UTC(), util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC).Time) {
		t.Errorf("expected \n%v\n, got \n%v", util.Now(), util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC))
	}
	if !reflect.DeepEqual(updatedNode.Status.Conditions[0].LastTransitionTime.Rfc3339Copy().UTC(), util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC).Time) {
		t.Errorf("expected \n%#v\n, got \n%#v", updatedNode.Status.Conditions[0].LastTransitionTime.Rfc3339Copy(),
			util.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC))
	}
	updatedNode.Status.Conditions[0].LastHeartbeatTime = util.Time{}
	updatedNode.Status.Conditions[0].LastTransitionTime = util.Time{}
	if !reflect.DeepEqual(expectedNode, updatedNode) {
		t.Errorf("expected \n%v\n, got \n%v", expectedNode, updatedNode)
	}
}

func TestUpdateNodeStatusWithoutContainerRuntime(t *testing.T) {
	testNodeManager := newTestNodeManager()
	nodeManager := testNodeManager.nodeManager
	client := testNodeManager.fakeClient
	fakeInfoGetter := testNodeManager.fakeInfoGetter

	client.ReactionChain = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname}},
	}}).ReactionChain

	fakeInfoGetter.machineInfo = &cadvisorApi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	fakeInfoGetter.versionInfo = &cadvisorApi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}

	expectedNode := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
		Spec:       api.NodeSpec{},
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{
					Type:               api.NodeReady,
					Status:             api.ConditionFalse,
					Reason:             "KubeletNotReady",
					Message:            fmt.Sprintf("container runtime is down"),
					LastHeartbeatTime:  util.Time{},
					LastTransitionTime: util.Time{},
				},
			},
			NodeInfo: api.NodeSystemInfo{
				MachineID:               "123",
				SystemUUID:              "abc",
				BootID:                  "1b3",
				KernelVersion:           "3.16.0-0.bpo.4-amd64",
				OsImage:                 "Debian GNU/Linux 7 (wheezy)",
				ContainerRuntimeVersion: "docker://1.5.0",
				KubeletVersion:          version.Get().String(),
				KubeProxyVersion:        version.Get().String(),
			},
			Capacity: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(2000, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(1024, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
			},
			Addresses: []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: "127.0.0.1"},
				{Type: api.NodeInternalIP, Address: "127.0.0.1"},
			},
		},
	}
	// Pretend that container runtime is down.
	fakeInfoGetter.runtimeUp = false
	fakeInfoGetter.networkConfigured = true

	if err := nodeManager.updateNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	actions := client.Actions()
	if len(actions) != 2 {
		t.Fatalf("unexpected actions: %v", actions)
	}
	if !actions[1].Matches("update", "nodes") || actions[1].GetSubresource() != "status" {
		t.Fatalf("unexpected actions: %v", actions)
	}
	updatedNode, ok := actions[1].(testclient.UpdateAction).GetObject().(*api.Node)
	if !ok {
		t.Errorf("unexpected action type.  expected UpdateAction, got %#v", actions[1])
	}

	if updatedNode.Status.Conditions[0].LastHeartbeatTime.IsZero() {
		t.Errorf("unexpected zero last probe timestamp")
	}
	if updatedNode.Status.Conditions[0].LastTransitionTime.IsZero() {
		t.Errorf("unexpected zero last transition timestamp")
	}
	updatedNode.Status.Conditions[0].LastHeartbeatTime = util.Time{}
	updatedNode.Status.Conditions[0].LastTransitionTime = util.Time{}
	if !reflect.DeepEqual(expectedNode, updatedNode) {
		t.Errorf("unexpected objects: %s", util.ObjectDiff(expectedNode, updatedNode))
	}
}

func TestUpdateNodeStatusError(t *testing.T) {
	testNodeManager := newTestNodeManager()
	nodeManager := testNodeManager.nodeManager
	client := testNodeManager.fakeClient
	// No matching node for the kubelet
	client.ReactionChain = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{}}).ReactionChain

	if err := nodeManager.updateNodeStatus(); err == nil {
		t.Errorf("unexpected non error: %v", err)
	}
	if len(client.Actions()) != nodeStatusUpdateRetry {
		t.Errorf("unexpected actions: %v", client.Actions())
	}
}

func TestRegisterExistingNodeWithApiserver(t *testing.T) {
	testNodeManager := newTestNodeManager()
	nodeManager := testNodeManager.nodeManager
	client := testNodeManager.fakeClient
	fakeInfoGetter := testNodeManager.fakeInfoGetter
	client.AddReactor("create", "nodes", func(action testclient.Action) (bool, runtime.Object, error) {
		// Return an error on create.
		return true, &api.Node{}, &apierrors.StatusError{
			ErrStatus: unversioned.Status{Reason: unversioned.StatusReasonAlreadyExists},
		}
	})
	client.AddReactor("get", "nodes", func(action testclient.Action) (bool, runtime.Object, error) {
		// Return an existing (matching) node on get.
		return true, &api.Node{
			ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
			Spec:       api.NodeSpec{ExternalID: testKubeletHostname},
		}, nil
	})
	client.AddReactor("*", "*", func(action testclient.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("no reaction implemented for %s", action)
	})
	fakeInfoGetter.machineInfo = &cadvisorApi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	fakeInfoGetter.versionInfo = &cadvisorApi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}

	done := make(chan struct{})
	go func() {
		nodeManager.registerWithApiserver()
		done <- struct{}{}
	}()
	select {
	case <-time.After(5 * time.Second):
		t.Errorf("timed out waiting for registration")
	case <-done:
		return
	}
}
