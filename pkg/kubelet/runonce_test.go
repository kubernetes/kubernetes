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
	"testing"
	"time"

	cadvisorApi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

func TestRunOnce(t *testing.T) {
	cadvisor := &cadvisor.Mock{}
	cadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)

	podManager, _ := newFakePodManager()
	diskSpaceManager, _ := newDiskSpaceManager(cadvisor, DiskSpacePolicy{})
	fakeRuntime := &kubecontainer.FakeRuntime{}

	kb := &Kubelet{
		rootDirectory:       "/tmp/kubelet",
		recorder:            &record.FakeRecorder{},
		cadvisor:            cadvisor,
		nodeLister:          testNodeLister{},
		statusManager:       status.NewManager(nil),
		containerRefManager: kubecontainer.NewRefManager(),
		podManager:          podManager,
		os:                  kubecontainer.FakeOS{},
		volumeManager:       newVolumeManager(),
		diskSpaceManager:    diskSpaceManager,
		containerRuntime:    fakeRuntime,
	}
	kb.containerManager, _ = newContainerManager(fakeContainerMgrMountInt(), cadvisor, "", "", "")

	kb.networkPlugin, _ = network.InitNetworkPlugin([]network.NetworkPlugin{}, "", network.NewFakeHost(nil))
	if err := kb.setupDataDirs(); err != nil {
		t.Errorf("Failed to init data dirs: %v", err)
	}

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar"},
				},
			},
		},
	}
	podManager.SetPods(pods)
	results, err := kb.runOnce(pods, time.Millisecond)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if results[0].Err != nil {
		t.Errorf("unexpected run pod error: %v", results[0].Err)
	}
	if results[0].Pod.Name != "foo" {
		t.Errorf("unexpected pod: %q", results[0].Pod.Name)
	}
}
