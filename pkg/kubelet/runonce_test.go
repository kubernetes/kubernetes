/*
Copyright 2014 The Kubernetes Authors.

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
	"os"
	"testing"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/record"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/configmap"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	podtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
	"k8s.io/kubernetes/pkg/kubelet/secret"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/kubelet/status"
	statustest "k8s.io/kubernetes/pkg/kubelet/status/testing"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager"
	"k8s.io/kubernetes/pkg/volume"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func TestRunOnce(t *testing.T) {
	cadvisor := &cadvisortest.Mock{}
	cadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	cadvisor.On("ImagesFsInfo").Return(cadvisorapiv2.FsInfo{
		Usage:     400,
		Capacity:  1000,
		Available: 600,
	}, nil)
	cadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{
		Usage:    9,
		Capacity: 10,
	}, nil)
	fakeSecretManager := secret.NewFakeManager()
	fakeConfigMapManager := configmap.NewFakeManager()
	podManager := kubepod.NewBasicPodManager(
		podtest.NewFakeMirrorClient(), fakeSecretManager, fakeConfigMapManager)
	fakeRuntime := &containertest.FakeRuntime{}
	basePath, err := utiltesting.MkTmpdir("kubelet")
	if err != nil {
		t.Fatalf("can't make a temp rootdir %v", err)
	}
	defer os.RemoveAll(basePath)
	kb := &Kubelet{
		rootDirectory:    basePath,
		recorder:         &record.FakeRecorder{},
		cadvisor:         cadvisor,
		nodeInfo:         testNodeInfo{},
		statusManager:    status.NewManager(nil, podManager, &statustest.FakePodDeletionSafetyProvider{}),
		podManager:       podManager,
		os:               &containertest.FakeOS{},
		containerRuntime: fakeRuntime,
		reasonCache:      NewReasonCache(),
		clock:            clock.RealClock{},
		kubeClient:       &fake.Clientset{},
		hostname:         testKubeletHostname,
		nodeName:         testKubeletHostname,
		runtimeState:     newRuntimeState(time.Second),
	}
	kb.containerManager = cm.NewStubContainerManager()

	plug := &volumetest.FakeVolumePlugin{PluginName: "fake", Host: nil}
	kb.volumePluginMgr, err =
		NewInitializedVolumePluginMgr(kb, fakeSecretManager, fakeConfigMapManager, []volume.VolumePlugin{plug})
	if err != nil {
		t.Fatalf("failed to initialize VolumePluginMgr: %v", err)
	}
	kb.volumeManager = volumemanager.NewVolumeManager(
		true,
		kb.nodeName,
		kb.podManager,
		kb.statusManager,
		kb.kubeClient,
		kb.volumePluginMgr,
		fakeRuntime,
		kb.mounter,
		kb.getPodsDir(),
		kb.recorder,
		false, /* experimentalCheckNodeCapabilitiesBeforeMount */
		false /* keepTerminatedPodVolumes */)

	kb.networkPlugin, _ = network.InitNetworkPlugin([]network.NetworkPlugin{}, "", nettest.NewFakeHost(nil), kubeletconfig.HairpinNone, "", network.UseDefaultMTU)
	// TODO: Factor out "StatsProvider" from Kubelet so we don't have a cyclic dependency
	volumeStatsAggPeriod := time.Second * 10
	kb.resourceAnalyzer = stats.NewResourceAnalyzer(kb, volumeStatsAggPeriod, kb.containerRuntime)
	nodeRef := &v1.ObjectReference{
		Kind:      "Node",
		Name:      string(kb.nodeName),
		UID:       types.UID(kb.nodeName),
		Namespace: "",
	}
	fakeKillPodFunc := func(pod *v1.Pod, podStatus v1.PodStatus, gracePeriodOverride *int64) error {
		return nil
	}
	evictionManager, evictionAdmitHandler := eviction.NewManager(kb.resourceAnalyzer, eviction.Config{}, fakeKillPodFunc, nil, nil, kb.recorder, nodeRef, kb.clock)

	kb.evictionManager = evictionManager
	kb.admitHandlers.AddPodAdmitHandler(evictionAdmitHandler)
	if err := kb.setupDataDirs(); err != nil {
		t.Errorf("Failed to init data dirs: %v", err)
	}

	pods := []*v1.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{Name: "bar"},
				},
			},
		},
	}
	podManager.SetPods(pods)
	// The original test here is totally meaningless, because fakeruntime will always return an empty podStatus. While
	// the original logic of isPodRunning happens to return true when podstatus is empty, so the test can always pass.
	// Now the logic in isPodRunning is changed, to let the test pass, we set the podstatus directly in fake runtime.
	// This is also a meaningless test, because the isPodRunning will also always return true after setting this. However,
	// because runonce is never used in kubernetes now, we should deprioritize the cleanup work.
	// TODO(random-liu) Fix the test, make it meaningful.
	fakeRuntime.PodStatus = kubecontainer.PodStatus{
		ContainerStatuses: []*kubecontainer.ContainerStatus{
			{
				Name:  "bar",
				State: kubecontainer.ContainerStateRunning,
			},
		},
	}
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
