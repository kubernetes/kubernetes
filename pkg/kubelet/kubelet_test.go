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
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"path"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	cadvisorApi "github.com/google/cadvisor/info/v1"
	cadvisorApiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"k8s.io/kubernetes/pkg/kubelet/container"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/prober"
	"k8s.io/kubernetes/pkg/kubelet/status"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/volume"
	_ "k8s.io/kubernetes/pkg/volume/host_path"
)

func init() {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
	util.ReallyCrash = true
}

const testKubeletHostname = "127.0.0.1"

type fakeHTTP struct {
	url string
	err error
}

func (f *fakeHTTP) Get(url string) (*http.Response, error) {
	f.url = url
	return nil, f.err
}

type TestKubelet struct {
	kubelet          *Kubelet
	fakeRuntime      *kubecontainer.FakeRuntime
	fakeCadvisor     *cadvisor.Mock
	fakeKubeClient   *testclient.Fake
	fakeMirrorClient *fakeMirrorClient
}

func newTestKubelet(t *testing.T) *TestKubelet {
	fakeRuntime := &kubecontainer.FakeRuntime{}
	fakeRuntime.VersionInfo = "1.15"
	fakeRecorder := &record.FakeRecorder{}
	fakeKubeClient := &testclient.Fake{}
	kubelet := &Kubelet{}
	kubelet.kubeClient = fakeKubeClient
	kubelet.os = kubecontainer.FakeOS{}

	kubelet.hostname = testKubeletHostname
	kubelet.nodeName = testKubeletHostname
	kubelet.runtimeUpThreshold = maxWaitForContainerRuntime
	kubelet.networkPlugin, _ = network.InitNetworkPlugin([]network.NetworkPlugin{}, "", network.NewFakeHost(nil))
	if tempDir, err := ioutil.TempDir("/tmp", "kubelet_test."); err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	} else {
		kubelet.rootDirectory = tempDir
	}
	if err := os.MkdirAll(kubelet.rootDirectory, 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %v", kubelet.rootDirectory, err)
	}
	kubelet.sourcesReady = func(_ sets.String) bool { return true }
	kubelet.masterServiceNamespace = api.NamespaceDefault
	kubelet.serviceLister = testServiceLister{}
	kubelet.nodeLister = testNodeLister{}
	kubelet.recorder = fakeRecorder
	kubelet.statusManager = status.NewManager(fakeKubeClient)
	if err := kubelet.setupDataDirs(); err != nil {
		t.Fatalf("can't initialize kubelet data dirs: %v", err)
	}
	kubelet.daemonEndpoints = &api.NodeDaemonEndpoints{}
	mockCadvisor := &cadvisor.Mock{}
	kubelet.cadvisor = mockCadvisor
	podManager, fakeMirrorClient := newFakePodManager()
	kubelet.podManager = podManager
	kubelet.containerRefManager = kubecontainer.NewRefManager()
	diskSpaceManager, err := newDiskSpaceManager(mockCadvisor, DiskSpacePolicy{})
	if err != nil {
		t.Fatalf("can't initialize disk space manager: %v", err)
	}
	kubelet.diskSpaceManager = diskSpaceManager

	kubelet.containerRuntime = fakeRuntime
	kubelet.runtimeCache = kubecontainer.NewFakeRuntimeCache(kubelet.containerRuntime)
	kubelet.podWorkers = &fakePodWorkers{
		syncPodFn:    kubelet.syncPod,
		runtimeCache: kubelet.runtimeCache,
		t:            t,
	}

	kubelet.prober = prober.FakeProber{}
	kubelet.probeManager = prober.FakeManager{}

	kubelet.volumeManager = newVolumeManager()
	kubelet.containerManager, _ = newContainerManager(fakeContainerMgrMountInt(), mockCadvisor, "", "", "")
	kubelet.networkConfigured = true
	fakeClock := &util.FakeClock{Time: time.Now()}
	kubelet.backOff = util.NewBackOff(time.Second, time.Minute)
	kubelet.backOff.Clock = fakeClock
	kubelet.podKillingCh = make(chan *kubecontainer.Pod, 20)
	return &TestKubelet{kubelet, fakeRuntime, mockCadvisor, fakeKubeClient, fakeMirrorClient}
}

func newTestPods(count int) []*api.Pod {
	pods := make([]*api.Pod, count)
	for i := 0; i < count; i++ {
		pods[i] = &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name: fmt.Sprintf("pod%d", i),
			},
		}
	}
	return pods
}

func TestKubeletDirs(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	root := kubelet.rootDirectory

	var exp, got string

	got = kubelet.getPodsDir()
	exp = path.Join(root, "pods")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPluginsDir()
	exp = path.Join(root, "plugins")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPluginDir("foobar")
	exp = path.Join(root, "plugins/foobar")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodDir("abc123")
	exp = path.Join(root, "pods/abc123")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodVolumesDir("abc123")
	exp = path.Join(root, "pods/abc123/volumes")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodVolumeDir("abc123", "plugin", "foobar")
	exp = path.Join(root, "pods/abc123/volumes/plugin/foobar")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodPluginsDir("abc123")
	exp = path.Join(root, "pods/abc123/plugins")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodPluginDir("abc123", "foobar")
	exp = path.Join(root, "pods/abc123/plugins/foobar")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodContainerDir("abc123", "def456")
	exp = path.Join(root, "pods/abc123/containers/def456")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}
}

func TestKubeletDirsCompat(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	root := kubelet.rootDirectory
	if err := os.MkdirAll(root, 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}

	var exp, got string

	// Old-style pod dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/oldpod", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	// New-style pod dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/pods/newpod", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	// Both-style pod dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/bothpod", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	if err := os.MkdirAll(fmt.Sprintf("%s/pods/bothpod", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}

	got = kubelet.getPodDir("oldpod")
	exp = path.Join(root, "oldpod")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodDir("newpod")
	exp = path.Join(root, "pods/newpod")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodDir("bothpod")
	exp = path.Join(root, "pods/bothpod")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodDir("neitherpod")
	exp = path.Join(root, "pods/neitherpod")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	root = kubelet.getPodDir("newpod")

	// Old-style container dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/oldctr", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	// New-style container dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/containers/newctr", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	// Both-style container dir.
	if err := os.MkdirAll(fmt.Sprintf("%s/bothctr", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}
	if err := os.MkdirAll(fmt.Sprintf("%s/containers/bothctr", root), 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %s", root, err)
	}

	got = kubelet.getPodContainerDir("newpod", "oldctr")
	exp = path.Join(root, "oldctr")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodContainerDir("newpod", "newctr")
	exp = path.Join(root, "containers/newctr")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodContainerDir("newpod", "bothctr")
	exp = path.Join(root, "containers/bothctr")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}

	got = kubelet.getPodContainerDir("newpod", "neitherctr")
	exp = path.Join(root, "containers/neitherctr")
	if got != exp {
		t.Errorf("expected %q', got %q", exp, got)
	}
}

var emptyPodUIDs map[types.UID]kubetypes.SyncPodType

func TestSyncLoopTimeUpdate(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	kubelet := testKubelet.kubelet

	loopTime1 := kubelet.LatestLoopEntryTime()
	if !loopTime1.IsZero() {
		t.Errorf("Unexpected sync loop time: %s, expected 0", loopTime1)
	}

	kubelet.syncLoopIteration(make(chan kubetypes.PodUpdate), kubelet)
	loopTime2 := kubelet.LatestLoopEntryTime()
	if loopTime2.IsZero() {
		t.Errorf("Unexpected sync loop time: 0, expected non-zero value.")
	}
	kubelet.syncLoopIteration(make(chan kubetypes.PodUpdate), kubelet)
	loopTime3 := kubelet.LatestLoopEntryTime()
	if !loopTime3.After(loopTime1) {
		t.Errorf("Sync Loop Time was not updated correctly. Second update timestamp should be greater than first update timestamp")
	}
}

func TestSyncLoopAbort(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	kubelet := testKubelet.kubelet
	kubelet.lastTimestampRuntimeUp = time.Now()
	kubelet.networkConfigured = true
	// The syncLoop waits on time.After(resyncInterval), set it really big so that we don't race for
	// the channel close
	kubelet.resyncInterval = time.Second * 30

	ch := make(chan kubetypes.PodUpdate)
	close(ch)

	// sanity check (also prevent this test from hanging in the next step)
	ok := kubelet.syncLoopIteration(ch, kubelet)
	if ok {
		t.Fatalf("expected syncLoopIteration to return !ok since update chan was closed")
	}

	// this should terminate immediately; if it hangs then the syncLoopIteration isn't aborting properly
	kubelet.syncLoop(ch, kubelet)
}

func TestSyncPodsStartPod(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
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
	kubelet.podManager.SetPods(pods)
	kubelet.HandlePodSyncs(pods)
	fakeRuntime.AssertStartedPods([]string{string(pods[0].UID)})
}

func TestSyncPodsDeletesWhenSourcesAreReady(t *testing.T) {
	ready := false

	testKubelet := newTestKubelet(t)
	fakeRuntime := testKubelet.fakeRuntime
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	kubelet.sourcesReady = func(_ sets.String) bool { return ready }

	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:   "12345678",
			Name: "foo", Namespace: "new",
			Containers: []*kubecontainer.Container{
				{Name: "bar"},
			},
		},
	}
	kubelet.HandlePodCleanups()
	// Sources are not ready yet. Don't remove any pods.
	fakeRuntime.AssertKilledPods([]string{})

	ready = true
	kubelet.HandlePodCleanups()

	// Sources are ready. Remove unwanted pods.
	fakeRuntime.AssertKilledPods([]string{"12345678"})
}

func TestMountExternalVolumes(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubelet.volumePluginMgr.InitPlugins([]volume.VolumePlugin{&volume.FakeVolumePlugin{PluginName: "fake", Host: nil}}, &volumeHost{kubelet})

	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "test",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name:         "vol1",
					VolumeSource: api.VolumeSource{},
				},
			},
		},
	}
	podVolumes, err := kubelet.mountExternalVolumes(&pod)
	if err != nil {
		t.Errorf("Expected success: %v", err)
	}
	expectedPodVolumes := []string{"vol1"}
	if len(expectedPodVolumes) != len(podVolumes) {
		t.Errorf("Unexpected volumes. Expected %#v got %#v.  Manifest was: %#v", expectedPodVolumes, podVolumes, pod)
	}
	for _, name := range expectedPodVolumes {
		if _, ok := podVolumes[name]; !ok {
			t.Errorf("api.Pod volumes map is missing key: %s. %#v", name, podVolumes)
		}
	}
}

func TestGetPodVolumesFromDisk(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	plug := &volume.FakeVolumePlugin{PluginName: "fake", Host: nil}
	kubelet.volumePluginMgr.InitPlugins([]volume.VolumePlugin{plug}, &volumeHost{kubelet})

	volsOnDisk := []struct {
		podUID  types.UID
		volName string
	}{
		{"pod1", "vol1"},
		{"pod1", "vol2"},
		{"pod2", "vol1"},
	}

	expectedPaths := []string{}
	for i := range volsOnDisk {
		fv := volume.FakeVolume{PodUID: volsOnDisk[i].podUID, VolName: volsOnDisk[i].volName, Plugin: plug}
		fv.SetUp()
		expectedPaths = append(expectedPaths, fv.GetPath())
	}

	volumesFound := kubelet.getPodVolumesFromDisk()
	if len(volumesFound) != len(expectedPaths) {
		t.Errorf("Expected to find %d cleaners, got %d", len(expectedPaths), len(volumesFound))
	}
	for _, ep := range expectedPaths {
		found := false
		for _, cl := range volumesFound {
			if ep == cl.GetPath() {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Could not find a volume with path %s", ep)
		}
	}
}

type stubVolume struct {
	path string
}

func (f *stubVolume) GetPath() string {
	return f.path
}

func TestMakeVolumeMounts(t *testing.T) {
	container := api.Container{
		VolumeMounts: []api.VolumeMount{
			{
				MountPath: "/mnt/path",
				Name:      "disk",
				ReadOnly:  false,
			},
			{
				MountPath: "/mnt/path3",
				Name:      "disk",
				ReadOnly:  true,
			},
			{
				MountPath: "/mnt/path4",
				Name:      "disk4",
				ReadOnly:  false,
			},
			{
				MountPath: "/mnt/path5",
				Name:      "disk5",
				ReadOnly:  false,
			},
		},
	}

	podVolumes := kubecontainer.VolumeMap{
		"disk":  &stubVolume{"/mnt/disk"},
		"disk4": &stubVolume{"/mnt/host"},
		"disk5": &stubVolume{"/var/lib/kubelet/podID/volumes/empty/disk5"},
	}

	mounts := makeMounts(&container, podVolumes)

	expectedMounts := []kubecontainer.Mount{
		{
			"disk",
			"/mnt/path",
			"/mnt/disk",
			false,
		},
		{
			"disk",
			"/mnt/path3",
			"/mnt/disk",
			true,
		},
		{
			"disk4",
			"/mnt/path4",
			"/mnt/host",
			false,
		},
		{
			"disk5",
			"/mnt/path5",
			"/var/lib/kubelet/podID/volumes/empty/disk5",
			false,
		},
	}
	if !reflect.DeepEqual(mounts, expectedMounts) {
		t.Errorf("Unexpected mounts: Expected %#v got %#v.  Container was: %#v", expectedMounts, mounts, container)
	}
}

func TestGetContainerInfo(t *testing.T) {
	containerID := "ab2cdf"
	containerPath := fmt.Sprintf("/docker/%v", containerID)
	containerInfo := cadvisorApi.ContainerInfo{
		ContainerReference: cadvisorApi.ContainerReference{
			Name: containerPath,
		},
	}

	testKubelet := newTestKubelet(t)
	fakeRuntime := testKubelet.fakeRuntime
	kubelet := testKubelet.kubelet
	cadvisorReq := &cadvisorApi.ContainerInfoRequest{}
	mockCadvisor := testKubelet.fakeCadvisor
	mockCadvisor.On("DockerContainer", containerID, cadvisorReq).Return(containerInfo, nil)
	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:        "12345678",
			Name:      "qux",
			Namespace: "ns",
			Containers: []*kubecontainer.Container{
				{
					Name: "foo",
					ID:   kubecontainer.ContainerID{"test", containerID},
				},
			},
		},
	}
	stats, err := kubelet.GetContainerInfo("qux_ns", "", "foo", cadvisorReq)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if stats == nil {
		t.Fatalf("stats should not be nil")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetRawContainerInfoRoot(t *testing.T) {
	containerPath := "/"
	containerInfo := &cadvisorApi.ContainerInfo{
		ContainerReference: cadvisorApi.ContainerReference{
			Name: containerPath,
		},
	}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorReq := &cadvisorApi.ContainerInfoRequest{}
	mockCadvisor.On("ContainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	_, err := kubelet.GetRawContainerInfo(containerPath, cadvisorReq, false)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetRawContainerInfoSubcontainers(t *testing.T) {
	containerPath := "/kubelet"
	containerInfo := map[string]*cadvisorApi.ContainerInfo{
		containerPath: {
			ContainerReference: cadvisorApi.ContainerReference{
				Name: containerPath,
			},
		},
		"/kubelet/sub": {
			ContainerReference: cadvisorApi.ContainerReference{
				Name: "/kubelet/sub",
			},
		},
	}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorReq := &cadvisorApi.ContainerInfoRequest{}
	mockCadvisor.On("SubcontainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	result, err := kubelet.GetRawContainerInfo(containerPath, cadvisorReq, true)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(result) != 2 {
		t.Errorf("Expected 2 elements, received: %+v", result)
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWhenCadvisorFailed(t *testing.T) {
	containerID := "ab2cdf"
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	fakeRuntime := testKubelet.fakeRuntime
	cadvisorApiFailure := fmt.Errorf("cAdvisor failure")
	containerInfo := cadvisorApi.ContainerInfo{}
	cadvisorReq := &cadvisorApi.ContainerInfoRequest{}
	mockCadvisor.On("DockerContainer", containerID, cadvisorReq).Return(containerInfo, cadvisorApiFailure)
	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:        "uuid",
			Name:      "qux",
			Namespace: "ns",
			Containers: []*kubecontainer.Container{
				{Name: "foo",
					ID: kubecontainer.ContainerID{"test", containerID},
				},
			},
		},
	}
	stats, err := kubelet.GetContainerInfo("qux_ns", "uuid", "foo", cadvisorReq)
	if stats != nil {
		t.Errorf("non-nil stats on error")
	}
	if err == nil {
		t.Errorf("expect error but received nil error")
		return
	}
	if err.Error() != cadvisorApiFailure.Error() {
		t.Errorf("wrong error message. expect %v, got %v", cadvisorApiFailure, err)
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoOnNonExistContainer(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	fakeRuntime := testKubelet.fakeRuntime
	fakeRuntime.PodList = []*kubecontainer.Pod{}

	stats, _ := kubelet.GetContainerInfo("qux", "", "foo", nil)
	if stats != nil {
		t.Errorf("non-nil stats on non exist container")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWhenContainerRuntimeFailed(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	fakeRuntime := testKubelet.fakeRuntime
	expectedErr := fmt.Errorf("List containers error")
	fakeRuntime.Err = expectedErr

	stats, err := kubelet.GetContainerInfo("qux", "", "foo", nil)
	if err == nil {
		t.Errorf("expected error from dockertools, got none")
	}
	if err.Error() != expectedErr.Error() {
		t.Errorf("expected error %v got %v", expectedErr.Error(), err.Error())
	}
	if stats != nil {
		t.Errorf("non-nil stats when dockertools failed")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWithNoContainers(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor

	stats, err := kubelet.GetContainerInfo("qux_ns", "", "foo", nil)
	if err == nil {
		t.Errorf("expected error from cadvisor client, got none")
	}
	if err != ErrContainerNotFound {
		t.Errorf("expected error %v, got %v", ErrContainerNotFound.Error(), err.Error())
	}
	if stats != nil {
		t.Errorf("non-nil stats when dockertools returned no containers")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWithNoMatchingContainers(t *testing.T) {
	testKubelet := newTestKubelet(t)
	fakeRuntime := testKubelet.fakeRuntime
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:        "12345678",
			Name:      "qux",
			Namespace: "ns",
			Containers: []*kubecontainer.Container{
				{Name: "bar",
					ID: kubecontainer.ContainerID{"test", "fakeID"},
				},
			}},
	}

	stats, err := kubelet.GetContainerInfo("qux_ns", "", "foo", nil)
	if err == nil {
		t.Errorf("Expected error from cadvisor client, got none")
	}
	if err != ErrContainerNotFound {
		t.Errorf("Expected error %v, got %v", ErrContainerNotFound.Error(), err.Error())
	}
	if stats != nil {
		t.Errorf("non-nil stats when dockertools returned no containers")
	}
	mockCadvisor.AssertExpectations(t)
}

type fakeContainerCommandRunner struct {
	Cmd    []string
	ID     kubecontainer.ContainerID
	PodID  types.UID
	E      error
	Stdin  io.Reader
	Stdout io.WriteCloser
	Stderr io.WriteCloser
	TTY    bool
	Port   uint16
	Stream io.ReadWriteCloser
}

func (f *fakeContainerCommandRunner) RunInContainer(id kubecontainer.ContainerID, cmd []string) ([]byte, error) {
	f.Cmd = cmd
	f.ID = id
	return []byte{}, f.E
}

func (f *fakeContainerCommandRunner) ExecInContainer(id kubecontainer.ContainerID, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool) error {
	f.Cmd = cmd
	f.ID = id
	f.Stdin = in
	f.Stdout = out
	f.Stderr = err
	f.TTY = tty
	return f.E
}

func (f *fakeContainerCommandRunner) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	f.PodID = pod.ID
	f.Port = port
	f.Stream = stream
	return nil
}

func TestRunInContainerNoSuchPod(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeRuntime.PodList = []*kubecontainer.Pod{}

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerName := "containerFoo"
	output, err := kubelet.RunInContainer(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{Name: podName, Namespace: podNamespace}}),
		"",
		containerName,
		[]string{"ls"})
	if output != nil {
		t.Errorf("unexpected non-nil command: %v", output)
	}
	if err == nil {
		t.Error("unexpected non-error")
	}
}

func TestRunInContainer(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner

	containerID := kubecontainer.ContainerID{"test", "abc1234"}
	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:        "12345678",
			Name:      "podFoo",
			Namespace: "nsFoo",
			Containers: []*kubecontainer.Container{
				{Name: "containerFoo",
					ID: containerID,
				},
			},
		},
	}
	cmd := []string{"ls"}
	_, err := kubelet.RunInContainer("podFoo_nsFoo", "", "containerFoo", cmd)
	if fakeCommandRunner.ID != containerID {
		t.Errorf("unexpected Name: %s", fakeCommandRunner.ID)
	}
	if !reflect.DeepEqual(fakeCommandRunner.Cmd, cmd) {
		t.Errorf("unexpected command: %s", fakeCommandRunner.Cmd)
	}
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestParseResolvConf(t *testing.T) {
	testCases := []struct {
		data        string
		nameservers []string
		searches    []string
	}{
		{"", []string{}, []string{}},
		{" ", []string{}, []string{}},
		{"\n", []string{}, []string{}},
		{"\t\n\t", []string{}, []string{}},
		{"#comment\n", []string{}, []string{}},
		{" #comment\n", []string{}, []string{}},
		{"#comment\n#comment", []string{}, []string{}},
		{"#comment\nnameserver", []string{}, []string{}},
		{"#comment\nnameserver\nsearch", []string{}, []string{}},
		{"nameserver 1.2.3.4", []string{"1.2.3.4"}, []string{}},
		{" nameserver 1.2.3.4", []string{"1.2.3.4"}, []string{}},
		{"\tnameserver 1.2.3.4", []string{"1.2.3.4"}, []string{}},
		{"nameserver\t1.2.3.4", []string{"1.2.3.4"}, []string{}},
		{"nameserver \t 1.2.3.4", []string{"1.2.3.4"}, []string{}},
		{"nameserver 1.2.3.4\nnameserver 5.6.7.8", []string{"1.2.3.4", "5.6.7.8"}, []string{}},
		{"search foo", []string{}, []string{"foo"}},
		{"search foo bar", []string{}, []string{"foo", "bar"}},
		{"search foo bar bat\n", []string{}, []string{"foo", "bar", "bat"}},
		{"search foo\nsearch bar", []string{}, []string{"bar"}},
		{"nameserver 1.2.3.4\nsearch foo bar", []string{"1.2.3.4"}, []string{"foo", "bar"}},
		{"nameserver 1.2.3.4\nsearch foo\nnameserver 5.6.7.8\nsearch bar", []string{"1.2.3.4", "5.6.7.8"}, []string{"bar"}},
		{"#comment\nnameserver 1.2.3.4\n#comment\nsearch foo\ncomment", []string{"1.2.3.4"}, []string{"foo"}},
	}
	for i, tc := range testCases {
		ns, srch, err := parseResolvConf(strings.NewReader(tc.data))
		if err != nil {
			t.Errorf("expected success, got %v", err)
			continue
		}
		if !reflect.DeepEqual(ns, tc.nameservers) {
			t.Errorf("[%d] expected nameservers %#v, got %#v", i, tc.nameservers, ns)
		}
		if !reflect.DeepEqual(srch, tc.searches) {
			t.Errorf("[%d] expected searches %#v, got %#v", i, tc.searches, srch)
		}
	}
}

func TestDNSConfigurationParams(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet

	clusterNS := "203.0.113.1"
	kubelet.clusterDomain = "kubernetes.io"
	kubelet.clusterDNS = net.ParseIP(clusterNS)

	pods := newTestPods(2)
	pods[0].Spec.DNSPolicy = api.DNSClusterFirst
	pods[1].Spec.DNSPolicy = api.DNSDefault

	options := make([]*kubecontainer.RunContainerOptions, 2)
	for i, pod := range pods {
		var err error
		kubelet.volumeManager.SetVolumes(pod.UID, make(kubecontainer.VolumeMap, 0))
		options[i], err = kubelet.GenerateRunContainerOptions(pod, &api.Container{})
		if err != nil {
			t.Fatalf("failed to generate container options: %v", err)
		}
	}
	if len(options[0].DNS) != 1 || options[0].DNS[0] != clusterNS {
		t.Errorf("expected nameserver %s, got %+v", clusterNS, options[0].DNS)
	}
	if len(options[0].DNSSearch) == 0 || options[0].DNSSearch[0] != ".svc."+kubelet.clusterDomain {
		t.Errorf("expected search %s, got %+v", ".svc."+kubelet.clusterDomain, options[0].DNSSearch)
	}
	if len(options[1].DNS) != 1 || options[1].DNS[0] != "127.0.0.1" {
		t.Errorf("expected nameserver 127.0.0.1, got %+v", options[1].DNS)
	}
	if len(options[1].DNSSearch) != 1 || options[1].DNSSearch[0] != "." {
		t.Errorf("expected search \".\", got %+v", options[1].DNSSearch)
	}

	kubelet.resolverConfig = "/etc/resolv.conf"
	for i, pod := range pods {
		var err error
		options[i], err = kubelet.GenerateRunContainerOptions(pod, &api.Container{})
		if err != nil {
			t.Fatalf("failed to generate container options: %v", err)
		}
	}
	t.Logf("nameservers %+v", options[1].DNS)
	if len(options[0].DNS) != len(options[1].DNS)+1 {
		t.Errorf("expected prepend of cluster nameserver, got %+v", options[0].DNS)
	} else if options[0].DNS[0] != clusterNS {
		t.Errorf("expected nameserver %s, got %v", clusterNS, options[0].DNS[0])
	}
	if len(options[0].DNSSearch) != len(options[1].DNSSearch)+3 {
		t.Errorf("expected prepend of cluster domain, got %+v", options[0].DNSSearch)
	} else if options[0].DNSSearch[0] != ".svc."+kubelet.clusterDomain {
		t.Errorf("expected domain %s, got %s", ".svc."+kubelet.clusterDomain, options[0].DNSSearch)
	}
}

type testServiceLister struct {
	services []api.Service
}

func (ls testServiceLister) List() (api.ServiceList, error) {
	return api.ServiceList{
		Items: ls.services,
	}, nil
}

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

type envs []kubecontainer.EnvVar

func (e envs) Len() int {
	return len(e)
}

func (e envs) Swap(i, j int) { e[i], e[j] = e[j], e[i] }

func (e envs) Less(i, j int) bool { return e[i].Name < e[j].Name }

func TestMakeEnvironmentVariables(t *testing.T) {
	services := []api.Service{
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: api.NamespaceDefault},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8081,
				}},
				ClusterIP: "1.2.3.1",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "test1"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8083,
				}},
				ClusterIP: "1.2.3.3",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "test2"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8084,
				}},
				ClusterIP: "1.2.3.4",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "test2"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8085,
				}},
				ClusterIP: "1.2.3.5",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "test2"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8085,
				}},
				ClusterIP: "None",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "test2"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8085,
				}},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8086,
				}},
				ClusterIP: "1.2.3.6",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "not-special", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8088,
				}},
				ClusterIP: "1.2.3.8",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "not-special", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8088,
				}},
				ClusterIP: "None",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "not-special", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8088,
				}},
				ClusterIP: "",
			},
		},
	}

	testCases := []struct {
		name            string                 // the name of the test case
		ns              string                 // the namespace to generate environment for
		container       *api.Container         // the container to use
		masterServiceNs string                 // the namespace to read master service info from
		nilLister       bool                   // whether the lister should be nil
		expectedEnvs    []kubecontainer.EnvVar // a set of expected environment vars
	}{
		{
			name: "api server = Y, kubelet = Y",
			ns:   "test1",
			container: &api.Container{
				Env: []api.EnvVar{
					{Name: "FOO", Value: "BAR"},
					{Name: "TEST_SERVICE_HOST", Value: "1.2.3.3"},
					{Name: "TEST_SERVICE_PORT", Value: "8083"},
					{Name: "TEST_PORT", Value: "tcp://1.2.3.3:8083"},
					{Name: "TEST_PORT_8083_TCP", Value: "tcp://1.2.3.3:8083"},
					{Name: "TEST_PORT_8083_TCP_PROTO", Value: "tcp"},
					{Name: "TEST_PORT_8083_TCP_PORT", Value: "8083"},
					{Name: "TEST_PORT_8083_TCP_ADDR", Value: "1.2.3.3"},
				},
			},
			masterServiceNs: api.NamespaceDefault,
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "FOO", Value: "BAR"},
				{Name: "TEST_SERVICE_HOST", Value: "1.2.3.3"},
				{Name: "TEST_SERVICE_PORT", Value: "8083"},
				{Name: "TEST_PORT", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP_PROTO", Value: "tcp"},
				{Name: "TEST_PORT_8083_TCP_PORT", Value: "8083"},
				{Name: "TEST_PORT_8083_TCP_ADDR", Value: "1.2.3.3"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8081"},
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.1"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.1:8081"},
				{Name: "KUBERNETES_PORT_8081_TCP", Value: "tcp://1.2.3.1:8081"},
				{Name: "KUBERNETES_PORT_8081_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8081_TCP_PORT", Value: "8081"},
				{Name: "KUBERNETES_PORT_8081_TCP_ADDR", Value: "1.2.3.1"},
			},
		},
		{
			name: "api server = Y, kubelet = N",
			ns:   "test1",
			container: &api.Container{
				Env: []api.EnvVar{
					{Name: "FOO", Value: "BAR"},
					{Name: "TEST_SERVICE_HOST", Value: "1.2.3.3"},
					{Name: "TEST_SERVICE_PORT", Value: "8083"},
					{Name: "TEST_PORT", Value: "tcp://1.2.3.3:8083"},
					{Name: "TEST_PORT_8083_TCP", Value: "tcp://1.2.3.3:8083"},
					{Name: "TEST_PORT_8083_TCP_PROTO", Value: "tcp"},
					{Name: "TEST_PORT_8083_TCP_PORT", Value: "8083"},
					{Name: "TEST_PORT_8083_TCP_ADDR", Value: "1.2.3.3"},
				},
			},
			masterServiceNs: api.NamespaceDefault,
			nilLister:       true,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "FOO", Value: "BAR"},
				{Name: "TEST_SERVICE_HOST", Value: "1.2.3.3"},
				{Name: "TEST_SERVICE_PORT", Value: "8083"},
				{Name: "TEST_PORT", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP_PROTO", Value: "tcp"},
				{Name: "TEST_PORT_8083_TCP_PORT", Value: "8083"},
				{Name: "TEST_PORT_8083_TCP_ADDR", Value: "1.2.3.3"},
			},
		},
		{
			name: "api server = N; kubelet = Y",
			ns:   "test1",
			container: &api.Container{
				Env: []api.EnvVar{
					{Name: "FOO", Value: "BAZ"},
				},
			},
			masterServiceNs: api.NamespaceDefault,
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "FOO", Value: "BAZ"},
				{Name: "TEST_SERVICE_HOST", Value: "1.2.3.3"},
				{Name: "TEST_SERVICE_PORT", Value: "8083"},
				{Name: "TEST_PORT", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP_PROTO", Value: "tcp"},
				{Name: "TEST_PORT_8083_TCP_PORT", Value: "8083"},
				{Name: "TEST_PORT_8083_TCP_ADDR", Value: "1.2.3.3"},
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.1"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8081"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.1:8081"},
				{Name: "KUBERNETES_PORT_8081_TCP", Value: "tcp://1.2.3.1:8081"},
				{Name: "KUBERNETES_PORT_8081_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8081_TCP_PORT", Value: "8081"},
				{Name: "KUBERNETES_PORT_8081_TCP_ADDR", Value: "1.2.3.1"},
			},
		},
		{
			name: "master service in pod ns",
			ns:   "test2",
			container: &api.Container{
				Env: []api.EnvVar{
					{Name: "FOO", Value: "ZAP"},
				},
			},
			masterServiceNs: "kubernetes",
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "FOO", Value: "ZAP"},
				{Name: "TEST_SERVICE_HOST", Value: "1.2.3.5"},
				{Name: "TEST_SERVICE_PORT", Value: "8085"},
				{Name: "TEST_PORT", Value: "tcp://1.2.3.5:8085"},
				{Name: "TEST_PORT_8085_TCP", Value: "tcp://1.2.3.5:8085"},
				{Name: "TEST_PORT_8085_TCP_PROTO", Value: "tcp"},
				{Name: "TEST_PORT_8085_TCP_PORT", Value: "8085"},
				{Name: "TEST_PORT_8085_TCP_ADDR", Value: "1.2.3.5"},
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.4"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8084"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.4:8084"},
				{Name: "KUBERNETES_PORT_8084_TCP", Value: "tcp://1.2.3.4:8084"},
				{Name: "KUBERNETES_PORT_8084_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8084_TCP_PORT", Value: "8084"},
				{Name: "KUBERNETES_PORT_8084_TCP_ADDR", Value: "1.2.3.4"},
			},
		},
		{
			name:            "pod in master service ns",
			ns:              "kubernetes",
			container:       &api.Container{},
			masterServiceNs: "kubernetes",
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "NOT_SPECIAL_SERVICE_HOST", Value: "1.2.3.8"},
				{Name: "NOT_SPECIAL_SERVICE_PORT", Value: "8088"},
				{Name: "NOT_SPECIAL_PORT", Value: "tcp://1.2.3.8:8088"},
				{Name: "NOT_SPECIAL_PORT_8088_TCP", Value: "tcp://1.2.3.8:8088"},
				{Name: "NOT_SPECIAL_PORT_8088_TCP_PROTO", Value: "tcp"},
				{Name: "NOT_SPECIAL_PORT_8088_TCP_PORT", Value: "8088"},
				{Name: "NOT_SPECIAL_PORT_8088_TCP_ADDR", Value: "1.2.3.8"},
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.6"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8086"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.6:8086"},
				{Name: "KUBERNETES_PORT_8086_TCP", Value: "tcp://1.2.3.6:8086"},
				{Name: "KUBERNETES_PORT_8086_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8086_TCP_PORT", Value: "8086"},
				{Name: "KUBERNETES_PORT_8086_TCP_ADDR", Value: "1.2.3.6"},
			},
		},
		{
			name: "downward api pod",
			ns:   "downward-api",
			container: &api.Container{
				Env: []api.EnvVar{
					{
						Name: "POD_NAME",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.Version(),
								FieldPath:  "metadata.name",
							},
						},
					},
					{
						Name: "POD_NAMESPACE",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.Version(),
								FieldPath:  "metadata.namespace",
							},
						},
					},
					{
						Name: "POD_IP",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.Version(),
								FieldPath:  "status.podIP",
							},
						},
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       true,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "POD_NAME", Value: "dapi-test-pod-name"},
				{Name: "POD_NAMESPACE", Value: "downward-api"},
				{Name: "POD_IP", Value: "1.2.3.4"},
			},
		},
		{
			name: "env expansion",
			ns:   "test1",
			container: &api.Container{
				Env: []api.EnvVar{
					{
						Name:  "TEST_LITERAL",
						Value: "test-test-test",
					},
					{
						Name: "POD_NAME",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.Version(),
								FieldPath:  "metadata.name",
							},
						},
					},
					{
						Name:  "OUT_OF_ORDER_TEST",
						Value: "$(OUT_OF_ORDER_TARGET)",
					},
					{
						Name:  "OUT_OF_ORDER_TARGET",
						Value: "FOO",
					},
					{
						Name: "EMPTY_VAR",
					},
					{
						Name:  "EMPTY_TEST",
						Value: "foo-$(EMPTY_VAR)",
					},
					{
						Name:  "POD_NAME_TEST2",
						Value: "test2-$(POD_NAME)",
					},
					{
						Name:  "POD_NAME_TEST3",
						Value: "$(POD_NAME_TEST2)-3",
					},
					{
						Name:  "LITERAL_TEST",
						Value: "literal-$(TEST_LITERAL)",
					},
					{
						Name:  "SERVICE_VAR_TEST",
						Value: "$(TEST_SERVICE_HOST):$(TEST_SERVICE_PORT)",
					},
					{
						Name:  "TEST_UNDEFINED",
						Value: "$(UNDEFINED_VAR)",
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "TEST_LITERAL",
					Value: "test-test-test",
				},
				{
					Name:  "POD_NAME",
					Value: "dapi-test-pod-name",
				},
				{
					Name:  "POD_NAME_TEST2",
					Value: "test2-dapi-test-pod-name",
				},
				{
					Name:  "POD_NAME_TEST3",
					Value: "test2-dapi-test-pod-name-3",
				},
				{
					Name:  "LITERAL_TEST",
					Value: "literal-test-test-test",
				},
				{
					Name:  "TEST_SERVICE_HOST",
					Value: "1.2.3.3",
				},
				{
					Name:  "TEST_SERVICE_PORT",
					Value: "8083",
				},
				{
					Name:  "TEST_PORT",
					Value: "tcp://1.2.3.3:8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP",
					Value: "tcp://1.2.3.3:8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP_PROTO",
					Value: "tcp",
				},
				{
					Name:  "TEST_PORT_8083_TCP_PORT",
					Value: "8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP_ADDR",
					Value: "1.2.3.3",
				},
				{
					Name:  "SERVICE_VAR_TEST",
					Value: "1.2.3.3:8083",
				},
				{
					Name:  "OUT_OF_ORDER_TEST",
					Value: "$(OUT_OF_ORDER_TARGET)",
				},
				{
					Name:  "OUT_OF_ORDER_TARGET",
					Value: "FOO",
				},
				{
					Name:  "TEST_UNDEFINED",
					Value: "$(UNDEFINED_VAR)",
				},
				{
					Name: "EMPTY_VAR",
				},
				{
					Name:  "EMPTY_TEST",
					Value: "foo-",
				},
			},
		},
	}

	for i, tc := range testCases {
		testKubelet := newTestKubelet(t)
		kl := testKubelet.kubelet
		kl.masterServiceNamespace = tc.masterServiceNs
		if tc.nilLister {
			kl.serviceLister = nil
		} else {
			kl.serviceLister = testServiceLister{services}
		}

		testPod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Namespace: tc.ns,
				Name:      "dapi-test-pod-name",
			},
		}
		testPod.Status.PodIP = "1.2.3.4"

		result, err := kl.makeEnvironmentVariables(testPod, tc.container)
		if err != nil {
			t.Errorf("[%v] Unexpected error: %v", tc.name, err)
		}

		sort.Sort(envs(result))
		sort.Sort(envs(tc.expectedEnvs))

		if !reflect.DeepEqual(result, tc.expectedEnvs) {
			t.Errorf("%d: [%v] Unexpected env entries; expected {%v}, got {%v}", i, tc.name, tc.expectedEnvs, result)
		}
	}
}

func runningState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
}
func stoppedState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Terminated: &api.ContainerStateTerminated{},
		},
	}
}
func succeededState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Terminated: &api.ContainerStateTerminated{
				ExitCode: 0,
			},
		},
	}
}
func failedState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Terminated: &api.ContainerStateTerminated{
				ExitCode: -1,
			},
		},
	}
}

func TestPodPhaseWithRestartAlways(t *testing.T) {
	desiredState := api.PodSpec{
		NodeName: "machine",
		Containers: []api.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: api.RestartPolicyAlways,
	}

	tests := []struct {
		pod    *api.Pod
		status api.PodPhase
		test   string
	}{
		{&api.Pod{Spec: desiredState, Status: api.PodStatus{}}, api.PodPending, "waiting"},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			api.PodRunning,
			"all running",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						stoppedState("containerA"),
						stoppedState("containerB"),
					},
				},
			},
			api.PodRunning,
			"all stopped with restart always",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						stoppedState("containerB"),
					},
				},
			},
			api.PodRunning,
			"mixed state #1 with restart always",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
					},
				},
			},
			api.PodPending,
			"mixed state #2 with restart always",
		},
	}
	for _, test := range tests {
		if status := GetPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses); status != test.status {
			t.Errorf("In test %s, expected %v, got %v", test.test, test.status, status)
		}
	}
}

func TestPodPhaseWithRestartNever(t *testing.T) {
	desiredState := api.PodSpec{
		NodeName: "machine",
		Containers: []api.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: api.RestartPolicyNever,
	}

	tests := []struct {
		pod    *api.Pod
		status api.PodPhase
		test   string
	}{
		{&api.Pod{Spec: desiredState, Status: api.PodStatus{}}, api.PodPending, "waiting"},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			api.PodRunning,
			"all running with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						succeededState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			api.PodSucceeded,
			"all succeeded with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						failedState("containerA"),
						failedState("containerB"),
					},
				},
			},
			api.PodFailed,
			"all failed with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			api.PodRunning,
			"mixed state #1 with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
					},
				},
			},
			api.PodPending,
			"mixed state #2 with restart never",
		},
	}
	for _, test := range tests {
		if status := GetPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses); status != test.status {
			t.Errorf("In test %s, expected %v, got %v", test.test, test.status, status)
		}
	}
}

func TestPodPhaseWithRestartOnFailure(t *testing.T) {
	desiredState := api.PodSpec{
		NodeName: "machine",
		Containers: []api.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: api.RestartPolicyOnFailure,
	}

	tests := []struct {
		pod    *api.Pod
		status api.PodPhase
		test   string
	}{
		{&api.Pod{Spec: desiredState, Status: api.PodStatus{}}, api.PodPending, "waiting"},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			api.PodRunning,
			"all running with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						succeededState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			api.PodSucceeded,
			"all succeeded with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						failedState("containerA"),
						failedState("containerB"),
					},
				},
			},
			api.PodRunning,
			"all failed with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			api.PodRunning,
			"mixed state #1 with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
					},
				},
			},
			api.PodPending,
			"mixed state #2 with restart onfailure",
		},
	}
	for _, test := range tests {
		if status := GetPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses); status != test.status {
			t.Errorf("In test %s, expected %v, got %v", test.test, test.status, status)
		}
	}
}

func getReadyStatus(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name:  cName,
		Ready: true,
	}
}
func getNotReadyStatus(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name:  cName,
		Ready: false,
	}
}
func getReadyCondition(status api.ConditionStatus, reason, message string) []api.PodCondition {
	return []api.PodCondition{{
		Type:    api.PodReady,
		Status:  status,
		Reason:  reason,
		Message: message,
	}}
}

func TestGetPodReadyCondition(t *testing.T) {
	tests := []struct {
		spec              *api.PodSpec
		containerStatuses []api.ContainerStatus
		expected          []api.PodCondition
	}{
		{
			spec:              nil,
			containerStatuses: nil,
			expected:          getReadyCondition(api.ConditionFalse, "UnknownContainerStatuses", ""),
		},
		{
			spec:              &api.PodSpec{},
			containerStatuses: []api.ContainerStatus{},
			expected:          getReadyCondition(api.ConditionTrue, "", ""),
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
				},
			},
			containerStatuses: []api.ContainerStatus{},
			expected:          getReadyCondition(api.ConditionFalse, "ContainersNotReady", "containers with unknown status: [1234]"),
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []api.ContainerStatus{
				getReadyStatus("1234"),
				getReadyStatus("5678"),
			},
			expected: getReadyCondition(api.ConditionTrue, "", ""),
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []api.ContainerStatus{
				getReadyStatus("1234"),
			},
			expected: getReadyCondition(api.ConditionFalse, "ContainersNotReady", "containers with unknown status: [5678]"),
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			containerStatuses: []api.ContainerStatus{
				getReadyStatus("1234"),
				getNotReadyStatus("5678"),
			},
			expected: getReadyCondition(api.ConditionFalse, "ContainersNotReady", "containers with unready status: [5678]"),
		},
	}

	for i, test := range tests {
		condition := getPodReadyCondition(test.spec, test.containerStatuses)
		if !reflect.DeepEqual(condition, test.expected) {
			t.Errorf("On test case %v, expected:\n%+v\ngot\n%+v\n", i, test.expected, condition)
		}
	}
}

func TestExecInContainerNoSuchPod(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner
	fakeRuntime.PodList = []*kubecontainer.Pod{}

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerID := "containerFoo"
	err := kubelet.ExecInContainer(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{Name: podName, Namespace: podNamespace}}),
		"",
		containerID,
		[]string{"ls"},
		nil,
		nil,
		nil,
		false,
	)
	if err == nil {
		t.Fatal("unexpected non-error")
	}
	if !fakeCommandRunner.ID.IsEmpty() {
		t.Fatal("unexpected invocation of runner.ExecInContainer")
	}
}

func TestExecInContainerNoSuchContainer(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerID := "containerFoo"
	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:        "12345678",
			Name:      podName,
			Namespace: podNamespace,
			Containers: []*kubecontainer.Container{
				{Name: "bar",
					ID: kubecontainer.ContainerID{"test", "barID"}},
			},
		},
	}

	err := kubelet.ExecInContainer(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      podName,
			Namespace: podNamespace,
		}}),
		"",
		containerID,
		[]string{"ls"},
		nil,
		nil,
		nil,
		false,
	)
	if err == nil {
		t.Fatal("unexpected non-error")
	}
	if !fakeCommandRunner.ID.IsEmpty() {
		t.Fatal("unexpected invocation of runner.ExecInContainer")
	}
}

type fakeReadWriteCloser struct{}

func (f *fakeReadWriteCloser) Write(data []byte) (int, error) {
	return 0, nil
}

func (f *fakeReadWriteCloser) Read(data []byte) (int, error) {
	return 0, nil
}

func (f *fakeReadWriteCloser) Close() error {
	return nil
}

func TestExecInContainer(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerID := "containerFoo"
	command := []string{"ls"}
	stdin := &bytes.Buffer{}
	stdout := &fakeReadWriteCloser{}
	stderr := &fakeReadWriteCloser{}
	tty := true
	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:        "12345678",
			Name:      podName,
			Namespace: podNamespace,
			Containers: []*kubecontainer.Container{
				{Name: containerID,
					ID: kubecontainer.ContainerID{"test", containerID},
				},
			},
		},
	}

	err := kubelet.ExecInContainer(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      podName,
			Namespace: podNamespace,
		}}),
		"",
		containerID,
		[]string{"ls"},
		stdin,
		stdout,
		stderr,
		tty,
	)
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	}
	if e, a := containerID, fakeCommandRunner.ID.ID; e != a {
		t.Fatalf("container name: expected %q, got %q", e, a)
	}
	if e, a := command, fakeCommandRunner.Cmd; !reflect.DeepEqual(e, a) {
		t.Fatalf("command: expected '%v', got '%v'", e, a)
	}
	if e, a := stdin, fakeCommandRunner.Stdin; e != a {
		t.Fatalf("stdin: expected %#v, got %#v", e, a)
	}
	if e, a := stdout, fakeCommandRunner.Stdout; e != a {
		t.Fatalf("stdout: expected %#v, got %#v", e, a)
	}
	if e, a := stderr, fakeCommandRunner.Stderr; e != a {
		t.Fatalf("stderr: expected %#v, got %#v", e, a)
	}
	if e, a := tty, fakeCommandRunner.TTY; e != a {
		t.Fatalf("tty: expected %t, got %t", e, a)
	}
}

func TestPortForwardNoSuchPod(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeRuntime.PodList = []*kubecontainer.Pod{}
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	var port uint16 = 5000

	err := kubelet.PortForward(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{Name: podName, Namespace: podNamespace}}),
		"",
		port,
		nil,
	)
	if err == nil {
		t.Fatal("unexpected non-error")
	}
	if !fakeCommandRunner.ID.IsEmpty() {
		t.Fatal("unexpected invocation of runner.PortForward")
	}
}

func TestPortForward(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime

	podName := "podFoo"
	podNamespace := "nsFoo"
	podID := types.UID("12345678")
	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:        podID,
			Name:      podName,
			Namespace: podNamespace,
			Containers: []*kubecontainer.Container{
				{
					Name: "foo",
					ID:   kubecontainer.ContainerID{"test", "containerFoo"},
				},
			},
		},
	}
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner

	var port uint16 = 5000
	stream := &fakeReadWriteCloser{}
	err := kubelet.PortForward(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      podName,
			Namespace: podNamespace,
		}}),
		"",
		port,
		stream,
	)
	if err != nil {
		t.Fatalf("unexpected error: %s", err)
	}
	if e, a := podID, fakeCommandRunner.PodID; e != a {
		t.Fatalf("container id: expected %q, got %q", e, a)
	}
	if e, a := port, fakeCommandRunner.Port; e != a {
		t.Fatalf("port: expected %v, got %v", e, a)
	}
	if e, a := stream, fakeCommandRunner.Stream; e != a {
		t.Fatalf("stream: expected %v, got %v", e, a)
	}
}

// Tests that identify the host port conflicts are detected correctly.
func TestGetHostPortConflicts(t *testing.T) {
	pods := []*api.Pod{
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 80}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 81}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 82}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 83}}}}}},
	}
	// Pods should not cause any conflict.
	if hasHostPortConflicts(pods) {
		t.Errorf("expected no conflicts, Got conflicts")
	}

	expected := &api.Pod{
		Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 81}}}}},
	}
	// The new pod should cause conflict and be reported.
	pods = append(pods, expected)
	if !hasHostPortConflicts(pods) {
		t.Errorf("expected no conflict, Got no conflicts")
	}
}

// Tests that we handle port conflicts correctly by setting the failed status in status map.
func TestHandlePortConflicts(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kl := testKubelet.kubelet
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)

	spec := api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 80}}}}}
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "123456789",
				Name:      "newpod",
				Namespace: "foo",
			},
			Spec: spec,
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "987654321",
				Name:      "oldpod",
				Namespace: "foo",
			},
			Spec: spec,
		},
	}
	// Make sure the Pods are in the reverse order of creation time.
	pods[1].CreationTimestamp = unversioned.NewTime(time.Now())
	pods[0].CreationTimestamp = unversioned.NewTime(time.Now().Add(1 * time.Second))
	// The newer pod should be rejected.
	conflictedPod := pods[0]

	kl.HandlePodAdditions(pods)
	// Check pod status stored in the status map.
	status, found := kl.statusManager.GetPodStatus(conflictedPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", conflictedPod.UID)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q. Got %q.", api.PodFailed, status.Phase)
	}
}

// Tests that we handle not matching labels selector correctly by setting the failed status in status map.
func TestHandleNodeSelector(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kl := testKubelet.kubelet
	kl.nodeLister = testNodeLister{nodes: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname, Labels: map[string]string{"key": "B"}}},
	}}
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "123456789",
				Name:      "podA",
				Namespace: "foo",
			},
			Spec: api.PodSpec{NodeSelector: map[string]string{"key": "A"}},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "987654321",
				Name:      "podB",
				Namespace: "foo",
			},
			Spec: api.PodSpec{NodeSelector: map[string]string{"key": "B"}},
		},
	}
	// The first pod should be rejected.
	notfittingPod := pods[0]

	kl.HandlePodAdditions(pods)
	// Check pod status stored in the status map.
	status, found := kl.statusManager.GetPodStatus(notfittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", notfittingPod.UID)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q. Got %q.", api.PodFailed, status.Phase)
	}
}

// Tests that we handle exceeded resources correctly by setting the failed status in status map.
func TestHandleMemExceeded(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kl := testKubelet.kubelet
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{MemoryCapacity: 100}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)

	spec := api.PodSpec{Containers: []api.Container{{Resources: api.ResourceRequirements{
		Requests: api.ResourceList{
			"memory": resource.MustParse("90"),
		},
	}}}}
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "123456789",
				Name:      "newpod",
				Namespace: "foo",
			},
			Spec: spec,
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "987654321",
				Name:      "oldpod",
				Namespace: "foo",
			},
			Spec: spec,
		},
	}
	// Make sure the Pods are in the reverse order of creation time.
	pods[1].CreationTimestamp = unversioned.NewTime(time.Now())
	pods[0].CreationTimestamp = unversioned.NewTime(time.Now().Add(1 * time.Second))
	// The newer pod should be rejected.
	notfittingPod := pods[0]

	kl.HandlePodAdditions(pods)
	// Check pod status stored in the status map.
	status, found := kl.statusManager.GetPodStatus(notfittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", notfittingPod.UID)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q. Got %q.", api.PodFailed, status.Phase)
	}
}

// TODO(filipg): This test should be removed once StatusSyncer can do garbage collection without external signal.
func TestPurgingObsoleteStatusMapEntries(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)

	kl := testKubelet.kubelet
	pods := []*api.Pod{
		{ObjectMeta: api.ObjectMeta{Name: "pod1", UID: "1234"}, Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 80}}}}}},
		{ObjectMeta: api.ObjectMeta{Name: "pod2", UID: "4567"}, Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 80}}}}}},
	}
	podToTest := pods[1]
	// Run once to populate the status map.
	kl.HandlePodAdditions(pods)
	if _, found := kl.statusManager.GetPodStatus(podToTest.UID); !found {
		t.Fatalf("expected to have status cached for pod2")
	}
	// Sync with empty pods so that the entry in status map will be removed.
	kl.podManager.SetPods([]*api.Pod{})
	kl.HandlePodCleanups()
	if _, found := kl.statusManager.GetPodStatus(podToTest.UID); found {
		t.Fatalf("expected to not have status cached for pod2")
	}
}

func TestValidatePodStatus(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	testCases := []struct {
		podPhase api.PodPhase
		success  bool
	}{
		{api.PodRunning, true},
		{api.PodSucceeded, true},
		{api.PodFailed, true},
		{api.PodPending, false},
		{api.PodUnknown, false},
	}

	for i, tc := range testCases {
		err := kubelet.validatePodPhase(&api.PodStatus{Phase: tc.podPhase})
		if tc.success {
			if err != nil {
				t.Errorf("[case %d]: unexpected failure - %v", i, err)
			}
		} else if err == nil {
			t.Errorf("[case %d]: unexpected success", i)
		}
	}
}

func TestValidateContainerStatus(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	containerName := "x"
	testCases := []struct {
		statuses []api.ContainerStatus
		success  bool
	}{
		{
			statuses: []api.ContainerStatus{
				{
					Name: containerName,
					State: api.ContainerState{
						Running: &api.ContainerStateRunning{},
					},
					LastTerminationState: api.ContainerState{
						Terminated: &api.ContainerStateTerminated{},
					},
				},
			},
			success: true,
		},
		{
			statuses: []api.ContainerStatus{
				{
					Name: containerName,
					State: api.ContainerState{
						Terminated: &api.ContainerStateTerminated{},
					},
				},
			},
			success: true,
		},
		{
			statuses: []api.ContainerStatus{
				{
					Name: containerName,
					State: api.ContainerState{
						Waiting: &api.ContainerStateWaiting{},
					},
				},
			},
			success: false,
		},
	}

	for i, tc := range testCases {
		_, err := kubelet.validateContainerStatus(&api.PodStatus{
			ContainerStatuses: tc.statuses,
		}, containerName, false)
		if tc.success {
			if err != nil {
				t.Errorf("[case %d]: unexpected failure - %v", i, err)
			}
		} else if err == nil {
			t.Errorf("[case %d]: unexpected success", i)
		}
	}
	if _, err := kubelet.validateContainerStatus(&api.PodStatus{
		ContainerStatuses: testCases[0].statuses,
	}, "blah", false); err == nil {
		t.Errorf("expected error with invalid container name")
	}
	if _, err := kubelet.validateContainerStatus(&api.PodStatus{
		ContainerStatuses: testCases[0].statuses,
	}, containerName, true); err != nil {
		t.Errorf("unexpected error with for previous terminated container - %v", err)
	}
	if _, err := kubelet.validateContainerStatus(&api.PodStatus{
		ContainerStatuses: testCases[1].statuses,
	}, containerName, true); err == nil {
		t.Errorf("expected error with for previous terminated container")
	}
}

func TestUpdateNewNodeStatus(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.ReactionChain = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname}},
	}}).ReactionChain
	machineInfo := &cadvisorApi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	mockCadvisor := testKubelet.fakeCadvisor
	mockCadvisor.On("MachineInfo").Return(machineInfo, nil)
	versionInfo := &cadvisorApi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}
	mockCadvisor.On("VersionInfo").Return(versionInfo, nil)
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
					LastHeartbeatTime:  unversioned.Time{},
					LastTransitionTime: unversioned.Time{},
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

	kubelet.updateRuntimeUp()
	if err := kubelet.updateNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	actions := kubeClient.Actions()
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
	updatedNode.Status.Conditions[0].LastHeartbeatTime = unversioned.Time{}
	updatedNode.Status.Conditions[0].LastTransitionTime = unversioned.Time{}
	if !reflect.DeepEqual(expectedNode, updatedNode) {
		t.Errorf("unexpected objects: %s", util.ObjectDiff(expectedNode, updatedNode))
	}
}

func TestUpdateExistingNodeStatus(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.ReactionChain = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{
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
						LastHeartbeatTime:  unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
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
	mockCadvisor := testKubelet.fakeCadvisor
	machineInfo := &cadvisorApi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	mockCadvisor.On("MachineInfo").Return(machineInfo, nil)
	versionInfo := &cadvisorApi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}
	mockCadvisor.On("VersionInfo").Return(versionInfo, nil)
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
					LastHeartbeatTime:  unversioned.Time{}, // placeholder
					LastTransitionTime: unversioned.Time{}, // placeholder
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

	kubelet.updateRuntimeUp()
	if err := kubelet.updateNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	actions := kubeClient.Actions()
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
	if reflect.DeepEqual(updatedNode.Status.Conditions[0].LastHeartbeatTime.Rfc3339Copy().UTC(), unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC).Time) {
		t.Errorf("expected \n%v\n, got \n%v", unversioned.Now(), unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC))
	}
	if !reflect.DeepEqual(updatedNode.Status.Conditions[0].LastTransitionTime.Rfc3339Copy().UTC(), unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC).Time) {
		t.Errorf("expected \n%#v\n, got \n%#v", updatedNode.Status.Conditions[0].LastTransitionTime.Rfc3339Copy(),
			unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC))
	}
	updatedNode.Status.Conditions[0].LastHeartbeatTime = unversioned.Time{}
	updatedNode.Status.Conditions[0].LastTransitionTime = unversioned.Time{}
	if !reflect.DeepEqual(expectedNode, updatedNode) {
		t.Errorf("expected \n%v\n, got \n%v", expectedNode, updatedNode)
	}
}

func TestUpdateNodeStatusWithoutContainerRuntime(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	fakeRuntime := testKubelet.fakeRuntime
	// This causes returning an error from GetContainerRuntimeVersion() which
	// simulates that container runtime is down.
	fakeRuntime.VersionInfo = ""

	kubeClient.ReactionChain = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname}},
	}}).ReactionChain
	mockCadvisor := testKubelet.fakeCadvisor
	machineInfo := &cadvisorApi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	mockCadvisor.On("MachineInfo").Return(machineInfo, nil)
	versionInfo := &cadvisorApi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}
	mockCadvisor.On("VersionInfo").Return(versionInfo, nil)

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
					LastHeartbeatTime:  unversioned.Time{},
					LastTransitionTime: unversioned.Time{},
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

	kubelet.runtimeUpThreshold = time.Duration(0)
	kubelet.updateRuntimeUp()
	if err := kubelet.updateNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	actions := kubeClient.Actions()
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
	updatedNode.Status.Conditions[0].LastHeartbeatTime = unversioned.Time{}
	updatedNode.Status.Conditions[0].LastTransitionTime = unversioned.Time{}
	if !reflect.DeepEqual(expectedNode, updatedNode) {
		t.Errorf("unexpected objects: %s", util.ObjectDiff(expectedNode, updatedNode))
	}
}

func TestUpdateNodeStatusError(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	// No matching node for the kubelet
	testKubelet.fakeKubeClient.ReactionChain = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{}}).ReactionChain

	if err := kubelet.updateNodeStatus(); err == nil {
		t.Errorf("unexpected non error: %v", err)
	}
	if len(testKubelet.fakeKubeClient.Actions()) != nodeStatusUpdateRetry {
		t.Errorf("unexpected actions: %v", testKubelet.fakeKubeClient.Actions())
	}
}

func TestCreateMirrorPod(t *testing.T) {
	for _, updateType := range []kubetypes.SyncPodType{kubetypes.SyncPodCreate, kubetypes.SyncPodUpdate} {
		testKubelet := newTestKubelet(t)
		kl := testKubelet.kubelet
		manager := testKubelet.fakeMirrorClient
		pod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "bar",
				Namespace: "foo",
				Annotations: map[string]string{
					kubetypes.ConfigSourceAnnotationKey: "file",
				},
			},
		}
		pods := []*api.Pod{pod}
		kl.podManager.SetPods(pods)
		err := kl.syncPod(pod, nil, container.Pod{}, updateType)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		podFullName := kubecontainer.GetPodFullName(pod)
		if !manager.HasPod(podFullName) {
			t.Errorf("expected mirror pod %q to be created", podFullName)
		}
		if manager.NumOfPods() != 1 || !manager.HasPod(podFullName) {
			t.Errorf("expected one mirror pod %q, got %v", podFullName, manager.GetPods())
		}
	}
}

func TestDeleteOutdatedMirrorPod(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kl := testKubelet.kubelet
	manager := testKubelet.fakeMirrorClient
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "ns",
			Annotations: map[string]string{
				kubetypes.ConfigSourceAnnotationKey: "file",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "1234", Image: "foo"},
			},
		},
	}
	// Mirror pod has an outdated spec.
	mirrorPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "11111111",
			Name:      "foo",
			Namespace: "ns",
			Annotations: map[string]string{
				kubetypes.ConfigSourceAnnotationKey: "api",
				kubetypes.ConfigMirrorAnnotationKey: "mirror",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "1234", Image: "bar"},
			},
		},
	}

	pods := []*api.Pod{pod, mirrorPod}
	kl.podManager.SetPods(pods)
	err := kl.syncPod(pod, mirrorPod, container.Pod{}, kubetypes.SyncPodUpdate)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	name := kubecontainer.GetPodFullName(pod)
	creates, deletes := manager.GetCounts(name)
	if creates != 0 || deletes != 1 {
		t.Errorf("expected 0 creation and 1 deletion of %q, got %d, %d", name, creates, deletes)
	}
}

func TestDeleteOrphanedMirrorPods(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kl := testKubelet.kubelet
	manager := testKubelet.fakeMirrorClient
	orphanPods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "pod1",
				Namespace: "ns",
				Annotations: map[string]string{
					kubetypes.ConfigSourceAnnotationKey: "api",
					kubetypes.ConfigMirrorAnnotationKey: "mirror",
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345679",
				Name:      "pod2",
				Namespace: "ns",
				Annotations: map[string]string{
					kubetypes.ConfigSourceAnnotationKey: "api",
					kubetypes.ConfigMirrorAnnotationKey: "mirror",
				},
			},
		},
	}

	kl.podManager.SetPods(orphanPods)
	// Sync with an empty pod list to delete all mirror pods.
	kl.HandlePodCleanups()
	if manager.NumOfPods() != 0 {
		t.Errorf("expected zero mirror pods, got %v", manager.GetPods())
	}
	for _, pod := range orphanPods {
		name := kubecontainer.GetPodFullName(pod)
		creates, deletes := manager.GetCounts(name)
		if creates != 0 || deletes != 1 {
			t.Errorf("expected 0 creation and one deletion of %q, got %d, %d", name, creates, deletes)
		}
	}
}

func TestGetContainerInfoForMirrorPods(t *testing.T) {
	// pods contain one static and one mirror pod with the same name but
	// different UIDs.
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "1234",
				Name:      "qux",
				Namespace: "ns",
				Annotations: map[string]string{
					kubetypes.ConfigSourceAnnotationKey: "file",
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "foo"},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "5678",
				Name:      "qux",
				Namespace: "ns",
				Annotations: map[string]string{
					kubetypes.ConfigSourceAnnotationKey: "api",
					kubetypes.ConfigMirrorAnnotationKey: "mirror",
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "foo"},
				},
			},
		},
	}

	containerID := "ab2cdf"
	containerPath := fmt.Sprintf("/docker/%v", containerID)
	containerInfo := cadvisorApi.ContainerInfo{
		ContainerReference: cadvisorApi.ContainerReference{
			Name: containerPath,
		},
	}

	testKubelet := newTestKubelet(t)
	fakeRuntime := testKubelet.fakeRuntime
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorReq := &cadvisorApi.ContainerInfoRequest{}
	mockCadvisor.On("DockerContainer", containerID, cadvisorReq).Return(containerInfo, nil)
	kubelet := testKubelet.kubelet

	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:        "1234",
			Name:      "qux",
			Namespace: "ns",
			Containers: []*kubecontainer.Container{
				{
					Name: "foo",
					ID:   kubecontainer.ContainerID{"test", containerID},
				},
			},
		},
	}

	kubelet.podManager.SetPods(pods)
	// Use the mirror pod UID to retrieve the stats.
	stats, err := kubelet.GetContainerInfo("qux_ns", "5678", "foo", cadvisorReq)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if stats == nil {
		t.Fatalf("stats should not be nil")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestDoNotCacheStatusForStaticPods(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "staticFoo",
				Namespace: "new",
				Annotations: map[string]string{
					kubetypes.ConfigSourceAnnotationKey: "file",
				},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar"},
				},
			},
		},
	}

	kubelet.podManager.SetPods(pods)
	kubelet.HandlePodSyncs(kubelet.podManager.GetPods())
	status, ok := kubelet.statusManager.GetPodStatus(pods[0].UID)
	if ok {
		t.Errorf("unexpected status %#v found for static pod %q", status, pods[0].UID)
	}
}

func TestHostNetworkAllowed(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet

	capabilities.SetForTests(capabilities.Capabilities{
		PrivilegedSources: capabilities.PrivilegedSources{
			HostNetworkSources: []string{kubetypes.ApiserverSource, kubetypes.FileSource},
		},
	})
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
			Annotations: map[string]string{
				kubetypes.ConfigSourceAnnotationKey: kubetypes.FileSource,
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "foo"},
			},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
		},
	}
	kubelet.podManager.SetPods([]*api.Pod{pod})
	err := kubelet.syncPod(pod, nil, container.Pod{}, kubetypes.SyncPodUpdate)
	if err != nil {
		t.Errorf("expected pod infra creation to succeed: %v", err)
	}
}

func TestHostNetworkDisallowed(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet

	capabilities.SetForTests(capabilities.Capabilities{
		PrivilegedSources: capabilities.PrivilegedSources{
			HostNetworkSources: []string{},
		},
	})
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
			Annotations: map[string]string{
				kubetypes.ConfigSourceAnnotationKey: kubetypes.FileSource,
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "foo"},
			},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
		},
	}
	err := kubelet.syncPod(pod, nil, container.Pod{}, kubetypes.SyncPodUpdate)
	if err == nil {
		t.Errorf("expected pod infra creation to fail")
	}
}

func TestPrivilegeContainerAllowed(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet

	capabilities.SetForTests(capabilities.Capabilities{
		AllowPrivileged: true,
	})
	privileged := true
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "foo", SecurityContext: &api.SecurityContext{Privileged: &privileged}},
			},
		},
	}
	kubelet.podManager.SetPods([]*api.Pod{pod})
	err := kubelet.syncPod(pod, nil, container.Pod{}, kubetypes.SyncPodUpdate)
	if err != nil {
		t.Errorf("expected pod infra creation to succeed: %v", err)
	}
}

func TestPrivilegeContainerDisallowed(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet

	capabilities.SetForTests(capabilities.Capabilities{
		AllowPrivileged: false,
	})
	privileged := true
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "foo", SecurityContext: &api.SecurityContext{Privileged: &privileged}},
			},
		},
	}
	err := kubelet.syncPod(pod, nil, container.Pod{}, kubetypes.SyncPodUpdate)
	if err == nil {
		t.Errorf("expected pod infra creation to fail")
	}
}

func TestFilterOutTerminatedPods(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	pods := newTestPods(5)
	pods[0].Status.Phase = api.PodFailed
	pods[1].Status.Phase = api.PodSucceeded
	pods[2].Status.Phase = api.PodRunning
	pods[3].Status.Phase = api.PodPending

	expected := []*api.Pod{pods[2], pods[3], pods[4]}
	kubelet.podManager.SetPods(pods)
	actual := kubelet.filterOutTerminatedPods(pods)
	if !reflect.DeepEqual(expected, actual) {
		t.Errorf("expected %#v, got %#v", expected, actual)
	}
}

func TestRegisterExistingNodeWithApiserver(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.AddReactor("create", "nodes", func(action testclient.Action) (bool, runtime.Object, error) {
		// Return an error on create.
		return true, &api.Node{}, &apierrors.StatusError{
			ErrStatus: unversioned.Status{Reason: unversioned.StatusReasonAlreadyExists},
		}
	})
	kubeClient.AddReactor("get", "nodes", func(action testclient.Action) (bool, runtime.Object, error) {
		// Return an existing (matching) node on get.
		return true, &api.Node{
			ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
			Spec:       api.NodeSpec{ExternalID: testKubeletHostname},
		}, nil
	})
	kubeClient.AddReactor("*", "*", func(action testclient.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("no reaction implemented for %s", action)
	})
	machineInfo := &cadvisorApi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	mockCadvisor := testKubelet.fakeCadvisor
	mockCadvisor.On("MachineInfo").Return(machineInfo, nil)
	versionInfo := &cadvisorApi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}
	mockCadvisor.On("VersionInfo").Return(versionInfo, nil)

	done := make(chan struct{})
	go func() {
		kubelet.registerWithApiserver()
		done <- struct{}{}
	}()
	select {
	case <-time.After(util.ForeverTestTimeout):
		t.Errorf("timed out waiting for registration")
	case <-done:
		return
	}
}

func TestMakePortMappings(t *testing.T) {
	tests := []struct {
		container            *api.Container
		expectedPortMappings []kubecontainer.PortMapping
	}{
		{
			&api.Container{
				Name: "fooContainer",
				Ports: []api.ContainerPort{
					{
						Protocol:      api.ProtocolTCP,
						ContainerPort: 80,
						HostPort:      8080,
						HostIP:        "127.0.0.1",
					},
					{
						Protocol:      api.ProtocolTCP,
						ContainerPort: 443,
						HostPort:      4343,
						HostIP:        "192.168.0.1",
					},
					{
						Name:          "foo",
						Protocol:      api.ProtocolUDP,
						ContainerPort: 555,
						HostPort:      5555,
					},
					{
						Name:          "foo", // Duplicated, should be ignored.
						Protocol:      api.ProtocolUDP,
						ContainerPort: 888,
						HostPort:      8888,
					},
					{
						Protocol:      api.ProtocolTCP, // Duplicated, should be ignored.
						ContainerPort: 80,
						HostPort:      8888,
					},
				},
			},
			[]kubecontainer.PortMapping{
				{
					Name:          "fooContainer-TCP:80",
					Protocol:      api.ProtocolTCP,
					ContainerPort: 80,
					HostPort:      8080,
					HostIP:        "127.0.0.1",
				},
				{
					Name:          "fooContainer-TCP:443",
					Protocol:      api.ProtocolTCP,
					ContainerPort: 443,
					HostPort:      4343,
					HostIP:        "192.168.0.1",
				},
				{
					Name:          "fooContainer-foo",
					Protocol:      api.ProtocolUDP,
					ContainerPort: 555,
					HostPort:      5555,
					HostIP:        "",
				},
			},
		},
	}

	for i, tt := range tests {
		actual := makePortMappings(tt.container)
		if !reflect.DeepEqual(tt.expectedPortMappings, actual) {
			t.Errorf("%d: Expected: %#v, saw: %#v", i, tt.expectedPortMappings, actual)
		}
	}
}

func TestIsPodPastActiveDeadline(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	pods := newTestPods(5)

	exceededActiveDeadlineSeconds := int64(30)
	notYetActiveDeadlineSeconds := int64(120)
	now := unversioned.Now()
	startTime := unversioned.NewTime(now.Time.Add(-1 * time.Minute))
	pods[0].Status.StartTime = &startTime
	pods[0].Spec.ActiveDeadlineSeconds = &exceededActiveDeadlineSeconds
	pods[1].Status.StartTime = &startTime
	pods[1].Spec.ActiveDeadlineSeconds = &notYetActiveDeadlineSeconds
	tests := []struct {
		pod      *api.Pod
		expected bool
	}{{pods[0], true}, {pods[1], false}, {pods[2], false}, {pods[3], false}, {pods[4], false}}

	kubelet.podManager.SetPods(pods)
	for i, tt := range tests {
		actual := kubelet.pastActiveDeadline(tt.pod)
		if actual != tt.expected {
			t.Errorf("[%d] expected %#v, got %#v", i, tt.expected, actual)
		}
	}
}

func TestSyncPodsSetStatusToFailedForPodsThatRunTooLong(t *testing.T) {
	testKubelet := newTestKubelet(t)
	fakeRuntime := testKubelet.fakeRuntime
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	kubelet := testKubelet.kubelet

	now := unversioned.Now()
	startTime := unversioned.NewTime(now.Time.Add(-1 * time.Minute))
	exceededActiveDeadlineSeconds := int64(30)

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "bar",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "foo"},
				},
				ActiveDeadlineSeconds: &exceededActiveDeadlineSeconds,
			},
			Status: api.PodStatus{
				StartTime: &startTime,
			},
		},
	}

	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:        "12345678",
			Name:      "bar",
			Namespace: "new",
			Containers: []*kubecontainer.Container{
				{Name: "foo"},
			},
		},
	}

	// Let the pod worker sets the status to fail after this sync.
	kubelet.HandlePodUpdates(pods)
	status, found := kubelet.statusManager.GetPodStatus(pods[0].UID)
	if !found {
		t.Errorf("expected to found status for pod %q", pods[0].UID)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q, ot %q.", api.PodFailed, status.Phase)
	}
}

func TestSyncPodsDoesNotSetPodsThatDidNotRunTooLongToFailed(t *testing.T) {
	testKubelet := newTestKubelet(t)
	fakeRuntime := testKubelet.fakeRuntime
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	kubelet := testKubelet.kubelet

	now := unversioned.Now()
	startTime := unversioned.NewTime(now.Time.Add(-1 * time.Minute))
	exceededActiveDeadlineSeconds := int64(300)

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "bar",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "foo"},
				},
				ActiveDeadlineSeconds: &exceededActiveDeadlineSeconds,
			},
			Status: api.PodStatus{
				StartTime: &startTime,
			},
		},
	}

	fakeRuntime.PodList = []*kubecontainer.Pod{
		{
			ID:        "12345678",
			Name:      "bar",
			Namespace: "new",
			Containers: []*kubecontainer.Container{
				{Name: "foo"},
			},
		},
	}

	kubelet.podManager.SetPods(pods)
	kubelet.HandlePodUpdates(pods)
	status, found := kubelet.statusManager.GetPodStatus(pods[0].UID)
	if !found {
		t.Errorf("expected to found status for pod %q", pods[0].UID)
	}
	if status.Phase == api.PodFailed {
		t.Fatalf("expected pod status to not be %q", status.Phase)
	}
}

func TestDeletePodDirsForDeletedPods(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kl := testKubelet.kubelet
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "pod1",
				Namespace: "ns",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345679",
				Name:      "pod2",
				Namespace: "ns",
			},
		},
	}

	kl.podManager.SetPods(pods)
	// Sync to create pod directories.
	kl.HandlePodSyncs(kl.podManager.GetPods())
	for i := range pods {
		if !dirExists(kl.getPodDir(pods[i].UID)) {
			t.Errorf("expected directory to exist for pod %d", i)
		}
	}

	// Pod 1 has been deleted and no longer exists.
	kl.podManager.SetPods([]*api.Pod{pods[0]})
	kl.HandlePodCleanups()
	if !dirExists(kl.getPodDir(pods[0].UID)) {
		t.Errorf("expected directory to exist for pod 0")
	}
	if dirExists(kl.getPodDir(pods[1].UID)) {
		t.Errorf("expected directory to be deleted for pod 1")
	}
}

func syncAndVerifyPodDir(t *testing.T, testKubelet *TestKubelet, pods []*api.Pod, podsToCheck []*api.Pod, shouldExist bool) {
	kl := testKubelet.kubelet

	kl.podManager.SetPods(pods)
	kl.HandlePodSyncs(pods)
	kl.HandlePodCleanups()
	for i, pod := range podsToCheck {
		exist := dirExists(kl.getPodDir(pod.UID))
		if shouldExist && !exist {
			t.Errorf("expected directory to exist for pod %d", i)
		} else if !shouldExist && exist {
			t.Errorf("expected directory to be removed for pod %d", i)
		}
	}
}

func TestDoesNotDeletePodDirsForTerminatedPods(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kl := testKubelet.kubelet
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "pod1",
				Namespace: "ns",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345679",
				Name:      "pod2",
				Namespace: "ns",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345680",
				Name:      "pod3",
				Namespace: "ns",
			},
		},
	}

	syncAndVerifyPodDir(t, testKubelet, pods, pods, true)
	// Pod 1 failed, and pod 2 succeeded. None of the pod directories should be
	// deleted.
	kl.statusManager.SetPodStatus(pods[1], api.PodStatus{Phase: api.PodFailed})
	kl.statusManager.SetPodStatus(pods[2], api.PodStatus{Phase: api.PodSucceeded})
	syncAndVerifyPodDir(t, testKubelet, pods, pods, true)
}

func TestDoesNotDeletePodDirsIfContainerIsRunning(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	runningPod := &kubecontainer.Pod{
		ID:        "12345678",
		Name:      "pod1",
		Namespace: "ns",
	}
	apiPod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       runningPod.ID,
			Name:      runningPod.Name,
			Namespace: runningPod.Namespace,
		},
	}
	// Sync once to create pod directory; confirm that the pod directory has
	// already been created.
	pods := []*api.Pod{apiPod}
	syncAndVerifyPodDir(t, testKubelet, pods, []*api.Pod{apiPod}, true)

	// Pretend the pod is deleted from apiserver, but is still active on the node.
	// The pod directory should not be removed.
	pods = []*api.Pod{}
	testKubelet.fakeRuntime.PodList = []*kubecontainer.Pod{runningPod}
	syncAndVerifyPodDir(t, testKubelet, pods, []*api.Pod{apiPod}, true)

	// The pod is deleted and also not active on the node. The pod directory
	// should be removed.
	pods = []*api.Pod{}
	testKubelet.fakeRuntime.PodList = []*kubecontainer.Pod{}
	syncAndVerifyPodDir(t, testKubelet, pods, []*api.Pod{apiPod}, false)
}

func TestCleanupBandwidthLimits(t *testing.T) {
	tests := []struct {
		status           *api.PodStatus
		pods             []*api.Pod
		inputCIDRs       []string
		expectResetCIDRs []string
		cacheStatus      bool
		expectedCalls    []string
		name             string
	}{
		{
			status: &api.PodStatus{
				PodIP: "1.2.3.4",
				Phase: api.PodRunning,
			},
			pods: []*api.Pod{
				{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
						Annotations: map[string]string{
							"kubernetes.io/ingress-bandwidth": "10M",
						},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name: "bar",
					},
				},
			},
			inputCIDRs:       []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectResetCIDRs: []string{"2.3.4.5/32", "5.6.7.8/32"},
			expectedCalls:    []string{"GetPodStatus"},
			name:             "pod running",
		},
		{
			status: &api.PodStatus{
				PodIP: "1.2.3.4",
				Phase: api.PodRunning,
			},
			pods: []*api.Pod{
				{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
						Annotations: map[string]string{
							"kubernetes.io/ingress-bandwidth": "10M",
						},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name: "bar",
					},
				},
			},
			inputCIDRs:       []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectResetCIDRs: []string{"2.3.4.5/32", "5.6.7.8/32"},
			expectedCalls:    []string{},
			cacheStatus:      true,
			name:             "pod running with cache",
		},
		{
			status: &api.PodStatus{
				PodIP: "1.2.3.4",
				Phase: api.PodFailed,
			},
			pods: []*api.Pod{
				{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
						Annotations: map[string]string{
							"kubernetes.io/ingress-bandwidth": "10M",
						},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name: "bar",
					},
				},
			},
			inputCIDRs:       []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectResetCIDRs: []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectedCalls:    []string{"GetPodStatus"},
			name:             "pod not running",
		},
		{
			status: &api.PodStatus{
				PodIP: "1.2.3.4",
				Phase: api.PodFailed,
			},
			pods: []*api.Pod{
				{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
						Annotations: map[string]string{
							"kubernetes.io/ingress-bandwidth": "10M",
						},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name: "bar",
					},
				},
			},
			inputCIDRs:       []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectResetCIDRs: []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectedCalls:    []string{},
			cacheStatus:      true,
			name:             "pod not running with cache",
		},
		{
			status: &api.PodStatus{
				PodIP: "1.2.3.4",
				Phase: api.PodRunning,
			},
			pods: []*api.Pod{
				{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
					},
				},
				{
					ObjectMeta: api.ObjectMeta{
						Name: "bar",
					},
				},
			},
			inputCIDRs:       []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			expectResetCIDRs: []string{"1.2.3.4/32", "2.3.4.5/32", "5.6.7.8/32"},
			name:             "no bandwidth limits",
		},
	}
	for _, test := range tests {
		shaper := &bandwidth.FakeShaper{
			CIDRs: test.inputCIDRs,
		}

		testKube := newTestKubelet(t)
		testKube.kubelet.shaper = shaper
		testKube.fakeRuntime.PodStatus = *test.status

		if test.cacheStatus {
			for _, pod := range test.pods {
				testKube.kubelet.statusManager.SetPodStatus(pod, *test.status)
			}
		}

		err := testKube.kubelet.cleanupBandwidthLimits(test.pods)
		if err != nil {
			t.Errorf("unexpected error: %v (%s)", test.name)
		}
		if !reflect.DeepEqual(shaper.ResetCIDRs, test.expectResetCIDRs) {
			t.Errorf("[%s]\nexpected: %v, saw: %v", test.name, test.expectResetCIDRs, shaper.ResetCIDRs)
		}

		if test.cacheStatus {
			if len(testKube.fakeRuntime.CalledFunctions) != 0 {
				t.Errorf("unexpected function calls: %v", testKube.fakeRuntime.CalledFunctions)
			}
		} else if !reflect.DeepEqual(testKube.fakeRuntime.CalledFunctions, test.expectedCalls) {
			t.Errorf("[%s], expected %v, saw %v", test.name, test.expectedCalls, testKube.fakeRuntime.CalledFunctions)
		}
	}
}

func TestExtractBandwidthResources(t *testing.T) {
	four, _ := resource.ParseQuantity("4M")
	ten, _ := resource.ParseQuantity("10M")
	twenty, _ := resource.ParseQuantity("20M")
	tests := []struct {
		pod             *api.Pod
		expectedIngress *resource.Quantity
		expectedEgress  *resource.Quantity
		expectError     bool
	}{
		{
			pod: &api.Pod{},
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubernetes.io/ingress-bandwidth": "10M",
					},
				},
			},
			expectedIngress: ten,
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubernetes.io/egress-bandwidth": "10M",
					},
				},
			},
			expectedEgress: ten,
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubernetes.io/ingress-bandwidth": "4M",
						"kubernetes.io/egress-bandwidth":  "20M",
					},
				},
			},
			expectedIngress: four,
			expectedEgress:  twenty,
		},
		{
			pod: &api.Pod{
				ObjectMeta: api.ObjectMeta{
					Annotations: map[string]string{
						"kubernetes.io/ingress-bandwidth": "foo",
					},
				},
			},
			expectError: true,
		},
	}
	for _, test := range tests {
		ingress, egress, err := extractBandwidthResources(test.pod)
		if test.expectError {
			if err == nil {
				t.Errorf("unexpected non-error")
			}
			continue
		}
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			continue
		}
		if !reflect.DeepEqual(ingress, test.expectedIngress) {
			t.Errorf("expected: %v, saw: %v", ingress, test.expectedIngress)
		}
		if !reflect.DeepEqual(egress, test.expectedEgress) {
			t.Errorf("expected: %v, saw: %v", egress, test.expectedEgress)
		}
	}
}
