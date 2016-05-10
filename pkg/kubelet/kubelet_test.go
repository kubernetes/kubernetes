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
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	cadvisorapi "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/client/testing/core"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/network"
	nettest "k8s.io/kubernetes/pkg/kubelet/network/testing"
	"k8s.io/kubernetes/pkg/kubelet/pleg"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	podtest "k8s.io/kubernetes/pkg/kubelet/pod/testing"
	proberesults "k8s.io/kubernetes/pkg/kubelet/prober/results"
	probetest "k8s.io/kubernetes/pkg/kubelet/prober/testing"
	"k8s.io/kubernetes/pkg/kubelet/status"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/queue"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/bandwidth"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/mount"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/volume"
	_ "k8s.io/kubernetes/pkg/volume/host_path"
	volumetest "k8s.io/kubernetes/pkg/volume/testing"
)

func init() {
	utilruntime.ReallyCrash = true
}

const testKubeletHostname = "127.0.0.1"

const testReservationCPU = "200m"
const testReservationMemory = "100M"

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
	fakeRuntime      *containertest.FakeRuntime
	fakeCadvisor     *cadvisortest.Mock
	fakeKubeClient   *fake.Clientset
	fakeMirrorClient *podtest.FakeMirrorClient
	fakeClock        *util.FakeClock
	mounter          mount.Interface
}

func newTestKubelet(t *testing.T) *TestKubelet {
	fakeRuntime := &containertest.FakeRuntime{}
	fakeRuntime.RuntimeType = "test"
	fakeRuntime.VersionInfo = "1.5.0"
	fakeRuntime.ImageList = []kubecontainer.Image{
		{
			ID:       "abc",
			RepoTags: []string{"gcr.io/google_containers:v1", "gcr.io/google_containers:v2"},
			Size:     123,
		},
		{
			ID:       "efg",
			RepoTags: []string{"gcr.io/google_containers:v3", "gcr.io/google_containers:v4"},
			Size:     456,
		},
	}
	fakeRecorder := &record.FakeRecorder{}
	fakeKubeClient := &fake.Clientset{}
	kubelet := &Kubelet{}
	kubelet.kubeClient = fakeKubeClient
	kubelet.os = containertest.FakeOS{}

	kubelet.hostname = testKubeletHostname
	kubelet.nodeName = testKubeletHostname
	kubelet.runtimeState = newRuntimeState(maxWaitForContainerRuntime)
	kubelet.runtimeState.setNetworkState(nil)
	kubelet.networkPlugin, _ = network.InitNetworkPlugin([]network.NetworkPlugin{}, "", nettest.NewFakeHost(nil))
	if tempDir, err := ioutil.TempDir("/tmp", "kubelet_test."); err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	} else {
		kubelet.rootDirectory = tempDir
	}
	if err := os.MkdirAll(kubelet.rootDirectory, 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %v", kubelet.rootDirectory, err)
	}
	kubelet.sourcesReady = config.NewSourcesReady(func(_ sets.String) bool { return true })
	kubelet.masterServiceNamespace = api.NamespaceDefault
	kubelet.serviceLister = testServiceLister{}
	kubelet.nodeLister = testNodeLister{}
	kubelet.nodeInfo = testNodeInfo{}
	kubelet.recorder = fakeRecorder
	if err := kubelet.setupDataDirs(); err != nil {
		t.Fatalf("can't initialize kubelet data dirs: %v", err)
	}
	kubelet.daemonEndpoints = &api.NodeDaemonEndpoints{}
	mockCadvisor := &cadvisortest.Mock{}
	kubelet.cadvisor = mockCadvisor
	fakeMirrorClient := podtest.NewFakeMirrorClient()
	kubelet.podManager = kubepod.NewBasicPodManager(fakeMirrorClient)
	kubelet.statusManager = status.NewManager(fakeKubeClient, kubelet.podManager)
	kubelet.containerRefManager = kubecontainer.NewRefManager()
	diskSpaceManager, err := newDiskSpaceManager(mockCadvisor, DiskSpacePolicy{})
	if err != nil {
		t.Fatalf("can't initialize disk space manager: %v", err)
	}
	kubelet.diskSpaceManager = diskSpaceManager

	kubelet.containerRuntime = fakeRuntime
	kubelet.runtimeCache = containertest.NewFakeRuntimeCache(kubelet.containerRuntime)
	kubelet.reasonCache = NewReasonCache()
	kubelet.podCache = containertest.NewFakeCache(kubelet.containerRuntime)
	kubelet.podWorkers = &fakePodWorkers{
		syncPodFn: kubelet.syncPod,
		cache:     kubelet.podCache,
		t:         t,
	}

	kubelet.probeManager = probetest.FakeManager{}
	kubelet.livenessManager = proberesults.NewManager()

	kubelet.volumeManager = newVolumeManager()
	kubelet.containerManager = cm.NewStubContainerManager()
	fakeNodeRef := &api.ObjectReference{
		Kind:      "Node",
		Name:      testKubeletHostname,
		UID:       types.UID(testKubeletHostname),
		Namespace: "",
	}
	fakeImageGCPolicy := ImageGCPolicy{
		HighThresholdPercent: 90,
		LowThresholdPercent:  80,
	}
	kubelet.imageManager, err = newImageManager(fakeRuntime, mockCadvisor, fakeRecorder, fakeNodeRef, fakeImageGCPolicy)
	fakeClock := util.NewFakeClock(time.Now())
	kubelet.backOff = flowcontrol.NewBackOff(time.Second, time.Minute)
	kubelet.backOff.Clock = fakeClock
	kubelet.podKillingCh = make(chan *kubecontainer.PodPair, 20)
	kubelet.resyncInterval = 10 * time.Second
	kubelet.reservation = kubetypes.Reservation{
		Kubernetes: api.ResourceList{
			api.ResourceCPU:    resource.MustParse(testReservationCPU),
			api.ResourceMemory: resource.MustParse(testReservationMemory),
		},
	}
	kubelet.workQueue = queue.NewBasicWorkQueue(fakeClock)
	// Relist period does not affect the tests.
	kubelet.pleg = pleg.NewGenericPLEG(fakeRuntime, 100, time.Hour, nil, util.RealClock{})
	kubelet.clock = fakeClock
	kubelet.setNodeStatusFuncs = kubelet.defaultNodeStatusFuncs()
	return &TestKubelet{kubelet, fakeRuntime, mockCadvisor, fakeKubeClient, fakeMirrorClient, fakeClock, nil}
}

func newTestPods(count int) []*api.Pod {
	pods := make([]*api.Pod, count)
	for i := 0; i < count; i++ {
		pods[i] = &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{
					HostNetwork: true,
				},
			},
			ObjectMeta: api.ObjectMeta{
				UID:  types.UID(10000 + i),
				Name: fmt.Sprintf("pod%d", i),
			},
		}
	}
	return pods
}

var emptyPodUIDs map[types.UID]kubetypes.SyncPodType

func TestSyncLoopTimeUpdate(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	kubelet := testKubelet.kubelet

	loopTime1 := kubelet.LatestLoopEntryTime()
	if !loopTime1.IsZero() {
		t.Errorf("Unexpected sync loop time: %s, expected 0", loopTime1)
	}

	// Start sync ticker.
	syncCh := make(chan time.Time, 1)
	housekeepingCh := make(chan time.Time, 1)
	plegCh := make(chan *pleg.PodLifecycleEvent)
	syncCh <- time.Now()
	kubelet.syncLoopIteration(make(chan kubetypes.PodUpdate), kubelet, syncCh, housekeepingCh, plegCh)
	loopTime2 := kubelet.LatestLoopEntryTime()
	if loopTime2.IsZero() {
		t.Errorf("Unexpected sync loop time: 0, expected non-zero value.")
	}

	syncCh <- time.Now()
	kubelet.syncLoopIteration(make(chan kubetypes.PodUpdate), kubelet, syncCh, housekeepingCh, plegCh)
	loopTime3 := kubelet.LatestLoopEntryTime()
	if !loopTime3.After(loopTime1) {
		t.Errorf("Sync Loop Time was not updated correctly. Second update timestamp should be greater than first update timestamp")
	}
}

func TestSyncLoopAbort(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	kubelet := testKubelet.kubelet
	kubelet.runtimeState.setRuntimeSync(time.Now())
	// The syncLoop waits on time.After(resyncInterval), set it really big so that we don't race for
	// the channel close
	kubelet.resyncInterval = time.Second * 30

	ch := make(chan kubetypes.PodUpdate)
	close(ch)

	// sanity check (also prevent this test from hanging in the next step)
	ok := kubelet.syncLoopIteration(ch, kubelet, make(chan time.Time), make(chan time.Time), make(chan *pleg.PodLifecycleEvent, 1))
	if ok {
		t.Fatalf("expected syncLoopIteration to return !ok since update chan was closed")
	}

	// this should terminate immediately; if it hangs then the syncLoopIteration isn't aborting properly
	kubelet.syncLoop(ch, kubelet)
}

func TestSyncPodsStartPod(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
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
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	kubelet.sourcesReady = config.NewSourcesReady(func(_ sets.String) bool { return ready })

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
	plug := &volumetest.FakeVolumePlugin{PluginName: "fake", Host: nil}
	kubelet.volumePluginMgr.InitPlugins([]volume.VolumePlugin{plug}, &volumeHost{kubelet})

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
	if plug.NewAttacherCallCount != 1 {
		t.Errorf("Expected plugin NewAttacher to be called %d times but got %d", 1, plug.NewAttacherCallCount)
	}

	attacher := plug.Attachers[0]
	if attacher.AttachCallCount != 1 {
		t.Errorf("Expected Attach to be called")
	}
	if attacher.WaitForAttachCallCount != 1 {
		t.Errorf("Expected WaitForAttach to be called")
	}
	if attacher.MountDeviceCallCount != 1 {
		t.Errorf("Expected MountDevice to be called")
	}
}

func TestGetPodVolumesFromDisk(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	plug := &volumetest.FakeVolumePlugin{PluginName: "fake", Host: nil}
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
		fv := volumetest.FakeVolume{PodUID: volsOnDisk[i].podUID, VolName: volsOnDisk[i].volName, Plugin: plug}
		fv.SetUp(nil)
		expectedPaths = append(expectedPaths, fv.GetPath())
	}

	volumesFound := kubelet.getPodVolumesFromDisk()
	if len(volumesFound) != len(expectedPaths) {
		t.Errorf("Expected to find %d unmounters, got %d", len(expectedPaths), len(volumesFound))
	}
	for _, ep := range expectedPaths {
		found := false
		for _, cl := range volumesFound {
			if ep == cl.Unmounter.GetPath() {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Could not find a volume with path %s", ep)
		}
	}
	if plug.NewDetacherCallCount != len(volsOnDisk) {
		t.Errorf("Expected plugin NewDetacher to be called %d times but got %d", len(volsOnDisk), plug.NewDetacherCallCount)
	}
}

// Test for https://github.com/kubernetes/kubernetes/pull/19600
func TestCleanupOrphanedVolumes(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubelet.mounter = &mount.FakeMounter{}
	kubeClient := testKubelet.fakeKubeClient
	plug := &volumetest.FakeVolumePlugin{PluginName: "fake", Host: nil}
	kubelet.volumePluginMgr.InitPlugins([]volume.VolumePlugin{plug}, &volumeHost{kubelet})

	// create a volume "on disk"
	volsOnDisk := []struct {
		podUID  types.UID
		volName string
	}{
		{"podUID", "myrealvol"},
	}

	pathsOnDisk := []string{}
	for i := range volsOnDisk {
		fv := volumetest.FakeVolume{PodUID: volsOnDisk[i].podUID, VolName: volsOnDisk[i].volName, Plugin: plug}
		fv.SetUp(nil)
		pathsOnDisk = append(pathsOnDisk, fv.GetPath())

		// Simulate the global mount so that the fakeMounter returns the
		// expected number of refs for the attached disk.
		kubelet.mounter.Mount(fv.GetPath(), fv.GetPath(), "fakefs", nil)
		kubelet.mounter.Mount(fv.GetPath(), "/path/fake/device", "fake", nil)
	}

	// store the claim in fake kubelet database
	claim := api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "myclaim",
			Namespace: "test",
		},
		Spec: api.PersistentVolumeClaimSpec{
			VolumeName: "myrealvol",
		},
		Status: api.PersistentVolumeClaimStatus{
			Phase: api.ClaimBound,
		},
	}
	kubeClient.ReactionChain = fake.NewSimpleClientset(&api.PersistentVolumeClaimList{Items: []api.PersistentVolumeClaim{
		claim,
	}}).ReactionChain

	// Create a pod referencing the volume via a PersistentVolumeClaim
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "podUID",
			Name:      "pod",
			Namespace: "test",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name: "myvolumeclaim",
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: "myclaim",
						},
					},
				},
			},
		},
	}

	// The pod is pending and not running yet. Test that cleanupOrphanedVolumes
	// won't remove the volume from disk if the volume is referenced only
	// indirectly by a claim.
	err := kubelet.cleanupOrphanedVolumes([]*api.Pod{&pod}, []*kubecontainer.Pod{})
	if err != nil {
		t.Errorf("cleanupOrphanedVolumes failed: %v", err)
	}

	if len(plug.Unmounters) != len(volsOnDisk) {
		t.Errorf("Unexpected number of unmounters created. Expected %d got %d", len(volsOnDisk), len(plug.Unmounters))
	}
	for _, unmounter := range plug.Unmounters {
		if unmounter.TearDownCallCount != 0 {
			t.Errorf("Unexpected number of calls to TearDown() %d for volume %v", unmounter.TearDownCallCount, unmounter)
		}
	}
	volumesFound := kubelet.getPodVolumesFromDisk()
	if len(volumesFound) != len(pathsOnDisk) {
		t.Errorf("Expected to find %d unmounters, got %d", len(pathsOnDisk), len(volumesFound))
	}
	for _, ep := range pathsOnDisk {
		found := false
		for _, cl := range volumesFound {
			if ep == cl.Unmounter.GetPath() {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("Could not find a volume with path %s", ep)
		}
	}

	// The pod is deleted -> kubelet should delete the volume
	err = kubelet.cleanupOrphanedVolumes([]*api.Pod{}, []*kubecontainer.Pod{})
	if err != nil {
		t.Errorf("cleanupOrphanedVolumes failed: %v", err)
	}
	volumesFound = kubelet.getPodVolumesFromDisk()
	if len(volumesFound) != 0 {
		t.Errorf("Expected to find 0 unmounters, got %d", len(volumesFound))
	}
	for _, cl := range volumesFound {
		t.Errorf("Found unexpected volume %s", cl.Unmounter.GetPath())
	}

	// Two unmounters created by the previous calls to cleanupOrphanedVolumes and getPodVolumesFromDisk
	expectedUnmounters := len(volsOnDisk) + 2
	if len(plug.Unmounters) != expectedUnmounters {
		t.Errorf("Unexpected number of unmounters created. Expected %d got  %d", expectedUnmounters, len(plug.Unmounters))
	}

	// This is the unmounter which was actually used to perform a tear down.
	unmounter := plug.Unmounters[2]

	if unmounter.TearDownCallCount != 1 {
		t.Errorf("Unexpected number of calls to TearDown() %d for volume %v", unmounter.TearDownCallCount, unmounter)
	}

	if plug.NewDetacherCallCount != expectedUnmounters {
		t.Errorf("Expected plugin NewDetacher to be called %d times but got %d", expectedUnmounters, plug.NewDetacherCallCount)
	}

	detacher := plug.Detachers[2]
	if detacher.DetachCallCount != 1 {
		t.Errorf("Expected Detach to be called")
	}
	if detacher.WaitForDetachCallCount != 1 {
		t.Errorf("Expected WaitForDetach to be called")
	}
	if detacher.UnmountDeviceCallCount != 1 {
		t.Errorf("Expected UnmountDevice to be called")
	}
}

type stubVolume struct {
	path string
	volume.MetricsNil
}

func (f *stubVolume) GetPath() string {
	return f.path
}

func (f *stubVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{}
}

func (f *stubVolume) SetUp(fsGroup *int64) error {
	return nil
}

func (f *stubVolume) SetUpAt(dir string, fsGroup *int64) error {
	return nil
}

func TestMakeVolumeMounts(t *testing.T) {
	container := api.Container{
		VolumeMounts: []api.VolumeMount{
			{
				MountPath: "/etc/hosts",
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
		"disk":  kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
		"disk4": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/host"}},
		"disk5": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/var/lib/kubelet/podID/volumes/empty/disk5"}},
	}

	pod := api.Pod{
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
		},
	}

	mounts, _ := makeMounts(&pod, "/pod", &container, "fakepodname", "", "", podVolumes)

	expectedMounts := []kubecontainer.Mount{
		{
			"disk",
			"/etc/hosts",
			"/mnt/disk",
			false,
			false,
		},
		{
			"disk",
			"/mnt/path3",
			"/mnt/disk",
			true,
			false,
		},
		{
			"disk4",
			"/mnt/path4",
			"/mnt/host",
			false,
			false,
		},
		{
			"disk5",
			"/mnt/path5",
			"/var/lib/kubelet/podID/volumes/empty/disk5",
			false,
			false,
		},
	}
	if !reflect.DeepEqual(mounts, expectedMounts) {
		t.Errorf("Unexpected mounts: Expected %#v got %#v.  Container was: %#v", expectedMounts, mounts, container)
	}
}

func TestNodeIPParam(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	tests := []struct {
		nodeIP   string
		success  bool
		testName string
	}{
		{
			nodeIP:   "",
			success:  true,
			testName: "IP not set",
		},
		{
			nodeIP:   "127.0.0.1",
			success:  false,
			testName: "loopback address",
		},
		{
			nodeIP:   "FE80::0202:B3FF:FE1E:8329",
			success:  false,
			testName: "IPv6 address",
		},
		{
			nodeIP:   "1.2.3.4",
			success:  false,
			testName: "IPv4 address that doesn't belong to host",
		},
	}
	for _, test := range tests {
		kubelet.nodeIP = net.ParseIP(test.nodeIP)
		err := kubelet.validateNodeIP()
		if err != nil && test.success {
			t.Errorf("Test: %s, expected no error but got: %v", test.testName, err)
		} else if err == nil && !test.success {
			t.Errorf("Test: %s, expected an error", test.testName)
		}
	}
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

	containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
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

type countingDNSScrubber struct {
	counter *int
}

func (cds countingDNSScrubber) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	(*cds.counter)++
	return nameservers, searches
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
		ns, srch, err := parseResolvConf(strings.NewReader(tc.data), nil)
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

		counter := 0
		cds := countingDNSScrubber{&counter}
		ns, srch, err = parseResolvConf(strings.NewReader(tc.data), cds)
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
		if counter != 1 {
			t.Errorf("[%d] expected dnsScrubber to have been called: got %d", i, counter)
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
		options[i], err = kubelet.GenerateRunContainerOptions(pod, &api.Container{}, "")
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
		options[i], err = kubelet.GenerateRunContainerOptions(pod, &api.Container{}, "")
		if err != nil {
			t.Fatalf("failed to generate container options: %v", err)
		}
	}
	t.Logf("nameservers %+v", options[1].DNS)
	if len(options[0].DNS) != 1 {
		t.Errorf("expected cluster nameserver only, got %+v", options[0].DNS)
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

type testNodeInfo struct {
	nodes []api.Node
}

func (ls testNodeInfo) GetNodeInfo(id string) (*api.Node, error) {
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
								APIVersion: testapi.Default.GroupVersion().String(),
								FieldPath:  "metadata.name",
							},
						},
					},
					{
						Name: "POD_NAMESPACE",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.GroupVersion().String(),
								FieldPath:  "metadata.namespace",
							},
						},
					},
					{
						Name: "POD_IP",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.GroupVersion().String(),
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
								APIVersion: testapi.Default.GroupVersion().String(),
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
		podIP := "1.2.3.4"

		result, err := kl.makeEnvironmentVariables(testPod, tc.container, podIP)
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

func waitingState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Waiting: &api.ContainerStateWaiting{},
		},
	}
}
func waitingStateWithLastTermination(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Waiting: &api.ContainerStateWaiting{},
		},
		LastTerminationState: api.ContainerState{
			Terminated: &api.ContainerStateTerminated{
				ExitCode: 0,
			},
		},
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
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			api.PodPending,
			"mixed state #3 with restart always",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						waitingStateWithLastTermination("containerB"),
					},
				},
			},
			api.PodRunning,
			"backoff crashloop container with restart always",
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
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			api.PodPending,
			"mixed state #3 with restart never",
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
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			api.PodPending,
			"mixed state #3 with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						waitingStateWithLastTermination("containerB"),
					},
				},
			},
			api.PodRunning,
			"backoff crashloop container with restart onfailure",
		},
	}
	for _, test := range tests {
		if status := GetPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses); status != test.status {
			t.Errorf("In test %s, expected %v, got %v", test.test, test.status, status)
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
					ID: kubecontainer.ContainerID{Type: "test", ID: "barID"}},
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
					ID: kubecontainer.ContainerID{Type: "test", ID: containerID},
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
					ID:   kubecontainer.ContainerID{Type: "test", ID: "containerFoo"},
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
		t.Errorf("expected conflict, Got no conflicts")
	}
}

// Tests that we handle port conflicts correctly by setting the failed status in status map.
func TestHandlePortConflicts(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kl := testKubelet.kubelet
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)

	kl.nodeLister = testNodeLister{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: kl.nodeName},
			Status: api.NodeStatus{
				Allocatable: api.ResourceList{
					api.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}
	kl.nodeInfo = testNodeInfo{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: kl.nodeName},
			Status: api.NodeStatus{
				Allocatable: api.ResourceList{
					api.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}

	spec := api.PodSpec{NodeName: kl.nodeName, Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 80}}}}}
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
	fittingPod := pods[1]

	kl.HandlePodAdditions(pods)
	// Check pod status stored in the status map.
	// notfittingPod should be Failed
	status, found := kl.statusManager.GetPodStatus(notfittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", notfittingPod.UID)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q. Got %q.", api.PodFailed, status.Phase)
	}
	// fittingPod should be Pending
	status, found = kl.statusManager.GetPodStatus(fittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", fittingPod.UID)
	}
	if status.Phase != api.PodPending {
		t.Fatalf("expected pod status %q. Got %q.", api.PodPending, status.Phase)
	}
}

// Tests that we handle host name conflicts correctly by setting the failed status in status map.
func TestHandleHostNameConflicts(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kl := testKubelet.kubelet
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)

	kl.nodeLister = testNodeLister{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: "127.0.0.1"},
			Status: api.NodeStatus{
				Allocatable: api.ResourceList{
					api.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}
	kl.nodeInfo = testNodeInfo{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: "127.0.0.1"},
			Status: api.NodeStatus{
				Allocatable: api.ResourceList{
					api.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "123456789",
				Name:      "notfittingpod",
				Namespace: "foo",
			},
			Spec: api.PodSpec{
				// default NodeName in test is 127.0.0.1
				NodeName: "127.0.0.2",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "987654321",
				Name:      "fittingpod",
				Namespace: "foo",
			},
			Spec: api.PodSpec{
				// default NodeName in test is 127.0.0.1
				NodeName: "127.0.0.1",
			},
		},
	}

	notfittingPod := pods[0]
	fittingPod := pods[1]

	kl.HandlePodAdditions(pods)
	// Check pod status stored in the status map.
	// notfittingPod should be Failed
	status, found := kl.statusManager.GetPodStatus(notfittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", notfittingPod.UID)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q. Got %q.", api.PodFailed, status.Phase)
	}
	// fittingPod should be Pending
	status, found = kl.statusManager.GetPodStatus(fittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", fittingPod.UID)
	}
	if status.Phase != api.PodPending {
		t.Fatalf("expected pod status %q. Got %q.", api.PodPending, status.Phase)
	}
}

// Tests that we handle not matching labels selector correctly by setting the failed status in status map.
func TestHandleNodeSelector(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kl := testKubelet.kubelet
	kl.nodeLister = testNodeLister{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: testKubeletHostname, Labels: map[string]string{"key": "B"}},
			Status: api.NodeStatus{
				Allocatable: api.ResourceList{
					api.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}
	kl.nodeInfo = testNodeInfo{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: testKubeletHostname, Labels: map[string]string{"key": "B"}},
			Status: api.NodeStatus{
				Allocatable: api.ResourceList{
					api.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
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
	fittingPod := pods[1]

	kl.HandlePodAdditions(pods)
	// Check pod status stored in the status map.
	// notfittingPod should be Failed
	status, found := kl.statusManager.GetPodStatus(notfittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", notfittingPod.UID)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q. Got %q.", api.PodFailed, status.Phase)
	}
	// fittingPod should be Pending
	status, found = kl.statusManager.GetPodStatus(fittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", fittingPod.UID)
	}
	if status.Phase != api.PodPending {
		t.Fatalf("expected pod status %q. Got %q.", api.PodPending, status.Phase)
	}
}

// Tests that we handle exceeded resources correctly by setting the failed status in status map.
func TestHandleMemExceeded(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kl := testKubelet.kubelet
	kl.nodeLister = testNodeLister{nodes: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
			Status: api.NodeStatus{Capacity: api.ResourceList{}, Allocatable: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(10, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(40, resource.DecimalSI),
			}}},
	}}
	kl.nodeInfo = testNodeInfo{nodes: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
			Status: api.NodeStatus{Capacity: api.ResourceList{}, Allocatable: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(10, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(100, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(40, resource.DecimalSI),
			}}},
	}}
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)

	spec := api.PodSpec{NodeName: kl.nodeName,
		Containers: []api.Container{{Resources: api.ResourceRequirements{
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
	fittingPod := pods[1]

	kl.HandlePodAdditions(pods)
	// Check pod status stored in the status map.
	// notfittingPod should be Failed
	status, found := kl.statusManager.GetPodStatus(notfittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", notfittingPod.UID)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q. Got %q.", api.PodFailed, status.Phase)
	}
	// fittingPod should be Pending
	status, found = kl.statusManager.GetPodStatus(fittingPod.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", fittingPod.UID)
	}
	if status.Phase != api.PodPending {
		t.Fatalf("expected pod status %q. Got %q.", api.PodPending, status.Phase)
	}
}

// TODO(filipg): This test should be removed once StatusSyncer can do garbage collection without external signal.
func TestPurgingObsoleteStatusMapEntries(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	versionInfo := &cadvisorapi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}
	testKubelet.fakeCadvisor.On("VersionInfo").Return(versionInfo, nil)

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

func TestValidateContainerLogStatus(t *testing.T) {
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
						Running: &api.ContainerStateRunning{},
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
		{
			statuses: []api.ContainerStatus{
				{
					Name:  containerName,
					State: api.ContainerState{Waiting: &api.ContainerStateWaiting{Reason: "ErrImagePull"}},
				},
			},
			success: false,
		},
		{
			statuses: []api.ContainerStatus{
				{
					Name:  containerName,
					State: api.ContainerState{Waiting: &api.ContainerStateWaiting{Reason: "ErrImagePullBackOff"}},
				},
			},
			success: false,
		},
	}

	for i, tc := range testCases {
		_, err := kubelet.validateContainerLogStatus("podName", &api.PodStatus{
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
	if _, err := kubelet.validateContainerLogStatus("podName", &api.PodStatus{
		ContainerStatuses: testCases[0].statuses,
	}, "blah", false); err == nil {
		t.Errorf("expected error with invalid container name")
	}
	if _, err := kubelet.validateContainerLogStatus("podName", &api.PodStatus{
		ContainerStatuses: testCases[0].statuses,
	}, containerName, true); err != nil {
		t.Errorf("unexpected error with for previous terminated container - %v", err)
	}
	if _, err := kubelet.validateContainerLogStatus("podName", &api.PodStatus{
		ContainerStatuses: testCases[0].statuses,
	}, containerName, false); err != nil {
		t.Errorf("unexpected error with for most recent container - %v", err)
	}
	if _, err := kubelet.validateContainerLogStatus("podName", &api.PodStatus{
		ContainerStatuses: testCases[1].statuses,
	}, containerName, true); err == nil {
		t.Errorf("expected error with for previous terminated container")
	}
	if _, err := kubelet.validateContainerLogStatus("podName", &api.PodStatus{
		ContainerStatuses: testCases[1].statuses,
	}, containerName, false); err != nil {
		t.Errorf("unexpected error with for most recent container")
	}
}

// updateDiskSpacePolicy creates a new DiskSpaceManager with a new policy. This new manager along
// with the mock FsInfo values added to Cadvisor should make the kubelet report that it has
// sufficient disk space or it is out of disk, depending on the capacity, availability and
// threshold values.
func updateDiskSpacePolicy(kubelet *Kubelet, mockCadvisor *cadvisortest.Mock, rootCap, dockerCap, rootAvail, dockerAvail uint64, rootThreshold, dockerThreshold int) error {
	dockerimagesFsInfo := cadvisorapiv2.FsInfo{Capacity: rootCap * mb, Available: rootAvail * mb}
	rootFsInfo := cadvisorapiv2.FsInfo{Capacity: dockerCap * mb, Available: dockerAvail * mb}
	mockCadvisor.On("DockerImagesFsInfo").Return(dockerimagesFsInfo, nil)
	mockCadvisor.On("RootFsInfo").Return(rootFsInfo, nil)

	dsp := DiskSpacePolicy{DockerFreeDiskMB: rootThreshold, RootFreeDiskMB: dockerThreshold}
	diskSpaceManager, err := newDiskSpaceManager(mockCadvisor, dsp)
	if err != nil {
		return err
	}
	kubelet.diskSpaceManager = diskSpaceManager
	return nil
}

func TestUpdateNewNodeStatus(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.ReactionChain = fake.NewSimpleClientset(&api.NodeList{Items: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname}},
	}}).ReactionChain
	machineInfo := &cadvisorapi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 10E9, // 10G
	}
	mockCadvisor := testKubelet.fakeCadvisor
	mockCadvisor.On("Start").Return(nil)
	mockCadvisor.On("MachineInfo").Return(machineInfo, nil)
	versionInfo := &cadvisorapi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
	}
	mockCadvisor.On("VersionInfo").Return(versionInfo, nil)

	// Make kubelet report that it has sufficient disk space.
	if err := updateDiskSpacePolicy(kubelet, mockCadvisor, 500, 500, 200, 200, 100, 100); err != nil {
		t.Fatalf("can't update disk space manager: %v", err)
	}

	expectedNode := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
		Spec:       api.NodeSpec{},
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{
					Type:               api.NodeOutOfDisk,
					Status:             api.ConditionFalse,
					Reason:             "KubeletHasSufficientDisk",
					Message:            fmt.Sprintf("kubelet has sufficient disk space available"),
					LastHeartbeatTime:  unversioned.Time{},
					LastTransitionTime: unversioned.Time{},
				},
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
				OSImage:                 "Debian GNU/Linux 7 (wheezy)",
				ContainerRuntimeVersion: "test://1.5.0",
				KubeletVersion:          version.Get().String(),
				KubeProxyVersion:        version.Get().String(),
			},
			Capacity: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(2000, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(10E9, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
			},
			Allocatable: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(1800, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(9900E6, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
			},
			Addresses: []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: "127.0.0.1"},
				{Type: api.NodeInternalIP, Address: "127.0.0.1"},
			},
			Images: []api.ContainerImage{
				{
					Names:     []string{"gcr.io/google_containers:v1", "gcr.io/google_containers:v2"},
					SizeBytes: 123,
				},
				{
					Names:     []string{"gcr.io/google_containers:v3", "gcr.io/google_containers:v4"},
					SizeBytes: 456,
				},
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
	updatedNode, ok := actions[1].(core.UpdateAction).GetObject().(*api.Node)
	if !ok {
		t.Errorf("unexpected object type")
	}
	for i, cond := range updatedNode.Status.Conditions {
		if cond.LastHeartbeatTime.IsZero() {
			t.Errorf("unexpected zero last probe timestamp for %v condition", cond.Type)
		}
		if cond.LastTransitionTime.IsZero() {
			t.Errorf("unexpected zero last transition timestamp for %v condition", cond.Type)
		}
		updatedNode.Status.Conditions[i].LastHeartbeatTime = unversioned.Time{}
		updatedNode.Status.Conditions[i].LastTransitionTime = unversioned.Time{}
	}

	// Version skew workaround. See: https://github.com/kubernetes/kubernetes/issues/16961
	if updatedNode.Status.Conditions[len(updatedNode.Status.Conditions)-1].Type != api.NodeReady {
		t.Errorf("unexpected node condition order. NodeReady should be last.")
	}

	if !api.Semantic.DeepEqual(expectedNode, updatedNode) {
		t.Errorf("unexpected objects: %s", diff.ObjectDiff(expectedNode, updatedNode))
	}
}

func TestUpdateNewNodeOutOfDiskStatusWithTransitionFrequency(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.ReactionChain = fake.NewSimpleClientset(&api.NodeList{Items: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname}},
	}}).ReactionChain
	machineInfo := &cadvisorapi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	mockCadvisor := testKubelet.fakeCadvisor
	mockCadvisor.On("Start").Return(nil)
	mockCadvisor.On("MachineInfo").Return(machineInfo, nil)
	versionInfo := &cadvisorapi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
	}
	mockCadvisor.On("VersionInfo").Return(versionInfo, nil)

	// Make Kubelet report that it has sufficient disk space.
	if err := updateDiskSpacePolicy(kubelet, mockCadvisor, 500, 500, 200, 200, 100, 100); err != nil {
		t.Fatalf("can't update disk space manager: %v", err)
	}

	kubelet.outOfDiskTransitionFrequency = 10 * time.Second

	expectedNodeOutOfDiskCondition := api.NodeCondition{
		Type:               api.NodeOutOfDisk,
		Status:             api.ConditionFalse,
		Reason:             "KubeletHasSufficientDisk",
		Message:            fmt.Sprintf("kubelet has sufficient disk space available"),
		LastHeartbeatTime:  unversioned.Time{},
		LastTransitionTime: unversioned.Time{},
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
	updatedNode, ok := actions[1].(core.UpdateAction).GetObject().(*api.Node)
	if !ok {
		t.Errorf("unexpected object type")
	}

	var oodCondition api.NodeCondition
	for i, cond := range updatedNode.Status.Conditions {
		if cond.LastHeartbeatTime.IsZero() {
			t.Errorf("unexpected zero last probe timestamp for %v condition", cond.Type)
		}
		if cond.LastTransitionTime.IsZero() {
			t.Errorf("unexpected zero last transition timestamp for %v condition", cond.Type)
		}
		updatedNode.Status.Conditions[i].LastHeartbeatTime = unversioned.Time{}
		updatedNode.Status.Conditions[i].LastTransitionTime = unversioned.Time{}
		if cond.Type == api.NodeOutOfDisk {
			oodCondition = updatedNode.Status.Conditions[i]
		}
	}

	if !reflect.DeepEqual(expectedNodeOutOfDiskCondition, oodCondition) {
		t.Errorf("unexpected objects: %s", diff.ObjectDiff(expectedNodeOutOfDiskCondition, oodCondition))
	}
}

func TestUpdateExistingNodeStatus(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.ReactionChain = fake.NewSimpleClientset(&api.NodeList{Items: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
			Spec:       api.NodeSpec{},
			Status: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:               api.NodeOutOfDisk,
						Status:             api.ConditionTrue,
						Reason:             "KubeletOutOfDisk",
						Message:            "out of disk space",
						LastHeartbeatTime:  unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
						LastTransitionTime: unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC),
					},
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
					api.ResourceMemory: *resource.NewQuantity(20E9, resource.BinarySI),
					api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
				},
				Allocatable: api.ResourceList{
					api.ResourceCPU:    *resource.NewMilliQuantity(2800, resource.DecimalSI),
					api.ResourceMemory: *resource.NewQuantity(19900E6, resource.BinarySI),
					api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
				},
			},
		},
	}}).ReactionChain
	mockCadvisor := testKubelet.fakeCadvisor
	mockCadvisor.On("Start").Return(nil)
	machineInfo := &cadvisorapi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 20E9,
	}
	mockCadvisor.On("MachineInfo").Return(machineInfo, nil)
	versionInfo := &cadvisorapi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
	}
	mockCadvisor.On("VersionInfo").Return(versionInfo, nil)

	// Make kubelet report that it is out of disk space.
	if err := updateDiskSpacePolicy(kubelet, mockCadvisor, 500, 500, 50, 50, 100, 100); err != nil {
		t.Fatalf("can't update disk space manager: %v", err)
	}

	expectedNode := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
		Spec:       api.NodeSpec{},
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{
					Type:               api.NodeOutOfDisk,
					Status:             api.ConditionTrue,
					Reason:             "KubeletOutOfDisk",
					Message:            "out of disk space",
					LastHeartbeatTime:  unversioned.Time{}, // placeholder
					LastTransitionTime: unversioned.Time{}, // placeholder
				},
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
				OSImage:                 "Debian GNU/Linux 7 (wheezy)",
				ContainerRuntimeVersion: "test://1.5.0",
				KubeletVersion:          version.Get().String(),
				KubeProxyVersion:        version.Get().String(),
			},
			Capacity: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(2000, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(20E9, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
			},
			Allocatable: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(1800, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(19900E6, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
			},
			Addresses: []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: "127.0.0.1"},
				{Type: api.NodeInternalIP, Address: "127.0.0.1"},
			},
			Images: []api.ContainerImage{
				{
					Names:     []string{"gcr.io/google_containers:v1", "gcr.io/google_containers:v2"},
					SizeBytes: 123,
				},
				{
					Names:     []string{"gcr.io/google_containers:v3", "gcr.io/google_containers:v4"},
					SizeBytes: 456,
				},
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
	updateAction, ok := actions[1].(core.UpdateAction)
	if !ok {
		t.Errorf("unexpected action type.  expected UpdateAction, got %#v", actions[1])
	}
	updatedNode, ok := updateAction.GetObject().(*api.Node)
	if !ok {
		t.Errorf("unexpected object type")
	}
	for i, cond := range updatedNode.Status.Conditions {
		// Expect LastProbeTime to be updated to Now, while LastTransitionTime to be the same.
		if old := unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC).Time; reflect.DeepEqual(cond.LastHeartbeatTime.Rfc3339Copy().UTC(), old) {
			t.Errorf("Condition %v LastProbeTime: expected \n%v\n, got \n%v", cond.Type, unversioned.Now(), old)
		}
		if got, want := cond.LastTransitionTime.Rfc3339Copy().UTC(), unversioned.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC).Time; !reflect.DeepEqual(got, want) {
			t.Errorf("Condition %v LastTransitionTime: expected \n%#v\n, got \n%#v", cond.Type, want, got)
		}
		updatedNode.Status.Conditions[i].LastHeartbeatTime = unversioned.Time{}
		updatedNode.Status.Conditions[i].LastTransitionTime = unversioned.Time{}
	}

	// Version skew workaround. See: https://github.com/kubernetes/kubernetes/issues/16961
	if updatedNode.Status.Conditions[len(updatedNode.Status.Conditions)-1].Type != api.NodeReady {
		t.Errorf("unexpected node condition order. NodeReady should be last.")
	}

	if !api.Semantic.DeepEqual(expectedNode, updatedNode) {
		t.Errorf("expected \n%v\n, got \n%v", expectedNode, updatedNode)
	}
}

func TestUpdateExistingNodeOutOfDiskStatusWithTransitionFrequency(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	clock := testKubelet.fakeClock
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.ReactionChain = fake.NewSimpleClientset(&api.NodeList{Items: []api.Node{
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
						LastHeartbeatTime:  unversioned.NewTime(clock.Now()),
						LastTransitionTime: unversioned.NewTime(clock.Now()),
					},
					{

						Type:               api.NodeOutOfDisk,
						Status:             api.ConditionTrue,
						Reason:             "KubeletOutOfDisk",
						Message:            "out of disk space",
						LastHeartbeatTime:  unversioned.NewTime(clock.Now()),
						LastTransitionTime: unversioned.NewTime(clock.Now()),
					},
				},
			},
		},
	}}).ReactionChain
	mockCadvisor := testKubelet.fakeCadvisor
	machineInfo := &cadvisorapi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	mockCadvisor.On("Start").Return(nil)
	mockCadvisor.On("MachineInfo").Return(machineInfo, nil)
	versionInfo := &cadvisorapi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}
	mockCadvisor.On("VersionInfo").Return(versionInfo, nil)

	kubelet.outOfDiskTransitionFrequency = 5 * time.Second

	ood := api.NodeCondition{
		Type:               api.NodeOutOfDisk,
		Status:             api.ConditionTrue,
		Reason:             "KubeletOutOfDisk",
		Message:            "out of disk space",
		LastHeartbeatTime:  unversioned.NewTime(clock.Now()), // placeholder
		LastTransitionTime: unversioned.NewTime(clock.Now()), // placeholder
	}
	noOod := api.NodeCondition{
		Type:               api.NodeOutOfDisk,
		Status:             api.ConditionFalse,
		Reason:             "KubeletHasSufficientDisk",
		Message:            fmt.Sprintf("kubelet has sufficient disk space available"),
		LastHeartbeatTime:  unversioned.NewTime(clock.Now()), // placeholder
		LastTransitionTime: unversioned.NewTime(clock.Now()), // placeholder
	}

	testCases := []struct {
		rootFsAvail   uint64
		dockerFsAvail uint64
		expected      api.NodeCondition
	}{
		{
			// NodeOutOfDisk==false
			rootFsAvail:   200,
			dockerFsAvail: 200,
			expected:      ood,
		},
		{
			// NodeOutOfDisk==true
			rootFsAvail:   50,
			dockerFsAvail: 200,
			expected:      ood,
		},
		{
			// NodeOutOfDisk==false
			rootFsAvail:   200,
			dockerFsAvail: 200,
			expected:      ood,
		},
		{
			// NodeOutOfDisk==true
			rootFsAvail:   200,
			dockerFsAvail: 50,
			expected:      ood,
		},
		{
			// NodeOutOfDisk==false
			rootFsAvail:   200,
			dockerFsAvail: 200,
			expected:      noOod,
		},
	}

	kubelet.updateRuntimeUp()
	for tcIdx, tc := range testCases {
		// Step by a second
		clock.Step(1 * time.Second)

		// Setup expected times.
		tc.expected.LastHeartbeatTime = unversioned.NewTime(clock.Now())
		// In the last case, there should be a status transition for NodeOutOfDisk
		if tcIdx == len(testCases)-1 {
			tc.expected.LastTransitionTime = unversioned.NewTime(clock.Now())
		}

		// Make kubelet report that it has sufficient disk space
		if err := updateDiskSpacePolicy(kubelet, mockCadvisor, 500, 500, tc.rootFsAvail, tc.dockerFsAvail, 100, 100); err != nil {
			t.Fatalf("can't update disk space manager: %v", err)
		}

		if err := kubelet.updateNodeStatus(); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		actions := kubeClient.Actions()
		if len(actions) != 2 {
			t.Errorf("%d. unexpected actions: %v", tcIdx, actions)
		}
		updateAction, ok := actions[1].(core.UpdateAction)
		if !ok {
			t.Errorf("%d. unexpected action type.  expected UpdateAction, got %#v", tcIdx, actions[1])
		}
		updatedNode, ok := updateAction.GetObject().(*api.Node)
		if !ok {
			t.Errorf("%d. unexpected object type", tcIdx)
		}
		kubeClient.ClearActions()

		var oodCondition api.NodeCondition
		for i, cond := range updatedNode.Status.Conditions {
			if cond.Type == api.NodeOutOfDisk {
				oodCondition = updatedNode.Status.Conditions[i]
			}
		}

		if !reflect.DeepEqual(tc.expected, oodCondition) {
			t.Errorf("%d.\nwant \n%v\n, got \n%v", tcIdx, tc.expected, oodCondition)
		}
	}
}

func TestUpdateNodeStatusWithRuntimeStateError(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	clock := testKubelet.fakeClock
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.ReactionChain = fake.NewSimpleClientset(&api.NodeList{Items: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname}},
	}}).ReactionChain
	mockCadvisor := testKubelet.fakeCadvisor
	mockCadvisor.On("Start").Return(nil)
	machineInfo := &cadvisorapi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 10E9,
	}
	mockCadvisor.On("MachineInfo").Return(machineInfo, nil)
	versionInfo := &cadvisorapi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
	}
	mockCadvisor.On("VersionInfo").Return(versionInfo, nil)

	// Make kubelet report that it has sufficient disk space.
	if err := updateDiskSpacePolicy(kubelet, mockCadvisor, 500, 500, 200, 200, 100, 100); err != nil {
		t.Fatalf("can't update disk space manager: %v", err)
	}

	expectedNode := &api.Node{
		ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
		Spec:       api.NodeSpec{},
		Status: api.NodeStatus{
			Conditions: []api.NodeCondition{
				{
					Type:               api.NodeOutOfDisk,
					Status:             api.ConditionFalse,
					Reason:             "KubeletHasSufficientDisk",
					Message:            "kubelet has sufficient disk space available",
					LastHeartbeatTime:  unversioned.Time{},
					LastTransitionTime: unversioned.Time{},
				},
				{}, //placeholder
			},
			NodeInfo: api.NodeSystemInfo{
				MachineID:               "123",
				SystemUUID:              "abc",
				BootID:                  "1b3",
				KernelVersion:           "3.16.0-0.bpo.4-amd64",
				OSImage:                 "Debian GNU/Linux 7 (wheezy)",
				ContainerRuntimeVersion: "test://1.5.0",
				KubeletVersion:          version.Get().String(),
				KubeProxyVersion:        version.Get().String(),
			},
			Capacity: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(2000, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(10E9, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
			},
			Allocatable: api.ResourceList{
				api.ResourceCPU:    *resource.NewMilliQuantity(1800, resource.DecimalSI),
				api.ResourceMemory: *resource.NewQuantity(9900E6, resource.BinarySI),
				api.ResourcePods:   *resource.NewQuantity(0, resource.DecimalSI),
			},
			Addresses: []api.NodeAddress{
				{Type: api.NodeLegacyHostIP, Address: "127.0.0.1"},
				{Type: api.NodeInternalIP, Address: "127.0.0.1"},
			},
			Images: []api.ContainerImage{
				{
					Names:     []string{"gcr.io/google_containers:v1", "gcr.io/google_containers:v2"},
					SizeBytes: 123,
				},
				{
					Names:     []string{"gcr.io/google_containers:v3", "gcr.io/google_containers:v4"},
					SizeBytes: 456,
				},
			},
		},
	}

	checkNodeStatus := func(status api.ConditionStatus, reason, message string) {
		kubeClient.ClearActions()
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
		updatedNode, ok := actions[1].(core.UpdateAction).GetObject().(*api.Node)
		if !ok {
			t.Errorf("unexpected action type.  expected UpdateAction, got %#v", actions[1])
		}

		for i, cond := range updatedNode.Status.Conditions {
			if cond.LastHeartbeatTime.IsZero() {
				t.Errorf("unexpected zero last probe timestamp")
			}
			if cond.LastTransitionTime.IsZero() {
				t.Errorf("unexpected zero last transition timestamp")
			}
			updatedNode.Status.Conditions[i].LastHeartbeatTime = unversioned.Time{}
			updatedNode.Status.Conditions[i].LastTransitionTime = unversioned.Time{}
		}

		// Version skew workaround. See: https://github.com/kubernetes/kubernetes/issues/16961
		if updatedNode.Status.Conditions[len(updatedNode.Status.Conditions)-1].Type != api.NodeReady {
			t.Errorf("unexpected node condition order. NodeReady should be last.")
		}
		expectedNode.Status.Conditions[1] = api.NodeCondition{
			Type:               api.NodeReady,
			Status:             status,
			Reason:             reason,
			Message:            message,
			LastHeartbeatTime:  unversioned.Time{},
			LastTransitionTime: unversioned.Time{},
		}
		if !api.Semantic.DeepEqual(expectedNode, updatedNode) {
			t.Errorf("unexpected objects: %s", diff.ObjectDiff(expectedNode, updatedNode))
		}
	}

	readyMessage := "kubelet is posting ready status"
	downMessage := "container runtime is down"

	// Should report kubelet not ready if the runtime check is out of date
	clock.SetTime(time.Now().Add(-maxWaitForContainerRuntime))
	kubelet.updateRuntimeUp()
	checkNodeStatus(api.ConditionFalse, "KubeletNotReady", downMessage)

	// Should report kubelet ready if the runtime check is updated
	clock.SetTime(time.Now())
	kubelet.updateRuntimeUp()
	checkNodeStatus(api.ConditionTrue, "KubeletReady", readyMessage)

	// Should report kubelet not ready if the runtime check is out of date
	clock.SetTime(time.Now().Add(-maxWaitForContainerRuntime))
	kubelet.updateRuntimeUp()
	checkNodeStatus(api.ConditionFalse, "KubeletNotReady", downMessage)

	// Should report kubelet not ready if the runtime check failed
	fakeRuntime := testKubelet.fakeRuntime
	// Inject error into fake runtime status check, node should be NotReady
	fakeRuntime.StatusErr = fmt.Errorf("injected runtime status error")
	clock.SetTime(time.Now())
	kubelet.updateRuntimeUp()
	checkNodeStatus(api.ConditionFalse, "KubeletNotReady", downMessage)
}

func TestUpdateNodeStatusError(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	// No matching node for the kubelet
	testKubelet.fakeKubeClient.ReactionChain = fake.NewSimpleClientset(&api.NodeList{Items: []api.Node{}}).ReactionChain

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
		err := kl.syncPod(pod, nil, &kubecontainer.PodStatus{}, updateType)
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
	testKubelet.fakeCadvisor.On("Start").Return(nil)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)

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
	err := kl.syncPod(pod, mirrorPod, &kubecontainer.PodStatus{}, kubetypes.SyncPodUpdate)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	name := kubecontainer.GetPodFullName(pod)
	creates, deletes := manager.GetCounts(name)
	if creates != 1 || deletes != 1 {
		t.Errorf("expected 1 creation and 1 deletion of %q, got %d, %d", name, creates, deletes)
	}
}

func TestDeleteOrphanedMirrorPods(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("Start").Return(nil)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)

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
	containerInfo := cadvisorapi.ContainerInfo{
		ContainerReference: cadvisorapi.ContainerReference{
			Name: containerPath,
		},
	}

	testKubelet := newTestKubelet(t)
	fakeRuntime := testKubelet.fakeRuntime
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorReq := &cadvisorapi.ContainerInfoRequest{}
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
					ID:   kubecontainer.ContainerID{Type: "test", ID: containerID},
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
	err := kubelet.syncPod(pod, nil, &kubecontainer.PodStatus{}, kubetypes.SyncPodUpdate)
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
	err := kubelet.syncPod(pod, nil, &kubecontainer.PodStatus{}, kubetypes.SyncPodUpdate)
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
	err := kubelet.syncPod(pod, nil, &kubecontainer.PodStatus{}, kubetypes.SyncPodUpdate)
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
	err := kubelet.syncPod(pod, nil, &kubecontainer.PodStatus{}, kubetypes.SyncPodUpdate)
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
	kubeClient.AddReactor("create", "nodes", func(action core.Action) (bool, runtime.Object, error) {
		// Return an error on create.
		return true, &api.Node{}, &apierrors.StatusError{
			ErrStatus: unversioned.Status{Reason: unversioned.StatusReasonAlreadyExists},
		}
	})
	kubeClient.AddReactor("get", "nodes", func(action core.Action) (bool, runtime.Object, error) {
		// Return an existing (matching) node on get.
		return true, &api.Node{
			ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
			Spec:       api.NodeSpec{ExternalID: testKubeletHostname},
		}, nil
	})
	kubeClient.AddReactor("*", "*", func(action core.Action) (bool, runtime.Object, error) {
		return true, nil, fmt.Errorf("no reaction implemented for %s", action)
	})
	machineInfo := &cadvisorapi.MachineInfo{
		MachineID:      "123",
		SystemUUID:     "abc",
		BootID:         "1b3",
		NumCores:       2,
		MemoryCapacity: 1024,
	}
	mockCadvisor := testKubelet.fakeCadvisor
	mockCadvisor.On("MachineInfo").Return(machineInfo, nil)
	versionInfo := &cadvisorapi.VersionInfo{
		KernelVersion:      "3.16.0-0.bpo.4-amd64",
		ContainerOsVersion: "Debian GNU/Linux 7 (wheezy)",
		DockerVersion:      "1.5.0",
	}
	mockCadvisor.On("VersionInfo").Return(versionInfo, nil)
	mockCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{
		Usage:     400 * mb,
		Capacity:  1000 * mb,
		Available: 600 * mb,
	}, nil)
	mockCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{
		Usage:    9 * mb,
		Capacity: 10 * mb,
	}, nil)

	done := make(chan struct{})
	go func() {
		kubelet.registerWithApiserver()
		done <- struct{}{}
	}()
	select {
	case <-time.After(wait.ForeverTestTimeout):
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
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
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
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
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
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
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
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
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
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
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
	// TODO(random-liu): We removed the test case for pod status not cached here. We should add a higher
	// layer status getter function and test that function instead.
	tests := []struct {
		status           *api.PodStatus
		pods             []*api.Pod
		inputCIDRs       []string
		expectResetCIDRs []string
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
			name:             "pod running",
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

		for _, pod := range test.pods {
			testKube.kubelet.statusManager.SetPodStatus(pod, *test.status)
		}

		err := testKube.kubelet.cleanupBandwidthLimits(test.pods)
		if err != nil {
			t.Errorf("unexpected error: %v (%s)", test.name, err)
		}
		if !reflect.DeepEqual(shaper.ResetCIDRs, test.expectResetCIDRs) {
			t.Errorf("[%s]\nexpected: %v, saw: %v", test.name, test.expectResetCIDRs, shaper.ResetCIDRs)
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
		ingress, egress, err := bandwidth.ExtractPodBandwidthResources(test.pod.Annotations)
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

func TestGetPodsToSync(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	clock := testKubelet.fakeClock
	pods := newTestPods(5)

	exceededActiveDeadlineSeconds := int64(30)
	notYetActiveDeadlineSeconds := int64(120)
	startTime := unversioned.NewTime(clock.Now())
	pods[0].Status.StartTime = &startTime
	pods[0].Spec.ActiveDeadlineSeconds = &exceededActiveDeadlineSeconds
	pods[1].Status.StartTime = &startTime
	pods[1].Spec.ActiveDeadlineSeconds = &notYetActiveDeadlineSeconds
	pods[2].Status.StartTime = &startTime
	pods[2].Spec.ActiveDeadlineSeconds = &exceededActiveDeadlineSeconds

	kubelet.podManager.SetPods(pods)
	kubelet.workQueue.Enqueue(pods[2].UID, 0)
	kubelet.workQueue.Enqueue(pods[3].UID, 30*time.Second)
	kubelet.workQueue.Enqueue(pods[4].UID, 2*time.Minute)

	clock.Step(1 * time.Minute)

	expectedPods := []*api.Pod{pods[0], pods[2], pods[3]}

	podsToSync := kubelet.getPodsToSync()

	if len(podsToSync) == len(expectedPods) {
		for _, expect := range expectedPods {
			var found bool
			for _, got := range podsToSync {
				if expect.UID == got.UID {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("expected pod not found: %+v", expect)
			}
		}
	} else {
		t.Errorf("expected %d pods to sync, got %d", len(expectedPods), len(podsToSync))
	}
}

func TestGenerateAPIPodStatusWithSortedContainers(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	numContainers := 10
	expectedOrder := []string{}
	cStatuses := []*kubecontainer.ContainerStatus{}
	specContainerList := []api.Container{}
	for i := 0; i < numContainers; i++ {
		id := fmt.Sprintf("%v", i)
		containerName := fmt.Sprintf("%vcontainer", id)
		expectedOrder = append(expectedOrder, containerName)
		cStatus := &kubecontainer.ContainerStatus{
			ID:   kubecontainer.BuildContainerID("test", id),
			Name: containerName,
		}
		// Rearrange container statuses
		if i%2 == 0 {
			cStatuses = append(cStatuses, cStatus)
		} else {
			cStatuses = append([]*kubecontainer.ContainerStatus{cStatus}, cStatuses...)
		}
		specContainerList = append(specContainerList, api.Container{Name: containerName})
	}
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       types.UID("uid1"),
			Name:      "foo",
			Namespace: "test",
		},
		Spec: api.PodSpec{
			Containers: specContainerList,
		},
	}
	status := &kubecontainer.PodStatus{
		ID:                pod.UID,
		Name:              pod.Name,
		Namespace:         pod.Namespace,
		ContainerStatuses: cStatuses,
	}
	for i := 0; i < 5; i++ {
		apiStatus := kubelet.generateAPIPodStatus(pod, status)
		for i, c := range apiStatus.ContainerStatuses {
			if expectedOrder[i] != c.Name {
				t.Fatalf("Container status not sorted, expected %v at index %d, but found %v", expectedOrder[i], i, c.Name)
			}
		}
	}
}

func TestGenerateAPIPodStatusWithReasonCache(t *testing.T) {
	// The following waiting reason and message  are generated in convertStatusToAPIStatus()
	startWaitingReason := "ContainerCreating"
	testTimestamp := time.Unix(123456789, 987654321)
	testErrorReason := fmt.Errorf("test-error")
	emptyContainerID := (&kubecontainer.ContainerID{}).String()
	verifyStatus := func(t *testing.T, c int, podStatus api.PodStatus, state, lastTerminationState map[string]api.ContainerState) {
		statuses := podStatus.ContainerStatuses
		for _, s := range statuses {
			if !reflect.DeepEqual(s.State, state[s.Name]) {
				t.Errorf("case #%d, unexpected state: %s", c, diff.ObjectDiff(state[s.Name], s.State))
			}
			if !reflect.DeepEqual(s.LastTerminationState, lastTerminationState[s.Name]) {
				t.Errorf("case #%d, unexpected last termination state %s", c, diff.ObjectDiff(
					lastTerminationState[s.Name], s.LastTerminationState))
			}
		}
	}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{RestartPolicy: api.RestartPolicyOnFailure},
	}
	podStatus := &kubecontainer.PodStatus{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}
	tests := []struct {
		containers                   []api.Container
		statuses                     []*kubecontainer.ContainerStatus
		reasons                      map[string]error
		oldStatuses                  []api.ContainerStatus
		expectedState                map[string]api.ContainerState
		expectedLastTerminationState map[string]api.ContainerState
	}{
		// For container with no historical record, State should be Waiting, LastTerminationState should be retrieved from
		// old status from apiserver.
		{
			[]api.Container{{Name: "without-old-record"}, {Name: "with-old-record"}},
			[]*kubecontainer.ContainerStatus{},
			map[string]error{},
			[]api.ContainerStatus{{
				Name:                 "with-old-record",
				LastTerminationState: api.ContainerState{Terminated: &api.ContainerStateTerminated{}},
			}},
			map[string]api.ContainerState{
				"without-old-record": {Waiting: &api.ContainerStateWaiting{
					Reason: startWaitingReason,
				}},
				"with-old-record": {Waiting: &api.ContainerStateWaiting{
					Reason: startWaitingReason,
				}},
			},
			map[string]api.ContainerState{
				"with-old-record": {Terminated: &api.ContainerStateTerminated{}},
			},
		},
		// For running container, State should be Running, LastTerminationState should be retrieved from latest terminated status.
		{
			[]api.Container{{Name: "running"}},
			[]*kubecontainer.ContainerStatus{
				{
					Name:      "running",
					State:     kubecontainer.ContainerStateRunning,
					StartedAt: testTimestamp,
				},
				{
					Name:     "running",
					State:    kubecontainer.ContainerStateExited,
					ExitCode: 1,
				},
			},
			map[string]error{},
			[]api.ContainerStatus{},
			map[string]api.ContainerState{
				"running": {Running: &api.ContainerStateRunning{
					StartedAt: unversioned.NewTime(testTimestamp),
				}},
			},
			map[string]api.ContainerState{
				"running": {Terminated: &api.ContainerStateTerminated{
					ExitCode:    1,
					ContainerID: emptyContainerID,
				}},
			},
		},
		// For terminated container:
		// * If there is no recent start error record, State should be Terminated, LastTerminationState should be retrieved from
		// second latest terminated status;
		// * If there is recent start error record, State should be Waiting, LastTerminationState should be retrieved from latest
		// terminated status;
		// * If ExitCode = 0, restart policy is RestartPolicyOnFailure, the container shouldn't be restarted. No matter there is
		// recent start error or not, State should be Terminated, LastTerminationState should be retrieved from second latest
		// terminated status.
		{
			[]api.Container{{Name: "without-reason"}, {Name: "with-reason"}},
			[]*kubecontainer.ContainerStatus{
				{
					Name:     "without-reason",
					State:    kubecontainer.ContainerStateExited,
					ExitCode: 1,
				},
				{
					Name:     "with-reason",
					State:    kubecontainer.ContainerStateExited,
					ExitCode: 2,
				},
				{
					Name:     "without-reason",
					State:    kubecontainer.ContainerStateExited,
					ExitCode: 3,
				},
				{
					Name:     "with-reason",
					State:    kubecontainer.ContainerStateExited,
					ExitCode: 4,
				},
				{
					Name:     "succeed",
					State:    kubecontainer.ContainerStateExited,
					ExitCode: 0,
				},
				{
					Name:     "succeed",
					State:    kubecontainer.ContainerStateExited,
					ExitCode: 5,
				},
			},
			map[string]error{"with-reason": testErrorReason, "succeed": testErrorReason},
			[]api.ContainerStatus{},
			map[string]api.ContainerState{
				"without-reason": {Terminated: &api.ContainerStateTerminated{
					ExitCode:    1,
					ContainerID: emptyContainerID,
				}},
				"with-reason": {Waiting: &api.ContainerStateWaiting{Reason: testErrorReason.Error()}},
				"succeed": {Terminated: &api.ContainerStateTerminated{
					ExitCode:    0,
					ContainerID: emptyContainerID,
				}},
			},
			map[string]api.ContainerState{
				"without-reason": {Terminated: &api.ContainerStateTerminated{
					ExitCode:    3,
					ContainerID: emptyContainerID,
				}},
				"with-reason": {Terminated: &api.ContainerStateTerminated{
					ExitCode:    2,
					ContainerID: emptyContainerID,
				}},
				"succeed": {Terminated: &api.ContainerStateTerminated{
					ExitCode:    5,
					ContainerID: emptyContainerID,
				}},
			},
		},
	}

	for i, test := range tests {
		kubelet.reasonCache = NewReasonCache()
		for n, e := range test.reasons {
			kubelet.reasonCache.add(pod.UID, n, e, "")
		}
		pod.Spec.Containers = test.containers
		pod.Status.ContainerStatuses = test.oldStatuses
		podStatus.ContainerStatuses = test.statuses
		apiStatus := kubelet.generateAPIPodStatus(pod, podStatus)
		verifyStatus(t, i, apiStatus, test.expectedState, test.expectedLastTerminationState)
	}
}

// testPodAdmitHandler is a lifecycle.PodAdmitHandler for testing.
type testPodAdmitHandler struct {
	// list of pods to reject.
	podsToReject []*api.Pod
}

// Admit rejects all pods in the podsToReject list with a matching UID.
func (a *testPodAdmitHandler) Admit(attrs *lifecycle.PodAdmitAttributes) lifecycle.PodAdmitResult {
	for _, podToReject := range a.podsToReject {
		if podToReject.UID == attrs.Pod.UID {
			return lifecycle.PodAdmitResult{Admit: false, Reason: "Rejected", Message: "Pod is rejected"}
		}
	}
	return lifecycle.PodAdmitResult{Admit: true}
}

// Test verifies that the kubelet invokes an admission handler during HandlePodAdditions.
func TestHandlePodAdditionsInvokesPodAdmitHandlers(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kl := testKubelet.kubelet
	kl.nodeLister = testNodeLister{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: kl.nodeName},
			Status: api.NodeStatus{
				Allocatable: api.ResourceList{
					api.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}
	kl.nodeInfo = testNodeInfo{nodes: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: kl.nodeName},
			Status: api.NodeStatus{
				Allocatable: api.ResourceList{
					api.ResourcePods: *resource.NewQuantity(110, resource.DecimalSI),
				},
			},
		},
	}}
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorapi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorapiv2.FsInfo{}, nil)

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "123456789",
				Name:      "podA",
				Namespace: "foo",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "987654321",
				Name:      "podB",
				Namespace: "foo",
			},
		},
	}
	podToReject := pods[0]
	podToAdmit := pods[1]
	podsToReject := []*api.Pod{podToReject}

	kl.AddPodAdmitHandler(&testPodAdmitHandler{podsToReject: podsToReject})

	kl.HandlePodAdditions(pods)
	// Check pod status stored in the status map.
	// podToReject should be Failed
	status, found := kl.statusManager.GetPodStatus(podToReject.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", podToReject.UID)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q. Got %q.", api.PodFailed, status.Phase)
	}
	// podToAdmit should be Pending
	status, found = kl.statusManager.GetPodStatus(podToAdmit.UID)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", podToAdmit.UID)
	}
	if status.Phase != api.PodPending {
		t.Fatalf("expected pod status %q. Got %q.", api.PodPending, status.Phase)
	}
}

// testPodSyncLoopHandler is a lifecycle.PodSyncLoopHandler that is used for testing.
type testPodSyncLoopHandler struct {
	// list of pods to sync
	podsToSync []*api.Pod
}

// ShouldSync evaluates if the pod should be synced from the kubelet.
func (a *testPodSyncLoopHandler) ShouldSync(pod *api.Pod) bool {
	for _, podToSync := range a.podsToSync {
		if podToSync.UID == pod.UID {
			return true
		}
	}
	return false
}

// TestGetPodsToSyncInvokesPodSyncLoopHandlers ensures that the get pods to sync routine invokes the handler.
func TestGetPodsToSyncInvokesPodSyncLoopHandlers(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	pods := newTestPods(5)
	podUIDs := []types.UID{}
	for _, pod := range pods {
		podUIDs = append(podUIDs, pod.UID)
	}
	podsToSync := []*api.Pod{pods[0]}
	kubelet.AddPodSyncLoopHandler(&testPodSyncLoopHandler{podsToSync})

	kubelet.podManager.SetPods(pods)

	expectedPodsUID := []types.UID{pods[0].UID}

	podsToSync = kubelet.getPodsToSync()

	if len(podsToSync) == len(expectedPodsUID) {
		var rightNum int
		for _, podUID := range expectedPodsUID {
			for _, podToSync := range podsToSync {
				if podToSync.UID == podUID {
					rightNum++
					break
				}
			}
		}
		if rightNum != len(expectedPodsUID) {
			// Just for report error
			podsToSyncUID := []types.UID{}
			for _, podToSync := range podsToSync {
				podsToSyncUID = append(podsToSyncUID, podToSync.UID)
			}
			t.Errorf("expected pods %v to sync, got %v", expectedPodsUID, podsToSyncUID)
		}

	} else {
		t.Errorf("expected %d pods to sync, got %d", 3, len(podsToSync))
	}
}

// testPodSyncHandler is a lifecycle.PodSyncHandler that is used for testing.
type testPodSyncHandler struct {
	// list of pods to evict.
	podsToEvict []*api.Pod
	// the reason for the eviction
	reason string
	// the mesage for the eviction
	message string
}

// ShouldEvict evaluates if the pod should be evicted from the kubelet.
func (a *testPodSyncHandler) ShouldEvict(pod *api.Pod) lifecycle.ShouldEvictResponse {
	for _, podToEvict := range a.podsToEvict {
		if podToEvict.UID == pod.UID {
			return lifecycle.ShouldEvictResponse{Evict: true, Reason: a.reason, Message: a.message}
		}
	}
	return lifecycle.ShouldEvictResponse{Evict: false}
}

// TestGenerateAPIPodStatusInvokesPodSyncHandlers invokes the handlers and reports the proper status
func TestGenerateAPIPodStatusInvokesPodSyncHandlers(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	pod := newTestPods(1)[0]
	podsToEvict := []*api.Pod{pod}
	kubelet.AddPodSyncHandler(&testPodSyncHandler{podsToEvict, "Evicted", "because"})
	status := &kubecontainer.PodStatus{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
	}
	apiStatus := kubelet.generateAPIPodStatus(pod, status)
	if apiStatus.Phase != api.PodFailed {
		t.Fatalf("Expected phase %v, but got %v", api.PodFailed, apiStatus.Phase)
	}
	if apiStatus.Reason != "Evicted" {
		t.Fatalf("Expected reason %v, but got %v", "Evicted", apiStatus.Reason)
	}
	if apiStatus.Message != "because" {
		t.Fatalf("Expected message %v, but got %v", "because", apiStatus.Message)
	}
}
