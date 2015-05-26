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
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/cadvisor"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/metrics"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/network"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/volume/host_path"
	"github.com/fsouza/go-dockerclient"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	cadvisorApiv2 "github.com/google/cadvisor/info/v2"
)

func init() {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
	util.ReallyCrash = true
}

type TestKubelet struct {
	kubelet          *Kubelet
	fakeDocker       *dockertools.FakeDockerClient
	fakeCadvisor     *cadvisor.Mock
	fakeKubeClient   *testclient.Fake
	fakeMirrorClient *fakeMirrorClient
}

const testKubeletHostname = "127.0.0.1"

func newTestKubelet(t *testing.T) *TestKubelet {
	fakeDocker := &dockertools.FakeDockerClient{Errors: make(map[string]error), RemovedImages: util.StringSet{}}
	fakeDocker.VersionInfo = []string{"ApiVersion=1.15"}
	fakeRecorder := &record.FakeRecorder{}
	fakeKubeClient := &testclient.Fake{}
	kubelet := &Kubelet{}
	kubelet.dockerClient = fakeDocker
	kubelet.kubeClient = fakeKubeClient
	kubelet.os = kubecontainer.FakeOS{}

	kubelet.hostname = testKubeletHostname
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
	kubelet.sourcesReady = func() bool { return true }
	kubelet.masterServiceNamespace = api.NamespaceDefault
	kubelet.serviceLister = testServiceLister{}
	kubelet.nodeLister = testNodeLister{}
	kubelet.readinessManager = kubecontainer.NewReadinessManager()
	kubelet.recorder = fakeRecorder
	kubelet.statusManager = newStatusManager(fakeKubeClient)
	if err := kubelet.setupDataDirs(); err != nil {
		t.Fatalf("can't initialize kubelet data dirs: %v", err)
	}
	mockCadvisor := &cadvisor.Mock{}
	kubelet.cadvisor = mockCadvisor
	podManager, fakeMirrorClient := newFakePodManager()
	kubelet.podManager = podManager
	kubelet.containerRefManager = kubecontainer.NewRefManager()
	runtimeHooks := newKubeletRuntimeHooks(kubelet.recorder)
	diskSpaceManager, err := newDiskSpaceManager(mockCadvisor, DiskSpacePolicy{})
	if err != nil {
		t.Fatalf("can't initialize disk space manager: %v", err)
	}
	kubelet.diskSpaceManager = diskSpaceManager

	kubelet.containerRuntime = dockertools.NewFakeDockerManager(fakeDocker, fakeRecorder, kubelet.readinessManager, kubelet.containerRefManager, dockertools.PodInfraContainerImage, 0, 0, "", kubelet.os, kubelet.networkPlugin, kubelet, &fakeHTTP{}, runtimeHooks)
	kubelet.runtimeCache = kubecontainer.NewFakeRuntimeCache(kubelet.containerRuntime)
	kubelet.podWorkers = &fakePodWorkers{
		syncPodFn:    kubelet.syncPod,
		runtimeCache: kubelet.runtimeCache,
		t:            t,
	}
	kubelet.volumeManager = newVolumeManager()
	return &TestKubelet{kubelet, fakeDocker, mockCadvisor, fakeKubeClient, fakeMirrorClient}
}

func verifyCalls(t *testing.T, fakeDocker *dockertools.FakeDockerClient, calls []string) {
	err := fakeDocker.AssertCalls(calls)
	if err != nil {
		t.Error(err)
	}
}

func verifyUnorderedCalls(t *testing.T, fakeDocker *dockertools.FakeDockerClient, calls []string) {
	err := fakeDocker.AssertUnorderedCalls(calls)
	if err != nil {
		t.Error(err)
	}
}

func verifyStringArrayEquals(t *testing.T, actual, expected []string) {
	invalid := len(actual) != len(expected)
	if !invalid {
		for ix, value := range actual {
			if expected[ix] != value {
				invalid = true
			}
		}
	}
	if invalid {
		t.Errorf("Expected: %#v, Actual: %#v", expected, actual)
	}
}

func verifyStringArrayEqualsAnyOrder(t *testing.T, actual, expected []string) {
	act := make([]string, len(actual))
	exp := make([]string, len(expected))
	copy(act, actual)
	copy(exp, expected)

	sort.StringSlice(act).Sort()
	sort.StringSlice(exp).Sort()

	if !reflect.DeepEqual(exp, act) {
		t.Errorf("Expected(sorted): %#v, Actual(sorted): %#v", exp, act)
	}
}

func verifyBoolean(t *testing.T, expected, value bool) {
	if expected != value {
		t.Errorf("Unexpected boolean.  Expected %t.  Found %t", expected, value)
	}
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

func apiContainerToContainer(c docker.APIContainers) container.Container {
	dockerName, hash, err := dockertools.ParseDockerName(c.Names[0])
	if err != nil {
		return container.Container{}
	}
	return container.Container{
		ID:   types.UID(c.ID),
		Name: dockerName.ContainerName,
		Hash: hash,
	}
}

var emptyPodUIDs map[types.UID]metrics.SyncPodType

func generatePodInfraContainerHash(pod *api.Pod) uint64 {
	var ports []api.ContainerPort
	if !pod.Spec.HostNetwork {
		for _, container := range pod.Spec.Containers {
			ports = append(ports, container.Ports...)
		}
	}

	container := &api.Container{
		Name:  dockertools.PodInfraContainerName,
		Image: dockertools.PodInfraContainerImage,
		Ports: ports,
	}
	return kubecontainer.HashContainer(container)
}

func TestSyncPodsDoesNothing(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	container := api.Container{Name: "bar"}
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					container,
				},
			},
		},
	}

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>_<random>
			Names: []string{"/k8s_bar." + strconv.FormatUint(kubecontainer.HashContainer(&container), 16) + "_foo_new_12345678_0"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			HostConfig: &docker.HostConfig{},
			Config:     &docker.Config{},
		},
		"9876": {
			ID:         "9876",
			HostConfig: &docker.HostConfig{},
			Config:     &docker.Config{},
		},
	}

	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	verifyCalls(t, fakeDocker, []string{
		"list", "list",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Check the pod infra contianer.
		"inspect_container",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Get pods for deleting orphaned volumes.
		"list",
	})
}

func TestSyncPodsWithTerminationLog(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	container := api.Container{
		Name: "bar",
		TerminationMessagePath: "/dev/somepath",
	}
	fakeDocker.ContainerList = []docker.APIContainers{}
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					container,
				},
			},
		},
	}
	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	verifyCalls(t, fakeDocker, []string{
		"list", "list",
		// Get pod status.
		"list", "inspect_image",
		// Create pod infra container.
		"create", "start", "inspect_container",
		// Create container.
		"create", "start",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Get pods for deleting orphaned volumes.
		"list",
	})

	fakeDocker.Lock()
	parts := strings.Split(fakeDocker.Container.HostConfig.Binds[0], ":")
	if !matchString(t, kubelet.getPodContainerDir("12345678", "bar")+"/k8s_bar\\.[a-f0-9]", parts[0]) {
		t.Errorf("Unexpected host path: %s", parts[0])
	}
	if parts[1] != "/dev/somepath" {
		t.Errorf("Unexpected container path: %s", parts[1])
	}
	fakeDocker.Unlock()
}

func matchString(t *testing.T, pattern, str string) bool {
	match, err := regexp.MatchString(pattern, str)
	if err != nil {
		t.Logf("unexpected error: %v", err)
	}
	return match
}

func TestSyncPodsCreatesNetAndContainer(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	// TODO: Move this test to dockertools so that we don't have to do the hacky
	// type assertion here.
	dm := kubelet.containerRuntime.(*dockertools.DockerManager)
	dm.PodInfraContainerImage = "custom_image_name"
	fakeDocker.ContainerList = []docker.APIContainers{}
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
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{
		"list", "list",
		// Get pod status.
		"list", "inspect_image",
		// Create pod infra container.
		"create", "start", "inspect_container",
		// Create container.
		"create", "start",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Get pods for deleting orphaned volumes.
		"list",
	})

	fakeDocker.Lock()

	found := false
	for _, c := range fakeDocker.ContainerList {
		if c.Image == "custom_image_name" && strings.HasPrefix(c.Names[0], "/k8s_POD") {
			found = true
		}
	}
	if !found {
		t.Errorf("Custom pod infra container not found: %v", fakeDocker.ContainerList)
	}

	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodsCreatesNetAndContainerPullsImage(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	// TODO: Move this test to dockertools so that we don't have to do the hacky
	// type assertion here.
	dm := kubelet.containerRuntime.(*dockertools.DockerManager)
	puller := dm.Puller.(*dockertools.FakeDockerPuller)
	puller.HasImages = []string{}
	dm.PodInfraContainerImage = "custom_image_name"
	fakeDocker.ContainerList = []docker.APIContainers{}
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar", Image: "something", ImagePullPolicy: "IfNotPresent"},
				},
			},
		},
	}
	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{
		"list", "list",
		// Get pod status.
		"list", "inspect_image",
		// Create pod infra container.
		"create", "start", "inspect_container",
		// Create container.
		"create", "start",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Get pods for deleting orphaned volumes.
		"list",
	})

	fakeDocker.Lock()

	if !reflect.DeepEqual(puller.ImagesPulled, []string{"custom_image_name", "something"}) {
		t.Errorf("Unexpected pulled containers: %v", puller.ImagesPulled)
	}

	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "k8s_POD\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[1]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodsWithPodInfraCreatesContainer(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
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
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}
	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{
		"list", "list",
		// Get pod status.
		"list", "inspect_container", "inspect_image",
		// Check the pod infra container.
		"inspect_container",
		// Create container.
		"create", "start",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Get pods for deleting orphaned volumes.
		"list",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 1 ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

type fakeHTTP struct {
	url string
	err error
}

func (f *fakeHTTP) Get(url string) (*http.Response, error) {
	f.url = url
	return nil, f.err
}

func TestSyncPodsWithPodInfraCreatesContainerCallsHandler(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	fakeHttp := fakeHTTP{}

	// Simulate HTTP failure. Re-create the containerRuntime to inject the failure.
	kubelet.httpClient = &fakeHttp
	runtimeHooks := newKubeletRuntimeHooks(kubelet.recorder)
	kubelet.containerRuntime = dockertools.NewFakeDockerManager(kubelet.dockerClient, kubelet.recorder, kubelet.readinessManager, kubelet.containerRefManager, dockertools.PodInfraContainerImage, 0, 0, "", kubelet.os, kubelet.networkPlugin, kubelet, kubelet.httpClient, runtimeHooks)

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: "bar",
						Lifecycle: &api.Lifecycle{
							PostStart: &api.Handler{
								HTTPGet: &api.HTTPGetAction{
									Host: "foo",
									Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
									Path: "bar",
								},
							},
						},
					},
				},
			},
		},
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}
	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{
		"list", "list",
		// Get pod status.
		"list", "inspect_container", "inspect_image",
		// Check the pod infra container.
		"inspect_container",
		// Create container.
		"create", "start",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Get pods for deleting orphaned volumes.
		"list",
	})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 1 ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo_new_", fakeDocker.Created[0]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
	if fakeHttp.url != "http://foo:8080/bar" {
		t.Errorf("Unexpected handler: %q", fakeHttp.url)
	}
}

func TestSyncPodsDeletesWithNoPodInfraContainer(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo1",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar1"},
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "87654321",
				Name:      "foo2",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar2"},
				},
			},
		},
	}
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_bar1_foo1_new_12345678_0"},
			ID:    "1234",
		},
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_bar2_foo2_new_87654321_0"},
			ID:    "5678",
		},
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_foo2_new_87654321_0"},
			ID:    "8765",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"5678": {
			ID:         "5678",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"8765": {
			ID:         "8765",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{
		"list",

		// foo1
		"list",
		// Get pod status.
		"list", "inspect_container",
		// Kill the container since pod infra container is not running.
		"inspect_container", "stop",
		// Create pod infra container.
		"create", "start", "inspect_container",
		// Create container.
		"create", "start",
		// Get pod status.
		"list", "inspect_container", "inspect_container", "inspect_container",

		// foo2
		"list",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Check the pod infra container.
		"inspect_container",
		// Get pod status.
		"list", "inspect_container", "inspect_container",

		// Get pods for deleting orphaned volumes.
		"list",
	})

	// A map iteration is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
	}
	fakeDocker.Lock()
	if len(fakeDocker.Stopped) != 1 || !expectedToStop[fakeDocker.Stopped[0]] {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
	fakeDocker.Unlock()
}

func TestSyncPodsDeletesWhenSourcesAreReady(t *testing.T) {
	ready := false
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	kubelet.sourcesReady = func() bool { return ready }

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar_new_12345678_42"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD_foo_new_12345678_42"},
			ID:    "9876",
		},
	}
	if err := kubelet.SyncPods([]*api.Pod{}, emptyPodUIDs, map[string]*api.Pod{}, time.Now()); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Validate nothing happened.
	verifyCalls(t, fakeDocker, []string{"list"})
	fakeDocker.ClearCalls()

	ready = true
	if err := kubelet.SyncPods([]*api.Pod{}, emptyPodUIDs, map[string]*api.Pod{}, time.Now()); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	verifyCalls(t, fakeDocker, []string{"list", "inspect_container", "stop", "inspect_container", "stop", "list"})

	// A map iteration is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
		"9876": true,
	}
	if len(fakeDocker.Stopped) != 2 ||
		!expectedToStop[fakeDocker.Stopped[0]] ||
		!expectedToStop[fakeDocker.Stopped[1]] {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

func TestSyncPodsDeletes(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar_new_12345678_42"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD_foo_new_12345678_42"},
			ID:    "9876",
		},
		{
			Names: []string{"foo"},
			ID:    "4567",
		},
	}
	err := kubelet.SyncPods([]*api.Pod{}, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "inspect_container", "stop", "inspect_container", "stop", "list"})

	// A map iteration is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
		"9876": true,
	}
	if len(fakeDocker.Stopped) != 2 ||
		!expectedToStop[fakeDocker.Stopped[0]] ||
		!expectedToStop[fakeDocker.Stopped[1]] {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

func TestSyncPodsDeletesDuplicate(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

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
			},
		},
	}

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar_new_12345678_1111"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_bar_new_12345678_2222"},
			ID:    "9876",
		},
		{
			// Duplicate for the same container.
			Names: []string{"/k8s_foo_bar_new_12345678_3333"},
			ID:    "4567",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"4567": {
			ID:         "4567",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{
		"list", "list",
		// Get pod status.
		"list", "inspect_container", "inspect_container", "inspect_container",
		// Check the pod infra container.
		"inspect_container",
		// Kill the duplicated container.
		"inspect_container", "stop",
		// Get pod status.
		"list", "inspect_container", "inspect_container", "inspect_container",
		// Get pods for deleting orphaned volumes.
		"list",
	})
	// Expect one of the duplicates to be killed.
	if len(fakeDocker.Stopped) != 1 || (fakeDocker.Stopped[0] != "1234" && fakeDocker.Stopped[0] != "4567") {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

func TestSyncPodsBadHash(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

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

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_bar.1234_foo_new_12345678_42"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_foo_new_12345678_42"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{
		"list", "list",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Check the pod infra container.
		"inspect_container",
		// Kill and restart the bad hash container.
		"inspect_container", "stop", "create", "start",
		// Get pod status.
		"list", "inspect_container", "inspect_container", "inspect_container",
		// Get pods for deleting orphaned volumes.
		"list",
	})

	if err := fakeDocker.AssertStopped([]string{"1234"}); err != nil {
		t.Errorf("%v", err)
	}
}

func TestSyncPodsUnhealthy(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar",
						LivenessProbe: &api.Probe{
						// Always returns healthy == false
						},
					},
				},
			},
		},
	}

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_bar_foo_new_12345678_42"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_foo_new_12345678_42"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}
	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{
		"list", "list",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Check the pod infra container.
		"inspect_container",
		// Kill the unhealthy container.
		"inspect_container", "stop",
		// Restart the unhealthy container.
		"create", "start",
		// Get pod status.
		"list", "inspect_container", "inspect_container", "inspect_container",
		// Get pods for deleting orphaned volumes.
		"list",
	})

	if err := fakeDocker.AssertStopped([]string{"1234"}); err != nil {
		t.Errorf("%v", err)
	}
}

func TestMountExternalVolumes(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubelet.volumePluginMgr.InitPlugins([]volume.VolumePlugin{&volume.FakeVolumePlugin{"fake", nil}}, &volumeHost{kubelet})

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
		t.Errorf("Expected sucess: %v", err)
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
	plug := &volume.FakeVolumePlugin{"fake", nil}
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
		fv := volume.FakeVolume{volsOnDisk[i].podUID, volsOnDisk[i].volName, plug}
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
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorReq := &cadvisorApi.ContainerInfoRequest{}
	mockCadvisor.On("DockerContainer", containerID, cadvisorReq).Return(containerInfo, nil)

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID: containerID,
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s_foo_qux_ns_1234_42"},
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
	fakeDocker := dockertools.FakeDockerClient{}

	mockCadvisor := &cadvisor.Mock{}
	cadvisorReq := &cadvisorApi.ContainerInfoRequest{}
	mockCadvisor.On("ContainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	kubelet := Kubelet{
		dockerClient: &fakeDocker,
		cadvisor:     mockCadvisor,
	}

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
	fakeDocker := dockertools.FakeDockerClient{}

	mockCadvisor := &cadvisor.Mock{}
	cadvisorReq := &cadvisorApi.ContainerInfoRequest{}
	mockCadvisor.On("SubcontainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	kubelet := Kubelet{
		dockerClient: &fakeDocker,
		cadvisor:     mockCadvisor,
	}

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
	fakeDocker := testKubelet.fakeDocker
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorApiFailure := fmt.Errorf("cAdvisor failure")
	containerInfo := cadvisorApi.ContainerInfo{}
	cadvisorReq := &cadvisorApi.ContainerInfoRequest{}
	mockCadvisor.On("DockerContainer", containerID, cadvisorReq).Return(containerInfo, cadvisorApiFailure)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID: containerID,
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s_foo_qux_ns_uuid_1234"},
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
	fakeDocker := testKubelet.fakeDocker
	mockCadvisor := testKubelet.fakeCadvisor
	fakeDocker.ContainerList = []docker.APIContainers{}

	stats, _ := kubelet.GetContainerInfo("qux", "", "foo", nil)
	if stats != nil {
		t.Errorf("non-nil stats on non exist container")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWhenDockerToolsFailed(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	fakeDocker := testKubelet.fakeDocker
	expectedErr := fmt.Errorf("List containers error")
	fakeDocker.Errors["list"] = expectedErr

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
	kubelet := testKubelet.kubelet
	mockCadvisor := testKubelet.fakeCadvisor
	fakeDocker := testKubelet.fakeDocker
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    "fakeId",
			Names: []string{"/k8s_bar_qux_ns_1234_42"},
		},
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
	ID     string
	E      error
	Stdin  io.Reader
	Stdout io.WriteCloser
	Stderr io.WriteCloser
	TTY    bool
	Port   uint16
	Stream io.ReadWriteCloser
}

func (f *fakeContainerCommandRunner) RunInContainer(id string, cmd []string) ([]byte, error) {
	f.Cmd = cmd
	f.ID = id
	return []byte{}, f.E
}

func (f *fakeContainerCommandRunner) ExecInContainer(id string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool) error {
	f.Cmd = cmd
	f.ID = id
	f.Stdin = in
	f.Stdout = out
	f.Stderr = err
	f.TTY = tty
	return f.E
}

func (f *fakeContainerCommandRunner) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	podInfraContainer := pod.FindContainerByName(dockertools.PodInfraContainerName)
	if podInfraContainer == nil {
		return fmt.Errorf("cannot find pod infra container in pod %q", kubecontainer.BuildPodFullName(pod.Name, pod.Namespace))
	}
	f.ID = string(podInfraContainer.ID)
	f.Port = port
	f.Stream = stream
	return nil
}

func TestRunInContainerNoSuchPod(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	fakeDocker.ContainerList = []docker.APIContainers{}
	kubelet.runner = &fakeCommandRunner

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
	fakeCommandRunner := fakeContainerCommandRunner{}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	kubelet.runner = &fakeCommandRunner

	containerID := "abc1234"
	podName := "podFoo"
	podNamespace := "nsFoo"
	containerName := "containerFoo"

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    containerID,
			Names: []string{"/k8s_" + containerName + "_" + podName + "_" + podNamespace + "_12345678_42"},
		},
	}

	cmd := []string{"ls"}
	_, err := kubelet.RunInContainer(
		kubecontainer.GetPodFullName(&api.Pod{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      podName,
				Namespace: podNamespace,
			},
		}),
		"",
		containerName,
		cmd)
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

func TestSyncPodEventHandlerFails(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	// Simulate HTTP failure. Re-create the containerRuntime to inject the failure.
	kubelet.httpClient = &fakeHTTP{
		err: fmt.Errorf("test error"),
	}
	runtimeHooks := newKubeletRuntimeHooks(kubelet.recorder)
	kubelet.containerRuntime = dockertools.NewFakeDockerManager(kubelet.dockerClient, kubelet.recorder, kubelet.readinessManager, kubelet.containerRefManager, dockertools.PodInfraContainerImage, 0, 0, "", kubelet.os, kubelet.networkPlugin, kubelet, kubelet.httpClient, runtimeHooks)

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar",
						Lifecycle: &api.Lifecycle{
							PostStart: &api.Handler{
								HTTPGet: &api.HTTPGetAction{
									Host: "does.no.exist",
									Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
									Path: "bar",
								},
							},
						},
					},
				},
			},
		},
	}

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_foo_new_12345678_42"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}
	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{
		"list", "list",
		// Get pod status.
		"list", "inspect_container", "inspect_image",
		// Check the pod infra container.
		"inspect_container",
		// Create the container.
		"create", "start",
		// Kill the container since event handler fails.
		"inspect_container", "stop",
		// Get pod status.
		"list", "inspect_container", "inspect_container",
		// Get pods for deleting orphaned volumes.
		"list",
	})

	// TODO(yifan): Check the stopped container's name.
	if len(fakeDocker.Stopped) != 1 {
		t.Fatalf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
	dockerName, _, err := dockertools.ParseDockerName(fakeDocker.Stopped[0])
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if dockerName.ContainerName != "bar" {
		t.Errorf("Wrong stopped container, expected: bar, get: %q", dockerName.ContainerName)
	}
}

func TestSyncPodsWithPullPolicy(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	// TODO: Move this test to dockertools so that we don't have to do the hacky
	// type assertion here.
	dm := kubelet.containerRuntime.(*dockertools.DockerManager)
	puller := dm.Puller.(*dockertools.FakeDockerPuller)
	puller.HasImages = []string{"existing_one", "want:latest"}
	dm.PodInfraContainerImage = "custom_image_name"
	fakeDocker.ContainerList = []docker.APIContainers{}

	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar", Image: "pull_always_image", ImagePullPolicy: api.PullAlways},
					{Name: "bar1", Image: "pull_never_image", ImagePullPolicy: api.PullNever},
					{Name: "bar2", Image: "pull_if_not_present_image", ImagePullPolicy: api.PullIfNotPresent},
					{Name: "bar3", Image: "existing_one", ImagePullPolicy: api.PullIfNotPresent},
					{Name: "bar4", Image: "want:latest", ImagePullPolicy: api.PullIfNotPresent},
				},
			},
		},
	}
	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	fakeDocker.Lock()

	pulledImageSet := make(map[string]empty)
	for v := range puller.ImagesPulled {
		pulledImageSet[puller.ImagesPulled[v]] = empty{}
	}

	if !reflect.DeepEqual(pulledImageSet, map[string]empty{
		"custom_image_name":         {},
		"pull_always_image":         {},
		"pull_if_not_present_image": {},
	}) {
		t.Errorf("Unexpected pulled containers: %v", puller.ImagesPulled)
	}

	if len(fakeDocker.Created) != 6 {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
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
	return nil, errors.New("not implemented")
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
				PortalIP: "1.2.3.1",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes-ro", Namespace: api.NamespaceDefault},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8082,
				}},
				PortalIP: "1.2.3.2",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes-ro", Namespace: api.NamespaceDefault},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8082,
				}},
				PortalIP: "None",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes-ro", Namespace: api.NamespaceDefault},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8082,
				}},
				PortalIP: "",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "test1"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8083,
				}},
				PortalIP: "1.2.3.3",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "test2"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8084,
				}},
				PortalIP: "1.2.3.4",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "test2"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8085,
				}},
				PortalIP: "1.2.3.5",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "test2"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8085,
				}},
				PortalIP: "None",
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
				PortalIP: "1.2.3.6",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes-ro", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8087,
				}},
				PortalIP: "1.2.3.7",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "not-special", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8088,
				}},
				PortalIP: "1.2.3.8",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "not-special", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8088,
				}},
				PortalIP: "None",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "not-special", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Ports: []api.ServicePort{{
					Protocol: "TCP",
					Port:     8088,
				}},
				PortalIP: "",
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
				{Name: "KUBERNETES_RO_SERVICE_HOST", Value: "1.2.3.2"},
				{Name: "KUBERNETES_RO_SERVICE_PORT", Value: "8082"},
				{Name: "KUBERNETES_RO_PORT", Value: "tcp://1.2.3.2:8082"},
				{Name: "KUBERNETES_RO_PORT_8082_TCP", Value: "tcp://1.2.3.2:8082"},
				{Name: "KUBERNETES_RO_PORT_8082_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_RO_PORT_8082_TCP_PORT", Value: "8082"},
				{Name: "KUBERNETES_RO_PORT_8082_TCP_ADDR", Value: "1.2.3.2"},
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
				{Name: "KUBERNETES_RO_SERVICE_HOST", Value: "1.2.3.2"},
				{Name: "KUBERNETES_RO_SERVICE_PORT", Value: "8082"},
				{Name: "KUBERNETES_RO_PORT", Value: "tcp://1.2.3.2:8082"},
				{Name: "KUBERNETES_RO_PORT_8082_TCP", Value: "tcp://1.2.3.2:8082"},
				{Name: "KUBERNETES_RO_PORT_8082_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_RO_PORT_8082_TCP_PORT", Value: "8082"},
				{Name: "KUBERNETES_RO_PORT_8082_TCP_ADDR", Value: "1.2.3.2"},
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
				{Name: "KUBERNETES_RO_SERVICE_HOST", Value: "1.2.3.7"},
				{Name: "KUBERNETES_RO_SERVICE_PORT", Value: "8087"},
				{Name: "KUBERNETES_RO_PORT", Value: "tcp://1.2.3.7:8087"},
				{Name: "KUBERNETES_RO_PORT_8087_TCP", Value: "tcp://1.2.3.7:8087"},
				{Name: "KUBERNETES_RO_PORT_8087_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_RO_PORT_8087_TCP_PORT", Value: "8087"},
				{Name: "KUBERNETES_RO_PORT_8087_TCP_ADDR", Value: "1.2.3.7"},
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
				{Name: "KUBERNETES_RO_SERVICE_HOST", Value: "1.2.3.7"},
				{Name: "KUBERNETES_RO_SERVICE_PORT", Value: "8087"},
				{Name: "KUBERNETES_RO_PORT", Value: "tcp://1.2.3.7:8087"},
				{Name: "KUBERNETES_RO_PORT_8087_TCP", Value: "tcp://1.2.3.7:8087"},
				{Name: "KUBERNETES_RO_PORT_8087_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_RO_PORT_8087_TCP_PORT", Value: "8087"},
				{Name: "KUBERNETES_RO_PORT_8087_TCP_ADDR", Value: "1.2.3.7"},
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
								APIVersion: "v1beta3",
								FieldPath:  "metadata.name",
							},
						},
					},
					{
						Name: "POD_NAMESPACE",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: "v1beta3",
								FieldPath:  "metadata.namespace",
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
			Termination: &api.ContainerStateTerminated{},
		},
	}
}
func succeededState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Termination: &api.ContainerStateTerminated{
				ExitCode: 0,
			},
		},
	}
}
func failedState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Termination: &api.ContainerStateTerminated{
				ExitCode: -1,
			},
		},
	}
}

func TestPodPhaseWithRestartAlways(t *testing.T) {
	desiredState := api.PodSpec{
		Host: "machine",
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
		if status := getPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses); status != test.status {
			t.Errorf("In test %s, expected %v, got %v", test.test, test.status, status)
		}
	}
}

func TestPodPhaseWithRestartNever(t *testing.T) {
	desiredState := api.PodSpec{
		Host: "machine",
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
		if status := getPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses); status != test.status {
			t.Errorf("In test %s, expected %v, got %v", test.test, test.status, status)
		}
	}
}

func TestPodPhaseWithRestartOnFailure(t *testing.T) {
	desiredState := api.PodSpec{
		Host: "machine",
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
		if status := getPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses); status != test.status {
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

func TestGetPodReadyCondition(t *testing.T) {
	ready := []api.PodCondition{{
		Type:   api.PodReady,
		Status: api.ConditionTrue,
	}}
	unready := []api.PodCondition{{
		Type:   api.PodReady,
		Status: api.ConditionFalse,
	}}
	tests := []struct {
		spec     *api.PodSpec
		info     []api.ContainerStatus
		expected []api.PodCondition
	}{
		{
			spec:     nil,
			info:     nil,
			expected: unready,
		},
		{
			spec:     &api.PodSpec{},
			info:     []api.ContainerStatus{},
			expected: ready,
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
				},
			},
			info:     []api.ContainerStatus{},
			expected: unready,
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
				},
			},
			info: []api.ContainerStatus{
				getReadyStatus("1234"),
			},
			expected: ready,
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			info: []api.ContainerStatus{
				getReadyStatus("1234"),
				getReadyStatus("5678"),
			},
			expected: ready,
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			info: []api.ContainerStatus{
				getReadyStatus("1234"),
			},
			expected: unready,
		},
		{
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "1234"},
					{Name: "5678"},
				},
			},
			info: []api.ContainerStatus{
				getReadyStatus("1234"),
				getNotReadyStatus("5678"),
			},
			expected: unready,
		},
	}

	for i, test := range tests {
		condition := getPodReadyCondition(test.spec, test.info)
		if !reflect.DeepEqual(condition, test.expected) {
			t.Errorf("On test case %v, expected:\n%+v\ngot\n%+v\n", i, test.expected, condition)
		}
	}
}

func TestExecInContainerNoSuchPod(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	fakeDocker.ContainerList = []docker.APIContainers{}
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerName := "containerFoo"
	err := kubelet.ExecInContainer(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{Name: podName, Namespace: podNamespace}}),
		"",
		containerName,
		[]string{"ls"},
		nil,
		nil,
		nil,
		false,
	)
	if err == nil {
		t.Fatal("unexpected non-error")
	}
	if fakeCommandRunner.ID != "" {
		t.Fatal("unexpected invocation of runner.ExecInContainer")
	}
}

func TestExecInContainerNoSuchContainer(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerID := "containerFoo"

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    "notfound",
			Names: []string{"/k8s_notfound_" + podName + "_" + podNamespace + "_12345678_42"},
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
	if fakeCommandRunner.ID != "" {
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
	fakeCommandRunner := fakeContainerCommandRunner{}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerID := "containerFoo"
	command := []string{"ls"}
	stdin := &bytes.Buffer{}
	stdout := &fakeReadWriteCloser{}
	stderr := &fakeReadWriteCloser{}
	tty := true

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    containerID,
			Names: []string{"/k8s_" + containerID + "_" + podName + "_" + podNamespace + "_12345678_42"},
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
	if e, a := containerID, fakeCommandRunner.ID; e != a {
		t.Fatalf("container id: expected %s, got %s", e, a)
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
	fakeCommandRunner := fakeContainerCommandRunner{}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	fakeDocker.ContainerList = []docker.APIContainers{}
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
	if fakeCommandRunner.ID != "" {
		t.Fatal("unexpected invocation of runner.PortForward")
	}
}

func TestPortForwardNoSuchContainer(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	var port uint16 = 5000

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    "notfound",
			Names: []string{"/k8s_notfound_" + podName + "_" + podNamespace + "_12345678_42"},
		},
	}

	err := kubelet.PortForward(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      podName,
			Namespace: podNamespace,
		}}),
		"",
		port,
		nil,
	)
	if err == nil {
		t.Fatal("unexpected non-error")
	}
	if fakeCommandRunner.ID != "" {
		t.Fatal("unexpected invocation of runner.PortForward")
	}
}

func TestPortForward(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerID := "containerFoo"
	var port uint16 = 5000
	stream := &fakeReadWriteCloser{}

	podInfraContainerImage := "POD"
	infraContainerID := "infra"
	// TODO: Move this test to dockertools so that we don't have to do the hacky
	// type assertion here.
	dm := kubelet.containerRuntime.(*dockertools.DockerManager)
	dm.PodInfraContainerImage = podInfraContainerImage

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    infraContainerID,
			Names: []string{"/k8s_" + podInfraContainerImage + "_" + podName + "_" + podNamespace + "_12345678_42"},
		},
		{
			ID:    containerID,
			Names: []string{"/k8s_" + containerID + "_" + podName + "_" + podNamespace + "_12345678_42"},
		},
	}

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
	if e, a := infraContainerID, fakeCommandRunner.ID; e != a {
		t.Fatalf("container id: expected %s, got %s", e, a)
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
	_, conflicts := checkHostPortConflicts(pods)
	if len(conflicts) != 0 {
		t.Errorf("expected no conflicts, Got %#v", conflicts)
	}

	// The new pod should cause conflict and be reported.
	expected := &api.Pod{
		Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 81}}}}},
	}
	pods = append(pods, expected)
	if _, actual := checkHostPortConflicts(pods); !reflect.DeepEqual(actual, []*api.Pod{expected}) {
		t.Errorf("expected %#v, Got %#v", expected, actual)
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
	pods[1].CreationTimestamp = util.NewTime(time.Now())
	pods[0].CreationTimestamp = util.NewTime(time.Now().Add(1 * time.Second))
	// The newer pod should be rejected.
	conflictedPodName := kubecontainer.GetPodFullName(pods[0])

	kl.handleNotFittingPods(pods)
	// Check pod status stored in the status map.
	status, found := kl.statusManager.GetPodStatus(conflictedPodName)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", conflictedPodName)
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
	notfittingPodName := kubecontainer.GetPodFullName(pods[0])

	kl.handleNotFittingPods(pods)
	// Check pod status stored in the status map.
	status, found := kl.statusManager.GetPodStatus(notfittingPodName)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", notfittingPodName)
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
		Limits: api.ResourceList{
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
	pods[1].CreationTimestamp = util.NewTime(time.Now())
	pods[0].CreationTimestamp = util.NewTime(time.Now().Add(1 * time.Second))
	// The newer pod should be rejected.
	notfittingPodName := kubecontainer.GetPodFullName(pods[0])

	kl.handleNotFittingPods(pods)
	// Check pod status stored in the status map.
	status, found := kl.statusManager.GetPodStatus(notfittingPodName)
	if !found {
		t.Fatalf("status of pod %q is not found in the status map", notfittingPodName)
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
		{ObjectMeta: api.ObjectMeta{Name: "pod1"}, Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 80}}}}}},
		{ObjectMeta: api.ObjectMeta{Name: "pod2"}, Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 80}}}}}},
	}
	// Run once to populate the status map.
	kl.handleNotFittingPods(pods)
	if _, found := kl.statusManager.GetPodStatus(kubecontainer.BuildPodFullName("pod2", "")); !found {
		t.Fatalf("expected to have status cached for pod2")
	}
	// Sync with empty pods so that the entry in status map will be removed.
	kl.SyncPods([]*api.Pod{}, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if _, found := kl.statusManager.GetPodStatus(kubecontainer.BuildPodFullName("pod2", "")); found {
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
						Termination: &api.ContainerStateTerminated{},
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
						Termination: &api.ContainerStateTerminated{},
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
	kubeClient.ReactFn = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname}},
	}}).ReactFn
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
					Reason:             fmt.Sprintf("kubelet is posting ready status"),
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
			Addresses: []api.NodeAddress{{Type: api.NodeLegacyHostIP, Address: "127.0.0.1"}},
		},
	}

	kubelet.updateRuntimeUp()
	if err := kubelet.updateNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(kubeClient.Actions) != 2 || kubeClient.Actions[1].Action != "update-status-node" {
		t.Fatalf("unexpected actions: %v", kubeClient.Actions)
	}
	updatedNode, ok := kubeClient.Actions[1].Value.(*api.Node)
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
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	kubeClient.ReactFn = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{
		{
			ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
			Spec:       api.NodeSpec{},
			Status: api.NodeStatus{
				Conditions: []api.NodeCondition{
					{
						Type:               api.NodeReady,
						Status:             api.ConditionTrue,
						Reason:             fmt.Sprintf("kubelet is posting ready status"),
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
	}}).ReactFn
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
					Reason:             fmt.Sprintf("kubelet is posting ready status"),
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
			Addresses: []api.NodeAddress{{Type: api.NodeLegacyHostIP, Address: "127.0.0.1"}},
		},
	}

	kubelet.updateRuntimeUp()
	if err := kubelet.updateNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(kubeClient.Actions) != 2 {
		t.Errorf("unexpected actions: %v", kubeClient.Actions)
	}
	updatedNode, ok := kubeClient.Actions[1].Value.(*api.Node)
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
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	kubeClient := testKubelet.fakeKubeClient
	fakeDocker := testKubelet.fakeDocker
	// This causes returning an error from GetContainerRuntimeVersion() which
	// simulates that container runtime is down.
	fakeDocker.VersionInfo = []string{}

	kubeClient.ReactFn = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{
		{ObjectMeta: api.ObjectMeta{Name: testKubeletHostname}},
	}}).ReactFn
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
					Reason:             fmt.Sprintf("container runtime is down"),
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
			Addresses: []api.NodeAddress{{Type: api.NodeLegacyHostIP, Address: "127.0.0.1"}},
		},
	}

	kubelet.runtimeUpThreshold = time.Duration(0)
	kubelet.updateRuntimeUp()
	if err := kubelet.updateNodeStatus(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if len(kubeClient.Actions) != 2 || kubeClient.Actions[1].Action != "update-status-node" {
		t.Fatalf("unexpected actions: %v", kubeClient.Actions)
	}
	updatedNode, ok := kubeClient.Actions[1].Value.(*api.Node)
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

func TestUpdateNodeStatusError(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet
	// No matching node for the kubelet
	testKubelet.fakeKubeClient.ReactFn = testclient.NewSimpleFake(&api.NodeList{Items: []api.Node{}}).ReactFn

	if err := kubelet.updateNodeStatus(); err == nil {
		t.Errorf("unexpected non error: %v", err)
	}
	if len(testKubelet.fakeKubeClient.Actions) != nodeStatusUpdateRetry {
		t.Errorf("unexpected actions: %v", testKubelet.fakeKubeClient.Actions)
	}
}

func TestCreateMirrorPod(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kl := testKubelet.kubelet
	manager := testKubelet.fakeMirrorClient
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "foo",
			Annotations: map[string]string{
				ConfigSourceAnnotationKey: "file",
			},
		},
	}
	pods := []*api.Pod{pod}
	kl.podManager.SetPods(pods)
	err := kl.syncPod(pod, nil, container.Pod{})
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
				ConfigSourceAnnotationKey: "file",
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
				ConfigSourceAnnotationKey: "api",
				ConfigMirrorAnnotationKey: "mirror",
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
	err := kl.syncPod(pod, mirrorPod, container.Pod{})
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
					ConfigSourceAnnotationKey: "api",
					ConfigMirrorAnnotationKey: "mirror",
				},
			},
		},
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345679",
				Name:      "pod2",
				Namespace: "ns",
				Annotations: map[string]string{
					ConfigSourceAnnotationKey: "api",
					ConfigMirrorAnnotationKey: "mirror",
				},
			},
		},
	}

	kl.podManager.SetPods(orphanPods)
	pods, mirrorMap := kl.podManager.GetPodsAndMirrorMap()
	// Sync with an empty pod list to delete all mirror pods.
	err := kl.SyncPods(pods, emptyPodUIDs, mirrorMap, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
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
					ConfigSourceAnnotationKey: "file",
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
					ConfigSourceAnnotationKey: "api",
					ConfigMirrorAnnotationKey: "mirror",
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
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker
	mockCadvisor := testKubelet.fakeCadvisor
	cadvisorReq := &cadvisorApi.ContainerInfoRequest{}
	mockCadvisor.On("DockerContainer", containerID, cadvisorReq).Return(containerInfo, nil)

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    containerID,
			Names: []string{"/k8s_foo_qux_ns_1234_42"},
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
				Name:      "foo",
				Namespace: "new",
				Annotations: map[string]string{
					ConfigSourceAnnotationKey: "file",
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
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	podFullName := kubecontainer.GetPodFullName(pods[0])
	status, ok := kubelet.statusManager.GetPodStatus(podFullName)
	if ok {
		t.Errorf("unexpected status %#v found for static pod %q", status, podFullName)
	}
}

func TestHostNetworkAllowed(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet

	capabilities.SetForTests(capabilities.Capabilities{
		HostNetworkSources: []string{ApiserverSource, FileSource},
	})
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
			Annotations: map[string]string{
				ConfigSourceAnnotationKey: FileSource,
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "foo"},
			},
			HostNetwork: true,
		},
	}
	kubelet.podManager.SetPods([]*api.Pod{pod})
	err := kubelet.syncPod(pod, nil, container.Pod{})
	if err != nil {
		t.Errorf("expected pod infra creation to succeed: %v", err)
	}
}

func TestHostNetworkDisallowed(t *testing.T) {
	testKubelet := newTestKubelet(t)
	kubelet := testKubelet.kubelet

	capabilities.SetForTests(capabilities.Capabilities{
		HostNetworkSources: []string{},
	})
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
			Annotations: map[string]string{
				ConfigSourceAnnotationKey: FileSource,
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "foo"},
			},
			HostNetwork: true,
		},
	}
	err := kubelet.syncPod(pod, nil, container.Pod{})
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
	err := kubelet.syncPod(pod, nil, container.Pod{})
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
	err := kubelet.syncPod(pod, nil, container.Pod{})
	if err == nil {
		t.Errorf("expected pod infra creation to fail")
	}
}

func TestSyncPodsWithRestartPolicy(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	containers := []api.Container{
		{Name: "succeeded"},
		{Name: "failed"},
	}
	pods := []*api.Pod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:       "12345678",
				Name:      "foo",
				Namespace: "new",
			},
			Spec: api.PodSpec{
				Containers: containers,
			},
		},
	}

	runningAPIContainers := []docker.APIContainers{
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	exitedAPIContainers := []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_succeeded." + strconv.FormatUint(kubecontainer.HashContainer(&containers[0]), 16) + "_foo_new_12345678_0"},
			ID:    "1234",
		},
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_failed." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_foo_new_12345678_0"},
			ID:    "5678",
		},
	}

	containerMap := map[string]*docker.Container{
		"9876": {
			ID:     "9876",
			Name:   "POD",
			Config: &docker.Config{},
			State: docker.State{
				StartedAt: time.Now(),
				Running:   true,
			},
		},
		"1234": {
			ID:     "1234",
			Name:   "succeeded",
			Config: &docker.Config{},
			State: docker.State{
				ExitCode:   0,
				StartedAt:  time.Now(),
				FinishedAt: time.Now(),
			},
		},
		"5678": {
			ID:     "5678",
			Name:   "failed",
			Config: &docker.Config{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  time.Now(),
				FinishedAt: time.Now(),
			},
		},
	}

	tests := []struct {
		policy  api.RestartPolicy
		calls   []string
		created []string
		stopped []string
	}{
		{
			api.RestartPolicyAlways,
			[]string{"list", "list",
				// Get pod status.
				"list", "inspect_container", "inspect_container", "inspect_container",
				// Check the pod infra container.
				"inspect_container",
				// Restart both containers.
				"create", "start", "create", "start",
				// Get pod status.
				"list", "inspect_container", "inspect_container", "inspect_container", "inspect_container", "inspect_container",
				// Get pods for deleting orphaned volumes.
				"list",
			},
			[]string{"succeeded", "failed"},
			[]string{},
		},
		{
			api.RestartPolicyOnFailure,
			[]string{"list", "list",
				// Get pod status.
				"list", "inspect_container", "inspect_container", "inspect_container",
				// Check the pod infra container.
				"inspect_container",
				// Restart the failed container.
				"create", "start",
				// Get pod status.
				"list", "inspect_container", "inspect_container", "inspect_container", "inspect_container",
				// Get pods for deleting orphaned volumes.
				"list",
			},
			[]string{"failed"},
			[]string{},
		},
		{
			api.RestartPolicyNever,
			[]string{"list", "list",
				// Get pod status.
				"list", "inspect_container", "inspect_container", "inspect_container",
				// Check the pod infra container.
				"inspect_container",
				// Stop the last pod infra container.
				"inspect_container", "stop",
				// Get pod status.
				"list", "inspect_container", "inspect_container", "inspect_container",
				// Get pods for deleting orphaned volumes.
				"list",
			},
			[]string{},
			[]string{"9876"},
		},
	}

	for i, tt := range tests {
		fakeDocker.ContainerList = runningAPIContainers
		fakeDocker.ExitedContainerList = exitedAPIContainers
		fakeDocker.ContainerMap = containerMap
		fakeDocker.ClearCalls()
		pods[0].Spec.RestartPolicy = tt.policy

		kubelet.podManager.SetPods(pods)
		err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		}

		// 'stop' is because the pod infra container is killed when no container is running.
		verifyCalls(t, fakeDocker, tt.calls)

		if err := fakeDocker.AssertCreated(tt.created); err != nil {
			t.Errorf("%d: %v", i, err)
		}
		if err := fakeDocker.AssertStopped(tt.stopped); err != nil {
			t.Errorf("%d: %v", i, err)
		}
	}
}

func TestGetPodStatusWithLastTermination(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	containers := []api.Container{
		{Name: "succeeded"},
		{Name: "failed"},
	}

	exitedAPIContainers := []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_succeeded." + strconv.FormatUint(kubecontainer.HashContainer(&containers[0]), 16) + "_foo_new_12345678_0"},
			ID:    "1234",
		},
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_failed." + strconv.FormatUint(kubecontainer.HashContainer(&containers[1]), 16) + "_foo_new_12345678_0"},
			ID:    "5678",
		},
	}

	containerMap := map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			Name:       "POD",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
			State: docker.State{
				StartedAt:  time.Now(),
				FinishedAt: time.Now(),
				Running:    true,
			},
		},
		"1234": {
			ID:         "1234",
			Name:       "succeeded",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
			State: docker.State{
				ExitCode:   0,
				StartedAt:  time.Now(),
				FinishedAt: time.Now(),
			},
		},
		"5678": {
			ID:         "5678",
			Name:       "failed",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  time.Now(),
				FinishedAt: time.Now(),
			},
		},
	}

	tests := []struct {
		policy           api.RestartPolicy
		created          []string
		stopped          []string
		lastTerminations []string
	}{
		{
			api.RestartPolicyAlways,
			[]string{"succeeded", "failed"},
			[]string{},
			[]string{"docker://1234", "docker://5678"},
		},
		{
			api.RestartPolicyOnFailure,
			[]string{"failed"},
			[]string{},
			[]string{"docker://5678"},
		},
		{
			api.RestartPolicyNever,
			[]string{},
			[]string{"9876"},
			[]string{},
		},
	}

	for i, tt := range tests {
		fakeDocker.ExitedContainerList = exitedAPIContainers
		fakeDocker.ContainerMap = containerMap
		fakeDocker.ClearCalls()
		pods := []*api.Pod{
			{
				ObjectMeta: api.ObjectMeta{
					UID:       "12345678",
					Name:      "foo",
					Namespace: "new",
				},
				Spec: api.PodSpec{
					Containers:    containers,
					RestartPolicy: tt.policy,
				},
			},
		}
		fakeDocker.ContainerList = []docker.APIContainers{
			{
				// pod infra container
				Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_foo_new_12345678_0"},
				ID:    "9876",
			},
		}
		kubelet.podManager.SetPods(pods)
		err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
		if err != nil {
			t.Errorf("%d: unexpected error: %v", i, err)
		}

		// Check if we can retrieve the pod status.
		podName := kubecontainer.GetPodFullName(pods[0])
		status, found := kubelet.statusManager.GetPodStatus(podName)
		if !found {
			t.Fatalf("unable to retrieve pod status for pod %q.", podName)
		} else {
			terminatedContainers := []string{}
			for _, cs := range status.ContainerStatuses {
				if cs.LastTerminationState.Termination != nil {
					terminatedContainers = append(terminatedContainers, cs.LastTerminationState.Termination.ContainerID)
				}
			}
			sort.StringSlice(terminatedContainers).Sort()
			sort.StringSlice(tt.lastTerminations).Sort()
			if !reflect.DeepEqual(terminatedContainers, tt.lastTerminations) {
				t.Errorf("Expected(sorted): %#v, Actual(sorted): %#v", tt.lastTerminations, terminatedContainers)
			}
		}

		if err := fakeDocker.AssertCreated(tt.created); err != nil {
			t.Errorf("%d: %v", i, err)
		}
		if err := fakeDocker.AssertStopped(tt.stopped); err != nil {
			t.Errorf("%d: %v", i, err)
		}
	}
}

func TestGetPodCreationFailureReason(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	// Inject the creation failure error to docker.
	failureReason := "creation failure"
	fakeDocker.Errors = map[string]error{
		"create": fmt.Errorf("%s", failureReason),
	}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{Name: "bar"}},
		},
	}

	// Pretend that the pod infra container has already been created, so that
	// we can run the user containers.
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			HostConfig: &docker.HostConfig{},
			Config:     &docker.Config{},
		},
	}

	pods := []*api.Pod{pod}
	kubelet.podManager.SetPods(pods)
	kubelet.volumeManager.SetVolumes(pod.UID, kubecontainer.VolumeMap{})
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	status, found := kubelet.statusManager.GetPodStatus(kubecontainer.GetPodFullName(pod))
	if !found {
		t.Fatalf("unexpected error %v", err)
	}
	if len(status.ContainerStatuses) < 1 {
		t.Errorf("expected 1 container status, got %d", len(status.ContainerStatuses))
	} else {
		state := status.ContainerStatuses[0].State
		if state.Waiting == nil {
			t.Errorf("expected waiting state, got %#v", state)
		} else if state.Waiting.Reason != failureReason {
			t.Errorf("expected reason %q, got %q", failureReason, state.Waiting.Reason)
		}
	}
}

func TestGetPodPullImageFailureReason(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	// Initialize the FakeDockerPuller so that it'd try to pull non-existent
	// images.
	dm := kubelet.containerRuntime.(*dockertools.DockerManager)
	puller := dm.Puller.(*dockertools.FakeDockerPuller)
	puller.HasImages = []string{}
	// Inject the pull image failure error.
	failureReason := "pull image faiulre"
	puller.ErrorsToInject = []error{fmt.Errorf("%s", failureReason)}

	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{{Name: "bar", Image: "realImage", ImagePullPolicy: api.PullAlways}},
		},
	}

	// Pretend that the pod infra container has already been created, so that
	// we can run the user containers.
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pod), 16) + "_foo_new_12345678_0"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"9876": {
			ID:         "9876",
			HostConfig: &docker.HostConfig{},
			Config:     &docker.Config{},
		},
	}

	pods := []*api.Pod{pod}
	kubelet.podManager.SetPods(pods)
	kubelet.volumeManager.SetVolumes(pod.UID, kubecontainer.VolumeMap{})
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	status, found := kubelet.statusManager.GetPodStatus(kubecontainer.GetPodFullName(pod))
	if !found {
		t.Errorf("expected status of pod %q to be found", kubecontainer.GetPodFullName(pod))
	}
	if len(status.ContainerStatuses) < 1 {
		t.Errorf("expected 1 container status, got %d", len(status.ContainerStatuses))
	} else {
		state := status.ContainerStatuses[0].State
		if state.Waiting == nil {
			t.Errorf("expected waiting state, got %#v", state)
		} else if state.Waiting.Reason != failureReason {
			t.Errorf("expected reason %q, got %q", failureReason, state.Waiting.Reason)
		}
	}
}

func TestGetRestartCount(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	testKubelet.fakeCadvisor.On("DockerImagesFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	testKubelet.fakeCadvisor.On("RootFsInfo").Return(cadvisorApiv2.FsInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	containers := []api.Container{
		{Name: "bar"},
	}
	pod := api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "new",
		},
		Spec: api.PodSpec{
			Containers: containers,
		},
	}

	// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
	names := []string{"/k8s_bar." + strconv.FormatUint(kubecontainer.HashContainer(&containers[0]), 16) + "_foo_new_12345678_0"}
	currTime := time.Now()
	containerMap := map[string]*docker.Container{
		"1234": {
			ID:     "1234",
			Name:   "bar",
			Config: &docker.Config{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  currTime.Add(-60 * time.Second),
				FinishedAt: currTime.Add(-60 * time.Second),
			},
		},
		"5678": {
			ID:     "5678",
			Name:   "bar",
			Config: &docker.Config{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  currTime.Add(-30 * time.Second),
				FinishedAt: currTime.Add(-30 * time.Second),
			},
		},
		"9101": {
			ID:     "9101",
			Name:   "bar",
			Config: &docker.Config{},
			State: docker.State{
				ExitCode:   42,
				StartedAt:  currTime.Add(30 * time.Minute),
				FinishedAt: currTime.Add(30 * time.Minute),
			},
		},
	}
	fakeDocker.ContainerMap = containerMap

	// Helper function for verifying the restart count.
	verifyRestartCount := func(pod *api.Pod, expectedCount int) api.PodStatus {
		status, err := kubelet.generatePodStatus(pod)
		if err != nil {
			t.Errorf("unexpected error %v", err)
		}
		restartCount := status.ContainerStatuses[0].RestartCount
		if restartCount != expectedCount {
			t.Errorf("expected %d restart count, got %d", expectedCount, restartCount)
		}
		return status
	}

	// Container "bar" has failed twice; create two dead docker containers.
	// TODO: container lists are expected to be sorted reversely by time.
	// We should fix FakeDockerClient to sort the list before returning.
	fakeDocker.ExitedContainerList = []docker.APIContainers{{Names: names, ID: "5678"}, {Names: names, ID: "1234"}}
	pod.Status = verifyRestartCount(&pod, 1)

	// Found a new dead container. The restart count should be incremented.
	fakeDocker.ExitedContainerList = []docker.APIContainers{
		{Names: names, ID: "9101"}, {Names: names, ID: "5678"}, {Names: names, ID: "1234"}}
	pod.Status = verifyRestartCount(&pod, 2)

	// All dead containers have been GC'd. The restart count should persist
	// (i.e., remain the same).
	fakeDocker.ExitedContainerList = []docker.APIContainers{}
	verifyRestartCount(&pod, 2)
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
	kubeClient.ReactFn = func(action testclient.FakeAction) (runtime.Object, error) {
		segments := strings.Split(action.Action, "-")
		if len(segments) < 2 {
			return nil, fmt.Errorf("unrecognized action, need two or three segments <verb>-<resource> or <verb>-<subresource>-<resource>: %s", action.Action)
		}
		verb := segments[0]
		switch verb {
		case "create":
			// Return an error on create.
			return &api.Node{}, &apierrors.StatusError{
				ErrStatus: api.Status{Reason: api.StatusReasonAlreadyExists},
			}
		case "get":
			// Return an existing (matching) node on get.
			return &api.Node{
				ObjectMeta: api.ObjectMeta{Name: testKubeletHostname},
				Spec:       api.NodeSpec{ExternalID: testKubeletHostname},
			}, nil
		default:
			return nil, fmt.Errorf("no reaction implemented for %s", action.Action)
		}
	}
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
	case <-time.After(5 * time.Second):
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
	now := util.Now()
	startTime := util.NewTime(now.Time.Add(-1 * time.Minute))
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
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	now := util.Now()
	startTime := util.NewTime(now.Time.Add(-1 * time.Minute))
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
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar_new_12345678_1111"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_bar_new_12345678_2222"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"9999": {
			ID:         "9999",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	// Let the pod worker sets the status to fail after this sync.
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	podFullName := kubecontainer.GetPodFullName(pods[0])
	status, found := kubelet.statusManager.GetPodStatus(podFullName)
	if !found {
		t.Errorf("expected to found status for pod %q", status)
	}
	if status.Phase != api.PodFailed {
		t.Fatalf("expected pod status %q, ot %q.", api.PodFailed, status.Phase)
	}
}

func TestSyncPodsDoesNotSetPodsThatDidNotRunTooLongToFailed(t *testing.T) {
	testKubelet := newTestKubelet(t)
	testKubelet.fakeCadvisor.On("MachineInfo").Return(&cadvisorApi.MachineInfo{}, nil)
	kubelet := testKubelet.kubelet
	fakeDocker := testKubelet.fakeDocker

	now := util.Now()
	startTime := util.NewTime(now.Time.Add(-1 * time.Minute))
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
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar_new_12345678_1111"},
			ID:    "1234",
		},
		{
			// pod infra container
			Names: []string{"/k8s_POD." + strconv.FormatUint(generatePodInfraContainerHash(pods[0]), 16) + "_bar_new_12345678_2222"},
			ID:    "9876",
		},
	}
	fakeDocker.ContainerMap = map[string]*docker.Container{
		"1234": {
			ID:         "1234",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"9876": {
			ID:         "9876",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
		"9999": {
			ID:         "9999",
			Config:     &docker.Config{},
			HostConfig: &docker.HostConfig{},
		},
	}

	kubelet.podManager.SetPods(pods)
	err := kubelet.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	podFullName := kubecontainer.GetPodFullName(pods[0])
	status, found := kubelet.statusManager.GetPodStatus(podFullName)
	if !found {
		t.Errorf("expected to found status for pod %q", status)
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
	err := kl.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	for i := range pods {
		if !dirExists(kl.getPodDir(pods[i].UID)) {
			t.Errorf("expected directory to exist for pod %d", i)
		}
	}

	// Pod 1 has been deleted and no longer exists.
	err = kl.SyncPods([]*api.Pod{pods[0]}, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if !dirExists(kl.getPodDir(pods[0].UID)) {
		t.Errorf("expected directory to exist for pod 0")
	}
	if dirExists(kl.getPodDir(pods[1].UID)) {
		t.Errorf("expected directory to be deleted for pod 1")
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

	kl.podManager.SetPods(pods)
	// Sync to create pod directories.
	err := kl.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	for i := range pods {
		if !dirExists(kl.getPodDir(pods[i].UID)) {
			t.Errorf("expected directory to exist for pod %d", i)
		}
	}
	// Pod 1 failed, and pod 2 succeeded. None of the pod directories should be
	// deleted.
	kl.statusManager.SetPodStatus(pods[1], api.PodStatus{Phase: api.PodFailed})
	kl.statusManager.SetPodStatus(pods[2], api.PodStatus{Phase: api.PodSucceeded})
	err = kl.SyncPods(pods, emptyPodUIDs, map[string]*api.Pod{}, time.Now())
	for i := range pods {
		if !dirExists(kl.getPodDir(pods[i].UID)) {
			t.Errorf("expected directory to exist for pod %d", i)
		}
	}
}
