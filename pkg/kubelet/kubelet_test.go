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

package kubelet

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/health"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume"
	_ "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/volume/host_path"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/fsouza/go-dockerclient"
	"github.com/google/cadvisor/info"
	"github.com/stretchr/testify/mock"
)

func init() {
	api.ForTesting_ReferencesAllowBlankSelfLinks = true
	util.ReallyCrash = true
}

func newTestKubelet(t *testing.T) (*Kubelet, *dockertools.FakeDockerClient) {
	fakeDocker := &dockertools.FakeDockerClient{
		RemovedImages: util.StringSet{},
	}

	kubelet := &Kubelet{}
	kubelet.dockerClient = fakeDocker
	kubelet.dockerPuller = &dockertools.FakeDockerPuller{}
	if tempDir, err := ioutil.TempDir("/tmp", "kubelet_test."); err != nil {
		t.Fatalf("can't make a temp rootdir: %v", err)
	} else {
		kubelet.rootDirectory = tempDir
	}
	if err := os.MkdirAll(kubelet.rootDirectory, 0750); err != nil {
		t.Fatalf("can't mkdir(%q): %v", kubelet.rootDirectory, err)
	}
	kubelet.podWorkers = newPodWorkers()
	kubelet.sourceReady = func(source string) bool { return true }
	kubelet.masterServiceNamespace = api.NamespaceDefault
	kubelet.serviceLister = testServiceLister{}
	if err := kubelet.setupDataDirs(); err != nil {
		t.Fatalf("can't initialize kubelet data dirs: %v", err)
	}

	return kubelet, fakeDocker
}

func verifyCalls(t *testing.T, fakeDocker *dockertools.FakeDockerClient, calls []string) {
	err := fakeDocker.AssertCalls(calls)
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

func verifyBoolean(t *testing.T, expected, value bool) {
	if expected != value {
		t.Errorf("Unexpected boolean.  Expected %t.  Found %t", expected, value)
	}
}

func TestKubeletDirs(t *testing.T) {
	kubelet, _ := newTestKubelet(t)
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
	kubelet, _ := newTestKubelet(t)
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

func TestKillContainerWithError(t *testing.T) {
	fakeDocker := &dockertools.FakeDockerClient{
		Err: fmt.Errorf("sample error"),
		ContainerList: []docker.APIContainers{
			{
				ID:    "1234",
				Names: []string{"/k8s_foo_qux_1234_42"},
			},
			{
				ID:    "5678",
				Names: []string{"/k8s_bar_qux_5678_42"},
			},
		},
	}
	kubelet, _ := newTestKubelet(t)
	kubelet.dockerClient = fakeDocker
	err := kubelet.killContainer(&fakeDocker.ContainerList[0])
	if err == nil {
		t.Errorf("expected error, found nil")
	}
	verifyCalls(t, fakeDocker, []string{"stop"})
}

func TestKillContainer(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    "1234",
			Names: []string{"/k8s_foo_qux_1234_42"},
		},
		{
			ID:    "5678",
			Names: []string{"/k8s_bar_qux_5678_42"},
		},
	}
	fakeDocker.Container = &docker.Container{
		Name: "foobar",
	}

	err := kubelet.killContainer(&fakeDocker.ContainerList[0])
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	verifyCalls(t, fakeDocker, []string{"stop"})
}

type channelReader struct {
	list [][]api.BoundPod
	wg   sync.WaitGroup
}

func startReading(channel <-chan interface{}) *channelReader {
	cr := &channelReader{}
	cr.wg.Add(1)
	go func() {
		for {
			update, ok := <-channel
			if !ok {
				break
			}
			cr.list = append(cr.list, update.(PodUpdate).Pods)
		}
		cr.wg.Done()
	}()
	return cr
}

func (cr *channelReader) GetList() [][]api.BoundPod {
	cr.wg.Wait()
	return cr.list
}

func TestSyncPodsDoesNothing(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	container := api.Container{Name: "bar"}
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>_<random>
			Names: []string{"/k8s_bar." + strconv.FormatUint(dockertools.HashContainer(&container), 16) + "_foo.new.test_12345678_0"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s_net_foo.new.test_12345678_0"},
			ID:    "9876",
		},
	}
	err := kubelet.SyncPods([]api.BoundPod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "12345678",
				Name:        "foo",
				Namespace:   "new",
				Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					container,
				},
			},
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()
	verifyCalls(t, fakeDocker, []string{"list", "list", "inspect_container", "inspect_container"})
}

func TestSyncPodsWithTerminationLog(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	container := api.Container{
		Name: "bar",
		TerminationMessagePath: "/dev/somepath",
	}
	fakeDocker.ContainerList = []docker.APIContainers{}
	err := kubelet.SyncPods([]api.BoundPod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "12345678",
				Name:        "foo",
				Namespace:   "new",
				Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					container,
				},
			},
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()
	verifyCalls(t, fakeDocker, []string{
		"list", "create", "start", "list", "inspect_container", "list", "create", "start"})

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

// drainWorkers waits until all workers are done.  Should only used for testing.
func (kl *Kubelet) drainWorkers() {
	for {
		kl.podWorkers.lock.Lock()
		length := len(kl.podWorkers.workers)
		kl.podWorkers.lock.Unlock()
		if length == 0 {
			return
		}
		time.Sleep(time.Millisecond * 100)
	}
}

func matchString(t *testing.T, pattern, str string) bool {
	match, err := regexp.MatchString(pattern, str)
	if err != nil {
		t.Logf("unexpected error: %v", err)
	}
	return match
}

func TestSyncPodsCreatesNetAndContainer(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.networkContainerImage = "custom_image_name"
	fakeDocker.ContainerList = []docker.APIContainers{}
	err := kubelet.SyncPods([]api.BoundPod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "12345678",
				Name:        "foo",
				Namespace:   "new",
				Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar"},
				},
			},
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	verifyCalls(t, fakeDocker, []string{
		"list", "create", "start", "list", "inspect_container", "list", "create", "start"})

	fakeDocker.Lock()

	found := false
	for _, c := range fakeDocker.ContainerList {
		if c.Image == "custom_image_name" && strings.HasPrefix(c.Names[0], "/k8s_net") {
			found = true
		}
	}
	if !found {
		t.Errorf("Custom net container not found: %v", fakeDocker.ContainerList)
	}

	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "k8s_net\\.[a-f0-9]+_foo.new.test_", fakeDocker.Created[0]) ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo.new.test_", fakeDocker.Created[1]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodsCreatesNetAndContainerPullsImage(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	puller := kubelet.dockerPuller.(*dockertools.FakeDockerPuller)
	puller.HasImages = []string{}
	kubelet.networkContainerImage = "custom_image_name"
	fakeDocker.ContainerList = []docker.APIContainers{}
	err := kubelet.SyncPods([]api.BoundPod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "12345678",
				Name:        "foo",
				Namespace:   "new",
				Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar"},
				},
			},
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	verifyCalls(t, fakeDocker, []string{
		"list", "create", "start", "list", "inspect_container", "list", "create", "start"})

	fakeDocker.Lock()

	if !reflect.DeepEqual(puller.ImagesPulled, []string{"custom_image_name", ""}) {
		t.Errorf("Unexpected pulled containers: %v", puller.ImagesPulled)
	}

	if len(fakeDocker.Created) != 2 ||
		!matchString(t, "k8s_net\\.[a-f0-9]+_foo.new.test_", fakeDocker.Created[0]) ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo.new.test_", fakeDocker.Created[1]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodsWithNetCreatesContainer(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// network container
			Names: []string{"/k8s_net_foo.new.test_12345678_0"},
			ID:    "9876",
		},
	}
	err := kubelet.SyncPods([]api.BoundPod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "12345678",
				Name:        "foo",
				Namespace:   "new",
				Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar"},
				},
			},
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	verifyCalls(t, fakeDocker, []string{
		"list", "list", "inspect_container", "list", "create", "start"})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 1 ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo.new.test_", fakeDocker.Created[0]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestSyncPodsWithNetCreatesContainerCallsHandler(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	fakeHttp := fakeHTTP{}
	kubelet.httpClient = &fakeHttp
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// network container
			Names: []string{"/k8s_net_foo.new.test_12345678_0"},
			ID:    "9876",
		},
	}
	err := kubelet.SyncPods([]api.BoundPod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "12345678",
				Name:        "foo",
				Namespace:   "new",
				Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
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
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	verifyCalls(t, fakeDocker, []string{
		"list", "list", "inspect_container", "list", "create", "start"})

	fakeDocker.Lock()
	if len(fakeDocker.Created) != 1 ||
		!matchString(t, "k8s_bar\\.[a-f0-9]+_foo.new.test_", fakeDocker.Created[0]) {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
	if fakeHttp.url != "http://foo:8080/bar" {
		t.Errorf("Unexpected handler: %s", fakeHttp.url)
	}
}

func TestSyncPodsDeletesWithNoNetContainer(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>
			Names: []string{"/k8s_bar_foo.new.test_12345678_0"},
			ID:    "1234",
		},
	}
	err := kubelet.SyncPods([]api.BoundPod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "12345678",
				Name:        "foo",
				Namespace:   "new",
				Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
			},
			Spec: api.PodSpec{
				Containers: []api.Container{
					{Name: "bar"},
				},
			},
		},
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	verifyCalls(t, fakeDocker, []string{
		"list", "stop", "create", "start", "list", "list", "inspect_container", "list", "create", "start"})

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
	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.sourceReady = func(source string) bool { return ready }

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar.new.test_12345678_42"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s_net_foo.new.test_12345678_42"},
			ID:    "9876",
		},
	}
	if err := kubelet.SyncPods([]api.BoundPod{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Validate nothing happened.
	verifyCalls(t, fakeDocker, []string{"list"})
	fakeDocker.ClearCalls()

	ready = true
	if err := kubelet.SyncPods([]api.BoundPod{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	verifyCalls(t, fakeDocker, []string{"list", "stop", "stop"})

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

func TestSyncPodsDeletesWhenContainerSourceReady(t *testing.T) {
	ready := false
	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.sourceReady = func(source string) bool {
		if source == "testSource" {
			return ready
		}
		return false
	}

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_boo_bar.default.testSource_12345678_42"},
			ID:    "7492",
		},
		{
			// network container
			Names: []string{"/k8s_net_boo.default.testSource_12345678_42"},
			ID:    "3542",
		},

		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar.new.otherSource_12345678_42"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s_net_foo.new.otherSource_12345678_42"},
			ID:    "9876",
		},
	}
	if err := kubelet.SyncPods([]api.BoundPod{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	// Validate nothing happened.
	verifyCalls(t, fakeDocker, []string{"list"})
	fakeDocker.ClearCalls()

	ready = true
	if err := kubelet.SyncPods([]api.BoundPod{}); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	verifyCalls(t, fakeDocker, []string{"list", "stop", "stop"})

	// Validate container for testSource are killed because testSource is reported as seen, but
	// containers for otherSource are not killed because otherSource has not.
	expectedToStop := map[string]bool{
		"7492": true,
		"3542": true,
		"1234": false,
		"9876": false,
	}
	if len(fakeDocker.Stopped) != 2 ||
		!expectedToStop[fakeDocker.Stopped[0]] ||
		!expectedToStop[fakeDocker.Stopped[1]] {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

func TestSyncPodsDeletes(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar.new.test_12345678_42"},
			ID:    "1234",
		},
		{
			// network container
			Names: []string{"/k8s_net_foo.new.test_12345678_42"},
			ID:    "9876",
		},
		{
			Names: []string{"foo"},
			ID:    "4567",
		},
	}
	err := kubelet.SyncPods([]api.BoundPod{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "stop", "stop"})

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

func TestSyncPodDeletesDuplicate(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	dockerContainers := dockertools.DockerContainers{
		"1234": &docker.APIContainers{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_foo_bar.new.test_12345678_1111"},
			ID:    "1234",
		},
		"9876": &docker.APIContainers{
			// network container
			Names: []string{"/k8s_net_bar.new.test_12345678_2222"},
			ID:    "9876",
		},
		"4567": &docker.APIContainers{
			// Duplicate for the same container.
			Names: []string{"/k8s_foo_bar.new.test_12345678_3333"},
			ID:    "4567",
		},
		"2304": &docker.APIContainers{
			// Container for another pod, untouched.
			Names: []string{"/k8s_baz_fiz.new.test_6_42"},
			ID:    "2304",
		},
	}
	err := kubelet.syncPod(&api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			UID:         "12345678",
			Name:        "bar",
			Namespace:   "new",
			Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "foo"},
			},
		},
	}, dockerContainers)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "stop"})

	// Expect one of the duplicates to be killed.
	if len(fakeDocker.Stopped) != 1 || (fakeDocker.Stopped[0] != "1234" && fakeDocker.Stopped[0] != "4567") {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

type FalseHealthChecker struct{}

func (f *FalseHealthChecker) HealthCheck(podFullName string, podUID types.UID, status api.PodStatus, container api.Container) (health.Status, error) {
	return health.Unhealthy, nil
}

func (f *FalseHealthChecker) CanCheck(probe *api.LivenessProbe) bool {
	return true
}

func TestSyncPodBadHash(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.healthChecker = &FalseHealthChecker{}
	dockerContainers := dockertools.DockerContainers{
		"1234": &docker.APIContainers{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_bar.1234_foo.new.test_12345678_42"},
			ID:    "1234",
		},
		"9876": &docker.APIContainers{
			// network container
			Names: []string{"/k8s_net_foo.new.test_12345678_42"},
			ID:    "9876",
		},
	}
	err := kubelet.syncPod(&api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			UID:         "12345678",
			Name:        "foo",
			Namespace:   "new",
			Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar"},
			},
		},
	}, dockerContainers)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "stop", "stop", "list", "create", "start"})

	// A map interation is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
		"9876": true,
	}
	if len(fakeDocker.Stopped) != 2 ||
		(!expectedToStop[fakeDocker.Stopped[0]] &&
			!expectedToStop[fakeDocker.Stopped[1]]) {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

func TestSyncPodUnhealthy(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.healthChecker = &FalseHealthChecker{}
	dockerContainers := dockertools.DockerContainers{
		"1234": &docker.APIContainers{
			// the k8s prefix is required for the kubelet to manage the container
			Names: []string{"/k8s_bar_foo.new.test_12345678_42"},
			ID:    "1234",
		},
		"9876": &docker.APIContainers{
			// network container
			Names: []string{"/k8s_net_foo.new.test_12345678_42"},
			ID:    "9876",
		},
	}
	err := kubelet.syncPod(&api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			UID:         "12345678",
			Name:        "foo",
			Namespace:   "new",
			Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{Name: "bar",
					LivenessProbe: &api.LivenessProbe{
					// Always returns healthy == false
					},
				},
			},
		},
	}, dockerContainers)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "stop", "stop", "list", "create", "start"})

	// A map interation is used to delete containers, so must not depend on
	// order here.
	expectedToStop := map[string]bool{
		"1234": true,
		"9876": true,
	}
	if len(fakeDocker.Stopped) != 2 ||
		(!expectedToStop[fakeDocker.Stopped[0]] &&
			expectedToStop[fakeDocker.Stopped[0]]) {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

func TestMountExternalVolumes(t *testing.T) {
	kubelet, _ := newTestKubelet(t)
	kubelet.volumePluginMgr.InitPlugins([]volume.Plugin{&volume.FakePlugin{"fake", nil}}, &volumeHost{kubelet})

	pod := api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "test",
		},
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{
					Name:   "vol1",
					Source: &api.VolumeSource{},
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
			t.Errorf("api.BoundPod volumes map is missing key: %s. %#v", name, podVolumes)
		}
	}
}

func TestGetPodVolumesFromDisk(t *testing.T) {
	kubelet, _ := newTestKubelet(t)
	plug := &volume.FakePlugin{"fake", nil}
	kubelet.volumePluginMgr.InitPlugins([]volume.Plugin{plug}, &volumeHost{kubelet})

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

func TestMakeVolumesAndBinds(t *testing.T) {
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

	pod := api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "pod",
			Namespace: "test",
		},
	}

	podVolumes := volumeMap{
		"disk":  &stubVolume{"/mnt/disk"},
		"disk4": &stubVolume{"/mnt/host"},
		"disk5": &stubVolume{"/var/lib/kubelet/podID/volumes/empty/disk5"},
	}

	binds := makeBinds(&pod, &container, podVolumes)

	expectedBinds := []string{
		"/mnt/disk:/mnt/path",
		"/mnt/disk:/mnt/path3:ro",
		"/mnt/host:/mnt/path4",
		"/var/lib/kubelet/podID/volumes/empty/disk5:/mnt/path5",
	}

	if len(binds) != len(expectedBinds) {
		t.Errorf("Unexpected binds: Expected %#v got %#v.  Container was: %#v", expectedBinds, binds, container)
	}
	verifyStringArrayEquals(t, binds, expectedBinds)
}

func TestMakePortsAndBindings(t *testing.T) {
	container := api.Container{
		Ports: []api.Port{
			{
				ContainerPort: 80,
				HostPort:      8080,
				HostIP:        "127.0.0.1",
			},
			{
				ContainerPort: 443,
				HostPort:      443,
				Protocol:      "tcp",
			},
			{
				ContainerPort: 444,
				HostPort:      444,
				Protocol:      "udp",
			},
			{
				ContainerPort: 445,
				HostPort:      445,
				Protocol:      "foobar",
			},
		},
	}
	exposedPorts, bindings := makePortsAndBindings(&container)
	if len(container.Ports) != len(exposedPorts) ||
		len(container.Ports) != len(bindings) {
		t.Errorf("Unexpected ports and bindings, %#v %#v %#v", container, exposedPorts, bindings)
	}
	for key, value := range bindings {
		switch value[0].HostPort {
		case "8080":
			if !reflect.DeepEqual(docker.Port("80/tcp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
			if value[0].HostIP != "127.0.0.1" {
				t.Errorf("Unexpected host IP: %s", value[0].HostIP)
			}
		case "443":
			if !reflect.DeepEqual(docker.Port("443/tcp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
			if value[0].HostIP != "" {
				t.Errorf("Unexpected host IP: %s", value[0].HostIP)
			}
		case "444":
			if !reflect.DeepEqual(docker.Port("444/udp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
			if value[0].HostIP != "" {
				t.Errorf("Unexpected host IP: %s", value[0].HostIP)
			}
		case "445":
			if !reflect.DeepEqual(docker.Port("445/tcp"), key) {
				t.Errorf("Unexpected docker port: %#v", key)
			}
			if value[0].HostIP != "" {
				t.Errorf("Unexpected host IP: %s", value[0].HostIP)
			}
		}
	}
}

func TestCheckHostPortConflicts(t *testing.T) {
	successCaseAll := []api.BoundPod{
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.Port{{HostPort: 80}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.Port{{HostPort: 81}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.Port{{HostPort: 82}}}}}},
	}
	successCaseNew := api.BoundPod{
		Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.Port{{HostPort: 83}}}}},
	}
	expected := append(successCaseAll, successCaseNew)
	if actual := filterHostPortConflicts(expected); !reflect.DeepEqual(actual, expected) {
		t.Errorf("Expected %#v, Got %#v", expected, actual)
	}

	failureCaseAll := []api.BoundPod{
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.Port{{HostPort: 80}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.Port{{HostPort: 81}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.Port{{HostPort: 82}}}}}},
	}
	failureCaseNew := api.BoundPod{
		Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.Port{{HostPort: 81}}}}},
	}
	if actual := filterHostPortConflicts(append(failureCaseAll, failureCaseNew)); !reflect.DeepEqual(failureCaseAll, actual) {
		t.Errorf("Expected %#v, Got %#v", expected, actual)
	}
}

func TestFieldPath(t *testing.T) {
	pod := &api.BoundPod{Spec: api.PodSpec{Containers: []api.Container{
		{Name: "foo"},
		{Name: "bar"},
		{Name: ""},
		{Name: "baz"},
	}}}
	table := map[string]struct {
		pod       *api.BoundPod
		container *api.Container
		path      string
		success   bool
	}{
		"basic":            {pod, &api.Container{Name: "foo"}, "spec.containers{foo}", true},
		"basic2":           {pod, &api.Container{Name: "baz"}, "spec.containers{baz}", true},
		"emptyName":        {pod, &api.Container{Name: ""}, "spec.containers[2]", true},
		"basicSamePointer": {pod, &pod.Spec.Containers[0], "spec.containers{foo}", true},
		"missing":          {pod, &api.Container{Name: "qux"}, "", false},
	}

	for name, item := range table {
		res, err := fieldPath(item.pod, item.container)
		if item.success == false {
			if err == nil {
				t.Errorf("%v: unexpected non-error", name)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: unexpected error: %v", name, err)
			continue
		}
		if e, a := item.path, res; e != a {
			t.Errorf("%v: wanted %v, got %v", name, e, a)
		}
	}
}

type mockCadvisorClient struct {
	mock.Mock
}

// ContainerInfo is a mock implementation of CadvisorInterface.ContainerInfo.
func (c *mockCadvisorClient) ContainerInfo(name string, req *info.ContainerInfoRequest) (*info.ContainerInfo, error) {
	args := c.Called(name, req)
	return args.Get(0).(*info.ContainerInfo), args.Error(1)
}

// DockerContainer is a mock implementation of CadvisorInterface.DockerContainer.
func (c *mockCadvisorClient) DockerContainer(name string, req *info.ContainerInfoRequest) (info.ContainerInfo, error) {
	args := c.Called(name, req)
	return args.Get(0).(info.ContainerInfo), args.Error(1)
}

// MachineInfo is a mock implementation of CadvisorInterface.MachineInfo.
func (c *mockCadvisorClient) MachineInfo() (*info.MachineInfo, error) {
	args := c.Called()
	return args.Get(0).(*info.MachineInfo), args.Error(1)
}

func TestGetContainerInfo(t *testing.T) {
	containerID := "ab2cdf"
	containerPath := fmt.Sprintf("/docker/%v", containerID)
	containerInfo := info.ContainerInfo{
		ContainerReference: info.ContainerReference{
			Name: containerPath,
		},
	}

	mockCadvisor := &mockCadvisorClient{}
	cadvisorReq := &info.ContainerInfoRequest{}
	mockCadvisor.On("DockerContainer", containerID, cadvisorReq).Return(containerInfo, nil)

	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.cadvisorClient = mockCadvisor
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID: containerID,
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s_foo_qux_1234_42"},
		},
	}

	stats, err := kubelet.GetContainerInfo("qux", "", "foo", cadvisorReq)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if stats == nil {
		t.Fatalf("stats should not be nil")
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetRootInfo(t *testing.T) {
	containerPath := "/"
	containerInfo := &info.ContainerInfo{
		ContainerReference: info.ContainerReference{
			Name: containerPath,
		},
	}
	fakeDocker := dockertools.FakeDockerClient{}

	mockCadvisor := &mockCadvisorClient{}
	cadvisorReq := &info.ContainerInfoRequest{}
	mockCadvisor.On("ContainerInfo", containerPath, cadvisorReq).Return(containerInfo, nil)

	kubelet := Kubelet{
		dockerClient:   &fakeDocker,
		dockerPuller:   &dockertools.FakeDockerPuller{},
		cadvisorClient: mockCadvisor,
		podWorkers:     newPodWorkers(),
	}

	// If the container name is an empty string, then it means the root container.
	_, err := kubelet.GetRootInfo(cadvisorReq)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoWithoutCadvisor(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID: "foobar",
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s_foo_qux_uuid_1234"},
		},
	}

	stats, _ := kubelet.GetContainerInfo("qux", "uuid", "foo", nil)
	// When there's no cAdvisor, the stats should be either nil or empty
	if stats == nil {
		return
	}
}

func TestGetContainerInfoWhenCadvisorFailed(t *testing.T) {
	containerID := "ab2cdf"

	containerInfo := info.ContainerInfo{}
	mockCadvisor := &mockCadvisorClient{}
	cadvisorReq := &info.ContainerInfoRequest{}
	expectedErr := fmt.Errorf("some error")
	mockCadvisor.On("DockerContainer", containerID, cadvisorReq).Return(containerInfo, expectedErr)

	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.cadvisorClient = mockCadvisor
	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID: containerID,
			// pod id: qux
			// container id: foo
			Names: []string{"/k8s_foo_qux_uuid_1234"},
		},
	}

	stats, err := kubelet.GetContainerInfo("qux", "uuid", "foo", cadvisorReq)
	if stats != nil {
		t.Errorf("non-nil stats on error")
	}
	if err == nil {
		t.Errorf("expect error but received nil error")
		return
	}
	if err.Error() != expectedErr.Error() {
		t.Errorf("wrong error message. expect %v, got %v", err, expectedErr)
	}
	mockCadvisor.AssertExpectations(t)
}

func TestGetContainerInfoOnNonExistContainer(t *testing.T) {
	mockCadvisor := &mockCadvisorClient{}

	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.cadvisorClient = mockCadvisor
	fakeDocker.ContainerList = []docker.APIContainers{}

	stats, _ := kubelet.GetContainerInfo("qux", "", "foo", nil)
	if stats != nil {
		t.Errorf("non-nil stats on non exist container")
	}
	mockCadvisor.AssertExpectations(t)
}

type fakeContainerCommandRunner struct {
	Cmd []string
	ID  string
	E   error
}

func (f *fakeContainerCommandRunner) RunInContainer(id string, cmd []string) ([]byte, error) {
	f.Cmd = cmd
	f.ID = id
	return []byte{}, f.E
}

func TestRunInContainerNoSuchPod(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet, fakeDocker := newTestKubelet(t)
	fakeDocker.ContainerList = []docker.APIContainers{}
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "etcd"
	containerName := "containerFoo"
	output, err := kubelet.RunInContainer(
		GetPodFullName(&api.BoundPod{ObjectMeta: api.ObjectMeta{Name: podName, Namespace: podNamespace}}),
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
	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.runner = &fakeCommandRunner

	containerID := "abc1234"
	podName := "podFoo"
	podNamespace := "etcd"
	containerName := "containerFoo"

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    containerID,
			Names: []string{"/k8s_" + containerName + "_" + podName + "." + podNamespace + ".test_12345678_42"},
		},
	}

	cmd := []string{"ls"}
	_, err := kubelet.RunInContainer(
		GetPodFullName(&api.BoundPod{
			ObjectMeta: api.ObjectMeta{
				UID:         "12345678",
				Name:        podName,
				Namespace:   podNamespace,
				Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
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

func TestRunHandlerExec(t *testing.T) {
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.runner = &fakeCommandRunner

	containerID := "abc1234"
	podName := "podFoo"
	podNamespace := "etcd"
	containerName := "containerFoo"

	fakeDocker.ContainerList = []docker.APIContainers{
		{
			ID:    containerID,
			Names: []string{"/k8s_" + containerName + "_" + podName + "." + podNamespace + "_12345678_42"},
		},
	}

	container := api.Container{
		Name: containerName,
		Lifecycle: &api.Lifecycle{
			PostStart: &api.Handler{
				Exec: &api.ExecAction{
					Command: []string{"ls", "-a"},
				},
			},
		},
	}
	err := kubelet.runHandler(podName+"."+podNamespace, "", &container, container.Lifecycle.PostStart)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fakeCommandRunner.ID != containerID ||
		!reflect.DeepEqual(container.Lifecycle.PostStart.Exec.Command, fakeCommandRunner.Cmd) {
		t.Errorf("unexpected commands: %v", fakeCommandRunner)
	}
}

type fakeHTTP struct {
	url string
	err error
}

func (f *fakeHTTP) Get(url string) (*http.Response, error) {
	f.url = url
	return nil, f.err
}

func TestRunHandlerHttp(t *testing.T) {
	fakeHttp := fakeHTTP{}

	kubelet, _ := newTestKubelet(t)
	kubelet.httpClient = &fakeHttp

	podName := "podFoo"
	podNamespace := "etcd"
	containerName := "containerFoo"

	container := api.Container{
		Name: containerName,
		Lifecycle: &api.Lifecycle{
			PostStart: &api.Handler{
				HTTPGet: &api.HTTPGetAction{
					Host: "foo",
					Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
					Path: "bar",
				},
			},
		},
	}
	err := kubelet.runHandler(podName+"."+podNamespace, "", &container, container.Lifecycle.PostStart)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if fakeHttp.url != "http://foo:8080/bar" {
		t.Errorf("unexpected url: %s", fakeHttp.url)
	}
}

func TestNewHandler(t *testing.T) {
	kubelet, _ := newTestKubelet(t)
	handler := &api.Handler{
		HTTPGet: &api.HTTPGetAction{
			Host: "foo",
			Port: util.IntOrString{IntVal: 8080, Kind: util.IntstrInt},
			Path: "bar",
		},
	}
	actionHandler := kubelet.newActionHandler(handler)
	if actionHandler == nil {
		t.Error("unexpected nil action handler.")
	}

	handler = &api.Handler{
		Exec: &api.ExecAction{
			Command: []string{"ls", "-l"},
		},
	}
	actionHandler = kubelet.newActionHandler(handler)
	if actionHandler == nil {
		t.Error("unexpected nil action handler.")
	}

	handler = &api.Handler{}
	actionHandler = kubelet.newActionHandler(handler)
	if actionHandler != nil {
		t.Errorf("unexpected non-nil action handler: %v", actionHandler)
	}
}

func TestSyncPodEventHandlerFails(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	kubelet.httpClient = &fakeHTTP{
		err: fmt.Errorf("test error"),
	}
	dockerContainers := dockertools.DockerContainers{
		"9876": &docker.APIContainers{
			// network container
			Names: []string{"/k8s_net_foo.new.test_12345678_42"},
			ID:    "9876",
		},
	}
	err := kubelet.syncPod(&api.BoundPod{
		ObjectMeta: api.ObjectMeta{
			UID:         "12345678",
			Name:        "foo",
			Namespace:   "new",
			Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
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
	}, dockerContainers)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	verifyCalls(t, fakeDocker, []string{"list", "list", "create", "start", "stop"})

	if len(fakeDocker.Stopped) != 1 {
		t.Errorf("Wrong containers were stopped: %v", fakeDocker.Stopped)
	}
}

func TestKubeletGarbageCollection(t *testing.T) {
	tests := []struct {
		containers       []docker.APIContainers
		containerDetails map[string]*docker.Container
		expectedRemoved  []string
	}{
		{
			containers: []docker.APIContainers{
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "1876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "2876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "3876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "4876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "5876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "6876",
				},
			},
			containerDetails: map[string]*docker.Container{
				"1876": {
					State: docker.State{
						Running: false,
					},
					ID:      "1876",
					Created: time.Now(),
				},
			},
			expectedRemoved: []string{"1876"},
		},
		{
			containers: []docker.APIContainers{
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "1876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "2876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "3876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "4876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "5876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "6876",
				},
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "7876",
				},
			},
			containerDetails: map[string]*docker.Container{
				"1876": {
					State: docker.State{
						Running: true,
					},
					ID:      "1876",
					Created: time.Now(),
				},
				"2876": {
					State: docker.State{
						Running: false,
					},
					ID:      "2876",
					Created: time.Now(),
				},
			},
			expectedRemoved: []string{"2876"},
		},
		{
			containers: []docker.APIContainers{
				{
					// network container
					Names: []string{"/k8s_net_foo.new.test_.deadbeef_42"},
					ID:    "1876",
				},
			},
		},
	}
	for _, test := range tests {
		kubelet, fakeDocker := newTestKubelet(t)
		kubelet.maxContainerCount = 5
		fakeDocker.ContainerList = test.containers
		fakeDocker.ContainerMap = test.containerDetails
		fakeDocker.Container = &docker.Container{ID: "error", Created: time.Now()}
		err := kubelet.GarbageCollectContainers()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		if !reflect.DeepEqual(fakeDocker.Removed, test.expectedRemoved) {
			t.Errorf("expected: %v, got: %v", test.expectedRemoved, fakeDocker.Removed)
		}
	}
}

func TestPurgeOldest(t *testing.T) {
	created := time.Now()
	tests := []struct {
		ids              []string
		containerDetails map[string]*docker.Container
		expectedRemoved  []string
	}{
		{
			ids: []string{"1", "2", "3", "4", "5"},
			containerDetails: map[string]*docker.Container{
				"1": {
					State: docker.State{
						Running: true,
					},
					ID:      "1",
					Created: created,
				},
				"2": {
					State: docker.State{
						Running: false,
					},
					ID:      "2",
					Created: created.Add(time.Second),
				},
				"3": {
					State: docker.State{
						Running: false,
					},
					ID:      "3",
					Created: created.Add(time.Second),
				},
				"4": {
					State: docker.State{
						Running: false,
					},
					ID:      "4",
					Created: created.Add(time.Second),
				},
				"5": {
					State: docker.State{
						Running: false,
					},
					ID:      "5",
					Created: created.Add(time.Second),
				},
			},
		},
		{
			ids: []string{"1", "2", "3", "4", "5", "6"},
			containerDetails: map[string]*docker.Container{
				"1": {
					State: docker.State{
						Running: false,
					},
					ID:      "1",
					Created: created.Add(time.Second),
				},
				"2": {
					State: docker.State{
						Running: false,
					},
					ID:      "2",
					Created: created.Add(time.Millisecond),
				},
				"3": {
					State: docker.State{
						Running: false,
					},
					ID:      "3",
					Created: created.Add(time.Second),
				},
				"4": {
					State: docker.State{
						Running: false,
					},
					ID:      "4",
					Created: created.Add(time.Second),
				},
				"5": {
					State: docker.State{
						Running: false,
					},
					ID:      "5",
					Created: created.Add(time.Second),
				},
				"6": {
					State: docker.State{
						Running: false,
					},
					ID:      "6",
					Created: created.Add(time.Second),
				},
			},
			expectedRemoved: []string{"2"},
		},
		{
			ids: []string{"1", "2", "3", "4", "5", "6", "7"},
			containerDetails: map[string]*docker.Container{
				"1": {
					State: docker.State{
						Running: false,
					},
					ID:      "1",
					Created: created.Add(time.Second),
				},
				"2": {
					State: docker.State{
						Running: false,
					},
					ID:      "2",
					Created: created.Add(time.Millisecond),
				},
				"3": {
					State: docker.State{
						Running: false,
					},
					ID:      "3",
					Created: created.Add(time.Second),
				},
				"4": {
					State: docker.State{
						Running: false,
					},
					ID:      "4",
					Created: created.Add(time.Second),
				},
				"5": {
					State: docker.State{
						Running: false,
					},
					ID:      "5",
					Created: created.Add(time.Second),
				},
				"6": {
					State: docker.State{
						Running: false,
					},
					ID:      "6",
					Created: created.Add(time.Microsecond),
				},
				"7": {
					State: docker.State{
						Running: false,
					},
					ID:      "7",
					Created: created.Add(time.Second),
				},
			},
			expectedRemoved: []string{"2", "6"},
		},
	}
	for _, test := range tests {
		kubelet, fakeDocker := newTestKubelet(t)
		kubelet.maxContainerCount = 5
		fakeDocker.ContainerMap = test.containerDetails
		kubelet.purgeOldest(test.ids)
		if !reflect.DeepEqual(fakeDocker.Removed, test.expectedRemoved) {
			t.Errorf("expected: %v, got: %v", test.expectedRemoved, fakeDocker.Removed)
		}
	}
}

func TestSyncPodsWithPullPolicy(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)
	puller := kubelet.dockerPuller.(*dockertools.FakeDockerPuller)
	puller.HasImages = []string{"existing_one", "want:latest"}
	kubelet.networkContainerImage = "custom_image_name"
	fakeDocker.ContainerList = []docker.APIContainers{}
	err := kubelet.SyncPods([]api.BoundPod{
		{
			ObjectMeta: api.ObjectMeta{
				UID:         "12345678",
				Name:        "foo",
				Namespace:   "new",
				Annotations: map[string]string{ConfigSourceAnnotationKey: "test"},
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
	})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	kubelet.drainWorkers()

	fakeDocker.Lock()

	if !reflect.DeepEqual(puller.ImagesPulled, []string{"custom_image_name", "pull_always_image", "pull_if_not_present_image", "want:latest"}) {
		t.Errorf("Unexpected pulled containers: %v", puller.ImagesPulled)
	}

	if len(fakeDocker.Created) != 6 {
		t.Errorf("Unexpected containers created %v", fakeDocker.Created)
	}
	fakeDocker.Unlock()
}

func TestGarbageCollectImages(t *testing.T) {
	kubelet, fakeDocker := newTestKubelet(t)

	fakeDocker.Images = []docker.APIImages{
		{
			ID: "foo",
		},
		{
			ID: "bar",
		},
	}

	if err := kubelet.GarbageCollectImages(); err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if len(fakeDocker.RemovedImages) != 2 ||
		!fakeDocker.RemovedImages.Has("foo") ||
		!fakeDocker.RemovedImages.Has("bar") {
		t.Errorf("unexpected images removed: %v", fakeDocker.RemovedImages)
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

type testServiceLister struct {
	services []api.Service
}

func (ls testServiceLister) List() (api.ServiceList, error) {
	return api.ServiceList{
		Items: ls.services,
	}, nil
}

func TestMakeEnvironmentVariables(t *testing.T) {
	services := []api.Service{
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: api.NamespaceDefault},
			Spec: api.ServiceSpec{
				Port:     8081,
				PortalIP: "1.2.3.1",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes-ro", Namespace: api.NamespaceDefault},
			Spec: api.ServiceSpec{
				Port:     8082,
				PortalIP: "1.2.3.2",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "test1"},
			Spec: api.ServiceSpec{
				Port:     8083,
				PortalIP: "1.2.3.3",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "test2"},
			Spec: api.ServiceSpec{
				Port:     8084,
				PortalIP: "1.2.3.4",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "test", Namespace: "test2"},
			Spec: api.ServiceSpec{
				Port:     8085,
				PortalIP: "1.2.3.5",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Port:     8086,
				PortalIP: "1.2.3.6",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "kubernetes-ro", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Port:     8087,
				PortalIP: "1.2.3.7",
			},
		},
		{
			ObjectMeta: api.ObjectMeta{Name: "not-special", Namespace: "kubernetes"},
			Spec: api.ServiceSpec{
				Port:     8088,
				PortalIP: "1.2.3.8",
			},
		},
	}

	testCases := []struct {
		name                   string         // the name of the test case
		ns                     string         // the namespace to generate environment for
		container              *api.Container // the container to use
		masterServiceNamespace string         // the namespace to read master service info from
		nilLister              bool           // whether the lister should be nil
		expectedEnvs           util.StringSet // a set of expected environment vars
		expectedEnvSize        int            // total number of expected env vars
	}{
		{
			"api server = Y, kubelet = Y",
			"test1",
			&api.Container{
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
			api.NamespaceDefault,
			false,
			util.NewStringSet("FOO=BAR",
				"TEST_SERVICE_HOST=1.2.3.3",
				"TEST_SERVICE_PORT=8083",
				"TEST_PORT=tcp://1.2.3.3:8083",
				"TEST_PORT_8083_TCP=tcp://1.2.3.3:8083",
				"TEST_PORT_8083_TCP_PROTO=tcp",
				"TEST_PORT_8083_TCP_PORT=8083",
				"TEST_PORT_8083_TCP_ADDR=1.2.3.3",
				"KUBERNETES_SERVICE_HOST=1.2.3.1",
				"KUBERNETES_SERVICE_PORT=8081",
				"KUBERNETES_PORT=tcp://1.2.3.1:8081",
				"KUBERNETES_PORT_8081_TCP=tcp://1.2.3.1:8081",
				"KUBERNETES_PORT_8081_TCP_PROTO=tcp",
				"KUBERNETES_PORT_8081_TCP_PORT=8081",
				"KUBERNETES_PORT_8081_TCP_ADDR=1.2.3.1",
				"KUBERNETES_RO_SERVICE_HOST=1.2.3.2",
				"KUBERNETES_RO_SERVICE_PORT=8082",
				"KUBERNETES_RO_PORT=tcp://1.2.3.2:8082",
				"KUBERNETES_RO_PORT_8082_TCP=tcp://1.2.3.2:8082",
				"KUBERNETES_RO_PORT_8082_TCP_PROTO=tcp",
				"KUBERNETES_RO_PORT_8082_TCP_PORT=8082",
				"KUBERNETES_RO_PORT_8082_TCP_ADDR=1.2.3.2"),
			22,
		},
		{
			"api server = Y, kubelet = N",
			"test1",
			&api.Container{
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
			api.NamespaceDefault,
			true,
			util.NewStringSet("FOO=BAR",
				"TEST_SERVICE_HOST=1.2.3.3",
				"TEST_SERVICE_PORT=8083",
				"TEST_PORT=tcp://1.2.3.3:8083",
				"TEST_PORT_8083_TCP=tcp://1.2.3.3:8083",
				"TEST_PORT_8083_TCP_PROTO=tcp",
				"TEST_PORT_8083_TCP_PORT=8083",
				"TEST_PORT_8083_TCP_ADDR=1.2.3.3"),
			8,
		},
		{
			"api server = N; kubelet = Y",
			"test1",
			&api.Container{
				Env: []api.EnvVar{
					{Name: "FOO", Value: "BAZ"},
				},
			},
			api.NamespaceDefault,
			false,
			util.NewStringSet("FOO=BAZ",
				"TEST_SERVICE_HOST=1.2.3.3",
				"TEST_SERVICE_PORT=8083",
				"TEST_PORT=tcp://1.2.3.3:8083",
				"TEST_PORT_8083_TCP=tcp://1.2.3.3:8083",
				"TEST_PORT_8083_TCP_PROTO=tcp",
				"TEST_PORT_8083_TCP_PORT=8083",
				"TEST_PORT_8083_TCP_ADDR=1.2.3.3",
				"KUBERNETES_SERVICE_HOST=1.2.3.1",
				"KUBERNETES_SERVICE_PORT=8081",
				"KUBERNETES_PORT=tcp://1.2.3.1:8081",
				"KUBERNETES_PORT_8081_TCP=tcp://1.2.3.1:8081",
				"KUBERNETES_PORT_8081_TCP_PROTO=tcp",
				"KUBERNETES_PORT_8081_TCP_PORT=8081",
				"KUBERNETES_PORT_8081_TCP_ADDR=1.2.3.1",
				"KUBERNETES_RO_SERVICE_HOST=1.2.3.2",
				"KUBERNETES_RO_SERVICE_PORT=8082",
				"KUBERNETES_RO_PORT=tcp://1.2.3.2:8082",
				"KUBERNETES_RO_PORT_8082_TCP=tcp://1.2.3.2:8082",
				"KUBERNETES_RO_PORT_8082_TCP_PROTO=tcp",
				"KUBERNETES_RO_PORT_8082_TCP_PORT=8082",
				"KUBERNETES_RO_PORT_8082_TCP_ADDR=1.2.3.2"),
			22,
		},
		{
			"master service in pod ns",
			"test2",
			&api.Container{
				Env: []api.EnvVar{
					{Name: "FOO", Value: "ZAP"},
				},
			},
			"kubernetes",
			false,
			util.NewStringSet("FOO=ZAP",
				"TEST_SERVICE_HOST=1.2.3.5",
				"TEST_SERVICE_PORT=8085",
				"TEST_PORT=tcp://1.2.3.5:8085",
				"TEST_PORT_8085_TCP=tcp://1.2.3.5:8085",
				"TEST_PORT_8085_TCP_PROTO=tcp",
				"TEST_PORT_8085_TCP_PORT=8085",
				"TEST_PORT_8085_TCP_ADDR=1.2.3.5",
				"KUBERNETES_SERVICE_HOST=1.2.3.4",
				"KUBERNETES_SERVICE_PORT=8084",
				"KUBERNETES_PORT=tcp://1.2.3.4:8084",
				"KUBERNETES_PORT_8084_TCP=tcp://1.2.3.4:8084",
				"KUBERNETES_PORT_8084_TCP_PROTO=tcp",
				"KUBERNETES_PORT_8084_TCP_PORT=8084",
				"KUBERNETES_PORT_8084_TCP_ADDR=1.2.3.4",
				"KUBERNETES_RO_SERVICE_HOST=1.2.3.7",
				"KUBERNETES_RO_SERVICE_PORT=8087",
				"KUBERNETES_RO_PORT=tcp://1.2.3.7:8087",
				"KUBERNETES_RO_PORT_8087_TCP=tcp://1.2.3.7:8087",
				"KUBERNETES_RO_PORT_8087_TCP_PROTO=tcp",
				"KUBERNETES_RO_PORT_8087_TCP_PORT=8087",
				"KUBERNETES_RO_PORT_8087_TCP_ADDR=1.2.3.7"),
			22,
		},
		{
			"pod in master service ns",
			"kubernetes",
			&api.Container{},
			"kubernetes",
			false,
			util.NewStringSet(
				"NOT_SPECIAL_SERVICE_HOST=1.2.3.8",
				"NOT_SPECIAL_SERVICE_PORT=8088",
				"NOT_SPECIAL_PORT=tcp://1.2.3.8:8088",
				"NOT_SPECIAL_PORT_8088_TCP=tcp://1.2.3.8:8088",
				"NOT_SPECIAL_PORT_8088_TCP_PROTO=tcp",
				"NOT_SPECIAL_PORT_8088_TCP_PORT=8088",
				"NOT_SPECIAL_PORT_8088_TCP_ADDR=1.2.3.8",
				"KUBERNETES_SERVICE_HOST=1.2.3.6",
				"KUBERNETES_SERVICE_PORT=8086",
				"KUBERNETES_PORT=tcp://1.2.3.6:8086",
				"KUBERNETES_PORT_8086_TCP=tcp://1.2.3.6:8086",
				"KUBERNETES_PORT_8086_TCP_PROTO=tcp",
				"KUBERNETES_PORT_8086_TCP_PORT=8086",
				"KUBERNETES_PORT_8086_TCP_ADDR=1.2.3.6",
				"KUBERNETES_RO_SERVICE_HOST=1.2.3.7",
				"KUBERNETES_RO_SERVICE_PORT=8087",
				"KUBERNETES_RO_PORT=tcp://1.2.3.7:8087",
				"KUBERNETES_RO_PORT_8087_TCP=tcp://1.2.3.7:8087",
				"KUBERNETES_RO_PORT_8087_TCP_PROTO=tcp",
				"KUBERNETES_RO_PORT_8087_TCP_PORT=8087",
				"KUBERNETES_RO_PORT_8087_TCP_ADDR=1.2.3.7"),
			21,
		},
	}

	for _, tc := range testCases {
		kl, _ := newTestKubelet(t)
		kl.masterServiceNamespace = tc.masterServiceNamespace
		if tc.nilLister {
			kl.serviceLister = nil
		} else {
			kl.serviceLister = testServiceLister{services}
		}

		result, err := kl.makeEnvironmentVariables(tc.ns, tc.container)
		if err != nil {
			t.Errorf("[%v] Unexpected error: %v", tc.name, err)
		}

		resultSet := util.NewStringSet(result...)
		if !resultSet.IsSuperset(tc.expectedEnvs) {
			t.Errorf("[%v] Unexpected env entries; expected {%v}, got {%v}", tc.name, tc.expectedEnvs, resultSet)
		}

		if a := len(resultSet); a != tc.expectedEnvSize {
			t.Errorf("[%v] Unexpected number of env vars; expected %v, got %v", tc.name, tc.expectedEnvSize, a)
		}
	}
}
