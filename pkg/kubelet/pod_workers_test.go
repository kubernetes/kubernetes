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
	"reflect"
	"sort"
	"sync"
	"testing"
	"time"

	docker "github.com/fsouza/go-dockerclient"
	cadvisorApi "github.com/google/cadvisor/info/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/types"
)

func newPod(uid, name string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:  types.UID(uid),
			Name: name,
		},
	}
}

func createFakeRuntimeCache(fakeRecorder *record.FakeRecorder) kubecontainer.RuntimeCache {
	fakeDocker := &dockertools.FakeDockerClient{}
	np, _ := network.InitNetworkPlugin([]network.NetworkPlugin{}, "", network.NewFakeHost(nil))
	dockerManager := dockertools.NewFakeDockerManager(fakeDocker, fakeRecorder, nil, nil, &cadvisorApi.MachineInfo{}, dockertools.PodInfraContainerImage, 0, 0, "", kubecontainer.FakeOS{}, np, nil, nil)
	return kubecontainer.NewFakeRuntimeCache(dockerManager)
}

func createPodWorkers() (*podWorkers, map[types.UID][]string) {
	lock := sync.Mutex{}
	processed := make(map[types.UID][]string)
	fakeRecorder := &record.FakeRecorder{}
	fakeRuntimeCache := createFakeRuntimeCache(fakeRecorder)
	podWorkers := newPodWorkers(
		fakeRuntimeCache,
		func(pod *api.Pod, mirrorPod *api.Pod, runningPod kubecontainer.Pod, updateType SyncPodType) error {
			func() {
				lock.Lock()
				defer lock.Unlock()
				processed[pod.UID] = append(processed[pod.UID], pod.Name)
			}()
			return nil
		},
		fakeRecorder,
	)
	return podWorkers, processed
}

func drainWorkers(podWorkers *podWorkers, numPods int) {
	for {
		stillWorking := false
		podWorkers.podLock.Lock()
		for i := 0; i < numPods; i++ {
			if podWorkers.isWorking[types.UID(string(i))] {
				stillWorking = true
			}
		}
		podWorkers.podLock.Unlock()
		if !stillWorking {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
}

func TestUpdatePod(t *testing.T) {
	podWorkers, processed := createPodWorkers()

	// Check whether all pod updates will be processed.
	numPods := 20
	for i := 0; i < numPods; i++ {
		for j := i; j < numPods; j++ {
			podWorkers.UpdatePod(newPod(string(j), string(i)), nil, func() {})
		}
	}
	drainWorkers(podWorkers, numPods)

	if len(processed) != 20 {
		t.Errorf("Not all pods processed: %v", len(processed))
		return
	}
	for i := 0; i < numPods; i++ {
		uid := types.UID(i)
		if len(processed[uid]) < 1 || len(processed[uid]) > i+1 {
			t.Errorf("Pod %v processed %v times", i, len(processed[uid]))
			continue
		}

		first := 0
		last := len(processed[uid]) - 1
		if processed[uid][first] != string(0) {
			t.Errorf("Pod %v: incorrect order %v, %v", i, first, processed[uid][first])

		}
		if processed[uid][last] != string(i) {
			t.Errorf("Pod %v: incorrect order %v, %v", i, last, processed[uid][last])
		}
	}
}

func TestUpdateType(t *testing.T) {
	syncType := make(chan SyncPodType)
	fakeRecorder := &record.FakeRecorder{}
	podWorkers := newPodWorkers(
		createFakeRuntimeCache(fakeRecorder),
		func(pod *api.Pod, mirrorPod *api.Pod, runningPod kubecontainer.Pod, updateType SyncPodType) error {
			func() {
				syncType <- updateType
			}()
			return nil
		},
		fakeRecorder,
	)
	cases := map[*api.Pod][]SyncPodType{
		newPod("u1", "n1"): {SyncPodCreate, SyncPodUpdate},
		newPod("u2", "n1"): {SyncPodCreate},
	}
	for p, expectedTypes := range cases {
		for i := range expectedTypes {
			podWorkers.UpdatePod(p, nil, func() {})
			select {
			case gotType := <-syncType:
				if gotType != expectedTypes[i] {
					t.Fatalf("Expected sync type %v got %v for pod with uid %v", expectedTypes[i], gotType, p.UID)
				}
			case <-time.After(100 * time.Millisecond):
				t.Errorf("Unexpected delay is running pod worker")
			}
		}
	}
}

func TestForgetNonExistingPodWorkers(t *testing.T) {
	podWorkers, _ := createPodWorkers()

	numPods := 20
	for i := 0; i < numPods; i++ {
		podWorkers.UpdatePod(newPod(string(i), "name"), nil, func() {})
	}
	drainWorkers(podWorkers, numPods)

	if len(podWorkers.podUpdates) != numPods {
		t.Errorf("Incorrect number of open channels %v", len(podWorkers.podUpdates))
	}

	desiredPods := map[types.UID]empty{}
	desiredPods[types.UID(2)] = empty{}
	desiredPods[types.UID(14)] = empty{}
	podWorkers.ForgetNonExistingPodWorkers(desiredPods)
	if len(podWorkers.podUpdates) != 2 {
		t.Errorf("Incorrect number of open channels %v", len(podWorkers.podUpdates))
	}
	if _, exists := podWorkers.podUpdates[types.UID(2)]; !exists {
		t.Errorf("No updates channel for pod 2")
	}
	if _, exists := podWorkers.podUpdates[types.UID(14)]; !exists {
		t.Errorf("No updates channel for pod 14")
	}

	podWorkers.ForgetNonExistingPodWorkers(map[types.UID]empty{})
	if len(podWorkers.podUpdates) != 0 {
		t.Errorf("Incorrect number of open channels %v", len(podWorkers.podUpdates))
	}
}

type simpleFakeKubelet struct {
	pod        *api.Pod
	mirrorPod  *api.Pod
	runningPod kubecontainer.Pod

	wg sync.WaitGroup
}

func (kl *simpleFakeKubelet) syncPod(pod *api.Pod, mirrorPod *api.Pod, runningPod kubecontainer.Pod, updateType SyncPodType) error {
	kl.pod, kl.mirrorPod, kl.runningPod = pod, mirrorPod, runningPod
	return nil
}

func (kl *simpleFakeKubelet) syncPodWithWaitGroup(pod *api.Pod, mirrorPod *api.Pod, runningPod kubecontainer.Pod, updateType SyncPodType) error {
	kl.pod, kl.mirrorPod, kl.runningPod = pod, mirrorPod, runningPod
	kl.wg.Done()
	return nil
}

// byContainerName sort the containers in a running pod by their names.
type byContainerName kubecontainer.Pod

func (b byContainerName) Len() int { return len(b.Containers) }

func (b byContainerName) Swap(i, j int) {
	b.Containers[i], b.Containers[j] = b.Containers[j], b.Containers[i]
}

func (b byContainerName) Less(i, j int) bool {
	return b.Containers[i].Name < b.Containers[j].Name
}

// TestFakePodWorkers verifies that the fakePodWorkers behaves the same way as the real podWorkers
// for their invocation of the syncPodFn.
func TestFakePodWorkers(t *testing.T) {
	// Create components for pod workers.
	fakeDocker := &dockertools.FakeDockerClient{}
	fakeRecorder := &record.FakeRecorder{}
	np, _ := network.InitNetworkPlugin([]network.NetworkPlugin{}, "", network.NewFakeHost(nil))
	dockerManager := dockertools.NewFakeDockerManager(fakeDocker, fakeRecorder, nil, nil, &cadvisorApi.MachineInfo{}, dockertools.PodInfraContainerImage, 0, 0, "", kubecontainer.FakeOS{}, np, nil, nil)
	fakeRuntimeCache := kubecontainer.NewFakeRuntimeCache(dockerManager)

	kubeletForRealWorkers := &simpleFakeKubelet{}
	kubeletForFakeWorkers := &simpleFakeKubelet{}

	realPodWorkers := newPodWorkers(fakeRuntimeCache, kubeletForRealWorkers.syncPodWithWaitGroup, fakeRecorder)
	fakePodWorkers := &fakePodWorkers{kubeletForFakeWorkers.syncPod, fakeRuntimeCache, t}

	tests := []struct {
		pod                    *api.Pod
		mirrorPod              *api.Pod
		containerList          []docker.APIContainers
		containersInRunningPod int
	}{
		{
			&api.Pod{},
			&api.Pod{},
			[]docker.APIContainers{},
			0,
		},

		{
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					UID:       "12345678",
					Name:      "foo",
					Namespace: "new",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "fooContainer",
						},
					},
				},
			},
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					UID:       "12345678",
					Name:      "fooMirror",
					Namespace: "new",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "fooContainerMirror",
						},
					},
				},
			},
			[]docker.APIContainers{
				{
					// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>_<random>
					Names: []string{"/k8s_bar.hash123_foo_new_12345678_0"},
					ID:    "1234",
				},
				{
					// pod infra container
					Names: []string{"/k8s_POD.hash123_foo_new_12345678_0"},
					ID:    "9876",
				},
			},
			2,
		},

		{
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					UID:       "98765",
					Name:      "bar",
					Namespace: "new",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "fooContainer",
						},
					},
				},
			},
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					UID:       "98765",
					Name:      "fooMirror",
					Namespace: "new",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "fooContainerMirror",
						},
					},
				},
			},
			[]docker.APIContainers{
				{
					// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>_<random>
					Names: []string{"/k8s_bar.hash123_bar_new_98765_0"},
					ID:    "1234",
				},
				{
					// pod infra container
					Names: []string{"/k8s_POD.hash123_foo_new_12345678_0"},
					ID:    "9876",
				},
			},
			1,
		},

		// Empty running pod.
		{
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					UID:       "98765",
					Name:      "baz",
					Namespace: "new",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bazContainer",
						},
					},
				},
			},
			&api.Pod{
				ObjectMeta: api.ObjectMeta{
					UID:       "98765",
					Name:      "bazMirror",
					Namespace: "new",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name: "bazContainerMirror",
						},
					},
				},
			},
			[]docker.APIContainers{
				{
					// format is // k8s_<container-id>_<pod-fullname>_<pod-uid>_<random>
					Names: []string{"/k8s_bar.hash123_bar_new_12345678_0"},
					ID:    "1234",
				},
				{
					// pod infra container
					Names: []string{"/k8s_POD.hash123_foo_new_12345678_0"},
					ID:    "9876",
				},
			},
			0,
		},
	}

	for i, tt := range tests {
		kubeletForRealWorkers.wg.Add(1)

		fakeDocker.ContainerList = tt.containerList
		realPodWorkers.UpdatePod(tt.pod, tt.mirrorPod, func() {})
		fakePodWorkers.UpdatePod(tt.pod, tt.mirrorPod, func() {})

		kubeletForRealWorkers.wg.Wait()

		if !reflect.DeepEqual(kubeletForRealWorkers.pod, kubeletForFakeWorkers.pod) {
			t.Errorf("%d: Expected: %#v, Actual: %#v", i, kubeletForRealWorkers.pod, kubeletForFakeWorkers.pod)
		}

		if !reflect.DeepEqual(kubeletForRealWorkers.mirrorPod, kubeletForFakeWorkers.mirrorPod) {
			t.Errorf("%d: Expected: %#v, Actual: %#v", i, kubeletForRealWorkers.mirrorPod, kubeletForFakeWorkers.mirrorPod)
		}

		if tt.containersInRunningPod != len(kubeletForFakeWorkers.runningPod.Containers) {
			t.Errorf("%d: Expected: %#v, Actual: %#v", i, tt.containersInRunningPod, len(kubeletForFakeWorkers.runningPod.Containers))
		}

		sort.Sort(byContainerName(kubeletForRealWorkers.runningPod))
		sort.Sort(byContainerName(kubeletForFakeWorkers.runningPod))
		if !reflect.DeepEqual(kubeletForRealWorkers.runningPod, kubeletForFakeWorkers.runningPod) {
			t.Errorf("%d: Expected: %#v, Actual: %#v", i, kubeletForRealWorkers.runningPod, kubeletForFakeWorkers.runningPod)
		}
	}
}
