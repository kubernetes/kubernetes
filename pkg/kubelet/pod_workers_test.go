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
	"sync"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/dockertools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/network"
	kubeletProber "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/prober"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
)

func newPod(uid, name string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:  types.UID(uid),
			Name: name,
		},
	}
}

func createPodWorkers() (*podWorkers, map[types.UID][]string) {
	fakeDocker := &dockertools.FakeDockerClient{}
	fakeRecorder := &record.FakeRecorder{}
	np, _ := network.InitNetworkPlugin([]network.NetworkPlugin{}, "", network.NewFakeHost(nil))
	dockerManager := dockertools.NewDockerManager(fakeDocker, fakeRecorder, nil, nil, dockertools.PodInfraContainerImage, 0, 0, "", kubecontainer.FakeOS{}, np, &kubeletProber.FakeProber{}, nil, nil, newKubeletRuntimeHooks(fakeRecorder))
	fakeRuntimeCache := kubecontainer.NewFakeRuntimeCache(dockerManager)

	lock := sync.Mutex{}
	processed := make(map[types.UID][]string)

	podWorkers := newPodWorkers(
		fakeRuntimeCache,
		func(pod *api.Pod, mirrorPod *api.Pod, runningPod kubecontainer.Pod) error {
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
