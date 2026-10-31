/*
Copyright 2023 The Kubernetes Authors.

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

package kuberuntime

import (
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2"
	"k8s.io/ktesting"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
)

func TestTerminationOrderingSidecarStopAfterMain(t *testing.T) {
	synctest.Test(t, testTerminationOrderingSidecarStopAfterMain)
}
func testTerminationOrderingSidecarStopAfterMain(t *testing.T) {
	restartPolicy := v1.ContainerRestartPolicyAlways
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name:            "init",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					RestartPolicy:   &restartPolicy,
				},
			},
			Containers: []v1.Container{
				{
					Name:            "main",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
				},
			},
		},
	}
	to := newTerminationOrdering(pod, getContainerNames(pod))

	var wg sync.WaitGroup
	wg.Add(1)
	var sidecarWaitDelay, mainWaitDelay int64
	syncChannel := make(chan struct{})

	go func() {
		close(syncChannel) // notifies other goroutine that this goroutine scheduled.
		sidecarWaitDelay = int64(to.waitForTurn("init", 30))
		to.containerTerminated("init")
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		<-syncChannel // make sure the other goroutine is scheduled.
		mainWaitDelay = int64(to.waitForTurn("main", 0))
		time.Sleep(1 * time.Second)
		to.containerTerminated("main")
		wg.Done()
	}()
	wg.Wait()
	if sidecarWaitDelay != 1 {
		t.Errorf("expected sidecar to wait for main container to exit, got delay of %d", sidecarWaitDelay)
	}
	if mainWaitDelay != 0 {
		t.Errorf("expected main container to not wait to exit, got delay of %d", mainWaitDelay)
	}
}

func TestTerminationOrderingSidecarsInReverseOrder(t *testing.T) {
	synctest.Test(t, testTerminationOrderingSidecarsInReverseOrder)
}
func testTerminationOrderingSidecarsInReverseOrder(t *testing.T) {
	restartPolicy := v1.ContainerRestartPolicyAlways
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name:            "sc1",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					RestartPolicy:   &restartPolicy,
				},
				{
					Name:            "sc2",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					RestartPolicy:   &restartPolicy,
				},
				{
					Name:            "sc3",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					RestartPolicy:   &restartPolicy,
				},
			},
			Containers: []v1.Container{
				{
					Name:            "main",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
				},
			},
		},
	}
	to := newTerminationOrdering(pod, getContainerNames(pod))

	var wg sync.WaitGroup
	var delays sync.Map

	waitAndExit := func(name string) {
		delay := int64(to.waitForTurn(name, 30))
		delays.Store(name, delay)
		time.Sleep(1250 * time.Millisecond)
		to.containerTerminated(name)
		wg.Done()
	}
	for _, ic := range pod.Spec.InitContainers {
		wg.Add(1)
		go waitAndExit(ic.Name)
	}
	for _, c := range pod.Spec.Containers {
		wg.Add(1)
		go waitAndExit(c.Name)
	}

	// wait for our simulated containers to exit
	wg.Wait()

	getDelay := func(name string) int64 {
		delay, ok := delays.Load(name)
		if !ok {
			t.Errorf("unable to find delay for container %s", name)
		}
		return delay.(int64)
	}

	for _, tc := range []struct {
		containerName string
		expectedDelay int64
	}{
		// sidecars should exit in reverse order, so
		// sc1 = 3 (main container + sc3 + sc2)
		{
			containerName: "sc1",
			expectedDelay: 3,
		},
		// sc2 = 2 (main container + sc3)
		{
			containerName: "sc2",
			expectedDelay: 2,
		},
		// sc3 = 1 (main container)
		{
			containerName: "sc3",
			expectedDelay: 1,
		},
		// main container = 0 delay, nothing to wait on
		{
			containerName: "main",
			expectedDelay: 0,
		},
	} {
		if got := getDelay(tc.containerName); got != tc.expectedDelay {
			t.Errorf("expected delay for container %s = %d, got %d", tc.containerName, tc.expectedDelay, got)
		}
	}
}

func TestTerminationOrderingObeysGrace(t *testing.T) {
	synctest.Test(t, testTerminationOrderingObeysGrace)
}
func testTerminationOrderingObeysGrace(t *testing.T) {
	restartPolicy := v1.ContainerRestartPolicyAlways
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID:       "12345678",
			Name:      "bar",
			Namespace: "new",
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name:            "sc1",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					RestartPolicy:   &restartPolicy,
				},
				{
					Name:            "sc2",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					RestartPolicy:   &restartPolicy,
				},
				{
					Name:            "sc3",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
					RestartPolicy:   &restartPolicy,
				},
			},
			Containers: []v1.Container{
				{
					Name:            "main",
					Image:           "busybox",
					ImagePullPolicy: v1.PullIfNotPresent,
				},
			},
		},
	}
	to := newTerminationOrdering(pod, getContainerNames(pod))

	var wg sync.WaitGroup
	var delays sync.Map

	waitAndExit := func(name string) {
		// just a two second grace period which is not long enough for all of the waits to finish
		delay := int64(to.waitForTurn(name, 2))
		delays.Store(name, delay)
		time.Sleep(1 * time.Second)
		to.containerTerminated(name)
		wg.Done()
	}
	for _, ic := range pod.Spec.InitContainers {
		wg.Add(1)
		go waitAndExit(ic.Name)
	}
	for _, c := range pod.Spec.Containers {
		wg.Add(1)
		go waitAndExit(c.Name)
	}

	// wait for our simulated containers to exit
	wg.Wait()

	getDelay := func(name string) int64 {
		delay, ok := delays.Load(name)
		if !ok {
			t.Errorf("unable to find delay for container %s", name)
		}
		return delay.(int64)
	}

	for _, tc := range []struct {
		containerName string
		expectedDelay int64
	}{
		{
			containerName: "sc1",
			// overall grace period limits the amount of time waited here
			expectedDelay: 2,
		},
		{
			containerName: "sc2",
			expectedDelay: 2,
		},
		{
			containerName: "sc3",
			expectedDelay: 1,
		},
		{
			containerName: "main",
			expectedDelay: 0,
		},
	} {
		if got := getDelay(tc.containerName); got != tc.expectedDelay {
			t.Errorf("expected delay for container %s = %d, got %d", tc.containerName, tc.expectedDelay, got)
		}
	}
}

func TestTerminationOrderingPod(t *testing.T) {
	restartPolicy := v1.ContainerRestartPolicyAlways
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "12345678", Name: "bar", Namespace: "new"},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{{Name: "sidecar", Image: "busybox", RestartPolicy: &restartPolicy}},
			Containers:     []v1.Container{{Name: "main", Image: "busybox"}},
		},
	}
	removed := v1.Container{Name: "removed", Image: "busybox"}
	running := []string{"sidecar", "main", "removed"}

	for name, tc := range map[string]struct {
		disabled bool
		// record sets up the actuated state.
		record  func(t *testing.T, m *kubeGenericRuntimeManager)
		running []string
		// wantRemoved is whether the removed container is added to the pod.
		wantRemoved bool
	}{
		"running removed container": {
			record: func(t *testing.T, m *kubeGenericRuntimeManager) {
				require.NoError(t, m.actuatedState.SetContainer(klog.Background(), pod.UID, podutil.Containers, removed))
			},
			running:     running,
			wantRemoved: true,
		},
		"removed container not running": {
			record: func(t *testing.T, m *kubeGenericRuntimeManager) {
				require.NoError(t, m.actuatedState.SetContainer(klog.Background(), pod.UID, podutil.Containers, removed))
			},
			running: []string{"sidecar", "main"},
		},
		"no record": {
			running: running,
		},
		"resources-only record": {
			// e.g. written by an older kubelet.
			record: func(t *testing.T, m *kubeGenericRuntimeManager) {
				require.NoError(t, m.actuatedState.SetContainerResources(klog.Background(), pod.UID, "removed", podutil.Containers, v1.ResourceRequirements{}))
			},
			running: running,
		},
		"gate disabled": {
			disabled: true,
			record: func(t *testing.T, m *kubeGenericRuntimeManager) {
				require.NoError(t, m.actuatedState.SetContainer(klog.Background(), pod.UID, podutil.Containers, removed))
			},
			running:     running,
			wantRemoved: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			setDynamicContainersGate(t, !tc.disabled)
			_, _, m, err := createTestRuntimeManager(ktesting.Init(t))
			require.NoError(t, err)
			if tc.record != nil {
				tc.record(t, m)
			}

			got := m.terminationOrderingPod(pod, tc.running)
			if tc.wantRemoved {
				// The removed container is added as a main container, so sidecars wait for it.
				assert.Equal(t, []v1.Container{pod.Spec.Containers[0], removed}, got.Spec.Containers)
				assert.Equal(t, pod.Spec.InitContainers, got.Spec.InitContainers)
				assert.Len(t, pod.Spec.Containers, 1, "the given pod is unchanged")
			} else {
				assert.Same(t, pod, got)
			}
		})
	}
	assert.Nil(t, (&kubeGenericRuntimeManager{}).terminationOrderingPod(nil, running))
}

func getContainerNames(p *v1.Pod) []string {
	var running []string
	for _, ic := range p.Spec.InitContainers {
		running = append(running, ic.Name)
	}
	for _, c := range p.Spec.Containers {
		running = append(running, c.Name)
	}
	return running
}
