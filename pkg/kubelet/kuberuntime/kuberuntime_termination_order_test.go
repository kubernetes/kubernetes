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
	"context"
	"sync"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
		sidecarWaitDelay = int64(to.waitForTurn(context.Background(), "init", 30))
		to.containerTerminated("init")
		wg.Done()
	}()

	wg.Add(1)
	go func() {
		<-syncChannel // make sure the other goroutine is scheduled.
		mainWaitDelay = int64(to.waitForTurn(context.Background(), "main", 0))
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
		delay := int64(to.waitForTurn(context.Background(), name, 30))
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
		delay := int64(to.waitForTurn(context.Background(), name, 2))
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

func TestTerminationOrderingCancellation(t *testing.T) {
	for _, tc := range []struct {
		name         string
		container    string
		mainStopped  bool
		cancelBefore bool
		gracePeriod  int64
	}{
		{name: "waiting for main", container: "sidecar-1", gracePeriod: 100},
		{name: "waiting for another sidecar", container: "sidecar-1", mainStopped: true, gracePeriod: 100},
		{name: "already canceled", container: "sidecar-1", cancelBefore: true, gracePeriod: 100},
		{name: "canceled without grace", container: "sidecar-1", cancelBefore: true},
		{name: "canceled without predecessors", container: "main", cancelBefore: true, gracePeriod: 100},
	} {
		t.Run(tc.name, func(t *testing.T) {
			synctest.Test(t, func(t *testing.T) {
				pod := newSidecarTerminationTestPod()
				ordering := newTerminationOrdering(pod, getContainerNames(pod))
				if tc.mainStopped {
					ordering.containerTerminated("main")
				}
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				if tc.cancelBefore {
					cancel()
				}
				done := make(chan struct{})
				start := time.Now()
				go func() {
					ordering.waitForTurn(ctx, tc.container, tc.gracePeriod)
					close(done)
				}()
				synctest.Wait()
				cancel()
				synctest.Wait()
				select {
				case <-done:
				default:
					t.Fatal("termination ordering did not return after cancellation")
				}
				require.Zero(t, time.Since(start), "cancellation must not wait for the original grace period")
				for name, notification := range ordering.terminated {
					select {
					case <-notification:
						t.Fatalf("cancellation marked container %q as terminated", name)
					default:
					}
				}
			})
		})
	}
}

func TestTerminationOrderingContextDeadline(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		pod := newSidecarTerminationTestPod()
		ordering := newTerminationOrdering(pod, getContainerNames(pod))
		ctx, cancel := context.WithTimeout(context.Background(), time.Second)
		defer cancel()
		waitTime := ordering.waitForTurn(ctx, "sidecar-1", 100)
		require.ErrorIs(t, ctx.Err(), context.DeadlineExceeded)
		require.InDelta(t, 1, waitTime, 0.001)
	})
}

func newSidecarTerminationTestPod() *v1.Pod {
	restartPolicy := v1.ContainerRestartPolicyAlways
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: "sidecar-pod", Name: "sidecar-pod", Namespace: "test"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{{Name: "main"}},
			InitContainers: []v1.Container{
				{Name: "sidecar-1", RestartPolicy: &restartPolicy},
				{Name: "sidecar-2", RestartPolicy: &restartPolicy},
			},
		},
	}
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
