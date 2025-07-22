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
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestTerminationOrderingSidecarStopAfterMain(t *testing.T) {
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
	var sidecarWaitDelay int64
	var mainWaitDelay int64
	go func() {
		sidecarWaitDelay = int64(to.waitForTurn("init", 30))
		to.containerTerminated("init")
		wg.Done()
	}()

	wg.Add(1)
	go func() {
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
