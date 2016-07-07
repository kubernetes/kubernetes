/*
Copyright 2016 The Kubernetes Authors.

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

package framework

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/gomega"
)

// TODO: Consolidate pod-specific framework functions here.

// Convenience method for getting a pod client interface in the framework's namespace.
func (f *Framework) PodClient() unversioned.PodInterface {
	return f.Client.Pods(f.Namespace.Name)
}

// Create a new pod according to the framework specifications, and wait for it to start.
func (f *Framework) CreatePod(pod *api.Pod) {
	f.CreatePodAsync(pod)
	ExpectNoError(f.WaitForPodRunning(pod.Name))
}

// Create a new pod according to the framework specifications (don't wait for it to start).
func (f *Framework) CreatePodAsync(pod *api.Pod) {
	f.MungePodSpec(pod)
	_, err := f.PodClient().Create(pod)
	ExpectNoError(err, "Error creating Pod")
}

// Batch version of CreatePod. All pods are created before waiting.
func (f *Framework) CreatePods(pods []*api.Pod) {
	for _, pod := range pods {
		f.CreatePodAsync(pod)
	}
	var wg sync.WaitGroup
	for _, pod := range pods {
		wg.Add(1)
		podName := pod.Name
		go func() {
			ExpectNoError(f.WaitForPodRunning(podName))
			wg.Done()
		}()
	}
	wg.Wait()
}

// Apply test-suite specific transformations to the pod spec.
// TODO: figure out a nicer, more generic way to tie this to framework instances.
func (f *Framework) MungePodSpec(pod *api.Pod) {
	if TestContext.NodeName != "" {
		Expect(pod.Spec.NodeName).To(Or(BeZero(), Equal(TestContext.NodeName)), "Test misconfigured")
		pod.Spec.NodeName = TestContext.NodeName
	}
}

// UpdatePod updates the pod object. It retries if there is a conflict, throw out error if
// there is any other errors. name is the pod name, updateFn is the function updating the
// pod object.
func (f *Framework) UpdatePod(name string, updateFn func(pod *api.Pod)) {
	ExpectNoError(wait.Poll(time.Millisecond*500, time.Second*30, func() (bool, error) {
		pod, err := f.PodClient().Get(name)
		if err != nil {
			return false, fmt.Errorf("failed to get pod %q: %v", name, err)
		}
		updateFn(pod)
		_, err = f.PodClient().Update(pod)
		if err == nil {
			Logf("Successfully updated pod %q", name)
			return true, nil
		}
		if errors.IsConflict(err) {
			Logf("Conflicting update to pod %q, re-get and re-update: %v", name, err)
			return false, nil
		}
		return false, fmt.Errorf("failed to update pod %q: %v", name, err)
	}))
}
