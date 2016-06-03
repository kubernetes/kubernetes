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
	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/unversioned"

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
