/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Framework supports common operations used by e2e tests; it will keep a client & a namespace for you.
// Eventual goal is to merge this with integration test framework.
type Framework struct {
	BaseName string

	Namespace *api.Namespace
	Client    *client.Client
}

// NewFramework makes a new framework and sets up a BeforeEach/AfterEach for
// you (you can write additional before/after each functions).
func NewFramework(baseName string) *Framework {
	f := &Framework{
		BaseName: baseName,
	}

	BeforeEach(f.beforeEach)
	AfterEach(f.afterEach)

	return f
}

// beforeEach gets a client and makes a namespace.
func (f *Framework) beforeEach() {
	By("Creating a kubernetes client")
	c, err := loadClient()
	Expect(err).NotTo(HaveOccurred())

	f.Client = c

	By("Building a namespace api object")
	namespace, err := createTestingNS(f.BaseName, f.Client)
	Expect(err).NotTo(HaveOccurred())

	f.Namespace = namespace

	By("Waiting for a default service account to be provisioned in namespace")
	err = waitForDefaultServiceAccountInNamespace(c, namespace.Name)
	Expect(err).NotTo(HaveOccurred())
}

// afterEach deletes the namespace, after reading its events.
func (f *Framework) afterEach() {
	// Print events if the test failed.
	if CurrentGinkgoTestDescription().Failed {
		By(fmt.Sprintf("Collecting events from namespace %q.", f.Namespace.Name))
		events, err := f.Client.Events(f.Namespace.Name).List(labels.Everything(), fields.Everything())
		Expect(err).NotTo(HaveOccurred())

		for _, e := range events.Items {
			Logf("event for %v: %v %v: %v", e.InvolvedObject.Name, e.Source, e.Reason, e.Message)
		}
		// Note that we don't wait for any cleanup to propagate, which means
		// that if you delete a bunch of pods right before ending your test,
		// you may or may not see the killing/deletion/cleanup events.
	}

	By(fmt.Sprintf("Destroying namespace %q for this suite.", f.Namespace.Name))
	if err := f.Client.Namespaces().Delete(f.Namespace.Name); err != nil {
		Failf("Couldn't delete ns %q: %s", f.Namespace.Name, err)
	}
	// Paranoia-- prevent reuse!
	f.Namespace = nil
	f.Client = nil
}

// WaitForPodRunning waits for the pod to run in the namespace.
func (f *Framework) WaitForPodRunning(podName string) error {
	return waitForPodRunningInNamespace(f.Client, podName, f.Namespace.Name)
}

// Runs the given pod and verifies that its output matches the desired output.
func (f *Framework) TestContainerOutput(scenarioName string, pod *api.Pod, expectedOutput []string) {
	testContainerOutputInNamespace(scenarioName, f.Client, pod, expectedOutput, f.Namespace.Name)
}
