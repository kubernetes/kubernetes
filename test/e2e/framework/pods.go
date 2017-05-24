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
	"regexp"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// ImageWhiteList is the images used in the current test suite. It should be initialized in test suite and
// the images in the white list should be pre-pulled in the test suite.  Currently, this is only used by
// node e2e test.
var ImageWhiteList sets.String

// Convenience method for getting a pod client interface in the framework's namespace,
// possibly applying test-suite specific transformations to the pod spec, e.g. for
// node e2e pod scheduling.
func (f *Framework) PodClient() *PodClient {
	return &PodClient{
		f:            f,
		PodInterface: f.ClientSet.Core().Pods(f.Namespace.Name),
	}
}

type PodClient struct {
	f *Framework
	unversionedcore.PodInterface
}

// Create creates a new pod according to the framework specifications (don't wait for it to start).
func (c *PodClient) Create(pod *api.Pod) *api.Pod {
	c.mungeSpec(pod)
	p, err := c.PodInterface.Create(pod)
	ExpectNoError(err, "Error creating Pod")
	return p
}

// CreateSync creates a new pod according to the framework specifications, and wait for it to start.
func (c *PodClient) CreateSync(pod *api.Pod) *api.Pod {
	p := c.Create(pod)
	ExpectNoError(c.f.WaitForPodRunning(p.Name))
	// Get the newest pod after it becomes running, some status may change after pod created, such as pod ip.
	p, err := c.Get(p.Name)
	ExpectNoError(err)
	return p
}

// CreateBatch create a batch of pods. All pods are created before waiting.
func (c *PodClient) CreateBatch(pods []*api.Pod) []*api.Pod {
	ps := make([]*api.Pod, len(pods))
	var wg sync.WaitGroup
	for i, pod := range pods {
		wg.Add(1)
		go func(i int, pod *api.Pod) {
			defer wg.Done()
			defer GinkgoRecover()
			ps[i] = c.CreateSync(pod)
		}(i, pod)
	}
	wg.Wait()
	return ps
}

// Update updates the pod object. It retries if there is a conflict, throw out error if
// there is any other errors. name is the pod name, updateFn is the function updating the
// pod object.
func (c *PodClient) Update(name string, updateFn func(pod *api.Pod)) {
	ExpectNoError(wait.Poll(time.Millisecond*500, time.Second*30, func() (bool, error) {
		pod, err := c.PodInterface.Get(name)
		if err != nil {
			return false, fmt.Errorf("failed to get pod %q: %v", name, err)
		}
		updateFn(pod)
		_, err = c.PodInterface.Update(pod)
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

// DeleteSync deletes the pod and wait for the pod to disappear for `timeout`. If the pod doesn't
// disappear before the timeout, it will fail the test.
func (c *PodClient) DeleteSync(name string, options *api.DeleteOptions, timeout time.Duration) {
	err := c.Delete(name, options)
	if err != nil && !errors.IsNotFound(err) {
		Failf("Failed to delete pod %q: %v", name, err)
	}
	Expect(WaitForPodToDisappear(c.f.ClientSet, c.f.Namespace.Name, name, labels.Everything(),
		2*time.Second, timeout)).To(Succeed(), "wait for pod %q to disappear", name)
}

// mungeSpec apply test-suite specific transformations to the pod spec.
func (c *PodClient) mungeSpec(pod *api.Pod) {
	if !TestContext.NodeE2E {
		return
	}

	Expect(pod.Spec.NodeName).To(Or(BeZero(), Equal(TestContext.NodeName)), "Test misconfigured")
	pod.Spec.NodeName = TestContext.NodeName
	// Node e2e does not support the default DNSClusterFirst policy. Set
	// the policy to DNSDefault, which is configured per node.
	pod.Spec.DNSPolicy = api.DNSDefault

	// PrepullImages only works for node e2e now. For cluster e2e, image prepull is not enforced,
	// we should not munge ImagePullPolicy for cluster e2e pods.
	if !TestContext.PrepullImages {
		return
	}
	// If prepull is enabled, munge the container spec to make sure the images are not pulled
	// during the test.
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		if c.ImagePullPolicy == api.PullAlways {
			// If the image pull policy is PullAlways, the image doesn't need to be in
			// the white list or pre-pulled, because the image is expected to be pulled
			// in the test anyway.
			continue
		}
		// If the image policy is not PullAlways, the image must be in the white list and
		// pre-pulled.
		Expect(ImageWhiteList.Has(c.Image)).To(BeTrue(), "Image %q is not in the white list, consider adding it to CommonImageWhiteList in test/e2e/common/util.go or NodeImageWhiteList in test/e2e_node/image_list.go", c.Image)
		// Do not pull images during the tests because the images in white list should have
		// been prepulled.
		c.ImagePullPolicy = api.PullNever
	}
}

// TODO(random-liu): Move pod wait function into this file
// WaitForSuccess waits for pod to success.
func (c *PodClient) WaitForSuccess(name string, timeout time.Duration) {
	f := c.f
	Expect(waitForPodCondition(f.ClientSet, f.Namespace.Name, name, "success or failure", timeout,
		func(pod *api.Pod) (bool, error) {
			switch pod.Status.Phase {
			case api.PodFailed:
				return true, fmt.Errorf("pod %q failed with reason: %q, message: %q", name, pod.Status.Reason, pod.Status.Message)
			case api.PodSucceeded:
				return true, nil
			default:
				return false, nil
			}
		},
	)).To(Succeed(), "wait for pod %q to success", name)
}

// MatchContainerOutput gest output of a container and match expected regexp in the output.
func (c *PodClient) MatchContainerOutput(name string, containerName string, expectedRegexp string) error {
	f := c.f
	output, err := GetPodLogs(f.ClientSet, f.Namespace.Name, name, containerName)
	if err != nil {
		return fmt.Errorf("failed to get output for container %q of pod %q", containerName, name)
	}
	regex, err := regexp.Compile(expectedRegexp)
	if err != nil {
		return fmt.Errorf("failed to compile regexp %q: %v", expectedRegexp, err)
	}
	if !regex.MatchString(output) {
		return fmt.Errorf("failed to match regexp %q in output %q", expectedRegexp, output)
	}
	return nil
}
