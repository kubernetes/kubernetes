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
	"context"
	"fmt"
	"regexp"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/sysctl"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	// TODO: Remove the following imports (ref: https://github.com/kubernetes/kubernetes/issues/81245)
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

// DefaultPodDeletionTimeout is the default timeout for deleting pod
const DefaultPodDeletionTimeout = 3 * time.Minute

// ImageWhiteList is the images used in the current test suite. It should be initialized in test suite and
// the images in the white list should be pre-pulled in the test suite.  Currently, this is only used by
// node e2e test.
var ImageWhiteList sets.String

// PodClient is a convenience method for getting a pod client interface in the framework's namespace,
// possibly applying test-suite specific transformations to the pod spec, e.g. for
// node e2e pod scheduling.
func (f *Framework) PodClient() *PodClient {
	return &PodClient{
		f:            f,
		PodInterface: f.ClientSet.CoreV1().Pods(f.Namespace.Name),
	}
}

// PodClientNS is a convenience method for getting a pod client interface in an alternative namespace,
// possibly applying test-suite specific transformations to the pod spec, e.g. for
// node e2e pod scheduling.
func (f *Framework) PodClientNS(namespace string) *PodClient {
	return &PodClient{
		f:            f,
		PodInterface: f.ClientSet.CoreV1().Pods(namespace),
	}
}

// PodClient is a struct for pod client.
type PodClient struct {
	f *Framework
	v1core.PodInterface
}

// Create creates a new pod according to the framework specifications (don't wait for it to start).
func (c *PodClient) Create(pod *v1.Pod) *v1.Pod {
	c.mungeSpec(pod)
	p, err := c.PodInterface.Create(context.TODO(), pod, metav1.CreateOptions{})
	ExpectNoError(err, "Error creating Pod")
	return p
}

// CreateSync creates a new pod according to the framework specifications, and wait for it to start.
func (c *PodClient) CreateSync(pod *v1.Pod) *v1.Pod {
	namespace := c.f.Namespace.Name
	p := c.Create(pod)
	ExpectNoError(e2epod.WaitForPodNameRunningInNamespace(c.f.ClientSet, p.Name, namespace))
	// Get the newest pod after it becomes running, some status may change after pod created, such as pod ip.
	p, err := c.Get(context.TODO(), p.Name, metav1.GetOptions{})
	ExpectNoError(err)
	return p
}

// CreateBatch create a batch of pods. All pods are created before waiting.
func (c *PodClient) CreateBatch(pods []*v1.Pod) []*v1.Pod {
	ps := make([]*v1.Pod, len(pods))
	var wg sync.WaitGroup
	for i, pod := range pods {
		wg.Add(1)
		go func(i int, pod *v1.Pod) {
			defer wg.Done()
			defer ginkgo.GinkgoRecover()
			ps[i] = c.CreateSync(pod)
		}(i, pod)
	}
	wg.Wait()
	return ps
}

// Update updates the pod object. It retries if there is a conflict, throw out error if
// there is any other apierrors. name is the pod name, updateFn is the function updating the
// pod object.
func (c *PodClient) Update(name string, updateFn func(pod *v1.Pod)) {
	ExpectNoError(wait.Poll(time.Millisecond*500, time.Second*30, func() (bool, error) {
		pod, err := c.PodInterface.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("failed to get pod %q: %v", name, err)
		}
		updateFn(pod)
		_, err = c.PodInterface.Update(context.TODO(), pod, metav1.UpdateOptions{})
		if err == nil {
			Logf("Successfully updated pod %q", name)
			return true, nil
		}
		if apierrors.IsConflict(err) {
			Logf("Conflicting update to pod %q, re-get and re-update: %v", name, err)
			return false, nil
		}
		return false, fmt.Errorf("failed to update pod %q: %v", name, err)
	}))
}

// DeleteSync deletes the pod and wait for the pod to disappear for `timeout`. If the pod doesn't
// disappear before the timeout, it will fail the test.
func (c *PodClient) DeleteSync(name string, options metav1.DeleteOptions, timeout time.Duration) {
	namespace := c.f.Namespace.Name
	err := c.Delete(context.TODO(), name, options)
	if err != nil && !apierrors.IsNotFound(err) {
		Failf("Failed to delete pod %q: %v", name, err)
	}
	gomega.Expect(e2epod.WaitForPodToDisappear(c.f.ClientSet, namespace, name, labels.Everything(),
		2*time.Second, timeout)).To(gomega.Succeed(), "wait for pod %q to disappear", name)
}

// mungeSpec apply test-suite specific transformations to the pod spec.
func (c *PodClient) mungeSpec(pod *v1.Pod) {
	if !TestContext.NodeE2E {
		return
	}

	gomega.Expect(pod.Spec.NodeName).To(gomega.Or(gomega.BeZero(), gomega.Equal(TestContext.NodeName)), "Test misconfigured")
	pod.Spec.NodeName = TestContext.NodeName
	// Node e2e does not support the default DNSClusterFirst policy. Set
	// the policy to DNSDefault, which is configured per node.
	pod.Spec.DNSPolicy = v1.DNSDefault

	// PrepullImages only works for node e2e now. For cluster e2e, image prepull is not enforced,
	// we should not munge ImagePullPolicy for cluster e2e pods.
	if !TestContext.PrepullImages {
		return
	}
	// If prepull is enabled, munge the container spec to make sure the images are not pulled
	// during the test.
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		if c.ImagePullPolicy == v1.PullAlways {
			// If the image pull policy is PullAlways, the image doesn't need to be in
			// the white list or pre-pulled, because the image is expected to be pulled
			// in the test anyway.
			continue
		}
		// If the image policy is not PullAlways, the image must be in the white list and
		// pre-pulled.
		gomega.Expect(ImageWhiteList.Has(c.Image)).To(gomega.BeTrue(), "Image %q is not in the white list, consider adding it to CommonImageWhiteList in test/e2e/common/util.go or NodeImageWhiteList in test/e2e_node/image_list.go", c.Image)
		// Do not pull images during the tests because the images in white list should have
		// been prepulled.
		c.ImagePullPolicy = v1.PullNever
	}
}

// WaitForSuccess waits for pod to succeed.
// TODO(random-liu): Move pod wait function into this file
func (c *PodClient) WaitForSuccess(name string, timeout time.Duration) {
	f := c.f
	gomega.Expect(e2epod.WaitForPodCondition(f.ClientSet, f.Namespace.Name, name, fmt.Sprintf("%s or %s", v1.PodSucceeded, v1.PodFailed), timeout,
		func(pod *v1.Pod) (bool, error) {
			switch pod.Status.Phase {
			case v1.PodFailed:
				return true, fmt.Errorf("pod %q failed with reason: %q, message: %q", name, pod.Status.Reason, pod.Status.Message)
			case v1.PodSucceeded:
				return true, nil
			default:
				return false, nil
			}
		},
	)).To(gomega.Succeed(), "wait for pod %q to success", name)
}

// WaitForFinish waits for pod to finish running, regardless of success or failure.
func (c *PodClient) WaitForFinish(name string, timeout time.Duration) {
	f := c.f
	gomega.Expect(e2epod.WaitForPodCondition(f.ClientSet, f.Namespace.Name, name, fmt.Sprintf("%s or %s", v1.PodSucceeded, v1.PodFailed), timeout,
		func(pod *v1.Pod) (bool, error) {
			switch pod.Status.Phase {
			case v1.PodFailed:
				return true, nil
			case v1.PodSucceeded:
				return true, nil
			default:
				return false, nil
			}
		},
	)).To(gomega.Succeed(), "wait for pod %q to finish running", name)
}

// WaitForErrorEventOrSuccess waits for pod to succeed or an error event for that pod.
func (c *PodClient) WaitForErrorEventOrSuccess(pod *v1.Pod) (*v1.Event, error) {
	var ev *v1.Event
	err := wait.Poll(Poll, PodStartTimeout, func() (bool, error) {
		evnts, err := c.f.ClientSet.CoreV1().Events(pod.Namespace).Search(scheme.Scheme, pod)
		if err != nil {
			return false, fmt.Errorf("error in listing events: %s", err)
		}
		for _, e := range evnts.Items {
			switch e.Reason {
			case events.KillingContainer, events.FailedToCreateContainer, sysctl.ForbiddenReason:
				ev = &e
				return true, nil
			case events.StartedContainer:
				return true, nil
			default:
				// ignore all other errors
			}
		}
		return false, nil
	})
	return ev, err
}

// MatchContainerOutput gets output of a container and match expected regexp in the output.
func (c *PodClient) MatchContainerOutput(name string, containerName string, expectedRegexp string) error {
	f := c.f
	output, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, name, containerName)
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

// PodIsReady returns true if the specified pod is ready. Otherwise false.
func (c *PodClient) PodIsReady(name string) bool {
	pod, err := c.Get(context.TODO(), name, metav1.GetOptions{})
	ExpectNoError(err)
	return podutil.IsPodReady(pod)
}
