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

package pod

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubectl/pkg/util/podutils"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/util/slice"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// DefaultPodDeletionTimeout is the default timeout for deleting pod
	DefaultPodDeletionTimeout = 3 * time.Minute

	// the status of container event, copied from k8s.io/kubernetes/pkg/kubelet/events
	killingContainer = "Killing"

	// the status of container event, copied from k8s.io/kubernetes/pkg/kubelet/events
	failedToCreateContainer = "Failed"

	// the status of container event, copied from k8s.io/kubernetes/pkg/kubelet/events
	startedContainer = "Started"

	// it is copied from k8s.io/kubernetes/pkg/kubelet/sysctl
	forbiddenReason = "SysctlForbidden"
)

// ImagePrePullList is the images used in the current test suite. It should be initialized in test suite and
// the images in the list should be pre-pulled in the test suite.  Currently, this is only used by
// node e2e test.
var ImagePrePullList sets.String

// NewPodClient is a convenience method for getting a pod client interface in the framework's namespace,
// possibly applying test-suite specific transformations to the pod spec, e.g. for
// node e2e pod scheduling.
func NewPodClient(f *framework.Framework) *PodClient {
	return &PodClient{
		f:            f,
		PodInterface: f.ClientSet.CoreV1().Pods(f.Namespace.Name),
	}
}

// PodClientNS is a convenience method for getting a pod client interface in an alternative namespace,
// possibly applying test-suite specific transformations to the pod spec, e.g. for
// node e2e pod scheduling.
func PodClientNS(f *framework.Framework, namespace string) *PodClient {
	return &PodClient{
		f:            f,
		PodInterface: f.ClientSet.CoreV1().Pods(namespace),
	}
}

// PodClient is a struct for pod client.
type PodClient struct {
	f *framework.Framework
	v1core.PodInterface
}

// Create creates a new pod according to the framework specifications (don't wait for it to start).
func (c *PodClient) Create(ctx context.Context, pod *v1.Pod) *v1.Pod {
	c.mungeSpec(pod)
	p, err := c.PodInterface.Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Error creating Pod")
	return p
}

// CreateSync creates a new pod according to the framework specifications, and wait for it to start and be running and ready.
func (c *PodClient) CreateSync(ctx context.Context, pod *v1.Pod) *v1.Pod {
	namespace := c.f.Namespace.Name
	p := c.Create(ctx, pod)
	framework.ExpectNoError(WaitTimeoutForPodReadyInNamespace(ctx, c.f.ClientSet, p.Name, namespace, framework.PodStartTimeout))
	// Get the newest pod after it becomes running and ready, some status may change after pod created, such as pod ip.
	p, err := c.Get(ctx, p.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return p
}

// CreateBatch create a batch of pods. All pods are created before waiting.
func (c *PodClient) CreateBatch(ctx context.Context, pods []*v1.Pod) []*v1.Pod {
	ps := make([]*v1.Pod, len(pods))
	var wg sync.WaitGroup
	for i, pod := range pods {
		wg.Add(1)
		go func(i int, pod *v1.Pod) {
			defer wg.Done()
			defer ginkgo.GinkgoRecover()
			ps[i] = c.CreateSync(ctx, pod)
		}(i, pod)
	}
	wg.Wait()
	return ps
}

// Update updates the pod object. It retries if there is a conflict, throw out error if
// there is any other apierrors. name is the pod name, updateFn is the function updating the
// pod object.
func (c *PodClient) Update(ctx context.Context, name string, updateFn func(pod *v1.Pod)) {
	framework.ExpectNoError(wait.PollWithContext(ctx, time.Millisecond*500, time.Second*30, func(ctx context.Context) (bool, error) {
		pod, err := c.PodInterface.Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("failed to get pod %q: %w", name, err)
		}
		updateFn(pod)
		_, err = c.PodInterface.Update(ctx, pod, metav1.UpdateOptions{})
		if err == nil {
			framework.Logf("Successfully updated pod %q", name)
			return true, nil
		}
		if apierrors.IsConflict(err) {
			framework.Logf("Conflicting update to pod %q, re-get and re-update: %v", name, err)
			return false, nil
		}
		return false, fmt.Errorf("failed to update pod %q: %w", name, err)
	}))
}

// AddEphemeralContainerSync adds an EphemeralContainer to a pod and waits for it to be running.
func (c *PodClient) AddEphemeralContainerSync(ctx context.Context, pod *v1.Pod, ec *v1.EphemeralContainer, timeout time.Duration) error {
	namespace := c.f.Namespace.Name

	podJS, err := json.Marshal(pod)
	framework.ExpectNoError(err, "error creating JSON for pod %q", format.Pod(pod))

	ecPod := pod.DeepCopy()
	ecPod.Spec.EphemeralContainers = append(ecPod.Spec.EphemeralContainers, *ec)
	ecJS, err := json.Marshal(ecPod)
	framework.ExpectNoError(err, "error creating JSON for pod with ephemeral container %q", format.Pod(pod))

	patch, err := strategicpatch.CreateTwoWayMergePatch(podJS, ecJS, pod)
	framework.ExpectNoError(err, "error creating patch to add ephemeral container %q", format.Pod(pod))

	// Clients may optimistically attempt to add an ephemeral container to determine whether the EphemeralContainers feature is enabled.
	if _, err := c.Patch(ctx, pod.Name, types.StrategicMergePatchType, patch, metav1.PatchOptions{}, "ephemeralcontainers"); err != nil {
		return err
	}

	framework.ExpectNoError(WaitForContainerRunning(ctx, c.f.ClientSet, namespace, pod.Name, ec.Name, timeout))
	return nil
}

// DeleteSync deletes the pod and wait for the pod to disappear for `timeout`. If the pod doesn't
// disappear before the timeout, it will fail the test.
func (c *PodClient) DeleteSync(ctx context.Context, name string, options metav1.DeleteOptions, timeout time.Duration) {
	namespace := c.f.Namespace.Name
	err := c.Delete(ctx, name, options)
	if err != nil && !apierrors.IsNotFound(err) {
		framework.Failf("Failed to delete pod %q: %v", name, err)
	}
	framework.ExpectNoError(WaitForPodNotFoundInNamespace(ctx, c.f.ClientSet, name, namespace, timeout), "wait for pod %q to disappear", name)
}

// mungeSpec apply test-suite specific transformations to the pod spec.
func (c *PodClient) mungeSpec(pod *v1.Pod) {
	if !framework.TestContext.NodeE2E {
		return
	}

	gomega.Expect(pod.Spec.NodeName).To(gomega.Or(gomega.BeZero(), gomega.Equal(framework.TestContext.NodeName)), "Test misconfigured")
	pod.Spec.NodeName = framework.TestContext.NodeName
	// Node e2e does not support the default DNSClusterFirst policy. Set
	// the policy to DNSDefault, which is configured per node.
	pod.Spec.DNSPolicy = v1.DNSDefault

	// PrepullImages only works for node e2e now. For cluster e2e, image prepull is not enforced,
	// we should not munge ImagePullPolicy for cluster e2e pods.
	if !framework.TestContext.PrepullImages {
		return
	}
	// If prepull is enabled, munge the container spec to make sure the images are not pulled
	// during the test.
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		if c.ImagePullPolicy == v1.PullAlways {
			// If the image pull policy is PullAlways, the image doesn't need to be in
			// the allow list or pre-pulled, because the image is expected to be pulled
			// in the test anyway.
			continue
		}
		// If the image policy is not PullAlways, the image must be in the pre-pull list and
		// pre-pulled.
		gomega.Expect(ImagePrePullList.Has(c.Image)).To(gomega.BeTrue(), "Image %q is not in the pre-pull list, consider adding it to PrePulledImages in test/e2e/common/util.go or NodePrePullImageList in test/e2e_node/image_list.go", c.Image)
		// Do not pull images during the tests because the images in pre-pull list should have
		// been prepulled.
		c.ImagePullPolicy = v1.PullNever
	}
}

// WaitForSuccess waits for pod to succeed.
// TODO(random-liu): Move pod wait function into this file
func (c *PodClient) WaitForSuccess(ctx context.Context, name string, timeout time.Duration) {
	f := c.f
	gomega.Expect(WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, name, fmt.Sprintf("%s or %s", v1.PodSucceeded, v1.PodFailed), timeout,
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
	)).To(gomega.Succeed(), "wait for pod %q to succeed", name)
}

// WaitForFinish waits for pod to finish running, regardless of success or failure.
func (c *PodClient) WaitForFinish(ctx context.Context, name string, timeout time.Duration) {
	f := c.f
	gomega.Expect(WaitForPodCondition(ctx, f.ClientSet, f.Namespace.Name, name, fmt.Sprintf("%s or %s", v1.PodSucceeded, v1.PodFailed), timeout,
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
func (c *PodClient) WaitForErrorEventOrSuccess(ctx context.Context, pod *v1.Pod) (*v1.Event, error) {
	var ev *v1.Event
	err := wait.PollWithContext(ctx, framework.Poll, framework.PodStartTimeout, func(ctx context.Context) (bool, error) {
		evnts, err := c.f.ClientSet.CoreV1().Events(pod.Namespace).Search(scheme.Scheme, pod)
		if err != nil {
			return false, fmt.Errorf("error in listing events: %w", err)
		}
		for _, e := range evnts.Items {
			switch e.Reason {
			case killingContainer, failedToCreateContainer, forbiddenReason:
				ev = &e
				return true, nil
			case startedContainer:
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
func (c *PodClient) MatchContainerOutput(ctx context.Context, name string, containerName string, expectedRegexp string) error {
	f := c.f
	output, err := GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, name, containerName)
	if err != nil {
		return fmt.Errorf("failed to get output for container %q of pod %q", containerName, name)
	}
	regex, err := regexp.Compile(expectedRegexp)
	if err != nil {
		return fmt.Errorf("failed to compile regexp %q: %w", expectedRegexp, err)
	}
	if !regex.MatchString(output) {
		return fmt.Errorf("failed to match regexp %q in output %q", expectedRegexp, output)
	}
	return nil
}

// PodIsReady returns true if the specified pod is ready. Otherwise false.
func (c *PodClient) PodIsReady(ctx context.Context, name string) bool {
	pod, err := c.Get(ctx, name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return podutils.IsPodReady(pod)
}

// RemovePodFinalizer removes the pod's finalizer
func (c *PodClient) RemoveFinalizer(ctx context.Context, podName string, finalizerName string) {
	framework.Logf("Removing pod's %q finalizer: %q", podName, finalizerName)
	c.Update(ctx, podName, func(pod *v1.Pod) {
		pod.ObjectMeta.Finalizers = slice.RemoveString(pod.ObjectMeta.Finalizers, finalizerName, nil)
	})
}
