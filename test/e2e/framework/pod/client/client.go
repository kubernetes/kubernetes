/*
Copyright 2021 The Kubernetes Authors.

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

package client

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"sigs.k8s.io/kustomize/kyaml/sets"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/podutils"
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

// PodClient is a struct for pod client.
type PodClient struct {
	f FrameworkShim
	v1core.PodInterface
}

// FrameworkShim exposes the same methods as test/e2e/framework, this is
// done for compatibility as work on #81245 continues.
type FrameworkShim interface {
	GetClientSet() clientset.Interface
	GetNamespace() *v1.Namespace
}

// MakePodClient is a convenience method for getting a pod client interface in the framework's namespace,
// possibly applying test-suite specific transformations to the pod spec, e.g. for
// node e2e pod scheduling.
func MakePodClient(f FrameworkShim) *PodClient {
	return &PodClient{
		f:            f,
		PodInterface: f.GetClientSet().CoreV1().Pods(f.GetNamespace().Name),
	}
}

// MakePodClientNS is a convenience method for getting a pod client interface in an alternative namespace,
// possibly applying test-suite specific transformations to the pod spec, e.g. for
// node e2e pod scheduling.
func MakePodClientNS(f FrameworkShim, namespace string) *PodClient {
	return &PodClient{
		f:            f,
		PodInterface: f.GetClientSet().CoreV1().Pods(namespace),
	}
}

// Create creates a new pod according to the framework specifications (don't wait for it to start).
func (c *PodClient) Create(pod *v1.Pod) *v1.Pod {
	c.mungeSpec(pod)
	p, err := c.PodInterface.Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Error creating Pod")
	return p
}

// CreateSync creates a new pod according to the framework specifications, and wait for it to start and be running and ready.
func (c *PodClient) CreateSync(pod *v1.Pod) *v1.Pod {
	namespace := c.f.GetNamespace().Name
	p := c.Create(pod)
	framework.ExpectNoError(WaitTimeoutForPodReadyInNamespace(c.f.GetClientSet(), p.Name, namespace, framework.PodStartTimeout))
	// Get the newest pod after it becomes running and ready, some status may change after pod created, such as pod ip.
	p, err := c.Get(context.TODO(), p.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
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
	framework.ExpectNoError(wait.Poll(time.Millisecond*500, time.Second*30, func() (bool, error) {
		pod, err := c.PodInterface.Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("failed to get pod %q: %v", name, err)
		}
		updateFn(pod)
		_, err = c.PodInterface.Update(context.TODO(), pod, metav1.UpdateOptions{})
		if err == nil {
			framework.Logf("Successfully updated pod %q", name)
			return true, nil
		}
		if apierrors.IsConflict(err) {
			framework.Logf("Conflicting update to pod %q, re-get and re-update: %v", name, err)
			return false, nil
		}
		return false, fmt.Errorf("failed to update pod %q: %v", name, err)
	}))
}

// DeleteSync deletes the pod and wait for the pod to disappear for `timeout`. If the pod doesn't
// disappear before the timeout, it will fail the test.
func (c *PodClient) DeleteSync(name string, options metav1.DeleteOptions, timeout time.Duration) {
	namespace := c.f.GetNamespace().Name
	err := c.Delete(context.TODO(), name, options)
	if err != nil && !apierrors.IsNotFound(err) {
		framework.Failf("Failed to delete pod %q: %v", name, err)
	}
	gomega.Expect(waitForPodToDisappear(c.f.GetClientSet(), namespace, name, labels.Everything(),
		2*time.Second, timeout)).To(gomega.Succeed(), "wait for pod %q to disappear", name)
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
			// the white list or pre-pulled, because the image is expected to be pulled
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
func (c *PodClient) WaitForSuccess(name string, timeout time.Duration) {
	f := c.f
	gomega.Expect(WaitForPodCondition(f.GetClientSet(), f.GetNamespace().Name, name, fmt.Sprintf("%s or %s", v1.PodSucceeded, v1.PodFailed), timeout,
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
func (c *PodClient) WaitForFinish(name string, timeout time.Duration) {
	f := c.f
	gomega.Expect(WaitForPodCondition(f.GetClientSet(), f.GetNamespace().Name, name, fmt.Sprintf("%s or %s", v1.PodSucceeded, v1.PodFailed), timeout,
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
	err := wait.Poll(framework.Poll, framework.PodStartTimeout, func() (bool, error) {
		evnts, err := c.f.GetClientSet().CoreV1().Events(pod.Namespace).Search(scheme.Scheme, pod)
		if err != nil {
			return false, fmt.Errorf("error in listing events: %s", err)
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
func (c *PodClient) MatchContainerOutput(name string, containerName string, expectedRegexp string) error {
	f := c.f
	output, err := GetPodLogs(f.GetClientSet(), f.GetNamespace().Name, name, containerName)
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
	framework.ExpectNoError(err)
	return podutils.IsPodReady(pod)
}

//
// These functions are duplicated from framework/pods and will be eventually eliminated
//
func waitForPodToDisappear(c clientset.Interface, ns, podName string, label labels.Selector, interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		framework.Logf("Waiting for pod %s to disappear", podName)
		options := metav1.ListOptions{LabelSelector: label.String()}
		pods, err := c.CoreV1().Pods(ns).List(context.TODO(), options)
		if err != nil {
			return false, err
		}
		found := false
		for _, pod := range pods.Items {
			if pod.Name == podName {
				framework.Logf("Pod %s still exists", podName)
				found = true
				break
			}
		}
		if !found {
			framework.Logf("Pod %s no longer exists", podName)
			return true, nil
		}
		return false, nil
	})
}

const (
	// poll is how often to poll pods, nodes and claims.
	poll = 2 * time.Second
)

var errPodCompleted = fmt.Errorf("pod ran to completion")

// WaitTimeoutForPodReadyInNamespace waits the given timeout duration for the
// specified pod to be ready and running.
func WaitTimeoutForPodReadyInNamespace(c clientset.Interface, podName, namespace string, timeout time.Duration) error {
	return wait.PollImmediate(poll, timeout, podRunningAndReady(c, podName, namespace))
}

func podRunningAndReady(c clientset.Interface, podName, namespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		switch pod.Status.Phase {
		case v1.PodFailed, v1.PodSucceeded:
			framework.Logf("The status of Pod %s is %s which is unexpected", podName, pod.Status.Phase)
			return false, errPodCompleted
		case v1.PodRunning:
			framework.Logf("The status of Pod %s is %s (Ready = %v)", podName, pod.Status.Phase, podutils.IsPodReady(pod))
			return podutils.IsPodReady(pod), nil
		}
		framework.Logf("The status of Pod %s is %s, waiting for it to be Running (with Ready = true)", podName, pod.Status.Phase)
		return false, nil
	}
}

type podCondition func(pod *v1.Pod) (bool, error)

// WaitForPodCondition waits a pods to be matched to the given condition.
func WaitForPodCondition(c clientset.Interface, ns, podName, desc string, timeout time.Duration, condition podCondition) error {
	framework.Logf("Waiting up to %v for pod %q in namespace %q to be %q", timeout, podName, ns, desc)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		pod, err := c.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			if apierrors.IsNotFound(err) {
				framework.Logf("Pod %q in namespace %q not found. Error: %v", podName, ns, err)
				return err
			}
			framework.Logf("Get pod %q in namespace %q failed, ignoring for %v. Error: %v", podName, ns, poll, err)
			continue
		}
		// log now so that current pod info is reported before calling `condition()`
		framework.Logf("Pod %q: Phase=%q, Reason=%q, readiness=%t. Elapsed: %v",
			podName, pod.Status.Phase, pod.Status.Reason, podutils.IsPodReady(pod), time.Since(start))
		if done, err := condition(pod); done {
			if err == nil {
				framework.Logf("Pod %q satisfied condition %q", podName, desc)
			}
			return err
		}
	}
	return fmt.Errorf("Gave up after waiting %v for pod %q to be %q", timeout, podName, desc)
}

// GetPodLogs returns the logs of the specified container (namespace/pod/container).
func GetPodLogs(c clientset.Interface, namespace, podName, containerName string) (string, error) {
	return getPodLogsInternal(c, namespace, podName, containerName, false, nil)
}

// utility function for gomega Eventually
func getPodLogsInternal(c clientset.Interface, namespace, podName, containerName string, previous bool, sinceTime *metav1.Time) (string, error) {
	request := c.CoreV1().RESTClient().Get().
		Resource("pods").
		Namespace(namespace).
		Name(podName).SubResource("log").
		Param("container", containerName).
		Param("previous", strconv.FormatBool(previous))
	if sinceTime != nil {
		request.Param("sinceTime", sinceTime.Format(time.RFC3339))
	}
	logs, err := request.Do(context.TODO()).Raw()
	if err != nil {
		return "", err
	}
	if strings.Contains(string(logs), "Internal Error") {
		return "", fmt.Errorf("Fetched log contains \"Internal Error\": %q", string(logs))
	}
	return string(logs), err
}

// WaitForPodSuccessInNamespaceTimeout returns nil if the pod reached state success, or an error if it reached failure or ran too long.
func WaitForPodSuccessInNamespaceTimeout(c clientset.Interface, podName, namespace string, timeout time.Duration) error {
	return WaitForPodCondition(c, namespace, podName, fmt.Sprintf("%s or %s", v1.PodSucceeded, v1.PodFailed), timeout, func(pod *v1.Pod) (bool, error) {
		if pod.Spec.RestartPolicy == v1.RestartPolicyAlways {
			return true, fmt.Errorf("pod %q will never terminate with a succeeded state since its restart policy is Always", podName)
		}
		switch pod.Status.Phase {
		case v1.PodSucceeded:
			ginkgo.By("Saw pod success")
			return true, nil
		case v1.PodFailed:
			return true, fmt.Errorf("pod %q failed with status: %+v", podName, pod.Status)
		default:
			return false, nil
		}
	})
}

// DumpAllPodInfoForNamespace logs all pod information for a given namespace.
func DumpAllPodInfoForNamespace(c clientset.Interface, namespace string) {
	pods, err := c.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		framework.Logf("unable to fetch pod debug info: %v", err)
	}
	LogPodStates(pods.Items)
	logPodTerminationMessages(pods.Items)
}

// LogPodStates logs basic info of provided pods for debugging.
func LogPodStates(pods []v1.Pod) {
	// Find maximum widths for pod, node, and phase strings for column printing.
	maxPodW, maxNodeW, maxPhaseW, maxGraceW := len("POD"), len("NODE"), len("PHASE"), len("GRACE")
	for i := range pods {
		pod := &pods[i]
		if len(pod.ObjectMeta.Name) > maxPodW {
			maxPodW = len(pod.ObjectMeta.Name)
		}
		if len(pod.Spec.NodeName) > maxNodeW {
			maxNodeW = len(pod.Spec.NodeName)
		}
		if len(pod.Status.Phase) > maxPhaseW {
			maxPhaseW = len(pod.Status.Phase)
		}
	}
	// Increase widths by one to separate by a single space.
	maxPodW++
	maxNodeW++
	maxPhaseW++
	maxGraceW++

	// Log pod info. * does space padding, - makes them left-aligned.
	framework.Logf("%-[1]*[2]s %-[3]*[4]s %-[5]*[6]s %-[7]*[8]s %[9]s",
		maxPodW, "POD", maxNodeW, "NODE", maxPhaseW, "PHASE", maxGraceW, "GRACE", "CONDITIONS")
	for _, pod := range pods {
		grace := ""
		if pod.DeletionGracePeriodSeconds != nil {
			grace = fmt.Sprintf("%ds", *pod.DeletionGracePeriodSeconds)
		}
		framework.Logf("%-[1]*[2]s %-[3]*[4]s %-[5]*[6]s %-[7]*[8]s %[9]s",
			maxPodW, pod.ObjectMeta.Name, maxNodeW, pod.Spec.NodeName, maxPhaseW, pod.Status.Phase, maxGraceW, grace, pod.Status.Conditions)
	}
	framework.Logf("") // Final empty line helps for readability.
}

// logPodTerminationMessages logs termination messages for failing pods.  It's a short snippet (much smaller than full logs), but it often shows
// why pods crashed and since it is in the API, it's fast to retrieve.
func logPodTerminationMessages(pods []v1.Pod) {
	for _, pod := range pods {
		for _, status := range pod.Status.InitContainerStatuses {
			if status.LastTerminationState.Terminated != nil && len(status.LastTerminationState.Terminated.Message) > 0 {
				framework.Logf("%s[%s].initContainer[%s]=%s", pod.Name, pod.Namespace, status.Name, status.LastTerminationState.Terminated.Message)
			}
		}
		for _, status := range pod.Status.ContainerStatuses {
			if status.LastTerminationState.Terminated != nil && len(status.LastTerminationState.Terminated.Message) > 0 {
				framework.Logf("%s[%s].container[%s]=%s", pod.Name, pod.Namespace, status.Name, status.LastTerminationState.Terminated.Message)
			}
		}
	}
}
