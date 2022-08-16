/*
Copyright 2019 The Kubernetes Authors.

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
	"bytes"
	"context"
	"errors"
	"fmt"
	"text/tabwriter"
	"time"

	"github.com/onsi/ginkgo/v2"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubectl/pkg/util/podutils"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	testutils "k8s.io/kubernetes/test/utils"
)

const (
	// defaultPodDeletionTimeout is the default timeout for deleting pod.
	defaultPodDeletionTimeout = 3 * time.Minute

	// podListTimeout is how long to wait for the pod to be listable.
	podListTimeout = time.Minute

	podRespondingTimeout = 15 * time.Minute

	// How long pods have to become scheduled onto nodes
	podScheduledBeforeTimeout = podListTimeout + (20 * time.Second)

	// podStartTimeout is how long to wait for the pod to be started.
	podStartTimeout = 5 * time.Minute

	// poll is how often to poll pods, nodes and claims.
	poll = 2 * time.Second

	// singleCallTimeout is how long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	singleCallTimeout = 5 * time.Minute

	// Some pods can take much longer to get ready due to volume attach/detach latency.
	slowPodStartTimeout = 15 * time.Minute
)

type podCondition func(pod *v1.Pod) (bool, error)

type timeoutError struct {
	msg             string
	observedObjects []interface{}
}

func (e *timeoutError) Error() string {
	return e.msg
}

func TimeoutError(msg string, observedObjects ...interface{}) *timeoutError {
	return &timeoutError{
		msg:             msg,
		observedObjects: observedObjects,
	}
}

// maybeTimeoutError returns a TimeoutError if err is a timeout. Otherwise, wrap err.
// taskFormat and taskArgs should be the task being performed when the error occurred,
// e.g. "waiting for pod to be running".
func maybeTimeoutError(err error, taskFormat string, taskArgs ...interface{}) error {
	if IsTimeout(err) {
		return TimeoutError(fmt.Sprintf("timed out while "+taskFormat, taskArgs...))
	} else if err != nil {
		return fmt.Errorf("error while %s: %w", fmt.Sprintf(taskFormat, taskArgs...), err)
	} else {
		return nil
	}
}

func IsTimeout(err error) bool {
	if err == wait.ErrWaitTimeout {
		return true
	}
	if _, ok := err.(*timeoutError); ok {
		return true
	}
	return false
}

// errorBadPodsStates create error message of basic info of bad pods for debugging.
func errorBadPodsStates(badPods []v1.Pod, desiredPods int, ns, desiredState string, timeout time.Duration, err error) error {
	errStr := fmt.Sprintf("%d / %d pods in namespace %s are NOT in %s state in %v\n", len(badPods), desiredPods, ns, desiredState, timeout)

	// Print bad pods info only if there are fewer than 10 bad pods
	if len(badPods) > 10 {
		errStr += "There are too many bad pods. Please check log for details."
	} else {
		buf := bytes.NewBuffer(nil)
		w := tabwriter.NewWriter(buf, 0, 0, 1, ' ', 0)
		fmt.Fprintln(w, "POD\tNODE\tPHASE\tGRACE\tCONDITIONS")
		for _, badPod := range badPods {
			grace := ""
			if badPod.DeletionGracePeriodSeconds != nil {
				grace = fmt.Sprintf("%ds", *badPod.DeletionGracePeriodSeconds)
			}
			podInfo := fmt.Sprintf("%s\t%s\t%s\t%s\t%+v",
				badPod.ObjectMeta.Name, badPod.Spec.NodeName, badPod.Status.Phase, grace, badPod.Status.Conditions)
			fmt.Fprintln(w, podInfo)
		}
		w.Flush()
		errStr += buf.String()
	}

	if err != nil && !IsTimeout(err) {
		return fmt.Errorf("%s\nLast error: %w", errStr, err)
	}
	return TimeoutError(errStr)
}

// WaitForPodsRunningReady waits up to timeout to ensure that all pods in
// namespace ns are either running and ready, or failed but controlled by a
// controller. Also, it ensures that at least minPods are running and
// ready. It has separate behavior from other 'wait for' pods functions in
// that it requests the list of pods on every iteration. This is useful, for
// example, in cluster startup, because the number of pods increases while
// waiting. All pods that are in SUCCESS state are not counted.
//
// If ignoreLabels is not empty, pods matching this selector are ignored.
//
// If minPods or allowedNotReadyPods are -1, this method returns immediately
// without waiting.
func WaitForPodsRunningReady(c clientset.Interface, ns string, minPods, allowedNotReadyPods int32, timeout time.Duration, ignoreLabels map[string]string) error {
	if minPods == -1 || allowedNotReadyPods == -1 {
		return nil
	}

	ignoreSelector := labels.SelectorFromSet(map[string]string{})
	start := time.Now()
	e2elog.Logf("Waiting up to %v for all pods (need at least %d) in namespace '%s' to be running and ready",
		timeout, minPods, ns)
	var ignoreNotReady bool
	badPods := []v1.Pod{}
	desiredPods := 0
	notReady := int32(0)
	var lastAPIError error

	if wait.PollImmediate(poll, timeout, func() (bool, error) {
		// We get the new list of pods, replication controllers, and
		// replica sets in every iteration because more pods come
		// online during startup and we want to ensure they are also
		// checked.
		replicas, replicaOk := int32(0), int32(0)
		// Clear API error from the last attempt in case the following calls succeed.
		lastAPIError = nil

		rcList, err := c.CoreV1().ReplicationControllers(ns).List(context.TODO(), metav1.ListOptions{})
		lastAPIError = err
		if err != nil {
			return handleWaitingAPIError(err, false, "listing replication controllers in namespace %s", ns)
		}
		for _, rc := range rcList.Items {
			replicas += *rc.Spec.Replicas
			replicaOk += rc.Status.ReadyReplicas
		}

		rsList, err := c.AppsV1().ReplicaSets(ns).List(context.TODO(), metav1.ListOptions{})
		lastAPIError = err
		if err != nil {
			return handleWaitingAPIError(err, false, "listing replication sets in namespace %s", ns)
		}
		for _, rs := range rsList.Items {
			replicas += *rs.Spec.Replicas
			replicaOk += rs.Status.ReadyReplicas
		}

		podList, err := c.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{})
		lastAPIError = err
		if err != nil {
			return handleWaitingAPIError(err, false, "listing pods in namespace %s", ns)
		}
		nOk := int32(0)
		notReady = int32(0)
		badPods = []v1.Pod{}
		desiredPods = len(podList.Items)
		for _, pod := range podList.Items {
			if len(ignoreLabels) != 0 && ignoreSelector.Matches(labels.Set(pod.Labels)) {
				continue
			}
			res, err := testutils.PodRunningReady(&pod)
			switch {
			case res && err == nil:
				nOk++
			case pod.Status.Phase == v1.PodSucceeded:
				e2elog.Logf("The status of Pod %s is Succeeded, skipping waiting", pod.ObjectMeta.Name)
				// it doesn't make sense to wait for this pod
				continue
			case pod.Status.Phase != v1.PodFailed:
				e2elog.Logf("The status of Pod %s is %s (Ready = false), waiting for it to be either Running (with Ready = true) or Failed", pod.ObjectMeta.Name, pod.Status.Phase)
				notReady++
				badPods = append(badPods, pod)
			default:
				if metav1.GetControllerOf(&pod) == nil {
					e2elog.Logf("Pod %s is Failed, but it's not controlled by a controller", pod.ObjectMeta.Name)
					badPods = append(badPods, pod)
				}
				//ignore failed pods that are controlled by some controller
			}
		}

		e2elog.Logf("%d / %d pods in namespace '%s' are running and ready (%d seconds elapsed)",
			nOk, len(podList.Items), ns, int(time.Since(start).Seconds()))
		e2elog.Logf("expected %d pod replicas in namespace '%s', %d are Running and Ready.", replicas, ns, replicaOk)

		if replicaOk == replicas && nOk >= minPods && len(badPods) == 0 {
			return true, nil
		}
		ignoreNotReady = (notReady <= allowedNotReadyPods)
		LogPodStates(badPods)
		return false, nil
	}) != nil {
		if !ignoreNotReady {
			return errorBadPodsStates(badPods, desiredPods, ns, "RUNNING and READY", timeout, lastAPIError)
		}
		e2elog.Logf("Number of not-ready pods (%d) is below the allowed threshold (%d).", notReady, allowedNotReadyPods)
	}
	return nil
}

// WaitForPodCondition waits a pods to be matched to the given condition.
func WaitForPodCondition(c clientset.Interface, ns, podName, conditionDesc string, timeout time.Duration, condition podCondition) error {
	e2elog.Logf("Waiting up to %v for pod %q in namespace %q to be %q", timeout, podName, ns, conditionDesc)
	var (
		lastPodError error
		lastPod      *v1.Pod
		start        = time.Now()
	)
	err := wait.PollImmediate(poll, timeout, func() (bool, error) {
		pod, err := c.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		lastPodError = err
		if err != nil {
			return handleWaitingAPIError(err, true, "getting pod %s", podIdentifier(ns, podName))
		}
		lastPod = pod // Don't overwrite if an error occurs after successfully retrieving.

		// log now so that current pod info is reported before calling `condition()`
		e2elog.Logf("Pod %q: Phase=%q, Reason=%q, readiness=%t. Elapsed: %v",
			podName, pod.Status.Phase, pod.Status.Reason, podutils.IsPodReady(pod), time.Since(start))
		if done, err := condition(pod); done {
			if err == nil {
				e2elog.Logf("Pod %q satisfied condition %q", podName, conditionDesc)
			}
			return true, err
		} else if err != nil {
			// TODO(#109732): stop polling and return the error in this case.
			e2elog.Logf("Error evaluating pod condition %s: %v", conditionDesc, err)
		}
		return false, nil
	})
	if err == nil {
		return nil
	}
	if IsTimeout(err) && lastPod != nil {
		return TimeoutError(fmt.Sprintf("timed out while waiting for pod %s to be %s", podIdentifier(ns, podName), conditionDesc),
			lastPod,
		)
	}
	if lastPodError != nil {
		// If the last API call was an error.
		err = lastPodError
	}
	return maybeTimeoutError(err, "waiting for pod %s to be %s", podIdentifier(ns, podName), conditionDesc)
}

// WaitForPodsCondition waits for the listed pods to match the given condition.
// To succeed, at least minPods must be listed, and all listed pods must match the condition.
func WaitForAllPodsCondition(c clientset.Interface, ns string, opts metav1.ListOptions, minPods int, conditionDesc string, timeout time.Duration, condition podCondition) (*v1.PodList, error) {
	e2elog.Logf("Waiting up to %v for at least %d pods in namespace %s to be %s", timeout, minPods, ns, conditionDesc)
	var pods *v1.PodList
	matched := 0
	err := wait.PollImmediate(poll, timeout, func() (done bool, err error) {
		pods, err = c.CoreV1().Pods(ns).List(context.TODO(), opts)
		if err != nil {
			return handleWaitingAPIError(err, true, "listing pods")
		}
		if len(pods.Items) < minPods {
			e2elog.Logf("found %d pods, waiting for at least %d", len(pods.Items), minPods)
			return false, nil
		}

		nonMatchingPods := []string{}
		for _, pod := range pods.Items {
			done, err := condition(&pod)
			if done && err != nil {
				return false, fmt.Errorf("error evaluating pod %s: %w", identifier(&pod), err)
			}
			if !done {
				nonMatchingPods = append(nonMatchingPods, identifier(&pod))
			}
		}
		matched = len(pods.Items) - len(nonMatchingPods)
		if len(nonMatchingPods) <= 0 {
			return true, nil // All pods match.
		}
		e2elog.Logf("%d pods are not %s: %v", len(nonMatchingPods), conditionDesc, nonMatchingPods)
		return false, nil
	})
	return pods, maybeTimeoutError(err, "waiting for at least %d pods to be %s (matched %d)", minPods, conditionDesc, matched)
}

// WaitForPodTerminatedInNamespace returns an error if it takes too long for the pod to terminate,
// if the pod Get api returns an error (IsNotFound or other), or if the pod failed (and thus did not
// terminate) with an unexpected reason. Typically called to test that the passed-in pod is fully
// terminated (reason==""), but may be called to detect if a pod did *not* terminate according to
// the supplied reason.
func WaitForPodTerminatedInNamespace(c clientset.Interface, podName, reason, namespace string) error {
	return WaitForPodCondition(c, namespace, podName, fmt.Sprintf("terminated with reason %s", reason), podStartTimeout, func(pod *v1.Pod) (bool, error) {
		// Only consider Failed pods. Successful pods will be deleted and detected in
		// waitForPodCondition's Get call returning `IsNotFound`
		if pod.Status.Phase == v1.PodFailed {
			if pod.Status.Reason == reason { // short-circuit waitForPodCondition's loop
				return true, nil
			}
			return true, fmt.Errorf("Expected pod %q in namespace %q to be terminated with reason %q, got reason: %q", podName, namespace, reason, pod.Status.Reason)
		}
		return false, nil
	})
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

// WaitForPodNameUnschedulableInNamespace returns an error if it takes too long for the pod to become Pending
// and have condition Status equal to Unschedulable,
// if the pod Get api returns an error (IsNotFound or other), or if the pod failed with an unexpected reason.
// Typically called to test that the passed-in pod is Pending and Unschedulable.
func WaitForPodNameUnschedulableInNamespace(c clientset.Interface, podName, namespace string) error {
	return WaitForPodCondition(c, namespace, podName, v1.PodReasonUnschedulable, podStartTimeout, func(pod *v1.Pod) (bool, error) {
		// Only consider Failed pods. Successful pods will be deleted and detected in
		// waitForPodCondition's Get call returning `IsNotFound`
		if pod.Status.Phase == v1.PodPending {
			for _, cond := range pod.Status.Conditions {
				if cond.Type == v1.PodScheduled && cond.Status == v1.ConditionFalse && cond.Reason == v1.PodReasonUnschedulable {
					return true, nil
				}
			}
		}
		if pod.Status.Phase == v1.PodRunning || pod.Status.Phase == v1.PodSucceeded || pod.Status.Phase == v1.PodFailed {
			return true, fmt.Errorf("Expected pod %q in namespace %q to be in phase Pending, but got phase: %v", podName, namespace, pod.Status.Phase)
		}
		return false, nil
	})
}

// WaitForPodNameRunningInNamespace waits default amount of time (PodStartTimeout) for the specified pod to become running.
// Returns an error if timeout occurs first, or pod goes in to failed state.
func WaitForPodNameRunningInNamespace(c clientset.Interface, podName, namespace string) error {
	return WaitTimeoutForPodRunningInNamespace(c, podName, namespace, podStartTimeout)
}

// WaitForPodRunningInNamespaceSlow waits an extended amount of time (slowPodStartTimeout) for the specified pod to become running.
// The resourceVersion is used when Watching object changes, it tells since when we care
// about changes to the pod. Returns an error if timeout occurs first, or pod goes in to failed state.
func WaitForPodRunningInNamespaceSlow(c clientset.Interface, podName, namespace string) error {
	return WaitTimeoutForPodRunningInNamespace(c, podName, namespace, slowPodStartTimeout)
}

// WaitTimeoutForPodRunningInNamespace waits the given timeout duration for the specified pod to become running.
func WaitTimeoutForPodRunningInNamespace(c clientset.Interface, podName, namespace string, timeout time.Duration) error {
	return WaitForPodCondition(c, namespace, podName, "running", timeout, func(pod *v1.Pod) (bool, error) {
		switch pod.Status.Phase {
		case v1.PodRunning:
			return true, nil
		case v1.PodFailed, v1.PodSucceeded:
			return false, errPodCompleted
		}
		return false, nil
	})
}

// WaitForPodRunningInNamespace waits default amount of time (podStartTimeout) for the specified pod to become running.
// Returns an error if timeout occurs first, or pod goes in to failed state.
func WaitForPodRunningInNamespace(c clientset.Interface, pod *v1.Pod) error {
	if pod.Status.Phase == v1.PodRunning {
		return nil
	}
	return WaitTimeoutForPodRunningInNamespace(c, pod.Name, pod.Namespace, podStartTimeout)
}

// WaitTimeoutForPodNoLongerRunningInNamespace waits the given timeout duration for the specified pod to stop.
func WaitTimeoutForPodNoLongerRunningInNamespace(c clientset.Interface, podName, namespace string, timeout time.Duration) error {
	return WaitForPodCondition(c, namespace, podName, "completed", timeout, func(pod *v1.Pod) (bool, error) {
		switch pod.Status.Phase {
		case v1.PodFailed, v1.PodSucceeded:
			return true, nil
		}
		return false, nil
	})
}

// WaitForPodNoLongerRunningInNamespace waits default amount of time (defaultPodDeletionTimeout) for the specified pod to stop running.
// Returns an error if timeout occurs first.
func WaitForPodNoLongerRunningInNamespace(c clientset.Interface, podName, namespace string) error {
	return WaitTimeoutForPodNoLongerRunningInNamespace(c, podName, namespace, defaultPodDeletionTimeout)
}

// WaitTimeoutForPodReadyInNamespace waits the given timeout duration for the
// specified pod to be ready and running.
func WaitTimeoutForPodReadyInNamespace(c clientset.Interface, podName, namespace string, timeout time.Duration) error {
	return WaitForPodCondition(c, namespace, podName, "running and ready", timeout, func(pod *v1.Pod) (bool, error) {
		switch pod.Status.Phase {
		case v1.PodFailed, v1.PodSucceeded:
			e2elog.Logf("The phase of Pod %s is %s which is unexpected, pod status: %#v", pod.Name, pod.Status.Phase, pod.Status)
			return false, errPodCompleted
		case v1.PodRunning:
			e2elog.Logf("The phase of Pod %s is %s (Ready = %v)", pod.Name, pod.Status.Phase, podutils.IsPodReady(pod))
			return podutils.IsPodReady(pod), nil
		}
		e2elog.Logf("The phase of Pod %s is %s, waiting for it to be Running (with Ready = true)", pod.Name, pod.Status.Phase)
		return false, nil
	})
}

// WaitForPodNotPending returns an error if it took too long for the pod to go out of pending state.
// The resourceVersion is used when Watching object changes, it tells since when we care
// about changes to the pod.
func WaitForPodNotPending(c clientset.Interface, ns, podName string) error {
	return WaitForPodCondition(c, ns, podName, "not pending", podStartTimeout, func(pod *v1.Pod) (bool, error) {
		switch pod.Status.Phase {
		case v1.PodPending:
			return false, nil
		default:
			return true, nil
		}
	})
}

// WaitForPodSuccessInNamespace returns nil if the pod reached state success, or an error if it reached failure or until podStartupTimeout.
func WaitForPodSuccessInNamespace(c clientset.Interface, podName string, namespace string) error {
	return WaitForPodSuccessInNamespaceTimeout(c, podName, namespace, podStartTimeout)
}

// WaitForPodSuccessInNamespaceSlow returns nil if the pod reached state success, or an error if it reached failure or until slowPodStartupTimeout.
func WaitForPodSuccessInNamespaceSlow(c clientset.Interface, podName string, namespace string) error {
	return WaitForPodSuccessInNamespaceTimeout(c, podName, namespace, slowPodStartTimeout)
}

// WaitForPodNotFoundInNamespace returns an error if it takes too long for the pod to fully terminate.
// Unlike `waitForPodTerminatedInNamespace`, the pod's Phase and Reason are ignored. If the pod Get
// api returns IsNotFound then the wait stops and nil is returned. If the Get api returns an error other
// than "not found" then that error is returned and the wait stops.
func WaitForPodNotFoundInNamespace(c clientset.Interface, podName, ns string, timeout time.Duration) error {
	var lastPod *v1.Pod
	err := wait.PollImmediate(poll, timeout, func() (bool, error) {
		pod, err := c.CoreV1().Pods(ns).Get(context.TODO(), podName, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return true, nil // done
		}
		if err != nil {
			return handleWaitingAPIError(err, true, "getting pod %s", podIdentifier(ns, podName))
		}
		lastPod = pod
		return false, nil
	})
	if err == nil {
		return nil
	}
	if IsTimeout(err) && lastPod != nil {
		return TimeoutError(fmt.Sprintf("timed out while waiting for pod %s to be Not Found", podIdentifier(ns, podName)),
			lastPod,
		)
	}
	return maybeTimeoutError(err, "waiting for pod %s not found", podIdentifier(ns, podName))
}

// WaitForPodToDisappear waits the given timeout duration for the specified pod to disappear.
func WaitForPodToDisappear(c clientset.Interface, ns, podName string, label labels.Selector, interval, timeout time.Duration) error {
	var lastPod *v1.Pod
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		e2elog.Logf("Waiting for pod %s to disappear", podName)
		options := metav1.ListOptions{LabelSelector: label.String()}
		pods, err := c.CoreV1().Pods(ns).List(context.TODO(), options)
		if err != nil {
			return handleWaitingAPIError(err, true, "listing pods")
		}
		found := false
		for i, pod := range pods.Items {
			if pod.Name == podName {
				e2elog.Logf("Pod %s still exists", podName)
				found = true
				lastPod = &(pods.Items[i])
				break
			}
		}
		if !found {
			e2elog.Logf("Pod %s no longer exists", podName)
			return true, nil
		}
		return false, nil
	})
	if err == nil {
		return nil
	}
	if IsTimeout(err) {
		return TimeoutError(fmt.Sprintf("timed out while waiting for pod %s to disappear", podIdentifier(ns, podName)),
			lastPod,
		)
	}
	return maybeTimeoutError(err, "waiting for pod %s to disappear", podIdentifier(ns, podName))
}

// PodsResponding waits for the pods to response.
func PodsResponding(c clientset.Interface, ns, name string, wantName bool, pods *v1.PodList) error {
	ginkgo.By("trying to dial each unique pod")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	err := wait.PollImmediate(poll, podRespondingTimeout, NewProxyResponseChecker(c, ns, label, name, wantName, pods).CheckAllResponses)
	return maybeTimeoutError(err, "waiting for pods to be responsive")
}

// WaitForNumberOfPods waits up to timeout to ensure there are exact
// `num` pods in namespace `ns`.
// It returns the matching Pods or a timeout error.
func WaitForNumberOfPods(c clientset.Interface, ns string, num int, timeout time.Duration) (pods *v1.PodList, err error) {
	actualNum := 0
	err = wait.PollImmediate(poll, timeout, func() (bool, error) {
		pods, err = c.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			return handleWaitingAPIError(err, false, "listing pods")
		}
		actualNum = len(pods.Items)
		return actualNum == num, nil
	})
	return pods, maybeTimeoutError(err, "waiting for there to be exactly %d pods in namespace (last seen %d)", num, actualNum)
}

// WaitForPodsWithLabelScheduled waits for all matching pods to become scheduled and at least one
// matching pod exists.  Return the list of matching pods.
func WaitForPodsWithLabelScheduled(c clientset.Interface, ns string, label labels.Selector) (pods *v1.PodList, err error) {
	opts := metav1.ListOptions{LabelSelector: label.String()}
	return WaitForAllPodsCondition(c, ns, opts, 1, "scheduled", podScheduledBeforeTimeout, func(pod *v1.Pod) (bool, error) {
		if pod.Spec.NodeName == "" {
			return false, nil
		}
		return true, nil
	})
}

// WaitForPodsWithLabel waits up to podListTimeout for getting pods with certain label
func WaitForPodsWithLabel(c clientset.Interface, ns string, label labels.Selector) (*v1.PodList, error) {
	opts := metav1.ListOptions{LabelSelector: label.String()}
	return WaitForAllPodsCondition(c, ns, opts, 1, "existent", podListTimeout, func(pod *v1.Pod) (bool, error) {
		return true, nil
	})
}

// WaitForPodsWithLabelRunningReady waits for exact amount of matching pods to become running and ready.
// Return the list of matching pods.
func WaitForPodsWithLabelRunningReady(c clientset.Interface, ns string, label labels.Selector, num int, timeout time.Duration) (pods *v1.PodList, err error) {
	opts := metav1.ListOptions{LabelSelector: label.String()}
	return WaitForAllPodsCondition(c, ns, opts, 1, "running and ready", podListTimeout, testutils.PodRunningReady)
}

// WaitForNRestartablePods tries to list restarting pods using ps until it finds expect of them,
// returning their names if it can do so before timeout.
func WaitForNRestartablePods(ps *testutils.PodStore, expect int, timeout time.Duration) ([]string, error) {
	var pods []*v1.Pod
	var errLast error
	found := wait.Poll(poll, timeout, func() (bool, error) {
		allPods := ps.List()
		pods = FilterNonRestartablePods(allPods)
		if len(pods) != expect {
			errLast = fmt.Errorf("expected to find %d pods but found only %d", expect, len(pods))
			e2elog.Logf("Error getting pods: %v", errLast)
			return false, nil
		}
		return true, nil
	}) == nil
	podNames := make([]string, len(pods))
	for i, p := range pods {
		podNames[i] = p.ObjectMeta.Name
	}
	if !found {
		return podNames, fmt.Errorf("couldn't find %d pods within %v; last error: %v",
			expect, timeout, errLast)
	}
	return podNames, nil
}

// WaitForPodContainerToFail waits for the given Pod container to fail with the given reason, specifically due to
// invalid container configuration. In this case, the container will remain in a waiting state with a specific
// reason set, which should match the given reason.
func WaitForPodContainerToFail(c clientset.Interface, namespace, podName string, containerIndex int, reason string, timeout time.Duration) error {
	conditionDesc := fmt.Sprintf("container %d failed with reason %s", containerIndex, reason)
	return WaitForPodCondition(c, namespace, podName, conditionDesc, timeout, func(pod *v1.Pod) (bool, error) {
		switch pod.Status.Phase {
		case v1.PodPending:
			if len(pod.Status.ContainerStatuses) == 0 {
				return false, nil
			}
			containerStatus := pod.Status.ContainerStatuses[containerIndex]
			if containerStatus.State.Waiting != nil && containerStatus.State.Waiting.Reason == reason {
				return true, nil
			}
			return false, nil
		case v1.PodFailed, v1.PodRunning, v1.PodSucceeded:
			return false, fmt.Errorf("pod was expected to be pending, but it is in the state: %s", pod.Status.Phase)
		}
		return false, nil
	})
}

// WaitForPodContainerStarted waits for the given Pod container to start, after a successful run of the startupProbe.
func WaitForPodContainerStarted(c clientset.Interface, namespace, podName string, containerIndex int, timeout time.Duration) error {
	conditionDesc := fmt.Sprintf("container %d started", containerIndex)
	return WaitForPodCondition(c, namespace, podName, conditionDesc, timeout, func(pod *v1.Pod) (bool, error) {
		if containerIndex > len(pod.Status.ContainerStatuses)-1 {
			return false, nil
		}
		containerStatus := pod.Status.ContainerStatuses[containerIndex]
		return *containerStatus.Started, nil
	})
}

// WaitForPodFailedReason wait for pod failed reason in status, for example "SysctlForbidden".
func WaitForPodFailedReason(c clientset.Interface, pod *v1.Pod, reason string, timeout time.Duration) error {
	conditionDesc := fmt.Sprintf("failed with reason %s", reason)
	return WaitForPodCondition(c, pod.Namespace, pod.Name, conditionDesc, timeout, func(pod *v1.Pod) (bool, error) {
		switch pod.Status.Phase {
		case v1.PodSucceeded:
			return true, errors.New("pod succeeded unexpectedly")
		case v1.PodFailed:
			if pod.Status.Reason == reason {
				return true, nil
			} else {
				return true, fmt.Errorf("pod failed with reason %s", pod.Status.Reason)
			}
		}
		return false, nil
	})
}

// WaitForContainerRunning waits for the given Pod container to have a state of running
func WaitForContainerRunning(c clientset.Interface, namespace, podName, containerName string, timeout time.Duration) error {
	conditionDesc := fmt.Sprintf("container %s running", containerName)
	return WaitForPodCondition(c, namespace, podName, conditionDesc, timeout, func(pod *v1.Pod) (bool, error) {
		for _, statuses := range [][]v1.ContainerStatus{pod.Status.ContainerStatuses, pod.Status.InitContainerStatuses, pod.Status.EphemeralContainerStatuses} {
			for _, cs := range statuses {
				if cs.Name == containerName {
					return cs.State.Running != nil, nil
				}
			}
		}
		return false, nil
	})
}

// handleWaitingAPIErrror handles an error from an API request in the context of a Wait function.
// If the error is retryable, sleep the recommended delay and ignore the error.
// If the erorr is terminal, return it.
func handleWaitingAPIError(err error, retryNotFound bool, taskFormat string, taskArgs ...interface{}) (bool, error) {
	taskDescription := fmt.Sprintf(taskFormat, taskArgs...)
	if retryNotFound && apierrors.IsNotFound(err) {
		e2elog.Logf("Ignoring NotFound error while " + taskDescription)
		return false, nil
	}
	if retry, delay := shouldRetry(err); retry {
		e2elog.Logf("Retryable error while %s, retrying after %v: %v", taskDescription, delay, err)
		if delay > 0 {
			time.Sleep(delay)
		}
		return false, nil
	}
	e2elog.Logf("Encountered non-retryable error while %s: %v", taskDescription, err)
	return false, err
}

// Decide whether to retry an API request. Optionally include a delay to retry after.
func shouldRetry(err error) (retry bool, retryAfter time.Duration) {
	// if the error sends the Retry-After header, we respect it as an explicit confirmation we should retry.
	if delay, shouldRetry := apierrors.SuggestsClientDelay(err); shouldRetry {
		return shouldRetry, time.Duration(delay) * time.Second
	}

	// these errors indicate a transient error that should be retried.
	if apierrors.IsInternalError(err) || apierrors.IsTimeout(err) || apierrors.IsTooManyRequests(err) {
		return true, 0
	}

	return false, 0
}
