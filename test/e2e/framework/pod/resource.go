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
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/kubectl/pkg/util/podutils"

	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// errPodCompleted is returned by PodRunning or PodContainerRunning to indicate that
// the pod has already reached completed state.
var errPodCompleted = fmt.Errorf("pod ran to completion")

// LabelLogOnPodFailure can be used to mark which Pods will have their logs logged in the case of
// a test failure. By default, if there are no Pods with this label, only the first 5 Pods will
// have their logs fetched.
const LabelLogOnPodFailure = "log-on-pod-failure"

// TODO: Move to its own subpkg.
// expectNoError checks if "err" is set, and if so, fails assertion while logging the error.
func expectNoError(err error, explain ...interface{}) {
	expectNoErrorWithOffset(1, err, explain...)
}

// TODO: Move to its own subpkg.
// expectNoErrorWithOffset checks if "err" is set, and if so, fails assertion while logging the error at "offset" levels above its caller
// (for example, for call chain f -> g -> expectNoErrorWithOffset(1, ...) error would be logged for "f").
func expectNoErrorWithOffset(offset int, err error, explain ...interface{}) {
	if err != nil {
		e2elog.Logf("Unexpected error occurred: %v", err)
	}
	gomega.ExpectWithOffset(1+offset, err).NotTo(gomega.HaveOccurred(), explain...)
}

func isElementOf(podUID types.UID, pods *v1.PodList) bool {
	for _, pod := range pods.Items {
		if pod.UID == podUID {
			return true
		}
	}
	return false
}

// ProxyResponseChecker is a context for checking pods responses by issuing GETs to them (via the API
// proxy) and verifying that they answer with their own pod name.
type ProxyResponseChecker struct {
	c              clientset.Interface
	ns             string
	label          labels.Selector
	controllerName string
	respondName    bool // Whether the pod should respond with its own name.
	pods           *v1.PodList
}

// NewProxyResponseChecker returns a context for checking pods responses.
func NewProxyResponseChecker(c clientset.Interface, ns string, label labels.Selector, controllerName string, respondName bool, pods *v1.PodList) ProxyResponseChecker {
	return ProxyResponseChecker{c, ns, label, controllerName, respondName, pods}
}

// CheckAllResponses issues GETs to all pods in the context and verify they
// reply with their own pod name.
func (r ProxyResponseChecker) CheckAllResponses() (done bool, err error) {
	successes := 0
	options := metav1.ListOptions{LabelSelector: r.label.String()}
	currentPods, err := r.c.CoreV1().Pods(r.ns).List(context.TODO(), options)
	expectNoError(err, "Failed to get list of currentPods in namespace: %s", r.ns)
	for i, pod := range r.pods.Items {
		// Check that the replica list remains unchanged, otherwise we have problems.
		if !isElementOf(pod.UID, currentPods) {
			return false, fmt.Errorf("pod with UID %s is no longer a member of the replica set.  Must have been restarted for some reason.  Current replica set: %v", pod.UID, currentPods)
		}

		ctx, cancel := context.WithTimeout(context.Background(), singleCallTimeout)
		defer cancel()

		body, err := r.c.CoreV1().RESTClient().Get().
			Namespace(r.ns).
			Resource("pods").
			SubResource("proxy").
			Name(string(pod.Name)).
			Do(ctx).
			Raw()

		if err != nil {
			if ctx.Err() != nil {
				// We may encounter errors here because of a race between the pod readiness and apiserver
				// proxy. So, we log the error and retry if this occurs.
				e2elog.Logf("Controller %s: Failed to Get from replica %d [%s]: %v\n pod status: %#v", r.controllerName, i+1, pod.Name, err, pod.Status)
				return false, nil
			}
			e2elog.Logf("Controller %s: Failed to GET from replica %d [%s]: %v\npod status: %#v", r.controllerName, i+1, pod.Name, err, pod.Status)
			continue
		}
		// The response checker expects the pod's name unless !respondName, in
		// which case it just checks for a non-empty response.
		got := string(body)
		what := ""
		if r.respondName {
			what = "expected"
			want := pod.Name
			if got != want {
				e2elog.Logf("Controller %s: Replica %d [%s] expected response %q but got %q",
					r.controllerName, i+1, pod.Name, want, got)
				continue
			}
		} else {
			what = "non-empty"
			if len(got) == 0 {
				e2elog.Logf("Controller %s: Replica %d [%s] expected non-empty response",
					r.controllerName, i+1, pod.Name)
				continue
			}
		}
		successes++
		e2elog.Logf("Controller %s: Got %s result from replica %d [%s]: %q, %d of %d required successes so far",
			r.controllerName, what, i+1, pod.Name, got, successes, len(r.pods.Items))
	}
	if successes < len(r.pods.Items) {
		return false, nil
	}
	return true, nil
}

func podRunning(c clientset.Interface, podName, namespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		switch pod.Status.Phase {
		case v1.PodRunning:
			return true, nil
		case v1.PodFailed, v1.PodSucceeded:
			return false, errPodCompleted
		}
		return false, nil
	}
}

func podCompleted(c clientset.Interface, podName, namespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		switch pod.Status.Phase {
		case v1.PodFailed, v1.PodSucceeded:
			return true, nil
		}
		return false, nil
	}
}

func podRunningAndReady(c clientset.Interface, podName, namespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		switch pod.Status.Phase {
		case v1.PodFailed, v1.PodSucceeded:
			e2elog.Logf("The status of Pod %s is %s which is unexpected", podName, pod.Status.Phase)
			return false, errPodCompleted
		case v1.PodRunning:
			e2elog.Logf("The status of Pod %s is %s (Ready = %v)", podName, pod.Status.Phase, podutils.IsPodReady(pod))
			return podutils.IsPodReady(pod), nil
		}
		e2elog.Logf("The status of Pod %s is %s, waiting for it to be Running (with Ready = true)", podName, pod.Status.Phase)
		return false, nil
	}
}

func podNotPending(c clientset.Interface, podName, namespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		switch pod.Status.Phase {
		case v1.PodPending:
			return false, nil
		default:
			return true, nil
		}
	}
}

// PodsCreated returns a pod list matched by the given name.
func PodsCreated(c clientset.Interface, ns, name string, replicas int32) (*v1.PodList, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	return PodsCreatedByLabel(c, ns, name, replicas, label)
}

// PodsCreatedByLabel returns a created pod list matched by the given label.
func PodsCreatedByLabel(c clientset.Interface, ns, name string, replicas int32, label labels.Selector) (*v1.PodList, error) {
	timeout := 2 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		options := metav1.ListOptions{LabelSelector: label.String()}

		// List the pods, making sure we observe all the replicas.
		pods, err := c.CoreV1().Pods(ns).List(context.TODO(), options)
		if err != nil {
			return nil, err
		}

		created := []v1.Pod{}
		for _, pod := range pods.Items {
			if pod.DeletionTimestamp != nil {
				continue
			}
			created = append(created, pod)
		}
		e2elog.Logf("Pod name %s: Found %d pods out of %d", name, len(created), replicas)

		if int32(len(created)) == replicas {
			pods.Items = created
			return pods, nil
		}
	}
	return nil, fmt.Errorf("Pod name %s: Gave up waiting %v for %d pods to come up", name, timeout, replicas)
}

// VerifyPods checks if the specified pod is responding.
func VerifyPods(c clientset.Interface, ns, name string, wantName bool, replicas int32) error {
	return podRunningMaybeResponding(c, ns, name, wantName, replicas, true)
}

// VerifyPodsRunning checks if the specified pod is running.
func VerifyPodsRunning(c clientset.Interface, ns, name string, wantName bool, replicas int32) error {
	return podRunningMaybeResponding(c, ns, name, wantName, replicas, false)
}

func podRunningMaybeResponding(c clientset.Interface, ns, name string, wantName bool, replicas int32, checkResponding bool) error {
	pods, err := PodsCreated(c, ns, name, replicas)
	if err != nil {
		return err
	}
	e := podsRunning(c, pods)
	if len(e) > 0 {
		return fmt.Errorf("failed to wait for pods running: %v", e)
	}
	if checkResponding {
		err = PodsResponding(c, ns, name, wantName, pods)
		if err != nil {
			return fmt.Errorf("failed to wait for pods responding: %v", err)
		}
	}
	return nil
}

func podsRunning(c clientset.Interface, pods *v1.PodList) []error {
	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	ginkgo.By("ensuring each pod is running")
	e := []error{}
	errorChan := make(chan error)

	for _, pod := range pods.Items {
		go func(p v1.Pod) {
			errorChan <- WaitForPodRunningInNamespace(c, &p)
		}(pod)
	}

	for range pods.Items {
		err := <-errorChan
		if err != nil {
			e = append(e, err)
		}
	}

	return e
}

func podContainerFailed(c clientset.Interface, namespace, podName string, containerIndex int, reason string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
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
	}
}

func podContainerStarted(c clientset.Interface, namespace, podName string, containerIndex int) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if containerIndex > len(pod.Status.ContainerStatuses)-1 {
			return false, nil
		}
		containerStatus := pod.Status.ContainerStatuses[containerIndex]
		return *containerStatus.Started, nil
	}
}

func isContainerRunning(c clientset.Interface, namespace, podName, containerName string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(context.TODO(), podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, statuses := range [][]v1.ContainerStatus{pod.Status.ContainerStatuses, pod.Status.InitContainerStatuses, pod.Status.EphemeralContainerStatuses} {
			for _, cs := range statuses {
				if cs.Name == containerName {
					return cs.State.Running != nil, nil
				}
			}
		}
		return false, nil
	}
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
	e2elog.Logf("%-[1]*[2]s %-[3]*[4]s %-[5]*[6]s %-[7]*[8]s %[9]s",
		maxPodW, "POD", maxNodeW, "NODE", maxPhaseW, "PHASE", maxGraceW, "GRACE", "CONDITIONS")
	for _, pod := range pods {
		grace := ""
		if pod.DeletionGracePeriodSeconds != nil {
			grace = fmt.Sprintf("%ds", *pod.DeletionGracePeriodSeconds)
		}
		e2elog.Logf("%-[1]*[2]s %-[3]*[4]s %-[5]*[6]s %-[7]*[8]s %[9]s",
			maxPodW, pod.ObjectMeta.Name, maxNodeW, pod.Spec.NodeName, maxPhaseW, pod.Status.Phase, maxGraceW, grace, pod.Status.Conditions)
	}
	e2elog.Logf("") // Final empty line helps for readability.
}

// logPodTerminationMessages logs termination messages for failing pods.  It's a short snippet (much smaller than full logs), but it often shows
// why pods crashed and since it is in the API, it's fast to retrieve.
func logPodTerminationMessages(pods []v1.Pod) {
	for _, pod := range pods {
		for _, status := range pod.Status.InitContainerStatuses {
			if status.LastTerminationState.Terminated != nil && len(status.LastTerminationState.Terminated.Message) > 0 {
				e2elog.Logf("%s[%s].initContainer[%s]=%s", pod.Name, pod.Namespace, status.Name, status.LastTerminationState.Terminated.Message)
			}
		}
		for _, status := range pod.Status.ContainerStatuses {
			if status.LastTerminationState.Terminated != nil && len(status.LastTerminationState.Terminated.Message) > 0 {
				e2elog.Logf("%s[%s].container[%s]=%s", pod.Name, pod.Namespace, status.Name, status.LastTerminationState.Terminated.Message)
			}
		}
	}
}

// logPodLogs logs the container logs from pods in the given namespace. This can be helpful for debugging
// issues that do not cause the container to fail (e.g.: network connectivity issues)
// We will log the Pods that have the LabelLogOnPodFailure label. If there aren't any, we default to
// logging only the first 5 Pods. This requires the reportDir to be set, and the pods are logged into:
// {report_dir}/pods/{namespace}/{pod}/{container_name}/logs.txt
func logPodLogs(c clientset.Interface, namespace string, pods []v1.Pod, reportDir string) {
	if reportDir == "" {
		return
	}

	var logPods []v1.Pod
	for _, pod := range pods {
		if _, ok := pod.Labels[LabelLogOnPodFailure]; ok {
			logPods = append(logPods, pod)
		}
	}
	maxPods := len(logPods)

	// There are no pods with the LabelLogOnPodFailure label, we default to the first 5 Pods.
	if maxPods == 0 {
		logPods = pods
		maxPods = len(pods)
		if maxPods > 5 {
			maxPods = 5
		}
	}

	tailLen := 42
	for i := 0; i < maxPods; i++ {
		pod := logPods[i]
		for _, container := range pod.Spec.Containers {
			logs, err := getPodLogsInternal(c, namespace, pod.Name, container.Name, false, nil, &tailLen)
			if err != nil {
				e2elog.Logf("Unable to fetch %s/%s/%s logs: %v", pod.Namespace, pod.Name, container.Name, err)
				continue
			}

			logDir := filepath.Join(reportDir, namespace, pod.Name, container.Name)
			err = os.MkdirAll(logDir, 0755)
			if err != nil {
				e2elog.Logf("Unable to create path '%s'. Err: %v", logDir, err)
				continue
			}

			logPath := filepath.Join(logDir, "logs.txt")
			err = os.WriteFile(logPath, []byte(logs), 0644)
			if err != nil {
				e2elog.Logf("Could not write the container logs in: %s. Err: %v", logPath, err)
			}
		}
	}
}

// DumpAllPodInfoForNamespace logs all pod information for a given namespace.
func DumpAllPodInfoForNamespace(c clientset.Interface, namespace, reportDir string) {
	pods, err := c.CoreV1().Pods(namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		e2elog.Logf("unable to fetch pod debug info: %v", err)
	}
	LogPodStates(pods.Items)
	logPodTerminationMessages(pods.Items)
	logPodLogs(c, namespace, pods.Items, reportDir)
}

// FilterNonRestartablePods filters out pods that will never get recreated if
// deleted after termination.
func FilterNonRestartablePods(pods []*v1.Pod) []*v1.Pod {
	var results []*v1.Pod
	for _, p := range pods {
		if isNotRestartAlwaysMirrorPod(p) {
			// Mirror pods with restart policy == Never will not get
			// recreated if they are deleted after the pods have
			// terminated. For now, we discount such pods.
			// https://github.com/kubernetes/kubernetes/issues/34003
			continue
		}
		results = append(results, p)
	}
	return results
}

func isNotRestartAlwaysMirrorPod(p *v1.Pod) bool {
	// Check if the pod is a mirror pod
	if _, ok := p.Annotations[v1.MirrorPodAnnotationKey]; !ok {
		return false
	}
	return p.Spec.RestartPolicy != v1.RestartPolicyAlways
}

// NewAgnhostPod returns a pod that uses the agnhost image. The image's binary supports various subcommands
// that behave the same, no matter the underlying OS. If no args are given, it defaults to the pause subcommand.
// For more information about agnhost subcommands, see: https://github.com/kubernetes/kubernetes/tree/master/test/images/agnhost#agnhost
func NewAgnhostPod(ns, podName string, volumes []v1.Volume, mounts []v1.VolumeMount, ports []v1.ContainerPort, args ...string) *v1.Pod {
	immediate := int64(0)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName,
			Namespace: ns,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				NewAgnhostContainer("agnhost-container", mounts, ports, args...),
			},
			Volumes:                       volumes,
			SecurityContext:               &v1.PodSecurityContext{},
			TerminationGracePeriodSeconds: &immediate,
		},
	}
	return pod
}

// NewAgnhostContainer returns the container Spec of an agnhost container.
func NewAgnhostContainer(containerName string, mounts []v1.VolumeMount, ports []v1.ContainerPort, args ...string) v1.Container {
	if len(args) == 0 {
		args = []string{"pause"}
	}
	return v1.Container{
		Name:            containerName,
		Image:           imageutils.GetE2EImage(imageutils.Agnhost),
		Args:            args,
		VolumeMounts:    mounts,
		Ports:           ports,
		SecurityContext: &v1.SecurityContext{},
		ImagePullPolicy: v1.PullIfNotPresent,
	}
}

// NewExecPodSpec returns the pod spec of hostexec pod
func NewExecPodSpec(ns, name string, hostNetwork bool) *v1.Pod {
	pod := NewAgnhostPod(ns, name, nil, nil, nil)
	pod.Spec.HostNetwork = hostNetwork
	return pod
}

// newExecPodSpec returns the pod spec of exec pod
func newExecPodSpec(ns, generateName string) *v1.Pod {
	// GenerateName is an optional prefix, used by the server,
	// to generate a unique name ONLY IF the Name field has not been provided
	pod := NewAgnhostPod(ns, "", nil, nil, nil)
	pod.ObjectMeta.GenerateName = generateName
	return pod
}

// CreateExecPodOrFail creates a agnhost pause pod used as a vessel for kubectl exec commands.
// Pod name is uniquely generated.
func CreateExecPodOrFail(client clientset.Interface, ns, generateName string, tweak func(*v1.Pod)) *v1.Pod {
	e2elog.Logf("Creating new exec pod")
	pod := newExecPodSpec(ns, generateName)
	if tweak != nil {
		tweak(pod)
	}
	execPod, err := client.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
	expectNoError(err, "failed to create new exec pod in namespace: %s", ns)
	err = wait.PollImmediate(poll, 5*time.Minute, func() (bool, error) {
		retrievedPod, err := client.CoreV1().Pods(execPod.Namespace).Get(context.TODO(), execPod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return retrievedPod.Status.Phase == v1.PodRunning, nil
	})
	expectNoError(err, "failed to create new exec pod in namespace: %s", ns)
	return execPod
}

// CheckPodsRunningReady returns whether all pods whose names are listed in
// podNames in namespace ns are running and ready, using c and waiting at most
// timeout.
func CheckPodsRunningReady(c clientset.Interface, ns string, podNames []string, timeout time.Duration) bool {
	return checkPodsCondition(c, ns, podNames, timeout, testutils.PodRunningReady, "running and ready")
}

// CheckPodsRunningReadyOrSucceeded returns whether all pods whose names are
// listed in podNames in namespace ns are running and ready, or succeeded; use
// c and waiting at most timeout.
func CheckPodsRunningReadyOrSucceeded(c clientset.Interface, ns string, podNames []string, timeout time.Duration) bool {
	return checkPodsCondition(c, ns, podNames, timeout, testutils.PodRunningReadyOrSucceeded, "running and ready, or succeeded")
}

// checkPodsCondition returns whether all pods whose names are listed in podNames
// in namespace ns are in the condition, using c and waiting at most timeout.
func checkPodsCondition(c clientset.Interface, ns string, podNames []string, timeout time.Duration, condition podCondition, desc string) bool {
	np := len(podNames)
	e2elog.Logf("Waiting up to %v for %d pods to be %s: %s", timeout, np, desc, podNames)
	type waitPodResult struct {
		success bool
		podName string
	}
	result := make(chan waitPodResult, len(podNames))
	for _, podName := range podNames {
		// Launch off pod readiness checkers.
		go func(name string) {
			err := WaitForPodCondition(c, ns, name, desc, timeout, condition)
			result <- waitPodResult{err == nil, name}
		}(podName)
	}
	// Wait for them all to finish.
	success := true
	for range podNames {
		res := <-result
		if !res.success {
			e2elog.Logf("Pod %[1]s failed to be %[2]s.", res.podName, desc)
			success = false
		}
	}
	e2elog.Logf("Wanted all %d pods to be %s. Result: %t. Pods: %v", np, desc, success, podNames)
	return success
}

// GetPodLogs returns the logs of the specified container (namespace/pod/container).
func GetPodLogs(c clientset.Interface, namespace, podName, containerName string) (string, error) {
	return getPodLogsInternal(c, namespace, podName, containerName, false, nil, nil)
}

// GetPodLogsSince returns the logs of the specified container (namespace/pod/container) since a timestamp.
func GetPodLogsSince(c clientset.Interface, namespace, podName, containerName string, since time.Time) (string, error) {
	sinceTime := metav1.NewTime(since)
	return getPodLogsInternal(c, namespace, podName, containerName, false, &sinceTime, nil)
}

// GetPreviousPodLogs returns the logs of the previous instance of the
// specified container (namespace/pod/container).
func GetPreviousPodLogs(c clientset.Interface, namespace, podName, containerName string) (string, error) {
	return getPodLogsInternal(c, namespace, podName, containerName, true, nil, nil)
}

// utility function for gomega Eventually
func getPodLogsInternal(c clientset.Interface, namespace, podName, containerName string, previous bool, sinceTime *metav1.Time, tailLines *int) (string, error) {
	request := c.CoreV1().RESTClient().Get().
		Resource("pods").
		Namespace(namespace).
		Name(podName).SubResource("log").
		Param("container", containerName).
		Param("previous", strconv.FormatBool(previous))
	if sinceTime != nil {
		request.Param("sinceTime", sinceTime.Format(time.RFC3339))
	}
	if tailLines != nil {
		request.Param("tailLines", strconv.Itoa(*tailLines))
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

// GetPodsInNamespace returns the pods in the given namespace.
func GetPodsInNamespace(c clientset.Interface, ns string, ignoreLabels map[string]string) ([]*v1.Pod, error) {
	pods, err := c.CoreV1().Pods(ns).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return []*v1.Pod{}, err
	}
	ignoreSelector := labels.SelectorFromSet(ignoreLabels)
	var filtered []*v1.Pod
	for i := range pods.Items {
		p := pods.Items[i]
		if len(ignoreLabels) != 0 && ignoreSelector.Matches(labels.Set(p.Labels)) {
			continue
		}
		filtered = append(filtered, &p)
	}
	return filtered, nil
}

// GetPods return the label matched pods in the given ns
func GetPods(c clientset.Interface, ns string, matchLabels map[string]string) ([]v1.Pod, error) {
	label := labels.SelectorFromSet(matchLabels)
	listOpts := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := c.CoreV1().Pods(ns).List(context.TODO(), listOpts)
	if err != nil {
		return []v1.Pod{}, err
	}
	return pods.Items, nil
}

// GetPodSecretUpdateTimeout returns the timeout duration for updating pod secret.
func GetPodSecretUpdateTimeout(c clientset.Interface) time.Duration {
	// With SecretManager(ConfigMapManager), we may have to wait up to full sync period +
	// TTL of secret(configmap) to elapse before the Kubelet projects the update into the
	// volume and the container picks it up.
	// So this timeout is based on default Kubelet sync period (1 minute) + maximum TTL for
	// secret(configmap) that's based on cluster size + additional time as a fudge factor.
	secretTTL, err := getNodeTTLAnnotationValue(c)
	if err != nil {
		e2elog.Logf("Couldn't get node TTL annotation (using default value of 0): %v", err)
	}
	podLogTimeout := 240*time.Second + secretTTL
	return podLogTimeout
}

func getNodeTTLAnnotationValue(c clientset.Interface) (time.Duration, error) {
	nodes, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil || len(nodes.Items) == 0 {
		return time.Duration(0), fmt.Errorf("Couldn't list any nodes to get TTL annotation: %v", err)
	}
	// Since TTL the kubelet is using is stored in node object, for the timeout
	// purpose we take it from the first node (all of them should be the same).
	node := &nodes.Items[0]
	if node.Annotations == nil {
		return time.Duration(0), fmt.Errorf("No annotations found on the node")
	}
	value, ok := node.Annotations[v1.ObjectTTLAnnotationKey]
	if !ok {
		return time.Duration(0), fmt.Errorf("No TTL annotation found on the node")
	}
	intValue, err := strconv.Atoi(value)
	if err != nil {
		return time.Duration(0), fmt.Errorf("Cannot convert TTL annotation from %#v to int", *node)
	}
	return time.Duration(intValue) * time.Second, nil
}

// FilterActivePods returns pods that have not terminated.
func FilterActivePods(pods []*v1.Pod) []*v1.Pod {
	var result []*v1.Pod
	for _, p := range pods {
		if IsPodActive(p) {
			result = append(result, p)
		} else {
			klog.V(4).Infof("Ignoring inactive pod %v/%v in state %v, deletion time %v",
				p.Namespace, p.Name, p.Status.Phase, p.DeletionTimestamp)
		}
	}
	return result
}

// IsPodActive return true if the pod meets certain conditions.
func IsPodActive(p *v1.Pod) bool {
	return v1.PodSucceeded != p.Status.Phase &&
		v1.PodFailed != p.Status.Phase &&
		p.DeletionTimestamp == nil
}
