/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"math"
	"math/rand"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/client/cache"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/davecgh/go-spew/spew"
	"github.com/prometheus/client_golang/extraction"
	"github.com/prometheus/client_golang/model"
	"golang.org/x/crypto/ssh"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// Initial pod start can be delayed O(minutes) by slow docker pulls
	// TODO: Make this 30 seconds once #4566 is resolved.
	podStartTimeout = 5 * time.Minute

	// How long to wait for a service endpoint to be resolvable.
	serviceStartTimeout = 1 * time.Minute

	// String used to mark pod deletion
	nonExist = "NonExist"

	// How often to poll pods and nodes.
	poll = 5 * time.Second

	// service accounts are provisioned after namespace creation
	// a service account is required to support pod creation in a namespace as part of admission control
	serviceAccountProvisionTimeout = 2 * time.Minute

	// How long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	singleCallTimeout = 30 * time.Second

	// How long nodes have to be "ready" when a test begins. They should already
	// be "ready" before the test starts, so this is small.
	nodeReadyInitialTimeout = 20 * time.Second

	// How long pods have to be "ready" when a test begins. They should already
	// be "ready" before the test starts, so this is small.
	podReadyBeforeTimeout = 20 * time.Second

	podRespondingTimeout     = 2 * time.Minute
	serviceRespondingTimeout = 2 * time.Minute

	// How wide to print pod names, by default. Useful for aligning printing to
	// quickly scan through output.
	podPrintWidth = 55
)

type CloudConfig struct {
	ProjectID         string
	Zone              string
	Cluster           string
	MasterName        string
	NodeInstanceGroup string
	NumNodes          int
	ClusterTag        string

	Provider cloudprovider.Interface
}

type TestContextType struct {
	KubeConfig            string
	KubeContext           string
	CertDir               string
	Host                  string
	RepoRoot              string
	Provider              string
	CloudConfig           CloudConfig
	KubectlPath           string
	OutputDir             string
	prefix                string
	MinStartupPods        int
	UpgradeTarget         string
	PrometheusPushGateway string
}

var testContext TestContextType

func SetTestContext(t TestContextType) {
	testContext = t
}

type ContainerFailures struct {
	status   *api.ContainerStateTerminated
	restarts int
}

// Convenient wrapper around cache.Store that returns list of api.Pod instead of interface{}.
type podStore struct {
	cache.Store
	stopCh chan struct{}
}

func newPodStore(c *client.Client, namespace string, label labels.Selector, field fields.Selector) *podStore {
	lw := &cache.ListWatch{
		ListFunc: func() (runtime.Object, error) {
			return c.Pods(namespace).List(label, field)
		},
		WatchFunc: func(rv string) (watch.Interface, error) {
			return c.Pods(namespace).Watch(label, field, rv)
		},
	}
	store := cache.NewStore(cache.MetaNamespaceKeyFunc)
	stopCh := make(chan struct{})
	cache.NewReflector(lw, &api.Pod{}, store, 0).RunUntil(stopCh)
	return &podStore{store, stopCh}
}

func (s *podStore) List() []*api.Pod {
	objects := s.Store.List()
	pods := make([]*api.Pod, 0)
	for _, o := range objects {
		pods = append(pods, o.(*api.Pod))
	}
	return pods
}

func (s *podStore) Stop() {
	close(s.stopCh)
}

type RCConfig struct {
	Client        *client.Client
	Image         string
	Command       []string
	Name          string
	Namespace     string
	PollInterval  time.Duration
	Timeout       time.Duration
	PodStatusFile *os.File
	Replicas      int
	CpuRequest    int64 // millicores
	CpuLimit      int64 // millicores
	MemRequest    int64 // bytes
	MemLimit      int64 // bytes

	// Env vars, set the same for every pod.
	Env map[string]string

	// Extra labels added to every pod.
	Labels map[string]string

	// Ports to declare in the container (map of name to containerPort).
	Ports map[string]int

	// Pointer to a list of pods; if non-nil, will be set to a list of pods
	// created by this RC by RunRC.
	CreatedPods *[]*api.Pod

	// Maximum allowable container failures. If exceeded, RunRC returns an error.
	// Defaults to replicas*0.1 if unspecified.
	MaxContainerFailures *int
}

func nowStamp() string {
	return time.Now().Format(time.StampMilli)
}

func Logf(format string, a ...interface{}) {
	fmt.Fprintf(GinkgoWriter, nowStamp()+": INFO: "+format+"\n", a...)
}

func Failf(format string, a ...interface{}) {
	Fail(nowStamp()+": "+fmt.Sprintf(format, a...), 1)
}

func Skipf(format string, args ...interface{}) {
	Skip(nowStamp() + ": " + fmt.Sprintf(format, args...))
}

func SkipUnlessNodeCountIsAtLeast(minNodeCount int) {
	if testContext.CloudConfig.NumNodes < minNodeCount {
		Skipf("Requires at least %d nodes (not %d)", minNodeCount, testContext.CloudConfig.NumNodes)
	}
}

func SkipIfProviderIs(unsupportedProviders ...string) {
	if providerIs(unsupportedProviders...) {
		Skipf("Not supported for providers %v (found %s)", unsupportedProviders, testContext.Provider)
	}
}

func SkipUnlessProviderIs(supportedProviders ...string) {
	if !providerIs(supportedProviders...) {
		Skipf("Only supported for providers %v (not %s)", supportedProviders, testContext.Provider)
	}
}

func providerIs(providers ...string) bool {
	for _, provider := range providers {
		if strings.ToLower(provider) == strings.ToLower(testContext.Provider) {
			return true
		}
	}
	return false
}

type podCondition func(pod *api.Pod) (bool, error)

// podReady returns whether pod has a condition of Ready with a status of true.
func podReady(pod *api.Pod) bool {
	for _, cond := range pod.Status.Conditions {
		if cond.Type == api.PodReady && cond.Status == api.ConditionTrue {
			return true
		}
	}
	return false
}

// logPodStates logs basic info of provided pods for debugging.
func logPodStates(pods []api.Pod) {
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
	Logf("%-[1]*[2]s %-[3]*[4]s %-[5]*[6]s %-[7]*[8]s %[9]s",
		maxPodW, "POD", maxNodeW, "NODE", maxPhaseW, "PHASE", maxGraceW, "GRACE", "CONDITIONS")
	for _, pod := range pods {
		grace := ""
		if pod.DeletionGracePeriodSeconds != nil {
			grace = fmt.Sprintf("%ds", *pod.DeletionGracePeriodSeconds)
		}
		Logf("%-[1]*[2]s %-[3]*[4]s %-[5]*[6]s %[7]s %[8]s",
			maxPodW, pod.ObjectMeta.Name, maxNodeW, pod.Spec.NodeName, maxPhaseW, pod.Status.Phase, grace, pod.Status.Conditions)
	}
	Logf("") // Final empty line helps for readability.
}

// podRunningReady checks whether pod p's phase is running and it has a ready
// condition of status true.
func podRunningReady(p *api.Pod) (bool, error) {
	// Check the phase is running.
	if p.Status.Phase != api.PodRunning {
		return false, fmt.Errorf("want pod '%s' on '%s' to be '%v' but was '%v'",
			p.ObjectMeta.Name, p.Spec.NodeName, api.PodRunning, p.Status.Phase)
	}
	// Check the ready condition is true.
	if !podReady(p) {
		return false, fmt.Errorf("pod '%s' on '%s' didn't have condition {%v %v}; conditions: %v",
			p.ObjectMeta.Name, p.Spec.NodeName, api.PodReady, api.ConditionTrue, p.Status.Conditions)
	}
	return true, nil
}

// check if a Pod is controlled by a Replication Controller in the List
func hasReplicationControllersForPod(rcs *api.ReplicationControllerList, pod api.Pod) bool {
	for _, rc := range rcs.Items {
		selector := labels.SelectorFromSet(rc.Spec.Selector)
		if selector.Matches(labels.Set(pod.ObjectMeta.Labels)) {
			return true
		}
	}
	return false
}

// waitForPodsRunningReady waits up to timeout to ensure that all pods in
// namespace ns are either running and ready, or failed but controlled by a
// replication controller. Also, it ensures that at least minPods are running
// and ready. It has separate behavior from other 'wait for' pods functions in
// that it requires the list of pods on every iteration. This is useful, for
// example, in cluster startup, because the number of pods increases while
// waiting.
func waitForPodsRunningReady(ns string, minPods int, timeout time.Duration) error {
	c, err := loadClient()
	if err != nil {
		return err
	}
	start := time.Now()
	Logf("Waiting up to %v for all pods (need at least %d) in namespace '%s' to be running and ready",
		timeout, minPods, ns)
	if wait.Poll(poll, timeout, func() (bool, error) {
		// We get the new list of pods and replication controllers in every
		// iteration because more pods come online during startup and we want to
		// ensure they are also checked.
		rcList, err := c.ReplicationControllers(ns).List(labels.Everything())
		if err != nil {
			Logf("Error getting replication controllers in namespace '%s': %v", ns, err)
			return false, nil
		}
		replicas := 0
		for _, rc := range rcList.Items {
			replicas += rc.Spec.Replicas
		}

		podList, err := c.Pods(ns).List(labels.Everything(), fields.Everything())
		if err != nil {
			Logf("Error getting pods in namespace '%s': %v", ns, err)
			return false, nil
		}
		nOk, replicaOk, badPods := 0, 0, []api.Pod{}
		for _, pod := range podList.Items {
			if res, err := podRunningReady(&pod); res && err == nil {
				nOk++
				if hasReplicationControllersForPod(rcList, pod) {
					replicaOk++
				}
			} else {
				if pod.Status.Phase != api.PodFailed {
					Logf("The status of Pod %s is %s, waiting for it to be either Running or Failed", pod.ObjectMeta.Name, pod.Status.Phase)
					badPods = append(badPods, pod)
				} else if !hasReplicationControllersForPod(rcList, pod) {
					Logf("Pod %s is Failed, but it's not controlled by a ReplicationController", pod.ObjectMeta.Name)
					badPods = append(badPods, pod)
				}
				//ignore failed pods that are controlled by a replication controller
			}
		}

		Logf("%d / %d pods in namespace '%s' are running and ready (%d seconds elapsed)",
			nOk, len(podList.Items), ns, int(time.Since(start).Seconds()))
		Logf("expected %d pod replicas in namespace '%s', %d are Running and Ready.", replicas, ns, replicaOk)

		if replicaOk == replicas && nOk >= minPods && len(badPods) == 0 {
			return true, nil
		}
		logPodStates(badPods)
		return false, nil
	}) != nil {
		return fmt.Errorf("Not all pods in namespace '%s' running and ready within %v", ns, timeout)
	}
	return nil
}

func waitForServiceAccountInNamespace(c *client.Client, ns, serviceAccountName string, timeout time.Duration) error {
	Logf("Waiting up to %v for service account %s to be provisioned in ns %s", timeout, serviceAccountName, ns)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		sa, err := c.ServiceAccounts(ns).Get(serviceAccountName)
		if apierrs.IsNotFound(err) {
			Logf("Get service account %s in ns %s failed, ignoring for %v: %v", serviceAccountName, ns, poll, err)
			continue
		}
		if err != nil {
			Logf("Get service account %s in ns %s failed: %v", serviceAccountName, ns, err)
			return err
		}
		if len(sa.Secrets) == 0 {
			Logf("Service account %s in ns %s had 0 secrets, ignoring for %v: %v", serviceAccountName, ns, poll, err)
			continue
		}
		Logf("Service account %s in ns %s with secrets found. (%v)", serviceAccountName, ns, time.Since(start))
		return nil
	}
	return fmt.Errorf("Service account %s in namespace %s not ready within %v", serviceAccountName, ns, timeout)
}

func waitForPodCondition(c *client.Client, ns, podName, desc string, timeout time.Duration, condition podCondition) error {
	Logf("Waiting up to %[1]v for pod %-[2]*[3]s status to be %[4]s", timeout, podPrintWidth, podName, desc)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		pod, err := c.Pods(ns).Get(podName)
		if err != nil {
			// Aligning this text makes it much more readable
			Logf("Get pod %-[1]*[2]s in namespace '%[3]s' failed, ignoring for %[4]v. Error: %[5]v",
				podPrintWidth, podName, ns, poll, err)
			continue
		}
		done, err := condition(pod)
		if done {
			return err
		}
		Logf("Waiting for pod %-[1]*[2]s in namespace '%[3]s' status to be '%[4]s'"+
			"(found phase: %[5]q, readiness: %[6]t) (%[7]v elapsed)",
			podPrintWidth, podName, ns, desc, pod.Status.Phase, podReady(pod), time.Since(start))
	}
	return fmt.Errorf("gave up waiting for pod '%s' to be '%s' after %v", podName, desc, timeout)
}

// waitForDefaultServiceAccountInNamespace waits for the default service account to be provisioned
// the default service account is what is associated with pods when they do not specify a service account
// as a result, pods are not able to be provisioned in a namespace until the service account is provisioned
func waitForDefaultServiceAccountInNamespace(c *client.Client, namespace string) error {
	return waitForServiceAccountInNamespace(c, namespace, "default", serviceAccountProvisionTimeout)
}

// waitForPersistentVolumePhase waits for a PersistentVolume to be in a specific phase or until timeout occurs, whichever comes first.
func waitForPersistentVolumePhase(phase api.PersistentVolumePhase, c *client.Client, pvName string, poll, timeout time.Duration) error {
	Logf("Waiting up to %v for PersistentVolume %s to have phase %s", timeout, pvName, phase)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		pv, err := c.PersistentVolumes().Get(pvName)
		if err != nil {
			Logf("Get persistent volume %s in failed, ignoring for %v: %v", pvName, poll, err)
			continue
		} else {
			if pv.Status.Phase == phase {
				Logf("PersistentVolume %s found and phase=%s (%v)", pvName, phase, time.Since(start))
				return nil
			} else {
				Logf("PersistentVolume %s found but phase is %s instead of %s.", pvName, pv.Status.Phase, phase)
			}
		}
	}
	return fmt.Errorf("PersistentVolume %s not in phase %s within %v", pvName, phase, timeout)
}

// createTestingNS should be used by every test, note that we append a common prefix to the provided test name.
// Please see NewFramework instead of using this directly.
func createTestingNS(baseName string, c *client.Client) (*api.Namespace, error) {
	namespaceObj := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			GenerateName: fmt.Sprintf("e2e-tests-%v-", baseName),
			Namespace:    "",
		},
		Status: api.NamespaceStatus{},
	}
	// Be robust about making the namespace creation call.
	var got *api.Namespace
	if err := wait.Poll(poll, singleCallTimeout, func() (bool, error) {
		var err error
		got, err = c.Namespaces().Create(namespaceObj)
		if err != nil {
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, err
	}

	if err := waitForDefaultServiceAccountInNamespace(c, got.Name); err != nil {
		return nil, err
	}
	return got, nil
}

// deleteTestingNS checks whether all e2e based existing namespaces are in the Terminating state
// and waits until they are finally deleted.
func deleteTestingNS(c *client.Client) error {
	// TODO: Since we don't have support for bulk resource deletion in the API,
	// while deleting a namespace we are deleting all objects from that namespace
	// one by one (one deletion == one API call). This basically exposes us to
	// throttling - currently controller-manager has a limit of max 20 QPS.
	// Once #10217 is implemented and used in namespace-controller, deleting all
	// object from a given namespace should be much faster and we will be able
	// to lower this timeout.
	// However, now Density test is producing ~26000 events and Load capacity test
	// is producing ~35000 events, thus assuming there are no other requests it will
	// take ~30 minutes to fully delete the namespace. Thus I'm setting it to 60
	// minutes to avoid any timeouts here.
	timeout := 60 * time.Minute

	Logf("Waiting for terminating namespaces to be deleted...")
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(15 * time.Second) {
		namespaces, err := c.Namespaces().List(labels.Everything(), fields.Everything())
		if err != nil {
			Logf("Listing namespaces failed: %v", err)
			continue
		}
		terminating := 0
		for _, ns := range namespaces.Items {
			if strings.HasPrefix(ns.ObjectMeta.Name, "e2e-tests-") {
				if ns.Status.Phase == api.NamespaceActive {
					return fmt.Errorf("Namespace %s is active", ns.ObjectMeta.Name)
				}
				terminating++
			}
		}
		if terminating == 0 {
			return nil
		}
	}
	return fmt.Errorf("Waiting for terminating namespaces to be deleted timed out")
}

// deleteNS deletes the provided namespace, waits for it to be completely deleted, and then checks
// whether there are any pods remaining in a non-terminating state.
func deleteNS(c *client.Client, namespace string) error {
	if err := c.Namespaces().Delete(namespace); err != nil {
		return err
	}

	err := wait.Poll(5*time.Second, 5*time.Minute, func() (bool, error) {
		if _, err := c.Namespaces().Get(namespace); err != nil {
			if apierrs.IsNotFound(err) {
				return true, nil
			}
			Logf("Error while waiting for namespace to be terminated: %v", err)
			return false, nil
		}
		return false, nil
	})

	// check for pods that were not deleted
	remaining := []string{}
	missingTimestamp := false
	if pods, perr := c.Pods(namespace).List(labels.Everything(), fields.Everything()); perr == nil {
		for _, pod := range pods.Items {
			Logf("Pod %s %s on node %s remains, has deletion timestamp %s", namespace, pod.Name, pod.Spec.NodeName, pod.DeletionTimestamp)
			remaining = append(remaining, pod.Name)
			if pod.DeletionTimestamp == nil {
				missingTimestamp = true
			}
		}
	}

	// a timeout occured
	if err != nil {
		if missingTimestamp {
			return fmt.Errorf("namespace %s was not deleted within limit: %v, some pods were not marked with a deletion timestamp, pods remaining: %v", namespace, err, remaining)
		}
		return fmt.Errorf("namespace %s was not deleted within limit: %v, pods remaining: %v", namespace, err, remaining)
	}
	// pods were not deleted but the namespace was deleted
	if len(remaining) > 0 {
		return fmt.Errorf("pods remained within namespace %s after deletion: %v", namespace, remaining)
	}
	return nil
}

func waitForPodRunningInNamespace(c *client.Client, podName string, namespace string) error {
	return waitForPodCondition(c, namespace, podName, "running", podStartTimeout, func(pod *api.Pod) (bool, error) {
		if pod.Status.Phase == api.PodRunning {
			Logf("Found pod '%s' on node '%s'", podName, pod.Spec.NodeName)
			return true, nil
		}
		if pod.Status.Phase == api.PodFailed {
			return true, fmt.Errorf("Giving up; pod went into failed status: \n%s", spew.Sprintf("%#v", pod))
		}
		return false, nil
	})
}

// waitForPodNotPending returns an error if it took too long for the pod to go out of pending state.
func waitForPodNotPending(c *client.Client, ns, podName string) error {
	return waitForPodCondition(c, ns, podName, "!pending", podStartTimeout, func(pod *api.Pod) (bool, error) {
		if pod.Status.Phase != api.PodPending {
			Logf("Saw pod '%s' in namespace '%s' out of pending state (found '%q')", podName, ns, pod.Status.Phase)
			return true, nil
		}
		return false, nil
	})
}

// waitForPodSuccessInNamespace returns nil if the pod reached state success, or an error if it reached failure or ran too long.
func waitForPodSuccessInNamespace(c *client.Client, podName string, contName string, namespace string) error {
	return waitForPodCondition(c, namespace, podName, "success or failure", podStartTimeout, func(pod *api.Pod) (bool, error) {
		// Cannot use pod.Status.Phase == api.PodSucceeded/api.PodFailed due to #2632
		ci, ok := api.GetContainerStatus(pod.Status.ContainerStatuses, contName)
		if !ok {
			Logf("No Status.Info for container '%s' in pod '%s' yet", contName, podName)
		} else {
			if ci.State.Terminated != nil {
				if ci.State.Terminated.ExitCode == 0 {
					By("Saw pod success")
					return true, nil
				}
				return true, fmt.Errorf("pod '%s' terminated with failure: %+v", podName, ci.State.Terminated)
			}
			Logf("Nil State.Terminated for container '%s' in pod '%s' in namespace '%s' so far", contName, podName, namespace)
		}
		return false, nil
	})
}

// waitForRCPodOnNode returns the pod from the given replication controller (described by rcName) which is scheduled on the given node.
// In case of failure or too long waiting time, an error is returned.
func waitForRCPodOnNode(c *client.Client, ns, rcName, node string) (*api.Pod, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": rcName}))
	var p *api.Pod = nil
	err := wait.Poll(10*time.Second, 5*time.Minute, func() (bool, error) {
		Logf("Waiting for pod %s to appear on node %s", rcName, node)
		pods, err := c.Pods(ns).List(label, fields.Everything())
		if err != nil {
			return false, err
		}
		for _, pod := range pods.Items {
			if pod.Spec.NodeName == node {
				Logf("Pod %s found on node %s", pod.Name, node)
				p = &pod
				return true, nil
			}
		}
		return false, nil
	})
	return p, err
}

// waitForRCPodToDisappear returns nil if the pod from the given replication controller (described by rcName) no longer exists.
// In case of failure or too long waiting time, an error is returned.
func waitForRCPodToDisappear(c *client.Client, ns, rcName, podName string) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": rcName}))
	return wait.Poll(20*time.Second, 5*time.Minute, func() (bool, error) {
		Logf("Waiting for pod %s to disappear", podName)
		pods, err := c.Pods(ns).List(label, fields.Everything())
		if err != nil {
			return false, err
		}
		found := false
		for _, pod := range pods.Items {
			if pod.Name == podName {
				Logf("Pod %s still exists", podName)
				found = true
			}
		}
		if !found {
			Logf("Pod %s no longer exists", podName)
			return true, nil
		}
		return false, nil
	})
}

// waitForService waits until the service appears (exist == true), or disappears (exist == false)
func waitForService(c *client.Client, namespace, name string, exist bool, interval, timeout time.Duration) error {
	err := wait.Poll(interval, timeout, func() (bool, error) {
		_, err := c.Services(namespace).Get(name)
		if err != nil {
			Logf("Get service %s in namespace %s failed (%v).", name, namespace, err)
			return !exist, nil
		} else {
			Logf("Service %s in namespace %s found.", name, namespace)
			return exist, nil
		}
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for service %s/%s %s: %v", namespace, name, stateMsg[exist], err)
	}
	return nil
}

// waitForReplicationController waits until the RC appears (exist == true), or disappears (exist == false)
func waitForReplicationController(c *client.Client, namespace, name string, exist bool, interval, timeout time.Duration) error {
	err := wait.Poll(interval, timeout, func() (bool, error) {
		_, err := c.ReplicationControllers(namespace).Get(name)
		if err != nil {
			Logf("Get ReplicationController %s in namespace %s failed (%v).", name, namespace, err)
			return !exist, nil
		} else {
			Logf("ReplicationController %s in namespace %s found.", name, namespace)
			return exist, nil
		}
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for ReplicationController %s/%s %s: %v", namespace, name, stateMsg[exist], err)
	}
	return nil
}

// Context for checking pods responses by issuing GETs to them and verifying if the answer with pod name.
type podResponseChecker struct {
	c              *client.Client
	ns             string
	label          labels.Selector
	controllerName string
	respondName    bool // Whether the pod should respond with its own name.
	pods           *api.PodList
}

// checkAllResponses issues GETs to all pods in the context and verify they reply with pod name.
func (r podResponseChecker) checkAllResponses() (done bool, err error) {
	successes := 0
	currentPods, err := r.c.Pods(r.ns).List(r.label, fields.Everything())
	Expect(err).NotTo(HaveOccurred())
	for i, pod := range r.pods.Items {
		// Check that the replica list remains unchanged, otherwise we have problems.
		if !isElementOf(pod.UID, currentPods) {
			return false, fmt.Errorf("pod with UID %s is no longer a member of the replica set.  Must have been restarted for some reason.  Current replica set: %v", pod.UID, currentPods)
		}
		body, err := r.c.Get().
			Prefix("proxy").
			Namespace(r.ns).
			Resource("pods").
			Name(string(pod.Name)).
			Do().
			Raw()
		if err != nil {
			Logf("Controller %s: Failed to GET from replica %d [%s]: %v:", r.controllerName, i+1, pod.Name, err)
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
				Logf("Controller %s: Replica %d [%s] expected response %q but got %q",
					r.controllerName, i+1, pod.Name, want, got)
				continue
			}
		} else {
			what = "non-empty"
			if len(got) == 0 {
				Logf("Controller %s: Replica %d [%s] expected non-empty response",
					r.controllerName, i+1, pod.Name)
				continue
			}
		}
		successes++
		Logf("Controller %s: Got %s result from replica %d [%s]: %q, %d of %d required successes so far",
			r.controllerName, what, i+1, pod.Name, got, successes, len(r.pods.Items))
	}
	if successes < len(r.pods.Items) {
		return false, nil
	}
	return true, nil
}

func podsResponding(c *client.Client, ns, name string, wantName bool, pods *api.PodList) error {
	By("trying to dial each unique pod")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	return wait.Poll(poll, podRespondingTimeout, podResponseChecker{c, ns, label, name, wantName, pods}.checkAllResponses)
}

func serviceResponding(c *client.Client, ns, name string) error {
	By(fmt.Sprintf("trying to dial the service %s.%s via the proxy", ns, name))

	return wait.Poll(poll, serviceRespondingTimeout, func() (done bool, err error) {
		body, err := c.Get().
			Prefix("proxy").
			Namespace(ns).
			Resource("services").
			Name(name).
			Do().
			Raw()
		if err != nil {
			Logf("Failed to GET from service %s: %v:", name, err)
			return false, nil
		}
		got := string(body)
		if len(got) == 0 {
			Logf("Service %s: expected non-empty response", name)
			return false, err // stop polling
		}
		Logf("Service %s: found nonempty answer: %s", name, got)
		return true, nil
	})
}

func loadConfig() (*client.Config, error) {
	switch {
	case testContext.KubeConfig != "":
		fmt.Printf(">>> testContext.KubeConfig: %s\n", testContext.KubeConfig)
		c, err := clientcmd.LoadFromFile(testContext.KubeConfig)
		if err != nil {
			return nil, fmt.Errorf("error loading KubeConfig: %v", err.Error())
		}
		if testContext.KubeContext != "" {
			fmt.Printf(">>> testContext.KubeContext: %s\n", testContext.KubeContext)
			c.CurrentContext = testContext.KubeContext
		}
		return clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: testContext.Host}}).ClientConfig()
	default:
		return nil, fmt.Errorf("KubeConfig must be specified to load client config")
	}
}

func loadClient() (*client.Client, error) {
	config, err := loadConfig()
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	c, err := client.New(config)
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	return c, nil
}

// randomSuffix provides a random string to append to pods,services,rcs.
// TODO: Allow service names to have the same form as names
//       for pods and replication controllers so we don't
//       need to use such a function and can instead
//       use the UUID utilty function.
func randomSuffix() string {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return strconv.Itoa(r.Int() % 10000)
}

func expectNoError(err error, explain ...interface{}) {
	ExpectWithOffset(1, err).NotTo(HaveOccurred(), explain...)
}

// Stops everything from filePath from namespace ns and checks if everything matching selectors from the given namespace is correctly stopped.
func cleanup(filePath string, ns string, selectors ...string) {
	By("using delete to clean up resources")
	var nsArg string
	if ns != "" {
		nsArg = fmt.Sprintf("--namespace=%s", ns)
	}
	runKubectl("stop", "--grace-period=0", "-f", filePath, nsArg)

	for _, selector := range selectors {
		resources := runKubectl("get", "rc,svc", "-l", selector, "--no-headers", nsArg)
		if resources != "" {
			Failf("Resources left running after stop:\n%s", resources)
		}
		pods := runKubectl("get", "pods", "-l", selector, nsArg, "-o", "go-template={{ range .items }}{{ if not .metadata.deletionTimestamp }}{{ .metadata.name }}{{ \"\\n\" }}{{ end }}{{ end }}")
		if pods != "" {
			Failf("Pods left unterminated after stop:\n%s", pods)
		}
	}
}

// validatorFn is the function which is individual tests will implement.
// we may want it to return more than just an error, at some point.
type validatorFn func(c *client.Client, podID string) error

// validateController is a generic mechanism for testing RC's that are running.
// It takes a container name, a test name, and a validator function which is plugged in by a specific test.
// "containername": this is grepped for.
// "containerImage" : this is the name of the image we expect to be launched.  Not to confuse w/ images (kitten.jpg)  which are validated.
// "testname":  which gets bubbled up to the logging/failure messages if errors happen.
// "validator" function: This function is given a podID and a client, and it can do some specific validations that way.
func validateController(c *client.Client, containerImage string, replicas int, containername string, testname string, validator validatorFn, ns string) {
	getPodsTemplate := "--template={{range.items}}{{.metadata.name}} {{end}}"
	// NB: kubectl adds the "exists" function to the standard template functions.
	// This lets us check to see if the "running" entry exists for each of the containers
	// we care about. Exists will never return an error and it's safe to check a chain of
	// things, any one of which may not exist. In the below template, all of info,
	// containername, and running might be nil, so the normal index function isn't very
	// helpful.
	// This template is unit-tested in kubectl, so if you change it, update the unit test.
	// You can read about the syntax here: http://golang.org/pkg/text/template/.
	getContainerStateTemplate := fmt.Sprintf(`--template={{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if (and (eq .name "%s") (exists . "state" "running"))}}true{{end}}{{end}}{{end}}`, containername)

	getImageTemplate := fmt.Sprintf(`--template={{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if eq .name "%s"}}{{.image}}{{end}}{{end}}{{end}}`, containername)

	By(fmt.Sprintf("waiting for all containers in %s pods to come up.", testname)) //testname should be selector
waitLoop:
	for start := time.Now(); time.Since(start) < podStartTimeout; time.Sleep(5 * time.Second) {
		getPodsOutput := runKubectl("get", "pods", "-o", "template", getPodsTemplate, "--api-version=v1", "-l", testname, fmt.Sprintf("--namespace=%v", ns))
		pods := strings.Fields(getPodsOutput)
		if numPods := len(pods); numPods != replicas {
			By(fmt.Sprintf("Replicas for %s: expected=%d actual=%d", testname, replicas, numPods))
			continue
		}
		var runningPods []string
		for _, podID := range pods {
			running := runKubectl("get", "pods", podID, "-o", "template", getContainerStateTemplate, "--api-version=v1", fmt.Sprintf("--namespace=%v", ns))
			if running != "true" {
				Logf("%s is created but not running", podID)
				continue waitLoop
			}

			currentImage := runKubectl("get", "pods", podID, "-o", "template", getImageTemplate, "--api-version=v1", fmt.Sprintf("--namespace=%v", ns))
			if currentImage != containerImage {
				Logf("%s is created but running wrong image; expected: %s, actual: %s", podID, containerImage, currentImage)
				continue waitLoop
			}

			// Call the generic validator function here.
			// This might validate for example, that (1) getting a url works and (2) url is serving correct content.
			if err := validator(c, podID); err != nil {
				Logf("%s is running right image but validator function failed: %v", podID, err)
				continue waitLoop
			}

			Logf("%s is verified up and running", podID)
			runningPods = append(runningPods, podID)
		}
		// If we reach here, then all our checks passed.
		if len(runningPods) == replicas {
			return
		}
	}
	// Reaching here means that one of more checks failed multiple times.  Assuming its not a race condition, something is broken.
	Failf("Timed out after %v seconds waiting for %s pods to reach valid state", podStartTimeout.Seconds(), testname)
}

// kubectlCmd runs the kubectl executable through the wrapper script.
func kubectlCmd(args ...string) *exec.Cmd {
	defaultArgs := []string{}

	// Reference a --server option so tests can run anywhere.
	if testContext.Host != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.FlagAPIServer+"="+testContext.Host)
	}
	if testContext.KubeConfig != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.RecommendedConfigPathFlag+"="+testContext.KubeConfig)

		// Reference the KubeContext
		if testContext.KubeContext != "" {
			defaultArgs = append(defaultArgs, "--"+clientcmd.FlagContext+"="+testContext.KubeContext)
		}

	} else {
		if testContext.CertDir != "" {
			defaultArgs = append(defaultArgs,
				fmt.Sprintf("--certificate-authority=%s", filepath.Join(testContext.CertDir, "ca.crt")),
				fmt.Sprintf("--client-certificate=%s", filepath.Join(testContext.CertDir, "kubecfg.crt")),
				fmt.Sprintf("--client-key=%s", filepath.Join(testContext.CertDir, "kubecfg.key")))
		}
	}
	kubectlArgs := append(defaultArgs, args...)

	//We allow users to specify path to kubectl, so you can test either "kubectl" or "cluster/kubectl.sh"
	//and so on.
	cmd := exec.Command(testContext.KubectlPath, kubectlArgs...)

	//caller will invoke this and wait on it.
	return cmd
}

// kubectlBuilder is used to build, custimize and execute a kubectl Command.
// Add more functions to customize the builder as needed.
type kubectlBuilder struct {
	cmd *exec.Cmd
}

func newKubectlCommand(args ...string) *kubectlBuilder {
	b := new(kubectlBuilder)
	b.cmd = kubectlCmd(args...)
	return b
}

func (b kubectlBuilder) withStdinData(data string) *kubectlBuilder {
	b.cmd.Stdin = strings.NewReader(data)
	return &b
}

func (b kubectlBuilder) withStdinReader(reader io.Reader) *kubectlBuilder {
	b.cmd.Stdin = reader
	return &b
}

func (b kubectlBuilder) exec() string {
	var stdout, stderr bytes.Buffer
	cmd := b.cmd
	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	Logf("Running '%s %s'", cmd.Path, strings.Join(cmd.Args[1:], " ")) // skip arg[0] as it is printed separately
	if err := cmd.Run(); err != nil {
		Failf("Error running %v:\nCommand stdout:\n%v\nstderr:\n%v\n", cmd, cmd.Stdout, cmd.Stderr)
		return ""
	}
	Logf(stdout.String())
	// TODO: trimspace should be unnecessary after switching to use kubectl binary directly
	return strings.TrimSpace(stdout.String())
}

// runKubectl is a convenience wrapper over kubectlBuilder
func runKubectl(args ...string) string {
	return newKubectlCommand(args...).exec()
}

func startCmdAndStreamOutput(cmd *exec.Cmd) (stdout, stderr io.ReadCloser, err error) {
	stdout, err = cmd.StdoutPipe()
	if err != nil {
		return
	}
	stderr, err = cmd.StderrPipe()
	if err != nil {
		return
	}
	Logf("Asynchronously running '%s %s'", cmd.Path, strings.Join(cmd.Args, " "))
	err = cmd.Start()
	return
}

// Rough equivalent of ctrl+c for cleaning up processes. Intended to be run in defer.
func tryKill(cmd *exec.Cmd) {
	if err := cmd.Process.Kill(); err != nil {
		Logf("ERROR failed to kill command %v! The process may leak", cmd)
	}
}

// testContainerOutputInNamespace runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status.  It retrieves
// the exact container log and searches for lines of expected output.
func testContainerOutputInNamespace(scenarioName string, c *client.Client, pod *api.Pod, containerIndex int, expectedOutput []string, ns string) {
	By(fmt.Sprintf("Creating a pod to test %v", scenarioName))

	defer c.Pods(ns).Delete(pod.Name, api.NewDeleteOptions(0))
	if _, err := c.Pods(ns).Create(pod); err != nil {
		Failf("Failed to create pod: %v", err)
	}

	// Wait for client pod to complete.
	var containerName string
	for id, container := range pod.Spec.Containers {
		expectNoError(waitForPodSuccessInNamespace(c, pod.Name, container.Name, ns))
		if id == containerIndex {
			containerName = container.Name
		}
	}
	if containerName == "" {
		Failf("Invalid container index: %d", containerIndex)
	}

	// Grab its logs.  Get host first.
	podStatus, err := c.Pods(ns).Get(pod.Name)
	if err != nil {
		Failf("Failed to get pod status: %v", err)
	}

	By(fmt.Sprintf("Trying to get logs from node %s pod %s container %s: %v",
		podStatus.Spec.NodeName, podStatus.Name, containerName, err))
	var logs []byte
	start := time.Now()

	// Sometimes the actual containers take a second to get started, try to get logs for 60s
	for time.Now().Sub(start) < (60 * time.Second) {
		err = nil
		logs, err = c.Get().
			Prefix("proxy").
			Resource("nodes").
			Name(podStatus.Spec.NodeName).
			Suffix("containerLogs", ns, podStatus.Name, containerName).
			Do().
			Raw()
		if err == nil && strings.Contains(string(logs), "Internal Error") {
			err = fmt.Errorf("Fetched log contains \"Internal Error\": %q.", string(logs))
		}
		if err != nil {
			By(fmt.Sprintf("Warning: Failed to get logs from node %q pod %q container %q. %v",
				podStatus.Spec.NodeName, podStatus.Name, containerName, err))
			time.Sleep(5 * time.Second)
			continue

		}
		By(fmt.Sprintf("Successfully fetched pod logs:%v\n", string(logs)))
		break
	}

	for _, m := range expectedOutput {
		Expect(string(logs)).To(ContainSubstring(m), "%q in container output", m)
	}
}

// podInfo contains pod information useful for debugging e2e tests.
type podInfo struct {
	oldHostname string
	oldPhase    string
	hostname    string
	phase       string
}

// PodDiff is a map of pod name to podInfos
type PodDiff map[string]*podInfo

// Print formats and prints the give PodDiff.
func (p PodDiff) Print(ignorePhases sets.String) {
	for name, info := range p {
		if ignorePhases.Has(info.phase) {
			continue
		}
		if info.phase == nonExist {
			Logf("Pod %v was deleted, had phase %v and host %v", name, info.oldPhase, info.oldHostname)
			continue
		}
		phaseChange, hostChange := false, false
		msg := fmt.Sprintf("Pod %v ", name)
		if info.oldPhase != info.phase {
			phaseChange = true
			if info.oldPhase == nonExist {
				msg += fmt.Sprintf("in phase %v ", info.phase)
			} else {
				msg += fmt.Sprintf("went from phase: %v -> %v ", info.oldPhase, info.phase)
			}
		}
		if info.oldHostname != info.hostname {
			hostChange = true
			if info.oldHostname == nonExist || info.oldHostname == "" {
				msg += fmt.Sprintf("assigned host %v ", info.hostname)
			} else {
				msg += fmt.Sprintf("went from host: %v -> %v ", info.oldHostname, info.hostname)
			}
		}
		if phaseChange || hostChange {
			Logf(msg)
		}
	}
}

// Diff computes a PodDiff given 2 lists of pods.
func Diff(oldPods []*api.Pod, curPods []*api.Pod) PodDiff {
	podInfoMap := PodDiff{}

	// New pods will show up in the curPods list but not in oldPods. They have oldhostname/phase == nonexist.
	for _, pod := range curPods {
		podInfoMap[pod.Name] = &podInfo{hostname: pod.Spec.NodeName, phase: string(pod.Status.Phase), oldHostname: nonExist, oldPhase: nonExist}
	}

	// Deleted pods will show up in the oldPods list but not in curPods. They have a hostname/phase == nonexist.
	for _, pod := range oldPods {
		if info, ok := podInfoMap[pod.Name]; ok {
			info.oldHostname, info.oldPhase = pod.Spec.NodeName, string(pod.Status.Phase)
		} else {
			podInfoMap[pod.Name] = &podInfo{hostname: nonExist, phase: nonExist, oldHostname: pod.Spec.NodeName, oldPhase: string(pod.Status.Phase)}
		}
	}
	return podInfoMap
}

// RunRC Launches (and verifies correctness) of a Replication Controller
// and will wait for all pods it spawns to become "Running".
// It's the caller's responsibility to clean up externally (i.e. use the
// namespace lifecycle for handling cleanup).
func RunRC(config RCConfig) error {

	// Don't force tests to fail if they don't care about containers restarting.
	var maxContainerFailures int
	if config.MaxContainerFailures == nil {
		maxContainerFailures = int(math.Max(1.0, float64(config.Replicas)*.01))
	} else {
		maxContainerFailures = *config.MaxContainerFailures
	}

	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": config.Name}))

	By(fmt.Sprintf("%v Creating replication controller %s", time.Now(), config.Name))
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: config.Name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: config.Replicas,
			Selector: map[string]string{
				"name": config.Name,
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"name": config.Name},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:    config.Name,
							Image:   config.Image,
							Command: config.Command,
							Ports:   []api.ContainerPort{{ContainerPort: 80}},
						},
					},
				},
			},
		},
	}
	if config.Env != nil {
		for k, v := range config.Env {
			c := &rc.Spec.Template.Spec.Containers[0]
			c.Env = append(c.Env, api.EnvVar{Name: k, Value: v})
		}
	}
	if config.Labels != nil {
		for k, v := range config.Labels {
			rc.Spec.Template.ObjectMeta.Labels[k] = v
		}
	}
	if config.Ports != nil {
		for k, v := range config.Ports {
			c := &rc.Spec.Template.Spec.Containers[0]
			c.Ports = append(c.Ports, api.ContainerPort{Name: k, ContainerPort: v})
		}
	}
	if config.CpuLimit > 0 || config.MemLimit > 0 {
		rc.Spec.Template.Spec.Containers[0].Resources.Limits = api.ResourceList{}
	}
	if config.CpuLimit > 0 {
		rc.Spec.Template.Spec.Containers[0].Resources.Limits[api.ResourceCPU] = *resource.NewMilliQuantity(config.CpuLimit, resource.DecimalSI)
	}
	if config.MemLimit > 0 {
		rc.Spec.Template.Spec.Containers[0].Resources.Limits[api.ResourceMemory] = *resource.NewQuantity(config.MemLimit, resource.DecimalSI)
	}
	if config.CpuRequest > 0 || config.MemRequest > 0 {
		rc.Spec.Template.Spec.Containers[0].Resources.Requests = api.ResourceList{}
	}
	if config.CpuRequest > 0 {
		rc.Spec.Template.Spec.Containers[0].Resources.Requests[api.ResourceCPU] = *resource.NewMilliQuantity(config.CpuRequest, resource.DecimalSI)
	}
	if config.MemRequest > 0 {
		rc.Spec.Template.Spec.Containers[0].Resources.Requests[api.ResourceMemory] = *resource.NewQuantity(config.MemRequest, resource.DecimalSI)
	}

	_, err := config.Client.ReplicationControllers(config.Namespace).Create(rc)
	if err != nil {
		return fmt.Errorf("Error creating replication controller: %v", err)
	}
	Logf("%v Created replication controller with name: %v, namespace: %v, replica count: %v", time.Now(), rc.Name, config.Namespace, rc.Spec.Replicas)
	podStore := newPodStore(config.Client, config.Namespace, label, fields.Everything())
	defer podStore.Stop()

	interval := config.PollInterval
	if interval <= 0 {
		interval = 10 * time.Second
	}
	timeout := config.Timeout
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	oldPods := make([]*api.Pod, 0)
	oldRunning := 0
	lastChange := time.Now()
	for oldRunning != config.Replicas {
		time.Sleep(interval)

		terminating := 0

		running := 0
		waiting := 0
		pending := 0
		unknown := 0
		inactive := 0
		failedContainers := 0
		containerRestartNodes := sets.NewString()

		pods := podStore.List()
		created := []*api.Pod{}
		for _, p := range pods {
			if p.DeletionTimestamp != nil {
				terminating++
				continue
			}
			created = append(created, p)
			if p.Status.Phase == api.PodRunning {
				running++
				for _, v := range FailedContainers(p) {
					failedContainers = failedContainers + v.restarts
					containerRestartNodes.Insert(p.Spec.NodeName)
				}
			} else if p.Status.Phase == api.PodPending {
				if p.Spec.NodeName == "" {
					waiting++
				} else {
					pending++
				}
			} else if p.Status.Phase == api.PodSucceeded || p.Status.Phase == api.PodFailed {
				inactive++
			} else if p.Status.Phase == api.PodUnknown {
				unknown++
			}
		}
		pods = created
		if config.CreatedPods != nil {
			*config.CreatedPods = pods
		}

		Logf("%v %v Pods: %d out of %d created, %d running, %d pending, %d waiting, %d inactive, %d terminating, %d unknown ",
			time.Now(), rc.Name, len(pods), config.Replicas, running, pending, waiting, inactive, terminating, unknown)

		promPushRunningPending(running, pending)

		if config.PodStatusFile != nil {
			fmt.Fprintf(config.PodStatusFile, "%s, %d, running, %d, pending, %d, waiting, %d, inactive, %d, unknown\n", time.Now(), running, pending, waiting, inactive, unknown)
		}

		if failedContainers > maxContainerFailures {
			dumpNodeDebugInfo(config.Client, containerRestartNodes.List())
			return fmt.Errorf("%d containers failed which is more than allowed %d", failedContainers, maxContainerFailures)
		}
		if len(pods) < len(oldPods) || len(pods) > config.Replicas {
			// This failure mode includes:
			// kubelet is dead, so node controller deleted pods and rc creates more
			//	- diagnose by noting the pod diff below.
			// pod is unhealthy, so replication controller creates another to take its place
			//	- diagnose by comparing the previous "2 Pod states" lines for inactive pods
			errorStr := fmt.Sprintf("Number of reported pods changed: %d vs %d", len(pods), len(oldPods))
			Logf("%v, pods that changed since the last iteration:", errorStr)
			Diff(oldPods, pods).Print(sets.NewString())
			return fmt.Errorf(errorStr)
		}

		if len(pods) > len(oldPods) || running > oldRunning {
			lastChange = time.Now()
		}
		oldPods = pods
		oldRunning = running

		if time.Since(lastChange) > timeout {
			dumpPodDebugInfo(config.Client, pods)
			break
		}
	}

	if oldRunning != config.Replicas {
		if pods, err := config.Client.Pods(api.NamespaceAll).List(labels.Everything(), fields.Everything()); err == nil {
			for _, pod := range pods.Items {
				Logf("Pod %s\t%s\t%s\t%s", pod.Namespace, pod.Name, pod.Spec.NodeName, pod.DeletionTimestamp)
			}
		} else {
			Logf("Can't list pod debug info: %v", err)
		}
		return fmt.Errorf("Only %d pods started out of %d", oldRunning, config.Replicas)
	}
	return nil
}

func dumpPodDebugInfo(c *client.Client, pods []*api.Pod) {
	badNodes := sets.NewString()
	for _, p := range pods {
		if p.Status.Phase != api.PodRunning {
			if p.Spec.NodeName != "" {
				Logf("Pod %v assigned to host %v (IP: %v) in %v", p.Name, p.Spec.NodeName, p.Status.HostIP, p.Status.Phase)
				badNodes.Insert(p.Spec.NodeName)
			} else {
				Logf("Pod %v still unassigned", p.Name)
			}
		}
	}
	dumpNodeDebugInfo(c, badNodes.List())
}

func dumpAllPodInfo(c *client.Client) {
	pods, err := c.Pods("").List(labels.Everything(), fields.Everything())
	if err != nil {
		Logf("unable to fetch pod debug info: %v", err)
	}
	logPodStates(pods.Items)
}

func dumpNodeDebugInfo(c *client.Client, nodeNames []string) {
	for _, n := range nodeNames {
		Logf("\nLogging kubelet events for node %v", n)
		for _, e := range getNodeEvents(c, n) {
			Logf("source %v message %v reason %v first ts %v last ts %v, involved obj %+v",
				e.Source, e.Message, e.Reason, e.FirstTimestamp, e.LastTimestamp, e.InvolvedObject)
		}
		Logf("\nLogging pods the kubelet thinks is on node %v", n)
		podList, err := GetKubeletPods(c, n)
		if err != nil {
			Logf("Unable to retrieve kubelet pods for node %v", n)
			continue
		}
		for _, p := range podList.Items {
			Logf("%v started at %v (%d container statuses recorded)", p.Name, p.Status.StartTime, len(p.Status.ContainerStatuses))
			for _, c := range p.Status.ContainerStatuses {
				Logf("\tContainer %v ready: %v, restart count %v",
					c.Name, c.Ready, c.RestartCount)
			}
		}
		HighLatencyKubeletOperations(c, 10*time.Second, n)
		// TODO: Log node resource info
	}
}

// logNodeEvents logs kubelet events from the given node. This includes kubelet
// restart and node unhealthy events. Note that listing events like this will mess
// with latency metrics, beware of calling it during a test.
func getNodeEvents(c *client.Client, nodeName string) []api.Event {
	events, err := c.Events(api.NamespaceSystem).List(
		labels.Everything(),
		fields.Set{
			"involvedObject.kind":      "Node",
			"involvedObject.name":      nodeName,
			"involvedObject.namespace": api.NamespaceAll,
			"source":                   "kubelet",
		}.AsSelector())
	if err != nil {
		Logf("Unexpected error retrieving node events %v", err)
		return []api.Event{}
	}
	return events.Items
}

func ScaleRC(c *client.Client, ns, name string, size uint, wait bool) error {
	By(fmt.Sprintf("%v Scaling replication controller %s in namespace %s to %d", time.Now(), name, ns, size))
	scaler, err := kubectl.ScalerFor("ReplicationController", kubectl.NewScalerClient(c))
	if err != nil {
		return err
	}
	waitForScale := kubectl.NewRetryParams(5*time.Second, 1*time.Minute)
	waitForReplicas := kubectl.NewRetryParams(5*time.Second, 5*time.Minute)
	if err = scaler.Scale(ns, name, size, nil, waitForScale, waitForReplicas); err != nil {
		return err
	}
	if !wait {
		return nil
	}
	return waitForRCPodsRunning(c, ns, name)
}

// Wait up to 10 minutes for pods to become Running.
func waitForRCPodsRunning(c *client.Client, ns, rcName string) error {
	running := false
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": rcName}))
	podStore := newPodStore(c, ns, label, fields.Everything())
	defer podStore.Stop()
waitLoop:
	for start := time.Now(); time.Since(start) < 10*time.Minute; time.Sleep(5 * time.Second) {
		pods := podStore.List()
		for _, p := range pods {
			if p.Status.Phase != api.PodRunning {
				continue waitLoop
			}
		}
		running = true
		break
	}
	if !running {
		return fmt.Errorf("Timeout while waiting for replication controller %s pods to be running", rcName)
	}
	return nil
}

// Wait up to 10 minutes for getting pods with certain label
func waitForPodsWithLabel(c *client.Client, ns string, label labels.Selector) (pods *api.PodList, err error) {
	for t := time.Now(); time.Since(t) < podListTimeout; time.Sleep(poll) {
		pods, err = c.Pods(ns).List(label, fields.Everything())
		Expect(err).NotTo(HaveOccurred())
		if len(pods.Items) > 0 {
			break
		}
	}
	if pods == nil || len(pods.Items) == 0 {
		err = fmt.Errorf("Timeout while waiting for pods with label %v", label)
	}
	return
}

// Delete a Replication Controller and all pods it spawned
func DeleteRC(c *client.Client, ns, name string) error {
	By(fmt.Sprintf("%v Deleting replication controller %s in namespace %s", time.Now(), name, ns))
	rc, err := c.ReplicationControllers(ns).Get(name)
	if err != nil {
		if apierrs.IsNotFound(err) {
			Logf("RC %s was already deleted: %v", name, err)
			return nil
		}
		return err
	}
	reaper, err := kubectl.ReaperForReplicationController(c, 10*time.Minute)
	if err != nil {
		if apierrs.IsNotFound(err) {
			Logf("RC %s was already deleted: %v", name, err)
			return nil
		}
		return err
	}
	startTime := time.Now()
	_, err = reaper.Stop(ns, name, 0, api.NewDeleteOptions(0))
	if apierrs.IsNotFound(err) {
		Logf("RC %s was already deleted: %v", name, err)
		return nil
	}
	deleteRCTime := time.Now().Sub(startTime)
	Logf("Deleting RC %s took: %v", name, deleteRCTime)
	if err == nil {
		err = waitForRCPodsGone(c, rc)
	}
	terminatePodTime := time.Now().Sub(startTime) - deleteRCTime
	Logf("Terminating RC %s pods took: %v", name, terminatePodTime)
	return err
}

// waitForRCPodsGone waits until there are no pods reported under an RC's selector (because the pods
// have completed termination).
func waitForRCPodsGone(c *client.Client, rc *api.ReplicationController) error {
	return wait.Poll(poll, 2*time.Minute, func() (bool, error) {
		if pods, err := c.Pods(rc.Namespace).List(labels.SelectorFromSet(rc.Spec.Selector), fields.Everything()); err == nil && len(pods.Items) == 0 {
			return true, nil
		}
		return false, nil
	})
}

// Convenient wrapper around listing nodes supporting retries.
func listNodes(c *client.Client, label labels.Selector, field fields.Selector) (*api.NodeList, error) {
	var nodes *api.NodeList
	var errLast error
	if wait.Poll(poll, singleCallTimeout, func() (bool, error) {
		nodes, errLast = c.Nodes().List(label, field)
		return errLast == nil, nil
	}) != nil {
		return nil, fmt.Errorf("listNodes() failed with last error: %v", errLast)
	}
	return nodes, nil
}

// FailedContainers inspects all containers in a pod and returns failure
// information for containers that have failed or been restarted.
// A map is returned where the key is the containerID and the value is a
// struct containing the restart and failure information
func FailedContainers(pod *api.Pod) map[string]ContainerFailures {
	var state ContainerFailures
	states := make(map[string]ContainerFailures)

	statuses := pod.Status.ContainerStatuses
	if len(statuses) == 0 {
		return nil
	} else {
		for _, status := range statuses {
			if status.State.Terminated != nil {
				states[status.ContainerID] = ContainerFailures{status: status.State.Terminated}
			} else if status.LastTerminationState.Terminated != nil {
				states[status.ContainerID] = ContainerFailures{status: status.LastTerminationState.Terminated}
			}
			if status.RestartCount > 0 {
				var ok bool
				if state, ok = states[status.ContainerID]; !ok {
					state = ContainerFailures{}
				}
				state.restarts = status.RestartCount
				states[status.ContainerID] = state
			}
		}
	}

	return states
}

// Prints the histogram of the events and returns the number of bad events.
func BadEvents(events []*api.Event) int {
	type histogramKey struct {
		reason string
		source string
	}
	histogram := make(map[histogramKey]int)
	for _, e := range events {
		histogram[histogramKey{reason: e.Reason, source: e.Source.Component}]++
	}
	for key, number := range histogram {
		Logf("- reason: %s, source: %s -> %d", key.reason, key.source, number)
	}
	badPatterns := []string{"kill", "fail"}
	badEvents := 0
	for key, number := range histogram {
		for _, s := range badPatterns {
			if strings.Contains(key.reason, s) {
				Logf("WARNING %d events from %s with reason: %s", number, key.source, key.reason)
				badEvents += number
				break
			}
		}
	}
	return badEvents
}

// NodeSSHHosts returns SSH-able host names for all nodes. It returns an error
// if it can't find an external IP for every node, though it still returns all
// hosts that it found in that case.
func NodeSSHHosts(c *client.Client) ([]string, error) {
	var hosts []string
	nodelist, err := c.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		return hosts, fmt.Errorf("error getting nodes: %v", err)
	}
	for _, n := range nodelist.Items {
		for _, addr := range n.Status.Addresses {
			// Use the first external IP address we find on the node, and
			// use at most one per node.
			// TODO(roberthbailey): Use the "preferred" address for the node, once
			// such a thing is defined (#2462).
			if addr.Type == api.NodeExternalIP {
				hosts = append(hosts, addr.Address+":22")
				break
			}
		}
	}

	// Error if any node didn't have an external IP.
	if len(hosts) != len(nodelist.Items) {
		return hosts, fmt.Errorf(
			"only found %d external IPs on nodes, but found %d nodes. Nodelist: %v",
			len(hosts), len(nodelist.Items), nodelist)
	}
	return hosts, nil
}

// SSH synchronously SSHs to a node running on provider and runs cmd. If there
// is no error performing the SSH, the stdout, stderr, and exit code are
// returned.
func SSH(cmd, host, provider string) (string, string, int, error) {
	return sshCore(cmd, host, provider, false)
}

// SSHVerbose is just like SSH, but it logs the command, user, host, stdout,
// stderr, exit code, and error.
func SSHVerbose(cmd, host, provider string) (string, string, int, error) {
	return sshCore(cmd, host, provider, true)
}

func sshCore(cmd, host, provider string, verbose bool) (string, string, int, error) {
	// Get a signer for the provider.
	signer, err := getSigner(provider)
	if err != nil {
		return "", "", 0, fmt.Errorf("error getting signer for provider %s: '%v'", provider, err)
	}

	// RunSSHCommand will default to Getenv("USER") if user == "", but we're
	// defaulting here as well for logging clarity.
	user := os.Getenv("KUBE_SSH_USER")
	if user == "" {
		user = os.Getenv("USER")
	}

	stdout, stderr, code, err := util.RunSSHCommand(cmd, user, host, signer)
	if verbose {
		remote := fmt.Sprintf("%s@%s", user, host)
		Logf("[%s] Running    `%s`", remote, cmd)
		Logf("[%s] stdout:    %q", remote, stdout)
		Logf("[%s] stderr:    %q", remote, stderr)
		Logf("[%s] exit code: %d", remote, code)
		Logf("[%s] error:     %v", remote, err)
	}
	return stdout, stderr, code, err
}

// getSigner returns an ssh.Signer for the provider ("gce", etc.) that can be
// used to SSH to their nodes.
func getSigner(provider string) (ssh.Signer, error) {
	// Get the directory in which SSH keys are located.
	keydir := filepath.Join(os.Getenv("HOME"), ".ssh")

	// Select the key itself to use. When implementing more providers here,
	// please also add them to any SSH tests that are disabled because of signer
	// support.
	keyfile := ""
	switch provider {
	case "gce", "gke":
		keyfile = "google_compute_engine"
	case "aws":
		keyfile = "kube_aws_rsa"
	default:
		return nil, fmt.Errorf("getSigner(...) not implemented for %s", provider)
	}
	key := filepath.Join(keydir, keyfile)

	return util.MakePrivateKeySignerFromFile(key)
}

// checkPodsRunning returns whether all pods whose names are listed in podNames
// in namespace ns are running and ready, using c and waiting at most timeout.
func checkPodsRunningReady(c *client.Client, ns string, podNames []string, timeout time.Duration) bool {
	np, desc := len(podNames), "running and ready"
	Logf("Waiting up to %v for the following %d pods to be %s: %s", timeout, np, desc, podNames)
	result := make(chan bool, len(podNames))
	for ix := range podNames {
		// Launch off pod readiness checkers.
		go func(name string) {
			err := waitForPodCondition(c, ns, name, desc, timeout, podRunningReady)
			result <- err == nil
		}(podNames[ix])
	}
	// Wait for them all to finish.
	success := true
	// TODO(a-robinson): Change to `for range` syntax and remove logging once we
	// support only Go >= 1.4.
	for _, podName := range podNames {
		if !<-result {
			Logf("Pod %-[1]*[2]s failed to be %[3]s.", podPrintWidth, podName, desc)
			success = false
		}
	}
	Logf("Wanted all %d pods to be %s. Result: %t. Pods: %v", np, desc, success, podNames)
	return success
}

// waitForNodeToBeReady returns whether node name is ready within timeout.
func waitForNodeToBeReady(c *client.Client, name string, timeout time.Duration) bool {
	return waitForNodeToBe(c, name, true, timeout)
}

// waitForNodeToBeNotReady returns whether node name is not ready (i.e. the
// readiness condition is anything but ready, e.g false or unknown) within
// timeout.
func waitForNodeToBeNotReady(c *client.Client, name string, timeout time.Duration) bool {
	return waitForNodeToBe(c, name, false, timeout)
}

func isNodeReadySetAsExpected(node *api.Node, wantReady bool) bool {
	// Check the node readiness condition (logging all).
	for i, cond := range node.Status.Conditions {
		Logf("Node %s condition %d/%d: type: %v, status: %v, reason: %q, message: %q, last transition time: %v",
			node.Name, i+1, len(node.Status.Conditions), cond.Type, cond.Status,
			cond.Reason, cond.Message, cond.LastTransitionTime)
		// Ensure that the condition type is readiness and the status
		// matches as desired.
		if cond.Type == api.NodeReady && (cond.Status == api.ConditionTrue) == wantReady {
			Logf("Successfully found node %s readiness to be %t", node.Name, wantReady)
			return true
		}
	}
	return false
}

// waitForNodeToBe returns whether node name's readiness state matches wantReady
// within timeout. If wantReady is true, it will ensure the node is ready; if
// it's false, it ensures the node is in any state other than ready (e.g. not
// ready or unknown).
func waitForNodeToBe(c *client.Client, name string, wantReady bool, timeout time.Duration) bool {
	Logf("Waiting up to %v for node %s readiness to be %t", timeout, name, wantReady)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		node, err := c.Nodes().Get(name)
		if err != nil {
			Logf("Couldn't get node %s", name)
			continue
		}

		if isNodeReadySetAsExpected(node, wantReady) {
			return true
		}
	}
	Logf("Node %s didn't reach desired readiness (%t) within %v", name, wantReady, timeout)
	return false
}

// checks whether all registered nodes are ready
func allNodesReady(c *client.Client, timeout time.Duration) error {
	Logf("Waiting up to %v for all nodes to be ready", timeout)

	var notReady []api.Node
	err := wait.Poll(poll, timeout, func() (bool, error) {
		notReady = nil
		nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
		if err != nil {
			return false, err
		}
		for _, node := range nodes.Items {
			if !isNodeReadySetAsExpected(&node, true) {
				notReady = append(notReady, node)
			}
		}
		return len(notReady) == 0, nil
	})

	if err != nil && err != wait.ErrWaitTimeout {
		return err
	}

	if len(notReady) > 0 {
		return fmt.Errorf("Not ready nodes: %v", notReady)
	}
	return nil
}

// Filters nodes in NodeList in place, removing nodes that do not
// satisfy the given condition
// TODO: consider merging with pkg/client/cache.NodeLister
func filterNodes(nodeList *api.NodeList, fn func(node api.Node) bool) {
	var l []api.Node

	for _, node := range nodeList.Items {
		if fn(node) {
			l = append(l, node)
		}
	}
	nodeList.Items = l
}

// LatencyMetrics stores data about request latency at a given quantile
// broken down by verb (e.g. GET, PUT, LIST) and resource (e.g. pods, services).
type LatencyMetric struct {
	Verb     string
	Resource string
	// 0 <= quantile <=1, e.g. 0.95 is 95%tile, 0.5 is median.
	Quantile float64
	Latency  time.Duration
}

// latencyMetricIngestor implements extraction.Ingester
type latencyMetricIngester []LatencyMetric

func (l *latencyMetricIngester) Ingest(samples model.Samples) error {
	for _, sample := range samples {
		// Example line:
		// apiserver_request_latencies_summary{resource="namespaces",verb="LIST",quantile="0.99"} 908
		if sample.Metric[model.MetricNameLabel] != "apiserver_request_latencies_summary" {
			continue
		}

		resource := string(sample.Metric["resource"])
		verb := string(sample.Metric["verb"])
		latency := sample.Value
		quantile, err := strconv.ParseFloat(string(sample.Metric[model.QuantileLabel]), 64)
		if err != nil {
			return err
		}
		*l = append(*l, LatencyMetric{
			verb,
			resource,
			quantile,
			time.Duration(int64(latency)) * time.Microsecond,
		})
	}
	return nil
}

// LatencyMetricByLatency implements sort.Interface for []LatencyMetric based on
// the latency field.
type LatencyMetricByLatency []LatencyMetric

func (a LatencyMetricByLatency) Len() int           { return len(a) }
func (a LatencyMetricByLatency) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a LatencyMetricByLatency) Less(i, j int) bool { return a[i].Latency < a[j].Latency }

func ReadLatencyMetrics(c *client.Client) ([]LatencyMetric, error) {
	body, err := getMetrics(c)
	if err != nil {
		return nil, err
	}
	var ingester latencyMetricIngester
	err = extraction.Processor004.ProcessSingle(strings.NewReader(body), &ingester, &extraction.ProcessOptions{})
	return ingester, err
}

// Prints summary metrics for request types with latency above threshold
// and returns number of such request types.
func HighLatencyRequests(c *client.Client, threshold time.Duration, ignoredResources sets.String) (int, error) {
	ignoredVerbs := sets.NewString("WATCHLIST", "PROXY")

	metrics, err := ReadLatencyMetrics(c)
	if err != nil {
		return 0, err
	}
	sort.Sort(sort.Reverse(LatencyMetricByLatency(metrics)))
	var badMetrics []LatencyMetric
	top := 5
	for _, metric := range metrics {
		if ignoredResources.Has(metric.Resource) || ignoredVerbs.Has(metric.Verb) {
			continue
		}
		isBad := false
		if metric.Latency > threshold &&
			// We are only interested in 99%tile, but for logging purposes
			// it's useful to have all the offending percentiles.
			metric.Quantile <= 0.99 {
			badMetrics = append(badMetrics, metric)
			isBad = true
		}
		if top > 0 || isBad {
			top--
			prefix := ""
			if isBad {
				prefix = "WARNING "
			}
			Logf("%vTop latency metric: %+v", prefix, metric)
		}
	}

	return len(badMetrics), nil
}

// Reset latency metrics in apiserver.
func resetMetrics(c *client.Client) error {
	Logf("Resetting latency metrics in apiserver...")
	body, err := c.Get().AbsPath("/resetMetrics").DoRaw()
	if err != nil {
		return err
	}
	if string(body) != "metrics reset\n" {
		return fmt.Errorf("Unexpected response: %q", string(body))
	}
	return nil
}

// Retrieve metrics information
func getMetrics(c *client.Client) (string, error) {
	body, err := c.Get().AbsPath("/metrics").DoRaw()
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// Retrieve debug information
func getDebugInfo(c *client.Client) (map[string]string, error) {
	data := make(map[string]string)
	for _, key := range []string{"block", "goroutine", "heap", "threadcreate"} {
		resp, err := http.Get(c.Get().AbsPath(fmt.Sprintf("debug/pprof/%s", key)).URL().String() + "?debug=2")
		if err != nil {
			Logf("Warning: Error trying to fetch %s debug data: %v", key, err)
			continue
		}
		body, err := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			Logf("Warning: Error trying to read %s debug data: %v", key, err)
		}
		data[key] = string(body)
	}
	return data, nil
}

func writePerfData(c *client.Client, dirName string, postfix string) error {
	fname := fmt.Sprintf("%s/metrics_%s.txt", dirName, postfix)

	handler, err := os.Create(fname)
	if err != nil {
		return fmt.Errorf("Error creating file '%s': %v", fname, err)
	}

	metrics, err := getMetrics(c)
	if err != nil {
		return fmt.Errorf("Error retrieving metrics: %v", err)
	}

	_, err = handler.WriteString(metrics)
	if err != nil {
		return fmt.Errorf("Error writing metrics: %v", err)
	}

	err = handler.Close()
	if err != nil {
		return fmt.Errorf("Error closing '%s': %v", fname, err)
	}

	debug, err := getDebugInfo(c)
	if err != nil {
		return fmt.Errorf("Error retrieving debug information: %v", err)
	}

	for key, value := range debug {
		fname := fmt.Sprintf("%s/%s_%s.txt", dirName, key, postfix)
		handler, err = os.Create(fname)
		if err != nil {
			return fmt.Errorf("Error creating file '%s': %v", fname, err)
		}
		_, err = handler.WriteString(value)
		if err != nil {
			return fmt.Errorf("Error writing %s: %v", key, err)
		}

		err = handler.Close()
		if err != nil {
			return fmt.Errorf("Error closing '%s': %v", fname, err)
		}
	}
	return nil
}

// parseKVLines parses output that looks like lines containing "<key>: <val>"
// and returns <val> if <key> is found. Otherwise, it returns the empty string.
func parseKVLines(output, key string) string {
	delim := ":"
	key = key + delim
	for _, line := range strings.Split(output, "\n") {
		pieces := strings.SplitAfterN(line, delim, 2)
		if len(pieces) != 2 {
			continue
		}
		k, v := pieces[0], pieces[1]
		if k == key {
			return strings.TrimSpace(v)
		}
	}
	return ""
}

func restartKubeProxy(host string) error {
	// TODO: Make it work for all providers.
	if !providerIs("gce", "gke", "aws") {
		return fmt.Errorf("unsupported provider: %s", testContext.Provider)
	}
	_, _, code, err := SSH("sudo /etc/init.d/kube-proxy restart", host, testContext.Provider)
	if err != nil || code != 0 {
		return fmt.Errorf("couldn't restart kube-proxy: %v (code %v)", err, code)
	}
	return nil
}

func restartApiserver() error {
	// TODO: Make it work for all providers.
	if !providerIs("gce", "gke", "aws") {
		return fmt.Errorf("unsupported provider: %s", testContext.Provider)
	}
	var command string
	if providerIs("gce", "gke") {
		command = "sudo docker ps | grep /kube-apiserver | cut -d ' ' -f 1 | xargs sudo docker kill"
	} else {
		command = "sudo /etc/init.d/kube-apiserver restart"
	}
	_, _, code, err := SSH(command, getMasterHost()+":22", testContext.Provider)
	if err != nil || code != 0 {
		return fmt.Errorf("couldn't restart apiserver: %v (code %v)", err, code)
	}
	return nil
}

func waitForApiserverUp(c *client.Client) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		body, err := c.Get().AbsPath("/healthz").Do().Raw()
		if err == nil && string(body) == "ok" {
			return nil
		}
	}
	return fmt.Errorf("waiting for apiserver timed out")
}

// waitForClusterSize waits until the cluster has desired size and there is no not-ready nodes in it.
func waitForClusterSize(c *client.Client, size int, timeout time.Duration) error {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		nodes, err := c.Nodes().List(labels.Everything(), fields.Everything())
		if err != nil {
			Logf("Failed to list nodes: %v", err)
			continue
		}
		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		filterNodes(nodes, func(node api.Node) bool {
			return isNodeReadySetAsExpected(&node, true)
		})
		numReady := len(nodes.Items)

		if numNodes == size && numReady == size {
			Logf("Cluster has reached the desired size %d", size)
			return nil
		}
		Logf("Waiting for cluster size %d, current size %d, not ready nodes %d", size, numNodes, numNodes-numReady)
	}
	return fmt.Errorf("timeout waiting %v for cluster size to be %d", timeout, size)
}
