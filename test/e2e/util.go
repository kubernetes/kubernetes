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
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	sshutil "k8s.io/kubernetes/pkg/ssh"
	"k8s.io/kubernetes/pkg/util"
	deploymentutil "k8s.io/kubernetes/pkg/util/deployment"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/blang/semver"
	"github.com/davecgh/go-spew/spew"
	"golang.org/x/crypto/ssh"
	"golang.org/x/net/websocket"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	gomegatypes "github.com/onsi/gomega/types"
)

const (
	// Initial pod start can be delayed O(minutes) by slow docker pulls
	// TODO: Make this 30 seconds once #4566 is resolved.
	podStartTimeout = 5 * time.Minute

	// How long to wait for the pod to no longer be running
	podNoLongerRunningTimeout = 30 * time.Second

	// If there are any orphaned namespaces to clean up, this test is running
	// on a long lived cluster. A long wait here is preferably to spurious test
	// failures caused by leaked resources from a previous test run.
	namespaceCleanupTimeout = 15 * time.Minute

	// Some pods can take much longer to get ready due to volume attach/detach latency.
	slowPodStartTimeout = 15 * time.Minute

	// How long to wait for a service endpoint to be resolvable.
	serviceStartTimeout = 1 * time.Minute

	// String used to mark pod deletion
	nonExist = "NonExist"

	// How often to poll pods and nodes.
	poll = 2 * time.Second

	// service accounts are provisioned after namespace creation
	// a service account is required to support pod creation in a namespace as part of admission control
	serviceAccountProvisionTimeout = 2 * time.Minute

	// How long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	singleCallTimeout = 30 * time.Second

	// How long nodes have to be "ready" when a test begins. They should already
	// be "ready" before the test starts, so this is small.
	nodeReadyInitialTimeout = 20 * time.Second

	// How long pods have to be "ready" when a test begins.
	podReadyBeforeTimeout = 2 * time.Minute

	podRespondingTimeout     = 2 * time.Minute
	serviceRespondingTimeout = 2 * time.Minute
	endpointRegisterTimeout  = time.Minute
)

// SubResource proxy should have been functional in v1.0.0, but SubResource
// proxy via tunneling is known to be broken in v1.0.  See
// https://github.com/kubernetes/kubernetes/pull/15224#issuecomment-146769463
//
// TODO(ihmccreery): remove once we don't care about v1.0 anymore, (tentatively
// in v1.3).
var subResourcePodProxyVersion = version.MustParse("v1.1.0")
var subResourceServiceAndNodeProxyVersion = version.MustParse("v1.2.0")

func getServicesProxyRequest(c *client.Client, request *client.Request) (*client.Request, error) {
	subResourceProxyAvailable, err := serverVersionGTE(subResourceServiceAndNodeProxyVersion, c)
	if err != nil {
		return nil, err
	}
	if subResourceProxyAvailable {
		return request.Resource("services").SubResource("proxy"), nil
	}
	return request.Prefix("proxy").Resource("services"), nil
}

func GetServicesProxyRequest(c *client.Client, request *client.Request) (*client.Request, error) {
	return getServicesProxyRequest(c, request)
}

type CloudConfig struct {
	ProjectID         string
	Zone              string
	Cluster           string
	MasterName        string
	NodeInstanceGroup string
	NumNodes          int
	ClusterTag        string
	ServiceAccount    string

	Provider cloudprovider.Interface
}

// unique identifier of the e2e run
var runId = util.NewUUID()

type CreateTestingNSFn func(baseName string, c *client.Client, labels map[string]string) (*api.Namespace, error)

type TestContextType struct {
	KubeConfig            string
	KubeContext           string
	KubeVolumeDir         string
	CertDir               string
	Host                  string
	RepoRoot              string
	Provider              string
	CloudConfig           CloudConfig
	KubectlPath           string
	OutputDir             string
	ReportDir             string
	prefix                string
	MinStartupPods        int
	UpgradeTarget         string
	PrometheusPushGateway string
	VerifyServiceAccount  bool
	DeleteNamespace       bool
	CleanStart            bool
	// If set to true framework will start a goroutine monitoring resource usage of system add-ons.
	// It will read the data every 30 seconds from all Nodes and print summary during afterEach.
	GatherKubeSystemResourceUsageData bool
	GatherLogsSizes                   bool
	GatherMetricsAfterTest            bool
	// Currently supported values are 'hr' for human-readable and 'json'. It's a comma separated list.
	OutputPrintType string
	// CreateTestingNS is responsible for creating namespace used for executing e2e tests.
	// It accepts namespace base name, which will be prepended with e2e prefix, kube client
	// and labels to be applied to a namespace.
	CreateTestingNS CreateTestingNSFn
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
		ListFunc: func(options api.ListOptions) (runtime.Object, error) {
			options.LabelSelector = label
			options.FieldSelector = field
			return c.Pods(namespace).List(options)
		},
		WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
			options.LabelSelector = label
			options.FieldSelector = field
			return c.Pods(namespace).Watch(options)
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
	Client         *client.Client
	Image          string
	Command        []string
	Name           string
	Namespace      string
	PollInterval   time.Duration
	Timeout        time.Duration
	PodStatusFile  *os.File
	Replicas       int
	CpuRequest     int64 // millicores
	CpuLimit       int64 // millicores
	MemRequest     int64 // bytes
	MemLimit       int64 // bytes
	ReadinessProbe *api.Probe

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

type DeploymentConfig struct {
	RCConfig
}

func nowStamp() string {
	return time.Now().Format(time.StampMilli)
}

func logf(level string, format string, args ...interface{}) {
	fmt.Fprintf(GinkgoWriter, nowStamp()+": "+level+": "+format+"\n", args...)
}

func Logf(format string, args ...interface{}) {
	logf("INFO", format, args...)
}

func Failf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	logf("FAIL", msg)
	Fail(nowStamp()+": "+msg, 1)
}

func Skipf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	logf("SKIP", msg)
	Skip(nowStamp() + ": " + msg)
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

func SkipUnlessServerVersionGTE(v semver.Version, c client.ServerVersionInterface) {
	gte, err := serverVersionGTE(v, c)
	if err != nil {
		Failf("Failed to get server version: %v", err)
	}
	if !gte {
		Skipf("Not supported for server versions before %q", v)
	}
}

// providersWithSSH are those providers where each node is accessible with SSH
var providersWithSSH = []string{"gce", "gke", "aws"}

// providersWithMasterSSH are those providers where master node is accessible with SSH
var providersWithMasterSSH = []string{"gce", "gke", "kubemark", "aws"}

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
		Logf("%-[1]*[2]s %-[3]*[4]s %-[5]*[6]s %-[7]*[8]s %[9]s",
			maxPodW, pod.ObjectMeta.Name, maxNodeW, pod.Spec.NodeName, maxPhaseW, pod.Status.Phase, maxGraceW, grace, pod.Status.Conditions)
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

// podNotReady checks whether pod p's has a ready condition of status false.
func podNotReady(p *api.Pod) (bool, error) {
	// Check the ready condition is false.
	if podReady(p) {
		return false, fmt.Errorf("pod '%s' on '%s' didn't have condition {%v %v}; conditions: %v",
			p.ObjectMeta.Name, p.Spec.NodeName, api.PodReady, api.ConditionFalse, p.Status.Conditions)
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
	if wait.PollImmediate(poll, timeout, func() (bool, error) {
		// We get the new list of pods and replication controllers in every
		// iteration because more pods come online during startup and we want to
		// ensure they are also checked.
		rcList, err := c.ReplicationControllers(ns).List(api.ListOptions{})
		if err != nil {
			Logf("Error getting replication controllers in namespace '%s': %v", ns, err)
			return false, nil
		}
		replicas := 0
		for _, rc := range rcList.Items {
			replicas += rc.Spec.Replicas
		}

		podList, err := c.Pods(ns).List(api.ListOptions{})
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

// deleteNamespaces deletes all namespaces that match the given delete and skip filters.
// Filter is by simple strings.Contains; first skip filter, then delete filter.
// Returns the list of deleted namespaces or an error.
func deleteNamespaces(c *client.Client, deleteFilter, skipFilter []string) ([]string, error) {
	By("Deleting namespaces")
	nsList, err := c.Namespaces().List(api.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	var deleted []string
	var wg sync.WaitGroup
OUTER:
	for _, item := range nsList.Items {
		if skipFilter != nil {
			for _, pattern := range skipFilter {
				if strings.Contains(item.Name, pattern) {
					continue OUTER
				}
			}
		}
		if deleteFilter != nil {
			var shouldDelete bool
			for _, pattern := range deleteFilter {
				if strings.Contains(item.Name, pattern) {
					shouldDelete = true
					break
				}
			}
			if !shouldDelete {
				continue OUTER
			}
		}
		wg.Add(1)
		deleted = append(deleted, item.Name)
		go func(nsName string) {
			defer wg.Done()
			defer GinkgoRecover()
			Expect(c.Namespaces().Delete(nsName)).To(Succeed())
			Logf("namespace : %v api call to delete is complete ", nsName)
		}(item.Name)
	}
	wg.Wait()
	return deleted, nil
}

func waitForNamespacesDeleted(c *client.Client, namespaces []string, timeout time.Duration) error {
	By("Waiting for namespaces to vanish")
	nsMap := map[string]bool{}
	for _, ns := range namespaces {
		nsMap[ns] = true
	}
	//Now POLL until all namespaces have been eradicated.
	return wait.Poll(2*time.Second, timeout,
		func() (bool, error) {
			nsList, err := c.Namespaces().List(api.ListOptions{})
			if err != nil {
				return false, err
			}
			for _, item := range nsList.Items {
				if _, ok := nsMap[item.Name]; ok {
					return false, nil
				}
			}
			return true, nil
		})
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
	Logf("Waiting up to %[1]v for pod %[2]s status to be %[3]s", timeout, podName, desc)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		pod, err := c.Pods(ns).Get(podName)
		if err != nil {
			if apierrs.IsNotFound(err) {
				Logf("Pod %q in namespace %q disappeared. Error: %v", podName, ns, err)
				return err
			}
			// Aligning this text makes it much more readable
			Logf("Get pod %[1]s in namespace '%[2]s' failed, ignoring for %[3]v. Error: %[4]v",
				podName, ns, poll, err)
			continue
		}
		done, err := condition(pod)
		if done {
			return err
		}
		Logf("Waiting for pod %[1]s in namespace '%[2]s' status to be '%[3]s'"+
			"(found phase: %[4]q, readiness: %[5]t) (%[6]v elapsed)",
			podName, ns, desc, pod.Status.Phase, podReady(pod), time.Since(start))
	}
	return fmt.Errorf("gave up waiting for pod '%s' to be '%s' after %v", podName, desc, timeout)
}

// waitForMatchPodsCondition finds match pods based on the input ListOptions.
// waits and checks if all match pods are in the given podCondition
func waitForMatchPodsCondition(c *client.Client, opts api.ListOptions, desc string, timeout time.Duration, condition podCondition) error {
	Logf("Waiting up to %v for matching pods' status to be %s", timeout, desc)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		pods, err := c.Pods(api.NamespaceAll).List(opts)
		if err != nil {
			return err
		}
		conditionNotMatch := []string{}
		for _, pod := range pods.Items {
			done, err := condition(&pod)
			if done && err != nil {
				return fmt.Errorf("Unexpected error: %v", err)
			}
			if !done {
				conditionNotMatch = append(conditionNotMatch, format.Pod(&pod))
			}
		}
		if len(conditionNotMatch) <= 0 {
			return err
		}
		Logf("%d pods are not %s", len(conditionNotMatch), desc)
	}
	return fmt.Errorf("gave up waiting for matching pods to be '%s' after %v", desc, timeout)
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

// CreateTestingNS should be used by every test, note that we append a common prefix to the provided test name.
// Please see NewFramework instead of using this directly.
func CreateTestingNS(baseName string, c *client.Client, labels map[string]string) (*api.Namespace, error) {
	if labels == nil {
		labels = map[string]string{}
	}
	labels["e2e-run"] = string(runId)

	namespaceObj := &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			GenerateName: fmt.Sprintf("e2e-tests-%v-", baseName),
			Namespace:    "",
			Labels:       labels,
		},
		Status: api.NamespaceStatus{},
	}
	// Be robust about making the namespace creation call.
	var got *api.Namespace
	if err := wait.PollImmediate(poll, singleCallTimeout, func() (bool, error) {
		var err error
		got, err = c.Namespaces().Create(namespaceObj)
		if err != nil {
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, err
	}

	if testContext.VerifyServiceAccount {
		if err := waitForDefaultServiceAccountInNamespace(c, got.Name); err != nil {
			return nil, err
		}
	}
	return got, nil
}

// checkTestingNSDeletedExcept checks whether all e2e based existing namespaces are in the Terminating state
// and waits until they are finally deleted. It ignores namespace skip.
func checkTestingNSDeletedExcept(c *client.Client, skip string) error {
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
		namespaces, err := c.Namespaces().List(api.ListOptions{})
		if err != nil {
			Logf("Listing namespaces failed: %v", err)
			continue
		}
		terminating := 0
		for _, ns := range namespaces.Items {
			if strings.HasPrefix(ns.ObjectMeta.Name, "e2e-tests-") && ns.ObjectMeta.Name != skip {
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
func deleteNS(c *client.Client, namespace string, timeout time.Duration) error {
	if err := c.Namespaces().Delete(namespace); err != nil {
		return err
	}

	err := wait.PollImmediate(5*time.Second, timeout, func() (bool, error) {
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
	if pods, perr := c.Pods(namespace).List(api.ListOptions{}); perr == nil {
		for _, pod := range pods.Items {
			Logf("Pod %s %s on node %s remains, has deletion timestamp %s", namespace, pod.Name, pod.Spec.NodeName, pod.DeletionTimestamp)
			remaining = append(remaining, pod.Name)
			if pod.DeletionTimestamp == nil {
				missingTimestamp = true
			}
		}
	}

	// a timeout occurred
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

// Waits default amount of time (podStartTimeout) for the specified pod to become running.
// Returns an error if timeout occurs first, or pod goes in to failed state.
func waitForPodRunningInNamespace(c *client.Client, podName string, namespace string) error {
	return waitTimeoutForPodRunningInNamespace(c, podName, namespace, podStartTimeout)
}

// Waits an extended amount of time (slowPodStartTimeout) for the specified pod to become running.
// Returns an error if timeout occurs first, or pod goes in to failed state.
func waitForPodRunningInNamespaceSlow(c *client.Client, podName string, namespace string) error {
	return waitTimeoutForPodRunningInNamespace(c, podName, namespace, slowPodStartTimeout)
}

func waitTimeoutForPodRunningInNamespace(c *client.Client, podName string, namespace string, timeout time.Duration) error {
	return waitForPodCondition(c, namespace, podName, "running", timeout, func(pod *api.Pod) (bool, error) {
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

// Waits default amount of time (podNoLongerRunningTimeout) for the specified pod to stop running.
// Returns an error if timeout occurs first.
func waitForPodNoLongerRunningInNamespace(c *client.Client, podName string, namespace string) error {
	return waitTimeoutForPodNoLongerRunningInNamespace(c, podName, namespace, podNoLongerRunningTimeout)
}

func waitTimeoutForPodNoLongerRunningInNamespace(c *client.Client, podName string, namespace string, timeout time.Duration) error {
	return waitForPodCondition(c, namespace, podName, "no longer running", timeout, func(pod *api.Pod) (bool, error) {
		if pod.Status.Phase == api.PodSucceeded || pod.Status.Phase == api.PodFailed {
			Logf("Found pod '%s' with status '%s' on node '%s'", podName, pod.Status.Phase, pod.Spec.NodeName)
			return true, nil
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

// waitForPodTerminatedInNamespace returns an error if it took too long for the pod
// to terminate or if the pod terminated with an unexpected reason.
func waitForPodTerminatedInNamespace(c *client.Client, podName, reason, namespace string) error {
	return waitForPodCondition(c, namespace, podName, "terminated due to deadline exceeded", podStartTimeout, func(pod *api.Pod) (bool, error) {
		if pod.Status.Phase == api.PodFailed {
			if pod.Status.Reason == reason {
				return true, nil
			} else {
				return true, fmt.Errorf("Expected pod %n/%n to be terminated with reason %v, got reason: ", namespace, podName, reason, pod.Status.Reason)
			}
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
	err := wait.PollImmediate(10*time.Second, 5*time.Minute, func() (bool, error) {
		Logf("Waiting for pod %s to appear on node %s", rcName, node)
		options := api.ListOptions{LabelSelector: label}
		pods, err := c.Pods(ns).List(options)
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

func waitForPodToDisappear(c *client.Client, ns, podName string, label labels.Selector, interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		Logf("Waiting for pod %s to disappear", podName)
		options := api.ListOptions{LabelSelector: label}
		pods, err := c.Pods(ns).List(options)
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

// waitForRCPodToDisappear returns nil if the pod from the given replication controller (described by rcName) no longer exists.
// In case of failure or too long waiting time, an error is returned.
func waitForRCPodToDisappear(c *client.Client, ns, rcName, podName string) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": rcName}))
	// NodeController evicts pod after 5 minutes, so we need timeout greater than that.
	// Additionally, there can be non-zero grace period, so we are setting 10 minutes
	// to be on the safe size.
	return waitForPodToDisappear(c, ns, podName, label, 20*time.Second, 10*time.Minute)
}

// waitForService waits until the service appears (exist == true), or disappears (exist == false)
func waitForService(c *client.Client, namespace, name string, exist bool, interval, timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		_, err := c.Services(namespace).Get(name)
		switch {
		case err == nil:
			if !exist {
				return false, nil
			}
			Logf("Service %s in namespace %s found.", name, namespace)
			return true, nil
		case apierrs.IsNotFound(err):
			if exist {
				return false, nil
			}
			Logf("Service %s in namespace %s disappeared.", name, namespace)
			return true, nil
		default:
			Logf("Get service %s in namespace %s failed: %v", name, namespace, err)
			return false, nil
		}
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for service %s/%s %s: %v", namespace, name, stateMsg[exist], err)
	}
	return nil
}

//waitForServiceEndpointsNum waits until the amount of endpoints that implement service to expectNum.
func waitForServiceEndpointsNum(c *client.Client, namespace, serviceName string, expectNum int, interval, timeout time.Duration) error {
	return wait.Poll(interval, timeout, func() (bool, error) {
		Logf("Waiting for amount of service:%s endpoints to %d", serviceName, expectNum)
		list, err := c.Endpoints(namespace).List(api.ListOptions{})
		if err != nil {
			return false, err
		}

		for _, e := range list.Items {
			if e.Name == serviceName && countEndpointsNum(&e) == expectNum {
				return true, nil
			}
		}
		return false, nil
	})
}

func countEndpointsNum(e *api.Endpoints) int {
	num := 0
	for _, sub := range e.Subsets {
		num += len(sub.Addresses)
	}
	return num
}

// waitForReplicationController waits until the RC appears (exist == true), or disappears (exist == false)
func waitForReplicationController(c *client.Client, namespace, name string, exist bool, interval, timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
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

func waitForEndpoint(c *client.Client, ns, name string) error {
	for t := time.Now(); time.Since(t) < endpointRegisterTimeout; time.Sleep(poll) {
		endpoint, err := c.Endpoints(ns).Get(name)
		Expect(err).NotTo(HaveOccurred())
		if len(endpoint.Subsets) == 0 || len(endpoint.Subsets[0].Addresses) == 0 {
			Logf("Endpoint %s/%s is not ready yet", ns, name)
			continue
		} else {
			return nil
		}
	}
	return fmt.Errorf("Failed to get entpoints for %s/%s", ns, name)
}

// Context for checking pods responses by issuing GETs to them (via the API
// proxy) and verifying that they answer with ther own pod name.
type podProxyResponseChecker struct {
	c              *client.Client
	ns             string
	label          labels.Selector
	controllerName string
	respondName    bool // Whether the pod should respond with its own name.
	pods           *api.PodList
}

// checkAllResponses issues GETs to all pods in the context and verify they
// reply with their own pod name.
func (r podProxyResponseChecker) checkAllResponses() (done bool, err error) {
	successes := 0
	options := api.ListOptions{LabelSelector: r.label}
	currentPods, err := r.c.Pods(r.ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	for i, pod := range r.pods.Items {
		// Check that the replica list remains unchanged, otherwise we have problems.
		if !isElementOf(pod.UID, currentPods) {
			return false, fmt.Errorf("pod with UID %s is no longer a member of the replica set.  Must have been restarted for some reason.  Current replica set: %v", pod.UID, currentPods)
		}
		subResourceProxyAvailable, err := serverVersionGTE(subResourcePodProxyVersion, r.c)
		if err != nil {
			return false, err
		}
		var body []byte
		if subResourceProxyAvailable {
			body, err = r.c.Get().
				Namespace(r.ns).
				Resource("pods").
				SubResource("proxy").
				Name(string(pod.Name)).
				Do().
				Raw()
		} else {
			body, err = r.c.Get().
				Prefix("proxy").
				Namespace(r.ns).
				Resource("pods").
				Name(string(pod.Name)).
				Do().
				Raw()
		}
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

// serverVersionGTE returns true if v is greater than or equal to the server
// version.
//
// TODO(18726): This should be incorporated into client.VersionInterface.
func serverVersionGTE(v semver.Version, c client.ServerVersionInterface) (bool, error) {
	serverVersion, err := c.ServerVersion()
	if err != nil {
		return false, fmt.Errorf("Unable to get server version: %v", err)
	}
	sv, err := version.Parse(serverVersion.GitVersion)
	if err != nil {
		return false, fmt.Errorf("Unable to parse server version %q: %v", serverVersion.GitVersion, err)
	}
	return sv.GTE(v), nil
}

func podsResponding(c *client.Client, ns, name string, wantName bool, pods *api.PodList) error {
	By("trying to dial each unique pod")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	return wait.PollImmediate(poll, podRespondingTimeout, podProxyResponseChecker{c, ns, label, name, wantName, pods}.checkAllResponses)
}

func serviceResponding(c *client.Client, ns, name string) error {
	By(fmt.Sprintf("trying to dial the service %s.%s via the proxy", ns, name))

	return wait.PollImmediate(poll, serviceRespondingTimeout, func() (done bool, err error) {
		proxyRequest, errProxy := getServicesProxyRequest(c, c.Get())
		if errProxy != nil {
			Logf("Failed to get services proxy request: %v:", errProxy)
			return false, nil
		}
		body, err := proxyRequest.Namespace(ns).
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
		Logf(">>> testContext.KubeConfig: %s\n", testContext.KubeConfig)
		c, err := clientcmd.LoadFromFile(testContext.KubeConfig)
		if err != nil {
			return nil, fmt.Errorf("error loading KubeConfig: %v", err.Error())
		}
		if testContext.KubeContext != "" {
			Logf(">>> testContext.KubeContext: %s\n", testContext.KubeContext)
			c.CurrentContext = testContext.KubeContext
		}
		return clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: testContext.Host}}).ClientConfig()
	default:
		return nil, fmt.Errorf("KubeConfig must be specified to load client config")
	}
}

func loadClientFromConfig(config *client.Config) (*client.Client, error) {
	c, err := client.New(config)
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	if c.Client.Timeout == 0 {
		c.Client.Timeout = singleCallTimeout
	}
	return c, nil
}

func loadClient() (*client.Client, error) {
	config, err := loadConfig()
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	return loadClientFromConfig(config)
}

// randomSuffix provides a random string to append to pods,services,rcs.
// TODO: Allow service names to have the same form as names
//       for pods and replication controllers so we don't
//       need to use such a function and can instead
//       use the UUID utility function.
func randomSuffix() string {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	return strconv.Itoa(r.Int() % 10000)
}

func expectNoError(err error, explain ...interface{}) {
	if err != nil {
		Logf("Unexpected error occurred: %v", err)
	}
	ExpectWithOffset(1, err).NotTo(HaveOccurred(), explain...)
}

// Stops everything from filePath from namespace ns and checks if everything matching selectors from the given namespace is correctly stopped.
func cleanup(filePath string, ns string, selectors ...string) {
	By("using delete to clean up resources")
	var nsArg string
	if ns != "" {
		nsArg = fmt.Sprintf("--namespace=%s", ns)
	}
	runKubectlOrDie("delete", "--grace-period=0", "-f", filePath, nsArg)

	for _, selector := range selectors {
		resources := runKubectlOrDie("get", "rc,svc", "-l", selector, "--no-headers", nsArg)
		if resources != "" {
			Failf("Resources left running after stop:\n%s", resources)
		}
		pods := runKubectlOrDie("get", "pods", "-l", selector, nsArg, "-o", "go-template={{ range .items }}{{ if not .metadata.deletionTimestamp }}{{ .metadata.name }}{{ \"\\n\" }}{{ end }}{{ end }}")
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
		getPodsOutput := runKubectlOrDie("get", "pods", "-o", "template", getPodsTemplate, "--api-version=v1", "-l", testname, fmt.Sprintf("--namespace=%v", ns))
		pods := strings.Fields(getPodsOutput)
		if numPods := len(pods); numPods != replicas {
			By(fmt.Sprintf("Replicas for %s: expected=%d actual=%d", testname, replicas, numPods))
			continue
		}
		var runningPods []string
		for _, podID := range pods {
			running := runKubectlOrDie("get", "pods", podID, "-o", "template", getContainerStateTemplate, "--api-version=v1", fmt.Sprintf("--namespace=%v", ns))
			if running != "true" {
				Logf("%s is created but not running", podID)
				continue waitLoop
			}

			currentImage := runKubectlOrDie("get", "pods", podID, "-o", "template", getImageTemplate, "--api-version=v1", fmt.Sprintf("--namespace=%v", ns))
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

// kubectlBuilder is used to build, customize and execute a kubectl Command.
// Add more functions to customize the builder as needed.
type kubectlBuilder struct {
	cmd     *exec.Cmd
	timeout <-chan time.Time
}

func newKubectlCommand(args ...string) *kubectlBuilder {
	b := new(kubectlBuilder)
	b.cmd = kubectlCmd(args...)
	return b
}

func (b *kubectlBuilder) withTimeout(t <-chan time.Time) *kubectlBuilder {
	b.timeout = t
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

func (b kubectlBuilder) execOrDie() string {
	str, err := b.exec()
	Expect(err).NotTo(HaveOccurred())
	return str
}

func (b kubectlBuilder) exec() (string, error) {
	var stdout, stderr bytes.Buffer
	cmd := b.cmd
	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	Logf("Running '%s %s'", cmd.Path, strings.Join(cmd.Args[1:], " ")) // skip arg[0] as it is printed separately
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("Error starting %v:\nCommand stdout:\n%v\nstderr:\n%v\nerror:\n%v\n", cmd, cmd.Stdout, cmd.Stderr, err)
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Wait()
	}()
	select {
	case err := <-errCh:
		if err != nil {
			return "", fmt.Errorf("Error running %v:\nCommand stdout:\n%v\nstderr:\n%v\nerror:\n%v\n", cmd, cmd.Stdout, cmd.Stderr, err)
		}
	case <-b.timeout:
		b.cmd.Process.Kill()
		return "", fmt.Errorf("Timed out waiting for command %v:\nCommand stdout:\n%v\nstderr:\n%v\n", cmd, cmd.Stdout, cmd.Stderr)
	}
	Logf("stdout: %q", stdout.String())
	Logf("stderr: %q", stderr.String())
	// TODO: trimspace should be unnecessary after switching to use kubectl binary directly
	return strings.TrimSpace(stdout.String()), nil
}

// runKubectlOrDie is a convenience wrapper over kubectlBuilder
func runKubectlOrDie(args ...string) string {
	return newKubectlCommand(args...).execOrDie()
}

// runKubectl is a convenience wrapper over kubectlBuilder
func runKubectl(args ...string) (string, error) {
	return newKubectlCommand(args...).exec()
}

// runKubectlOrDieInput is a convenience wrapper over kubectlBuilder that takes input to stdin
func runKubectlOrDieInput(data string, args ...string) string {
	return newKubectlCommand(args...).withStdinData(data).execOrDie()
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

// testContainerOutput runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status, and tests
// the specified container log against the given expected output using a substring matcher.
func testContainerOutput(scenarioName string, c *client.Client, pod *api.Pod, containerIndex int, expectedOutput []string, ns string) {
	testContainerOutputMatcher(scenarioName, c, pod, containerIndex, expectedOutput, ns, ContainSubstring)
}

// testContainerOutputRegexp runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status, and tests
// the specified container log against the given expected output using a regexp matcher.
func testContainerOutputRegexp(scenarioName string, c *client.Client, pod *api.Pod, containerIndex int, expectedOutput []string, ns string) {
	testContainerOutputMatcher(scenarioName, c, pod, containerIndex, expectedOutput, ns, MatchRegexp)
}

// testContainerOutputMatcher runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status, and tests
// the specified container log against the given expected output using the given matcher.
func testContainerOutputMatcher(scenarioName string,
	c *client.Client,
	pod *api.Pod,
	containerIndex int,
	expectedOutput []string, ns string,
	matcher func(string, ...interface{}) gomegatypes.GomegaMatcher) {
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
	var logs string
	start := time.Now()

	// Sometimes the actual containers take a second to get started, try to get logs for 60s
	for time.Now().Sub(start) < (60 * time.Second) {
		err = nil
		logs, err = getPodLogs(c, ns, pod.Name, containerName)
		if err != nil {
			By(fmt.Sprintf("Warning: Failed to get logs from node %q pod %q container %q. %v",
				podStatus.Spec.NodeName, podStatus.Name, containerName, err))
			time.Sleep(5 * time.Second)
			continue

		}
		By(fmt.Sprintf("Successfully fetched pod logs:%v\n", logs))
		break
	}

	for _, m := range expectedOutput {
		Expect(logs).To(matcher(m), "%q in container output", m)
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

// RunDeployment Launches (and verifies correctness) of a Deployment
// and will wait for all pods it spawns to become "Running".
// It's the caller's responsibility to clean up externally (i.e. use the
// namespace lifecycle for handling cleanup).
func RunDeployment(config DeploymentConfig) error {
	err := config.create()
	if err != nil {
		return err
	}
	return config.start()
}

func (config *DeploymentConfig) create() error {
	By(fmt.Sprintf("creating deployment %s in namespace %s", config.Name, config.Namespace))
	deployment := &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: config.Name,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: config.Replicas,
			Selector: &unversioned.LabelSelector{
				MatchLabels: map[string]string{
					"name": config.Name,
				},
			},
			Template: api.PodTemplateSpec{
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

	config.applyTo(&deployment.Spec.Template)

	_, err := config.Client.Deployments(config.Namespace).Create(deployment)
	if err != nil {
		return fmt.Errorf("Error creating deployment: %v", err)
	}
	Logf("Created deployment with name: %v, namespace: %v, replica count: %v", deployment.Name, config.Namespace, deployment.Spec.Replicas)
	return nil
}

// RunRC Launches (and verifies correctness) of a Replication Controller
// and will wait for all pods it spawns to become "Running".
// It's the caller's responsibility to clean up externally (i.e. use the
// namespace lifecycle for handling cleanup).
func RunRC(config RCConfig) error {
	err := config.create()
	if err != nil {
		return err
	}
	return config.start()
}

func (config *RCConfig) create() error {
	By(fmt.Sprintf("creating replication controller %s in namespace %s", config.Name, config.Namespace))
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
							Name:           config.Name,
							Image:          config.Image,
							Command:        config.Command,
							Ports:          []api.ContainerPort{{ContainerPort: 80}},
							ReadinessProbe: config.ReadinessProbe,
						},
					},
					DNSPolicy: api.DNSDefault,
				},
			},
		},
	}

	config.applyTo(rc.Spec.Template)

	_, err := config.Client.ReplicationControllers(config.Namespace).Create(rc)
	if err != nil {
		return fmt.Errorf("Error creating replication controller: %v", err)
	}
	Logf("Created replication controller with name: %v, namespace: %v, replica count: %v", rc.Name, config.Namespace, rc.Spec.Replicas)
	return nil
}

func (config *RCConfig) applyTo(template *api.PodTemplateSpec) {
	if config.Env != nil {
		for k, v := range config.Env {
			c := &template.Spec.Containers[0]
			c.Env = append(c.Env, api.EnvVar{Name: k, Value: v})
		}
	}
	if config.Labels != nil {
		for k, v := range config.Labels {
			template.ObjectMeta.Labels[k] = v
		}
	}
	if config.Ports != nil {
		for k, v := range config.Ports {
			c := &template.Spec.Containers[0]
			c.Ports = append(c.Ports, api.ContainerPort{Name: k, ContainerPort: v})
		}
	}
	if config.CpuLimit > 0 || config.MemLimit > 0 {
		template.Spec.Containers[0].Resources.Limits = api.ResourceList{}
	}
	if config.CpuLimit > 0 {
		template.Spec.Containers[0].Resources.Limits[api.ResourceCPU] = *resource.NewMilliQuantity(config.CpuLimit, resource.DecimalSI)
	}
	if config.MemLimit > 0 {
		template.Spec.Containers[0].Resources.Limits[api.ResourceMemory] = *resource.NewQuantity(config.MemLimit, resource.DecimalSI)
	}
	if config.CpuRequest > 0 || config.MemRequest > 0 {
		template.Spec.Containers[0].Resources.Requests = api.ResourceList{}
	}
	if config.CpuRequest > 0 {
		template.Spec.Containers[0].Resources.Requests[api.ResourceCPU] = *resource.NewMilliQuantity(config.CpuRequest, resource.DecimalSI)
	}
	if config.MemRequest > 0 {
		template.Spec.Containers[0].Resources.Requests[api.ResourceMemory] = *resource.NewQuantity(config.MemRequest, resource.DecimalSI)
	}
}

func (config *RCConfig) start() error {
	// Don't force tests to fail if they don't care about containers restarting.
	var maxContainerFailures int
	if config.MaxContainerFailures == nil {
		maxContainerFailures = int(math.Max(1.0, float64(config.Replicas)*.01))
	} else {
		maxContainerFailures = *config.MaxContainerFailures
	}

	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": config.Name}))

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
		runningButNotReady := 0
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
				ready := false
				for _, c := range p.Status.Conditions {
					if c.Type == api.PodReady && c.Status == api.ConditionTrue {
						ready = true
						break
					}
				}
				if ready {
					// Only count a pod is running when it is also ready.
					running++
				} else {
					runningButNotReady++
				}
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

		Logf("%v Pods: %d out of %d created, %d running, %d pending, %d waiting, %d inactive, %d terminating, %d unknown, %d runningButNotReady ",
			config.Name, len(pods), config.Replicas, running, pending, waiting, inactive, terminating, unknown, runningButNotReady)

		promPushRunningPending(running, pending)

		if config.PodStatusFile != nil {
			fmt.Fprintf(config.PodStatusFile, "%d, running, %d, pending, %d, waiting, %d, inactive, %d, unknown, %d, runningButNotReady\n", running, pending, waiting, inactive, unknown, runningButNotReady)
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
			errorStr := fmt.Sprintf("Number of reported pods for %s changed: %d vs %d", config.Name, len(pods), len(oldPods))
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
		// List only pods from a given replication controller.
		options := api.ListOptions{LabelSelector: label}
		if pods, err := config.Client.Pods(api.NamespaceAll).List(options); err == nil {

			for _, pod := range pods.Items {
				Logf("Pod %s\t%s\t%s\t%s", pod.Name, pod.Spec.NodeName, pod.Status.Phase, pod.DeletionTimestamp)
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

func dumpAllNamespaceInfo(c *client.Client, namespace string) {
	By(fmt.Sprintf("Collecting events from namespace %q.", namespace))
	events, err := c.Events(namespace).List(api.ListOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Sort events by their first timestamp
	sortedEvents := events.Items
	if len(sortedEvents) > 1 {
		sort.Sort(byFirstTimestamp(sortedEvents))
	}
	for _, e := range sortedEvents {
		Logf("At %v - event for %v: %v %v: %v", e.FirstTimestamp, e.InvolvedObject.Name, e.Source, e.Reason, e.Message)
	}
	// Note that we don't wait for any cleanup to propagate, which means
	// that if you delete a bunch of pods right before ending your test,
	// you may or may not see the killing/deletion/cleanup events.

	dumpAllPodInfo(c)

	dumpAllNodeInfo(c)
}

// byFirstTimestamp sorts a slice of events by first timestamp, using their involvedObject's name as a tie breaker.
type byFirstTimestamp []api.Event

func (o byFirstTimestamp) Len() int      { return len(o) }
func (o byFirstTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byFirstTimestamp) Less(i, j int) bool {
	if o[i].FirstTimestamp.Equal(o[j].FirstTimestamp) {
		return o[i].InvolvedObject.Name < o[j].InvolvedObject.Name
	}
	return o[i].FirstTimestamp.Before(o[j].FirstTimestamp)
}

func dumpAllPodInfo(c *client.Client) {
	pods, err := c.Pods("").List(api.ListOptions{})
	if err != nil {
		Logf("unable to fetch pod debug info: %v", err)
	}
	logPodStates(pods.Items)
}

func dumpAllNodeInfo(c *client.Client) {
	// It should be OK to list unschedulable Nodes here.
	nodes, err := c.Nodes().List(api.ListOptions{})
	if err != nil {
		Logf("unable to fetch node list: %v", err)
		return
	}
	names := make([]string, len(nodes.Items))
	for ix := range nodes.Items {
		names[ix] = nodes.Items[ix].Name
	}
	dumpNodeDebugInfo(c, names)
}

func dumpNodeDebugInfo(c *client.Client, nodeNames []string) {
	for _, n := range nodeNames {
		Logf("\nLogging node info for node %v", n)
		node, err := c.Nodes().Get(n)
		if err != nil {
			Logf("Error getting node info %v", err)
		}
		Logf("Node Info: %v", node)

		Logf("\nLogging kubelet events for node %v", n)
		for _, e := range getNodeEvents(c, n) {
			Logf("source %v type %v message %v reason %v first ts %v last ts %v, involved obj %+v",
				e.Source, e.Type, e.Message, e.Reason, e.FirstTimestamp, e.LastTimestamp, e.InvolvedObject)
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
	selector := fields.Set{
		"involvedObject.kind":      "Node",
		"involvedObject.name":      nodeName,
		"involvedObject.namespace": api.NamespaceAll,
		"source":                   "kubelet",
	}.AsSelector()
	options := api.ListOptions{FieldSelector: selector}
	events, err := c.Events(api.NamespaceSystem).List(options)
	if err != nil {
		Logf("Unexpected error retrieving node events %v", err)
		return []api.Event{}
	}
	return events.Items
}

// Convenient wrapper around listing nodes supporting retries.
func ListSchedulableNodesOrDie(c *client.Client) *api.NodeList {
	var nodes *api.NodeList
	var err error
	if wait.PollImmediate(poll, singleCallTimeout, func() (bool, error) {
		nodes, err = c.Nodes().List(api.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector()})
		return err == nil, nil
	}) != nil {
		expectNoError(err, "Timed out while listing nodes for e2e cluster.")
	}
	return nodes
}

func ScaleRC(c *client.Client, ns, name string, size uint, wait bool) error {
	By(fmt.Sprintf("Scaling replication controller %s in namespace %s to %d", name, ns, size))
	scaler, err := kubectl.ScalerFor(api.Kind("ReplicationController"), c)
	if err != nil {
		return err
	}
	waitForScale := kubectl.NewRetryParams(5*time.Second, 1*time.Minute)
	waitForReplicas := kubectl.NewRetryParams(5*time.Second, 5*time.Minute)
	if err = scaler.Scale(ns, name, size, nil, waitForScale, waitForReplicas); err != nil {
		return fmt.Errorf("error while scaling RC %s to %d replicas: %v", name, size, err)
	}
	if !wait {
		return nil
	}
	return waitForRCPodsRunning(c, ns, name)
}

// Wait up to 10 minutes for pods to become Running. Assume that the pods of the
// rc are labels with {"name":rcName}.
func waitForRCPodsRunning(c *client.Client, ns, rcName string) error {
	selector := labels.SelectorFromSet(labels.Set(map[string]string{"name": rcName}))
	err := waitForPodsWithLabelRunning(c, ns, selector)
	if err != nil {
		return fmt.Errorf("Error while waiting for replication controller %s pods to be running: %v", rcName, err)
	}
	return nil
}

// Wait up to 10 minutes for all matching pods to become Running and at least one
// matching pod exists.
func waitForPodsWithLabelRunning(c *client.Client, ns string, label labels.Selector) error {
	running := false
	podStore := newPodStore(c, ns, label, fields.Everything())
	defer podStore.Stop()
waitLoop:
	for start := time.Now(); time.Since(start) < 10*time.Minute; time.Sleep(5 * time.Second) {
		pods := podStore.List()
		if len(pods) == 0 {
			continue waitLoop
		}
		for _, p := range pods {
			if p.Status.Phase != api.PodRunning {
				continue waitLoop
			}
		}
		running = true
		break
	}
	if !running {
		return fmt.Errorf("Timeout while waiting for pods with labels %q to be running", label.String())
	}
	return nil
}

// Wait up to 10 minutes for getting pods with certain label
func waitForPodsWithLabel(c *client.Client, ns string, label labels.Selector) (pods *api.PodList, err error) {
	for t := time.Now(); time.Since(t) < podListTimeout; time.Sleep(poll) {
		options := api.ListOptions{LabelSelector: label}
		pods, err = c.Pods(ns).List(options)
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
	By(fmt.Sprintf("deleting replication controller %s in namespace %s", name, ns))
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
	err = reaper.Stop(ns, name, 0, api.NewDeleteOptions(0))
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
	return wait.PollImmediate(poll, 2*time.Minute, func() (bool, error) {
		selector := labels.SelectorFromSet(rc.Spec.Selector)
		options := api.ListOptions{LabelSelector: selector}
		if pods, err := c.Pods(rc.Namespace).List(options); err == nil && len(pods.Items) == 0 {
			return true, nil
		}
		return false, nil
	})
}

// Delete a ReplicaSet and all pods it spawned
func DeleteReplicaSet(c *client.Client, ns, name string) error {
	By(fmt.Sprintf("deleting ReplicaSet %s in namespace %s", name, ns))
	rc, err := c.Extensions().ReplicaSets(ns).Get(name)
	if err != nil {
		if apierrs.IsNotFound(err) {
			Logf("ReplicaSet %s was already deleted: %v", name, err)
			return nil
		}
		return err
	}
	reaper, err := kubectl.ReaperFor(extensions.Kind("ReplicaSet"), c)
	if err != nil {
		if apierrs.IsNotFound(err) {
			Logf("ReplicaSet %s was already deleted: %v", name, err)
			return nil
		}
		return err
	}
	startTime := time.Now()
	err = reaper.Stop(ns, name, 0, api.NewDeleteOptions(0))
	if apierrs.IsNotFound(err) {
		Logf("ReplicaSet %s was already deleted: %v", name, err)
		return nil
	}
	deleteRSTime := time.Now().Sub(startTime)
	Logf("Deleting RS %s took: %v", name, deleteRSTime)
	if err == nil {
		err = waitForReplicaSetPodsGone(c, rc)
	}
	terminatePodTime := time.Now().Sub(startTime) - deleteRSTime
	Logf("Terminating ReplicaSet %s pods took: %v", name, terminatePodTime)
	return err
}

// waitForReplicaSetPodsGone waits until there are no pods reported under a
// ReplicaSet selector (because the pods have completed termination).
func waitForReplicaSetPodsGone(c *client.Client, rs *extensions.ReplicaSet) error {
	return wait.PollImmediate(poll, 2*time.Minute, func() (bool, error) {
		selector, err := unversioned.LabelSelectorAsSelector(rs.Spec.Selector)
		expectNoError(err)
		options := api.ListOptions{LabelSelector: selector}
		if pods, err := c.Pods(rs.Namespace).List(options); err == nil && len(pods.Items) == 0 {
			return true, nil
		}
		return false, nil
	})
}

// Waits for the deployment to reach desired state.
// Returns an error if minAvailable or maxCreated is broken at any times.
func waitForDeploymentStatus(c clientset.Interface, ns, deploymentName string, desiredUpdatedReplicas, minAvailable, maxCreated, minReadySeconds int) error {
	var oldRSs, allOldRSs, allRSs []*extensions.ReplicaSet
	var newRS *extensions.ReplicaSet
	var deployment *extensions.Deployment
	err := wait.Poll(poll, 5*time.Minute, func() (bool, error) {

		var err error
		deployment, err = c.Extensions().Deployments(ns).Get(deploymentName)
		if err != nil {
			return false, err
		}
		oldRSs, allOldRSs, err = deploymentutil.GetOldReplicaSets(*deployment, c)
		if err != nil {
			return false, err
		}
		newRS, err = deploymentutil.GetNewReplicaSet(*deployment, c)
		if err != nil {
			return false, err
		}
		if newRS == nil {
			// New RC hasn't been created yet.
			return false, nil
		}
		allRSs = append(oldRSs, newRS)
		totalCreated := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
		totalAvailable, err := deploymentutil.GetAvailablePodsForReplicaSets(c, allRSs, minReadySeconds)
		if err != nil {
			return false, err
		}
		if totalCreated > maxCreated {
			logReplicaSetsOfDeployment(deployment, allOldRSs, newRS)
			logPodsOfReplicaSets(c, allRSs, minReadySeconds)
			return false, fmt.Errorf("total pods created: %d, more than the max allowed: %d", totalCreated, maxCreated)
		}
		if totalAvailable < minAvailable {
			logReplicaSetsOfDeployment(deployment, allOldRSs, newRS)
			logPodsOfReplicaSets(c, allRSs, minReadySeconds)
			return false, fmt.Errorf("total pods available: %d, less than the min required: %d", totalAvailable, minAvailable)
		}

		// When the deployment status and its underlying resources reach the desired state, we're done
		if deployment.Status.Replicas == desiredUpdatedReplicas &&
			deployment.Status.UpdatedReplicas == desiredUpdatedReplicas &&
			deploymentutil.GetReplicaCountForReplicaSets(oldRSs) == 0 &&
			deploymentutil.GetReplicaCountForReplicaSets([]*extensions.ReplicaSet{newRS}) == desiredUpdatedReplicas {
			return true, nil
		}
		return false, nil
	})

	if err == wait.ErrWaitTimeout {
		logReplicaSetsOfDeployment(deployment, allOldRSs, newRS)
		logPodsOfReplicaSets(c, allRSs, minReadySeconds)
	}
	return err
}

func waitForPodsReady(c *clientset.Clientset, ns, name string, minReadySeconds int) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	options := api.ListOptions{LabelSelector: label}
	return wait.Poll(poll, 5*time.Minute, func() (bool, error) {
		pods, err := c.Pods(ns).List(options)
		if err != nil {
			return false, nil
		}
		for _, pod := range pods.Items {
			if !deploymentutil.IsPodAvailable(&pod, minReadySeconds) {
				return false, nil
			}
		}
		return true, nil
	})
}

// Waits for the deployment to clean up old rcs.
func waitForDeploymentOldRSsNum(c *clientset.Clientset, ns, deploymentName string, desiredRSNum int) error {
	return wait.Poll(poll, 5*time.Minute, func() (bool, error) {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		if err != nil {
			return false, err
		}
		_, oldRSs, err := deploymentutil.GetOldReplicaSets(*deployment, c)
		if err != nil {
			return false, err
		}
		return len(oldRSs) == desiredRSNum, nil
	})
}

func logReplicaSetsOfDeployment(deployment *extensions.Deployment, allOldRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet) {
	Logf("Deployment: %+v. Selector = %+v", deployment, deployment.Spec.Selector)
	for i := range allOldRSs {
		Logf("All old ReplicaSets (%d/%d) of deployment %s: %+v. Selector = %+v", i+1, len(allOldRSs), deployment.Name, allOldRSs[i], allOldRSs[i].Spec.Selector)
	}
	Logf("New ReplicaSet of deployment %s: %+v. Selector = %+v", deployment.Name, newRS, newRS.Spec.Selector)
}

func waitForObservedDeployment(c *clientset.Clientset, ns, deploymentName string, desiredGeneration int64) error {
	return deploymentutil.WaitForObservedDeployment(func() (*extensions.Deployment, error) { return c.Extensions().Deployments(ns).Get(deploymentName) }, desiredGeneration, poll, 1*time.Minute)
}

func logPodsOfReplicaSets(c clientset.Interface, rss []*extensions.ReplicaSet, minReadySeconds int) {
	allPods, err := deploymentutil.GetPodsForReplicaSets(c, rss)
	if err == nil {
		for _, pod := range allPods {
			availability := "not available"
			if deploymentutil.IsPodAvailable(&pod, minReadySeconds) {
				availability = "available"
			}
			Logf("Pod %s is %s: %+v", pod.Name, availability, pod)
		}
	}
}

// Waits for the number of events on the given object to reach a desired count.
func waitForEvents(c *client.Client, ns string, objOrRef runtime.Object, desiredEventsCount int) error {
	return wait.Poll(poll, 5*time.Minute, func() (bool, error) {
		events, err := c.Events(ns).Search(objOrRef)
		if err != nil {
			return false, fmt.Errorf("error in listing events: %s", err)
		}
		eventsCount := len(events.Items)
		if eventsCount == desiredEventsCount {
			return true, nil
		}
		if eventsCount < desiredEventsCount {
			return false, nil
		}
		// Number of events has exceeded the desired count.
		return false, fmt.Errorf("number of events has exceeded the desired count, eventsCount: %d, desiredCount: %d", eventsCount, desiredEventsCount)
	})
}

// Waits for the number of events on the given object to be at least a desired count.
func waitForPartialEvents(c *client.Client, ns string, objOrRef runtime.Object, atLeastEventsCount int) error {
	return wait.Poll(poll, 5*time.Minute, func() (bool, error) {
		events, err := c.Events(ns).Search(objOrRef)
		if err != nil {
			return false, fmt.Errorf("error in listing events: %s", err)
		}
		eventsCount := len(events.Items)
		if eventsCount >= atLeastEventsCount {
			return true, nil
		}
		return false, nil
	})
}

// waitForRollbackDone waits for the given deployment finishes rollback.
func waitForRollbackDone(c *clientset.Clientset, deployment *extensions.Deployment) (err error) {
	deployments := c.Extensions().Deployments(deployment.Namespace)
	name := deployment.Name
	return wait.Poll(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		if deployment, err = deployments.Get(name); err != nil {
			return false, err
		}
		// When deployment's RollbackTo is empty, the rollback is done.
		if deployment.Spec.RollbackTo == nil {
			return true, nil
		}
		return false, nil
	})
}

type updateDeploymentFunc func(d *extensions.Deployment)

func updateDeploymentWithRetries(c *clientset.Clientset, namespace, name string, applyUpdate updateDeploymentFunc) (deployment *extensions.Deployment, err error) {
	deployments := c.Extensions().Deployments(namespace)
	err = wait.Poll(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		if deployment, err = deployments.Get(name); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(deployment)
		if deployment, err = deployments.Update(deployment); err == nil {
			Logf("updating deployment %s", name)
			return true, nil
		}
		return false, nil
	})
	return deployment, err
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

// NodeAddresses returns the first address of the given type of each node.
func NodeAddresses(nodelist *api.NodeList, addrType api.NodeAddressType) []string {
	hosts := []string{}
	for _, n := range nodelist.Items {
		for _, addr := range n.Status.Addresses {
			// Use the first external IP address we find on the node, and
			// use at most one per node.
			// TODO(roberthbailey): Use the "preferred" address for the node, once
			// such a thing is defined (#2462).
			if addr.Type == addrType {
				hosts = append(hosts, addr.Address)
				break
			}
		}
	}
	return hosts
}

// NodeSSHHosts returns SSH-able host names for all nodes. It returns an error
// if it can't find an external IP for every node, though it still returns all
// hosts that it found in that case.
func NodeSSHHosts(c *client.Client) ([]string, error) {
	// It should be OK to list unschedulable Nodes here.
	nodelist, err := c.Nodes().List(api.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("error getting nodes: %v", err)
	}

	// TODO(roberthbailey): Use the "preferred" address for the node, once such a thing is defined (#2462).
	hosts := NodeAddresses(nodelist, api.NodeExternalIP)

	// Error if any node didn't have an external IP.
	if len(hosts) != len(nodelist.Items) {
		return hosts, fmt.Errorf(
			"only found %d external IPs on nodes, but found %d nodes. Nodelist: %v",
			len(hosts), len(nodelist.Items), nodelist)
	}

	sshHosts := make([]string, 0, len(hosts))
	for _, h := range hosts {
		sshHosts = append(sshHosts, net.JoinHostPort(h, "22"))
	}
	return sshHosts, nil
}

type SSHResult struct {
	User   string
	Host   string
	Cmd    string
	Stdout string
	Stderr string
	Code   int
}

// SSH synchronously SSHs to a node running on provider and runs cmd. If there
// is no error performing the SSH, the stdout, stderr, and exit code are
// returned.
func SSH(cmd, host, provider string) (SSHResult, error) {
	result := SSHResult{Host: host, Cmd: cmd}

	// Get a signer for the provider.
	signer, err := getSigner(provider)
	if err != nil {
		return result, fmt.Errorf("error getting signer for provider %s: '%v'", provider, err)
	}

	// RunSSHCommand will default to Getenv("USER") if user == "", but we're
	// defaulting here as well for logging clarity.
	result.User = os.Getenv("KUBE_SSH_USER")
	if result.User == "" {
		result.User = os.Getenv("USER")
	}

	stdout, stderr, code, err := sshutil.RunSSHCommand(cmd, result.User, host, signer)
	result.Stdout = stdout
	result.Stderr = stderr
	result.Code = code

	return result, err
}

func LogSSHResult(result SSHResult) {
	remote := fmt.Sprintf("%s@%s", result.User, result.Host)
	Logf("ssh %s: command:   %s", remote, result.Cmd)
	Logf("ssh %s: stdout:    %q", remote, result.Stdout)
	Logf("ssh %s: stderr:    %q", remote, result.Stderr)
	Logf("ssh %s: exit code: %d", remote, result.Code)
}

func issueSSHCommand(cmd, provider string, node *api.Node) error {
	Logf("Getting external IP address for %s", node.Name)
	host := ""
	for _, a := range node.Status.Addresses {
		if a.Type == api.NodeExternalIP {
			host = a.Address + ":22"
			break
		}
	}
	if host == "" {
		return fmt.Errorf("couldn't find external IP address for node %s", node.Name)
	}
	Logf("Calling %s on %s(%s)", cmd, node.Name, host)
	result, err := SSH(cmd, host, provider)
	LogSSHResult(result)
	if result.Code != 0 || err != nil {
		return fmt.Errorf("failed running %q: %v (exit code %d)", cmd, err, result.Code)
	}
	return nil
}

// NewHostExecPodSpec returns the pod spec of hostexec pod
func NewHostExecPodSpec(ns, name string) *api.Pod {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: registered.GroupOrDie(api.GroupName).GroupVersion.String(),
		},
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:            "hostexec",
					Image:           "gcr.io/google_containers/hostexec:1.2",
					ImagePullPolicy: api.PullIfNotPresent,
				},
			},
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
		},
	}
	return pod
}

// RunHostCmd runs the given cmd in the context of the given pod using `kubectl exec`
// inside of a shell.
func RunHostCmd(ns, name, cmd string) (string, error) {
	return runKubectl("exec", fmt.Sprintf("--namespace=%v", ns), name, "--", "/bin/sh", "-c", cmd)
}

// RunHostCmdOrDie calls RunHostCmd and dies on error.
func RunHostCmdOrDie(ns, name, cmd string) string {
	stdout, err := RunHostCmd(ns, name, cmd)
	expectNoError(err)
	return stdout
}

// LaunchHostExecPod launches a hostexec pod in the given namespace and waits
// until it's Running
func LaunchHostExecPod(client *client.Client, ns, name string) *api.Pod {
	hostExecPod := NewHostExecPodSpec(ns, name)
	pod, err := client.Pods(ns).Create(hostExecPod)
	expectNoError(err)
	err = waitForPodRunningInNamespace(client, pod.Name, pod.Namespace)
	expectNoError(err)
	return pod
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
	case "gce", "gke", "kubemark":
		keyfile = "google_compute_engine"
	case "aws":
		// If there is an env. variable override, use that.
		aws_keyfile := os.Getenv("AWS_SSH_KEY")
		if len(aws_keyfile) != 0 {
			return sshutil.MakePrivateKeySignerFromFile(aws_keyfile)
		}
		// Otherwise revert to home dir
		keyfile = "kube_aws_rsa"
	default:
		return nil, fmt.Errorf("getSigner(...) not implemented for %s", provider)
	}
	key := filepath.Join(keydir, keyfile)

	return sshutil.MakePrivateKeySignerFromFile(key)
}

// checkPodsRunning returns whether all pods whose names are listed in podNames
// in namespace ns are running and ready, using c and waiting at most timeout.
func checkPodsRunningReady(c *client.Client, ns string, podNames []string, timeout time.Duration) bool {
	np, desc := len(podNames), "running and ready"
	Logf("Waiting up to %v for %d pods to be %s: %s", timeout, np, desc, podNames)
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
			Logf("Pod %[1]s failed to be %[2]s.", podName, desc)
			success = false
		}
	}
	Logf("Wanted all %d pods to be %s. Result: %t. Pods: %v", np, desc, success, podNames)
	return success
}

// waitForNodeToBeReady returns whether node name is ready within timeout.
func waitForNodeToBeReady(c *client.Client, name string, timeout time.Duration) bool {
	return waitForNodeToBe(c, name, api.NodeReady, true, timeout)
}

// waitForNodeToBeNotReady returns whether node name is not ready (i.e. the
// readiness condition is anything but ready, e.g false or unknown) within
// timeout.
func waitForNodeToBeNotReady(c *client.Client, name string, timeout time.Duration) bool {
	return waitForNodeToBe(c, name, api.NodeReady, false, timeout)
}

func isNodeConditionSetAsExpected(node *api.Node, conditionType api.NodeConditionType, wantTrue bool) bool {
	// Check the node readiness condition (logging all).
	for _, cond := range node.Status.Conditions {
		// Ensure that the condition type and the status matches as desired.
		if cond.Type == conditionType {
			if (cond.Status == api.ConditionTrue) == wantTrue {
				return true
			} else {
				Logf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
					conditionType, node.Name, cond.Status == api.ConditionTrue, wantTrue, cond.Reason, cond.Message)
				return false
			}
		}
	}
	Logf("Couldn't find condition %v on node %v", conditionType, node.Name)
	return false
}

// waitForNodeToBe returns whether node "name's" condition state matches wantTrue
// within timeout. If wantTrue is true, it will ensure the node condition status
// is ConditionTrue; if it's false, it ensures the node condition is in any state
// other than ConditionTrue (e.g. not true or unknown).
func waitForNodeToBe(c *client.Client, name string, conditionType api.NodeConditionType, wantTrue bool, timeout time.Duration) bool {
	Logf("Waiting up to %v for node %s condition %s to be %t", timeout, name, conditionType, wantTrue)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		node, err := c.Nodes().Get(name)
		if err != nil {
			Logf("Couldn't get node %s", name)
			continue
		}

		if isNodeConditionSetAsExpected(node, conditionType, wantTrue) {
			return true
		}
	}
	Logf("Node %s didn't reach desired %s condition status (%t) within %v", name, conditionType, wantTrue, timeout)
	return false
}

// checks whether all registered nodes are ready
func allNodesReady(c *client.Client, timeout time.Duration) error {
	Logf("Waiting up to %v for all nodes to be ready", timeout)

	var notReady []api.Node
	err := wait.PollImmediate(poll, timeout, func() (bool, error) {
		notReady = nil
		// It should be OK to list unschedulable Nodes here.
		nodes, err := c.Nodes().List(api.ListOptions{})
		if err != nil {
			return false, err
		}
		for _, node := range nodes.Items {
			if !isNodeConditionSetAsExpected(&node, api.NodeReady, true) {
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
	// kubelet will restart the kube-proxy since it's running in a static pod
	result, err := SSH("sudo pkill kube-proxy", host, testContext.Provider)
	if err != nil || result.Code != 0 {
		LogSSHResult(result)
		return fmt.Errorf("couldn't restart kube-proxy: %v", err)
	}
	// wait for kube-proxy to come back up
	err = wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
		result, err := SSH("sudo /bin/sh -c 'pgrep kube-proxy | wc -l'", host, testContext.Provider)
		if err != nil {
			return false, err
		}
		if result.Code != 0 {
			LogSSHResult(result)
			return false, fmt.Errorf("failed to run command, exited %d", result.Code)
		}
		if result.Stdout == "0\n" {
			return false, nil
		}
		Logf("kube-proxy is back up.")
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("kube-proxy didn't recover: %v", err)
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
	result, err := SSH(command, getMasterHost()+":22", testContext.Provider)
	if err != nil || result.Code != 0 {
		LogSSHResult(result)
		return fmt.Errorf("couldn't restart apiserver: %v", err)
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
// By cluster size we mean number of Nodes excluding Master Node.
func waitForClusterSize(c *client.Client, size int, timeout time.Duration) error {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		nodes, err := c.Nodes().List(api.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector()})
		if err != nil {
			Logf("Failed to list nodes: %v", err)
			continue
		}
		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		filterNodes(nodes, func(node api.Node) bool {
			return isNodeConditionSetAsExpected(&node, api.NodeReady, true)
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

// getHostExternalAddress gets the node for a pod and returns the first External
// address. Returns an error if the node the pod is on doesn't have an External
// address.
func getHostExternalAddress(client *client.Client, p *api.Pod) (externalAddress string, err error) {
	node, err := client.Nodes().Get(p.Spec.NodeName)
	if err != nil {
		return "", err
	}
	for _, address := range node.Status.Addresses {
		if address.Type == api.NodeExternalIP {
			if address.Address != "" {
				externalAddress = address.Address
				break
			}
		}
	}
	if externalAddress == "" {
		err = fmt.Errorf("No external address for pod %v on node %v",
			p.Name, p.Spec.NodeName)
	}
	return
}

type extractRT struct {
	http.Header
}

func (rt *extractRT) RoundTrip(req *http.Request) (*http.Response, error) {
	rt.Header = req.Header
	return nil, nil
}

// headersForConfig extracts any http client logic necessary for the provided
// config.
func headersForConfig(c *client.Config) (http.Header, error) {
	extract := &extractRT{}
	rt, err := client.HTTPWrappersForConfig(c, extract)
	if err != nil {
		return nil, err
	}
	if _, err := rt.RoundTrip(&http.Request{}); err != nil {
		return nil, err
	}
	return extract.Header, nil
}

// OpenWebSocketForURL constructs a websocket connection to the provided URL, using the client
// config, with the specified protocols.
func OpenWebSocketForURL(url *url.URL, config *client.Config, protocols []string) (*websocket.Conn, error) {
	tlsConfig, err := client.TLSConfigFor(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create tls config: %v", err)
	}
	if tlsConfig != nil {
		url.Scheme = "wss"
		if !strings.Contains(url.Host, ":") {
			url.Host += ":443"
		}
	} else {
		url.Scheme = "ws"
		if !strings.Contains(url.Host, ":") {
			url.Host += ":80"
		}
	}
	headers, err := headersForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to load http headers: %v", err)
	}
	cfg, err := websocket.NewConfig(url.String(), "http://localhost")
	if err != nil {
		return nil, fmt.Errorf("failed to create websocket config: %v", err)
	}
	cfg.Header = headers
	cfg.TlsConfig = tlsConfig
	cfg.Protocol = protocols
	return websocket.DialConfig(cfg)
}

// getIngressAddress returns the ips/hostnames associated with the Ingress.
func getIngressAddress(client *client.Client, ns, name string) ([]string, error) {
	ing, err := client.Extensions().Ingress(ns).Get(name)
	if err != nil {
		return nil, err
	}
	addresses := []string{}
	for _, a := range ing.Status.LoadBalancer.Ingress {
		if a.IP != "" {
			addresses = append(addresses, a.IP)
		}
		if a.Hostname != "" {
			addresses = append(addresses, a.Hostname)
		}
	}
	return addresses, nil
}

// waitForIngressAddress waits for the Ingress to acquire an address.
func waitForIngressAddress(c *client.Client, ns, ingName string, timeout time.Duration) (string, error) {
	var address string
	err := wait.PollImmediate(10*time.Second, timeout, func() (bool, error) {
		ipOrNameList, err := getIngressAddress(c, ns, ingName)
		if err != nil || len(ipOrNameList) == 0 {
			Logf("Waiting for Ingress %v to acquire IP, error %v", ingName, err)
			return false, nil
		}
		address = ipOrNameList[0]
		return true, nil
	})
	return address, err
}

// Looks for the given string in the log of a specific pod container
func lookForStringInLog(ns, podName, container, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		return runKubectlOrDie("log", podName, container, fmt.Sprintf("--namespace=%v", ns))
	})
}

// Looks for the given string in a file in a specific pod container
func lookForStringInFile(ns, podName, container, file, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		return runKubectlOrDie("exec", podName, "-c", container, fmt.Sprintf("--namespace=%v", ns), "--", "cat", file)
	})
}

// Looks for the given string in the output of a command executed in a specific pod container
func lookForStringInPodExec(ns, podName string, command []string, expectedString string, timeout time.Duration) (result string, err error) {
	return lookForString(expectedString, timeout, func() string {
		// use the first container
		args := []string{"exec", podName, fmt.Sprintf("--namespace=%v", ns), "--"}
		args = append(args, command...)
		return runKubectlOrDie(args...)
	})
}

// Looks for the given string in the output of fn, repeatedly calling fn until
// the timeout is reached or the string is found. Returns last log and possibly
// error if the string was not found.
func lookForString(expectedString string, timeout time.Duration, fn func() string) (result string, err error) {
	for t := time.Now(); time.Since(t) < timeout; time.Sleep(poll) {
		result = fn()
		if strings.Contains(result, expectedString) {
			return
		}
	}
	err = fmt.Errorf("Failed to find \"%s\", last result: \"%s\"", expectedString, result)
	return
}

// getSvcNodePort returns the node port for the given service:port.
func getSvcNodePort(client *client.Client, ns, name string, svcPort int) (int, error) {
	svc, err := client.Services(ns).Get(name)
	if err != nil {
		return 0, err
	}
	for _, p := range svc.Spec.Ports {
		if p.Port == svcPort {
			if p.NodePort != 0 {
				return p.NodePort, nil
			}
		}
	}
	return 0, fmt.Errorf(
		"No node port found for service %v, port %v", name, svcPort)
}

// getNodePortURL returns the url to a nodeport Service.
func getNodePortURL(client *client.Client, ns, name string, svcPort int) (string, error) {
	nodePort, err := getSvcNodePort(client, ns, name, svcPort)
	if err != nil {
		return "", err
	}
	// It should be OK to list unschedulable Node here.
	nodes, err := client.Nodes().List(api.ListOptions{})
	if err != nil {
		return "", err
	}
	if len(nodes.Items) == 0 {
		return "", fmt.Errorf("Unable to list nodes in cluster.")
	}
	for _, node := range nodes.Items {
		for _, address := range node.Status.Addresses {
			if address.Type == api.NodeExternalIP {
				if address.Address != "" {
					return fmt.Sprintf("http://%v:%v", address.Address, nodePort), nil
				}
			}
		}
	}
	return "", fmt.Errorf("Failed to find external address for service %v", name)
}

// scaleRCByLabels scales an RC via ns/label lookup. If replicas == 0 it waits till
// none are running, otherwise it does what a synchronous scale operation would do.
func scaleRCByLabels(client *client.Client, ns string, l map[string]string, replicas uint) error {
	listOpts := api.ListOptions{LabelSelector: labels.SelectorFromSet(labels.Set(l))}
	rcs, err := client.ReplicationControllers(ns).List(listOpts)
	if err != nil {
		return err
	}
	if len(rcs.Items) == 0 {
		return fmt.Errorf("RC with labels %v not found in ns %v", l, ns)
	}
	Logf("Scaling %v RCs with labels %v in ns %v to %v replicas.", len(rcs.Items), l, ns, replicas)
	for _, labelRC := range rcs.Items {
		name := labelRC.Name
		if err := ScaleRC(client, ns, name, replicas, false); err != nil {
			return err
		}
		rc, err := client.ReplicationControllers(ns).Get(name)
		if err != nil {
			return err
		}
		if replicas == 0 {
			if err := waitForRCPodsGone(client, rc); err != nil {
				return err
			}
		} else {
			if err := waitForPodsWithLabelRunning(
				client, ns, labels.SelectorFromSet(labels.Set(rc.Spec.Selector))); err != nil {
				return err
			}
		}
	}
	return nil
}

func getPodLogs(c *client.Client, namespace, podName, containerName string) (string, error) {
	return getPodLogsInternal(c, namespace, podName, containerName, false)
}

func getPreviousPodLogs(c *client.Client, namespace, podName, containerName string) (string, error) {
	return getPodLogsInternal(c, namespace, podName, containerName, true)
}

// utility function for gomega Eventually
func getPodLogsInternal(c *client.Client, namespace, podName, containerName string, previous bool) (string, error) {
	logs, err := c.Get().
		Resource("pods").
		Namespace(namespace).
		Name(podName).SubResource("log").
		Param("container", containerName).
		Param("previous", strconv.FormatBool(previous)).
		Do().
		Raw()
	if err != nil {
		return "", err
	}
	if err == nil && strings.Contains(string(logs), "Internal Error") {
		return "", fmt.Errorf("Fetched log contains \"Internal Error\": %q.", string(logs))
	}
	return string(logs), err
}

// EnsureLoadBalancerResourcesDeleted ensures that cloud load balancer resources that were created
// are actually cleaned up.  Currently only implemented for GCE/GKE.
func EnsureLoadBalancerResourcesDeleted(ip, portRange string) error {
	if testContext.Provider == "gce" || testContext.Provider == "gke" {
		return ensureGCELoadBalancerResourcesDeleted(ip, portRange)
	}
	return nil
}

func ensureGCELoadBalancerResourcesDeleted(ip, portRange string) error {
	gceCloud, ok := testContext.CloudConfig.Provider.(*gcecloud.GCECloud)
	if !ok {
		return fmt.Errorf("failed to convert CloudConfig.Provider to GCECloud: %#v", testContext.CloudConfig.Provider)
	}
	project := testContext.CloudConfig.ProjectID
	region, err := gcecloud.GetGCERegion(testContext.CloudConfig.Zone)
	if err != nil {
		return fmt.Errorf("could not get region for zone %q: %v", testContext.CloudConfig.Zone, err)
	}

	return wait.Poll(10*time.Second, 5*time.Minute, func() (bool, error) {
		service := gceCloud.GetComputeService()
		list, err := service.ForwardingRules.List(project, region).Do()
		if err != nil {
			return false, err
		}
		for ix := range list.Items {
			item := list.Items[ix]
			if item.PortRange == portRange && item.IPAddress == ip {
				Logf("found a load balancer: %v", item)
				return false, nil
			}
		}
		return true, nil
	})
}

// The following helper functions can block/unblock network from source
// host to destination host by manipulating iptable rules.
// This function assumes it can ssh to the source host.
//
// Caution:
// Recommend to input IP instead of hostnames. Using hostnames will cause iptables to
// do a DNS lookup to resolve the name to an IP address, which will
// slow down the test and cause it to fail if DNS is absent or broken.
//
// Suggested usage pattern:
// func foo() {
//	...
//	defer unblockNetwork(from, to)
//	blockNetwork(from, to)
//	...
// }
//
func blockNetwork(from string, to string) {
	Logf("block network traffic from %s to %s", from, to)
	iptablesRule := fmt.Sprintf("OUTPUT --destination %s --jump REJECT", to)
	dropCmd := fmt.Sprintf("sudo iptables --insert %s", iptablesRule)
	if result, err := SSH(dropCmd, from, testContext.Provider); result.Code != 0 || err != nil {
		LogSSHResult(result)
		Failf("Unexpected error: %v", err)
	}
}

func unblockNetwork(from string, to string) {
	Logf("Unblock network traffic from %s to %s", from, to)
	iptablesRule := fmt.Sprintf("OUTPUT --destination %s --jump REJECT", to)
	undropCmd := fmt.Sprintf("sudo iptables --delete %s", iptablesRule)
	// Undrop command may fail if the rule has never been created.
	// In such case we just lose 30 seconds, but the cluster is healthy.
	// But if the rule had been created and removing it failed, the node is broken and
	// not coming back. Subsequent tests will run or fewer nodes (some of the tests
	// may fail). Manual intervention is required in such case (recreating the
	// cluster solves the problem too).
	err := wait.Poll(time.Millisecond*100, time.Second*30, func() (bool, error) {
		result, err := SSH(undropCmd, from, testContext.Provider)
		if result.Code == 0 && err == nil {
			return true, nil
		}
		LogSSHResult(result)
		if err != nil {
			Logf("Unexpected error: %v", err)
		}
		return false, nil
	})
	if err != nil {
		Failf("Failed to remove the iptable REJECT rule. Manual intervention is "+
			"required on host %s: remove rule %s, if exists", from, iptablesRule)
	}
}
