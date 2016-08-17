/*
Copyright 2014 The Kubernetes Authors.

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
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	goRuntime "runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/controller"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/runtime"
	sshutil "k8s.io/kubernetes/pkg/ssh"
	"k8s.io/kubernetes/pkg/types"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/system"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/util/wait"
	utilyaml "k8s.io/kubernetes/pkg/util/yaml"
	"k8s.io/kubernetes/pkg/version"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/blang/semver"
	"golang.org/x/crypto/ssh"
	"golang.org/x/net/websocket"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	gomegatypes "github.com/onsi/gomega/types"
)

const (
	// How long to wait for the pod to be listable
	PodListTimeout = time.Minute
	// Initial pod start can be delayed O(minutes) by slow docker pulls
	// TODO: Make this 30 seconds once #4566 is resolved.
	PodStartTimeout = 5 * time.Minute

	// How long to wait for the pod to no longer be running
	podNoLongerRunningTimeout = 30 * time.Second

	// If there are any orphaned namespaces to clean up, this test is running
	// on a long lived cluster. A long wait here is preferably to spurious test
	// failures caused by leaked resources from a previous test run.
	NamespaceCleanupTimeout = 15 * time.Minute

	// Some pods can take much longer to get ready due to volume attach/detach latency.
	slowPodStartTimeout = 15 * time.Minute

	// How long to wait for a service endpoint to be resolvable.
	ServiceStartTimeout = 1 * time.Minute

	// String used to mark pod deletion
	nonExist = "NonExist"

	// How often to Poll pods, nodes and claims.
	Poll = 2 * time.Second

	// service accounts are provisioned after namespace creation
	// a service account is required to support pod creation in a namespace as part of admission control
	ServiceAccountProvisionTimeout = 2 * time.Minute

	// How long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	// TODO: client should not apply this timeout to Watch calls. Increased from 30s until that is fixed.
	SingleCallTimeout = 5 * time.Minute

	// How long nodes have to be "ready" when a test begins. They should already
	// be "ready" before the test starts, so this is small.
	NodeReadyInitialTimeout = 20 * time.Second

	// How long pods have to be "ready" when a test begins.
	PodReadyBeforeTimeout = 5 * time.Minute

	// How long pods have to become scheduled onto nodes
	podScheduledBeforeTimeout = PodListTimeout + (20 * time.Second)

	podRespondingTimeout     = 2 * time.Minute
	ServiceRespondingTimeout = 2 * time.Minute
	EndpointRegisterTimeout  = time.Minute

	// How long claims have to become dynamically provisioned
	ClaimProvisionTimeout = 5 * time.Minute

	// When these values are updated, also update cmd/kubelet/app/options/options.go
	currentPodInfraContainerImageName    = "gcr.io/google_containers/pause"
	currentPodInfraContainerImageVersion = "3.0"

	// How long each node is given during a process that restarts all nodes
	// before the test is considered failed. (Note that the total time to
	// restart all nodes will be this number times the number of nodes.)
	RestartPerNodeTimeout = 5 * time.Minute

	// How often to Poll the statues of a restart.
	RestartPoll = 20 * time.Second

	// How long a node is allowed to become "Ready" after it is restarted before
	// the test is considered failed.
	RestartNodeReadyAgainTimeout = 5 * time.Minute

	// How long a pod is allowed to become "running" and "ready" after a node
	// restart before test is considered failed.
	RestartPodReadyAgainTimeout = 5 * time.Minute

	// Number of times we want to retry Updates in case of conflict
	UpdateRetries = 5
)

var (
	// Label allocated to the image puller static pod that runs on each node
	// before e2es.
	ImagePullerLabels = map[string]string{"name": "e2e-image-puller"}

	// For parsing Kubectl version for version-skewed testing.
	gitVersionRegexp = regexp.MustCompile("GitVersion:\"(v.+?)\"")
)

// GetServerArchitecture fetches the architecture of the cluster's apiserver.
func GetServerArchitecture(c *client.Client) string {
	arch := ""
	sVer, err := c.Discovery().ServerVersion()
	if err != nil || sVer.Platform == "" {
		// If we failed to get the server version for some reason, default to amd64.
		arch = "amd64"
	} else {
		// Split the platform string into OS and Arch separately.
		// The platform string may for example be "linux/amd64", "linux/arm" or "windows/amd64".
		osArchArray := strings.Split(sVer.Platform, "/")
		arch = osArchArray[1]
	}
	return arch
}

// GetPauseImageName fetches the pause image name for the same architecture as the apiserver.
func GetPauseImageName(c *client.Client) string {
	return currentPodInfraContainerImageName + "-" + GetServerArchitecture(c) + ":" + currentPodInfraContainerImageVersion
}

// GetPauseImageNameForHostArch fetches the pause image name for the same architecture the test is running on.
func GetPauseImageNameForHostArch() string {
	return currentPodInfraContainerImageName + "-" + goRuntime.GOARCH + ":" + currentPodInfraContainerImageVersion
}

// SubResource proxy should have been functional in v1.0.0, but SubResource
// proxy via tunneling is known to be broken in v1.0.  See
// https://github.com/kubernetes/kubernetes/pull/15224#issuecomment-146769463
//
// TODO(ihmccreery): remove once we don't care about v1.0 anymore, (tentatively
// in v1.3).
var SubResourcePodProxyVersion = version.MustParse("v1.1.0")
var subResourceServiceAndNodeProxyVersion = version.MustParse("v1.2.0")

func GetServicesProxyRequest(c *client.Client, request *restclient.Request) (*restclient.Request, error) {
	subResourceProxyAvailable, err := ServerVersionGTE(subResourceServiceAndNodeProxyVersion, c)
	if err != nil {
		return nil, err
	}
	if subResourceProxyAvailable {
		return request.Resource("services").SubResource("proxy"), nil
	}
	return request.Prefix("proxy").Resource("services"), nil
}

// unique identifier of the e2e run
var RunId = uuid.NewUUID()

type CreateTestingNSFn func(baseName string, c *client.Client, labels map[string]string) (*api.Namespace, error)

type ContainerFailures struct {
	status   *api.ContainerStateTerminated
	Restarts int
}

func GetMasterHost() string {
	masterUrl, err := url.Parse(TestContext.Host)
	ExpectNoError(err)
	return masterUrl.Host
}

// Convenient wrapper around cache.Store that returns list of api.Pod instead of interface{}.
type PodStore struct {
	cache.Store
	stopCh    chan struct{}
	reflector *cache.Reflector
}

func NewPodStore(c *client.Client, namespace string, label labels.Selector, field fields.Selector) *PodStore {
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
	reflector := cache.NewReflector(lw, &api.Pod{}, store, 0)
	reflector.RunUntil(stopCh)
	return &PodStore{store, stopCh, reflector}
}

func (s *PodStore) List() []*api.Pod {
	objects := s.Store.List()
	pods := make([]*api.Pod, 0)
	for _, o := range objects {
		pods = append(pods, o.(*api.Pod))
	}
	return pods
}

func (s *PodStore) Stop() {
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
	DNSPolicy      *api.DNSPolicy

	// Env vars, set the same for every pod.
	Env map[string]string

	// Extra labels added to every pod.
	Labels map[string]string

	// Node selector for pods in the RC.
	NodeSelector map[string]string

	// Ports to declare in the container (map of name to containerPort).
	Ports map[string]int
	// Ports to declare in the container as host and container ports.
	HostPorts map[string]int

	Volumes      []api.Volume
	VolumeMounts []api.VolumeMount

	// Pointer to a list of pods; if non-nil, will be set to a list of pods
	// created by this RC by RunRC.
	CreatedPods *[]*api.Pod

	// Maximum allowable container failures. If exceeded, RunRC returns an error.
	// Defaults to replicas*0.1 if unspecified.
	MaxContainerFailures *int

	// If set to false starting RC will print progress, otherwise only errors will be printed.
	Silent bool
}

type DeploymentConfig struct {
	RCConfig
}

type ReplicaSetConfig struct {
	RCConfig
}

func nowStamp() string {
	return time.Now().Format(time.StampMilli)
}

func log(level string, format string, args ...interface{}) {
	fmt.Fprintf(GinkgoWriter, nowStamp()+": "+level+": "+format+"\n", args...)
}

func Logf(format string, args ...interface{}) {
	log("INFO", format, args...)
}

func Failf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	log("INFO", msg)
	Fail(nowStamp()+": "+msg, 1)
}

func Skipf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	log("INFO", msg)
	Skip(nowStamp() + ": " + msg)
}

func SkipUnlessNodeCountIsAtLeast(minNodeCount int) {
	if TestContext.CloudConfig.NumNodes < minNodeCount {
		Skipf("Requires at least %d nodes (not %d)", minNodeCount, TestContext.CloudConfig.NumNodes)
	}
}

func SkipUnlessAtLeast(value int, minValue int, message string) {
	if value < minValue {
		Skipf(message)
	}
}

func SkipIfProviderIs(unsupportedProviders ...string) {
	if ProviderIs(unsupportedProviders...) {
		Skipf("Not supported for providers %v (found %s)", unsupportedProviders, TestContext.Provider)
	}
}

func SkipUnlessProviderIs(supportedProviders ...string) {
	if !ProviderIs(supportedProviders...) {
		Skipf("Only supported for providers %v (not %s)", supportedProviders, TestContext.Provider)
	}
}

func SkipIfContainerRuntimeIs(runtimes ...string) {
	for _, runtime := range runtimes {
		if runtime == TestContext.ContainerRuntime {
			Skipf("Not supported under container runtime %s", runtime)
		}
	}
}

func ProviderIs(providers ...string) bool {
	for _, provider := range providers {
		if strings.ToLower(provider) == strings.ToLower(TestContext.Provider) {
			return true
		}
	}
	return false
}

func SkipUnlessServerVersionGTE(v semver.Version, c discovery.ServerVersionInterface) {
	gte, err := ServerVersionGTE(v, c)
	if err != nil {
		Failf("Failed to get server version: %v", err)
	}
	if !gte {
		Skipf("Not supported for server versions before %q", v)
	}
}

// Detects whether the federation namespace exists in the underlying cluster
func SkipUnlessFederated(c *client.Client) {
	federationNS := os.Getenv("FEDERATION_NAMESPACE")
	if federationNS == "" {
		federationNS = "federation"
	}

	_, err := c.Namespaces().Get(federationNS)
	if err != nil {
		if apierrs.IsNotFound(err) {
			Skipf("Could not find federation namespace %s: skipping federated test", federationNS)
		} else {
			Failf("Unexpected error getting namespace: %v", err)
		}
	}
}

// ProvidersWithSSH are those providers where each node is accessible with SSH
var ProvidersWithSSH = []string{"gce", "gke", "aws"}

// providersWithMasterSSH are those providers where master node is accessible with SSH
var providersWithMasterSSH = []string{"gce", "gke", "kubemark", "aws"}

type podCondition func(pod *api.Pod) (bool, error)

// podReady returns whether pod has a condition of Ready with a status of true.
// TODO: should be replaced with api.IsPodReady
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

// PodRunningReady checks whether pod p's phase is running and it has a ready
// condition of status true.
func PodRunningReady(p *api.Pod) (bool, error) {
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

func PodRunningReadyOrSucceeded(p *api.Pod) (bool, error) {
	// Check if the phase is succeeded.
	if p.Status.Phase == api.PodSucceeded {
		return true, nil
	}
	return PodRunningReady(p)
}

// PodNotReady checks whether pod p's has a ready condition of status false.
func PodNotReady(p *api.Pod) (bool, error) {
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

// WaitForPodsSuccess waits till all labels matching the given selector enter
// the Success state. The caller is expected to only invoke this method once the
// pods have been created.
func WaitForPodsSuccess(c *client.Client, ns string, successPodLabels map[string]string, timeout time.Duration) error {
	successPodSelector := labels.SelectorFromSet(successPodLabels)
	start, badPods := time.Now(), []api.Pod{}

	if wait.PollImmediate(30*time.Second, timeout, func() (bool, error) {
		podList, err := c.Pods(ns).List(api.ListOptions{LabelSelector: successPodSelector})
		if err != nil {
			Logf("Error getting pods in namespace %q: %v", ns, err)
			return false, nil
		}
		if len(podList.Items) == 0 {
			Logf("Waiting for pods to enter Success, but no pods in %q match label %v", ns, successPodLabels)
			return true, nil
		}
		badPods = []api.Pod{}
		for _, pod := range podList.Items {
			if pod.Status.Phase != api.PodSucceeded {
				badPods = append(badPods, pod)
			}
		}
		successPods := len(podList.Items) - len(badPods)
		Logf("%d / %d pods in namespace %q are in Success state (%d seconds elapsed)",
			successPods, len(podList.Items), ns, int(time.Since(start).Seconds()))
		if len(badPods) == 0 {
			return true, nil
		}
		return false, nil
	}) != nil {
		logPodStates(badPods)
		LogPodsWithLabels(c, ns, successPodLabels)
		return fmt.Errorf("Not all pods in namespace %q are successful within %v", ns, timeout)
	}
	return nil
}

// WaitForPodsRunningReady waits up to timeout to ensure that all pods in
// namespace ns are either running and ready, or failed but controlled by a
// replication controller. Also, it ensures that at least minPods are running
// and ready. It has separate behavior from other 'wait for' pods functions in
// that it requires the list of pods on every iteration. This is useful, for
// example, in cluster startup, because the number of pods increases while
// waiting.
// If ignoreLabels is not empty, pods matching this selector are ignored and
// this function waits for minPods to enter Running/Ready and for all pods
// matching ignoreLabels to enter Success phase. Otherwise an error is returned
// even if there are minPods pods, some of which are in Running/Ready
// and some in Success. This is to allow the client to decide if "Success"
// means "Ready" or not.
func WaitForPodsRunningReady(c *client.Client, ns string, minPods int32, timeout time.Duration, ignoreLabels map[string]string) error {
	ignoreSelector := labels.SelectorFromSet(ignoreLabels)
	start := time.Now()
	Logf("Waiting up to %v for all pods (need at least %d) in namespace '%s' to be running and ready",
		timeout, minPods, ns)
	wg := sync.WaitGroup{}
	wg.Add(1)
	var waitForSuccessError error
	go func() {
		waitForSuccessError = WaitForPodsSuccess(c, ns, ignoreLabels, timeout)
		wg.Done()
	}()

	if wait.PollImmediate(Poll, timeout, func() (bool, error) {
		// We get the new list of pods and replication controllers in every
		// iteration because more pods come online during startup and we want to
		// ensure they are also checked.
		rcList, err := c.ReplicationControllers(ns).List(api.ListOptions{})
		if err != nil {
			Logf("Error getting replication controllers in namespace '%s': %v", ns, err)
			return false, nil
		}
		replicas := int32(0)
		for _, rc := range rcList.Items {
			replicas += rc.Spec.Replicas
		}

		podList, err := c.Pods(ns).List(api.ListOptions{})
		if err != nil {
			Logf("Error getting pods in namespace '%s': %v", ns, err)
			return false, nil
		}
		nOk, replicaOk, badPods := int32(0), int32(0), []api.Pod{}
		for _, pod := range podList.Items {
			if len(ignoreLabels) != 0 && ignoreSelector.Matches(labels.Set(pod.Labels)) {
				Logf("%v in state %v, ignoring", pod.Name, pod.Status.Phase)
				continue
			}
			if res, err := PodRunningReady(&pod); res && err == nil {
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
	wg.Wait()
	if waitForSuccessError != nil {
		return waitForSuccessError
	}
	return nil
}

func podFromManifest(filename string) (*api.Pod, error) {
	var pod api.Pod
	Logf("Parsing pod from %v", filename)
	data := ReadOrDie(filename)
	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, err
	}
	if err := runtime.DecodeInto(api.Codecs.UniversalDecoder(), json, &pod); err != nil {
		return nil, err
	}
	return &pod, nil
}

// Run a test container to try and contact the Kubernetes api-server from a pod, wait for it
// to flip to Ready, log its output and delete it.
func RunKubernetesServiceTestContainer(c *client.Client, ns string) {
	path := "test/images/clusterapi-tester/pod.yaml"
	p, err := podFromManifest(path)
	if err != nil {
		Logf("Failed to parse clusterapi-tester from manifest %v: %v", path, err)
		return
	}
	p.Namespace = ns
	if _, err := c.Pods(ns).Create(p); err != nil {
		Logf("Failed to create %v: %v", p.Name, err)
		return
	}
	defer func() {
		if err := c.Pods(ns).Delete(p.Name, nil); err != nil {
			Logf("Failed to delete pod %v: %v", p.Name, err)
		}
	}()
	timeout := 5 * time.Minute
	if err := waitForPodCondition(c, ns, p.Name, "clusterapi-tester", timeout, PodRunningReady); err != nil {
		Logf("Pod %v took longer than %v to enter running/ready: %v", p.Name, timeout, err)
		return
	}
	logs, err := GetPodLogs(c, ns, p.Name, p.Spec.Containers[0].Name)
	if err != nil {
		Logf("Failed to retrieve logs from %v: %v", p.Name, err)
	} else {
		Logf("Output of clusterapi-tester:\n%v", logs)
	}
}

func kubectlLogPod(c *client.Client, pod api.Pod, containerNameSubstr string) {
	for _, container := range pod.Spec.Containers {
		if strings.Contains(container.Name, containerNameSubstr) {
			// Contains() matches all strings if substr is empty
			logs, err := GetPodLogs(c, pod.Namespace, pod.Name, container.Name)
			if err != nil {
				logs, err = getPreviousPodLogs(c, pod.Namespace, pod.Name, container.Name)
				if err != nil {
					Logf("Failed to get logs of pod %v, container %v, err: %v", pod.Name, container.Name, err)
				}
			}
			By(fmt.Sprintf("Logs of %v/%v:%v on node %v", pod.Namespace, pod.Name, container.Name, pod.Spec.NodeName))
			Logf("%s : STARTLOG\n%s\nENDLOG for container %v:%v:%v", containerNameSubstr, logs, pod.Namespace, pod.Name, container.Name)
		}
	}
}

func LogFailedContainers(c *client.Client, ns string) {
	podList, err := c.Pods(ns).List(api.ListOptions{})
	if err != nil {
		Logf("Error getting pods in namespace '%s': %v", ns, err)
		return
	}
	Logf("Running kubectl logs on non-ready containers in %v", ns)
	for _, pod := range podList.Items {
		if res, err := PodRunningReady(&pod); !res || err != nil {
			kubectlLogPod(c, pod, "")
		}
	}
}

func LogPodsWithLabels(c *client.Client, ns string, match map[string]string) {
	podList, err := c.Pods(ns).List(api.ListOptions{LabelSelector: labels.SelectorFromSet(match)})
	if err != nil {
		Logf("Error getting pods in namespace %q: %v", ns, err)
		return
	}
	Logf("Running kubectl logs on pods with labels %v in %v", match, ns)
	for _, pod := range podList.Items {
		kubectlLogPod(c, pod, "")
	}
}

func LogContainersInPodsWithLabels(c *client.Client, ns string, match map[string]string, containerSubstr string) {
	podList, err := c.Pods(ns).List(api.ListOptions{LabelSelector: labels.SelectorFromSet(match)})
	if err != nil {
		Logf("Error getting pods in namespace %q: %v", ns, err)
		return
	}
	for _, pod := range podList.Items {
		kubectlLogPod(c, pod, containerSubstr)
	}
}

// DeleteNamespaces deletes all namespaces that match the given delete and skip filters.
// Filter is by simple strings.Contains; first skip filter, then delete filter.
// Returns the list of deleted namespaces or an error.
func DeleteNamespaces(c *client.Client, deleteFilter, skipFilter []string) ([]string, error) {
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

func WaitForNamespacesDeleted(c *client.Client, namespaces []string, timeout time.Duration) error {
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
	w, err := c.ServiceAccounts(ns).Watch(api.SingleObject(api.ObjectMeta{Name: serviceAccountName}))
	if err != nil {
		return err
	}
	_, err = watch.Until(timeout, w, client.ServiceAccountHasSecrets)
	return err
}

func waitForPodCondition(c *client.Client, ns, podName, desc string, timeout time.Duration, condition podCondition) error {
	Logf("Waiting up to %[1]v for pod %[2]s status to be %[3]s", timeout, podName, desc)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pod, err := c.Pods(ns).Get(podName)
		if err != nil {
			if apierrs.IsNotFound(err) {
				Logf("Pod %q in namespace %q disappeared. Error: %v", podName, ns, err)
				return err
			}
			// Aligning this text makes it much more readable
			Logf("Get pod %[1]s in namespace '%[2]s' failed, ignoring for %[3]v. Error: %[4]v",
				podName, ns, Poll, err)
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

// WaitForMatchPodsCondition finds match pods based on the input ListOptions.
// waits and checks if all match pods are in the given podCondition
func WaitForMatchPodsCondition(c *client.Client, opts api.ListOptions, desc string, timeout time.Duration, condition podCondition) error {
	Logf("Waiting up to %v for matching pods' status to be %s", timeout, desc)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
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

// WaitForDefaultServiceAccountInNamespace waits for the default service account to be provisioned
// the default service account is what is associated with pods when they do not specify a service account
// as a result, pods are not able to be provisioned in a namespace until the service account is provisioned
func WaitForDefaultServiceAccountInNamespace(c *client.Client, namespace string) error {
	return waitForServiceAccountInNamespace(c, namespace, "default", ServiceAccountProvisionTimeout)
}

// WaitForFederationApiserverReady waits for the federation apiserver to be ready.
// It tests the readiness by sending a GET request and expecting a non error response.
func WaitForFederationApiserverReady(c *federation_release_1_4.Clientset) error {
	return wait.PollImmediate(time.Second, 1*time.Minute, func() (bool, error) {
		_, err := c.Federation().Clusters().List(api.ListOptions{})
		if err != nil {
			return false, nil
		}
		return true, nil
	})
}

// WaitForPersistentVolumePhase waits for a PersistentVolume to be in a specific phase or until timeout occurs, whichever comes first.
func WaitForPersistentVolumePhase(phase api.PersistentVolumePhase, c *client.Client, pvName string, Poll, timeout time.Duration) error {
	Logf("Waiting up to %v for PersistentVolume %s to have phase %s", timeout, pvName, phase)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pv, err := c.PersistentVolumes().Get(pvName)
		if err != nil {
			Logf("Get persistent volume %s in failed, ignoring for %v: %v", pvName, Poll, err)
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

// WaitForPersistentVolumeDeleted waits for a PersistentVolume to get deleted or until timeout occurs, whichever comes first.
func WaitForPersistentVolumeDeleted(c *client.Client, pvName string, Poll, timeout time.Duration) error {
	Logf("Waiting up to %v for PersistentVolume %s to get deleted", timeout, pvName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pv, err := c.PersistentVolumes().Get(pvName)
		if err == nil {
			Logf("PersistentVolume %s found and phase=%s (%v)", pvName, pv.Status.Phase, time.Since(start))
			continue
		} else {
			if apierrs.IsNotFound(err) {
				Logf("PersistentVolume %s was removed", pvName)
				return nil
			} else {
				Logf("Get persistent volume %s in failed, ignoring for %v: %v", pvName, Poll, err)
			}
		}
	}
	return fmt.Errorf("PersistentVolume %s still exists within %v", pvName, timeout)
}

// WaitForPersistentVolumeClaimPhase waits for a PersistentVolumeClaim to be in a specific phase or until timeout occurs, whichever comes first.
func WaitForPersistentVolumeClaimPhase(phase api.PersistentVolumeClaimPhase, c *client.Client, ns string, pvcName string, Poll, timeout time.Duration) error {
	Logf("Waiting up to %v for PersistentVolumeClaim %s to have phase %s", timeout, pvcName, phase)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pvc, err := c.PersistentVolumeClaims(ns).Get(pvcName)
		if err != nil {
			Logf("Get persistent volume claim %s in failed, ignoring for %v: %v", pvcName, Poll, err)
			continue
		} else {
			if pvc.Status.Phase == phase {
				Logf("PersistentVolumeClaim %s found and phase=%s (%v)", pvcName, phase, time.Since(start))
				return nil
			} else {
				Logf("PersistentVolumeClaim %s found but phase is %s instead of %s.", pvcName, pvc.Status.Phase, phase)
			}
		}
	}
	return fmt.Errorf("PersistentVolumeClaim %s not in phase %s within %v", pvcName, phase, timeout)
}

// CreateTestingNS should be used by every test, note that we append a common prefix to the provided test name.
// Please see NewFramework instead of using this directly.
func CreateTestingNS(baseName string, c *client.Client, labels map[string]string) (*api.Namespace, error) {
	if labels == nil {
		labels = map[string]string{}
	}
	labels["e2e-run"] = string(RunId)

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
	if err := wait.PollImmediate(Poll, SingleCallTimeout, func() (bool, error) {
		var err error
		got, err = c.Namespaces().Create(namespaceObj)
		if err != nil {
			Logf("Unexpected error while creating namespace: %v", err)
			return false, nil
		}
		return true, nil
	}); err != nil {
		return nil, err
	}

	if TestContext.VerifyServiceAccount {
		if err := WaitForDefaultServiceAccountInNamespace(c, got.Name); err != nil {
			return nil, err
		}
	}
	return got, nil
}

// CheckTestingNSDeletedExcept checks whether all e2e based existing namespaces are in the Terminating state
// and waits until they are finally deleted. It ignores namespace skip.
func CheckTestingNSDeletedExcept(c *client.Client, skip string) error {
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
	remainingPods := []api.Pod{}
	missingTimestamp := false
	if pods, perr := c.Pods(namespace).List(api.ListOptions{}); perr == nil {
		for _, pod := range pods.Items {
			Logf("Pod %s %s on node %s remains, has deletion timestamp %s", namespace, pod.Name, pod.Spec.NodeName, pod.DeletionTimestamp)
			remaining = append(remaining, fmt.Sprintf("%s{Reason=%s}", pod.Name, pod.Status.Reason))
			remainingPods = append(remainingPods, pod)
			if pod.DeletionTimestamp == nil {
				missingTimestamp = true
			}
		}
	}

	// log pod status
	if len(remainingPods) > 0 {
		logPodStates(remainingPods)
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

func ContainerInitInvariant(older, newer runtime.Object) error {
	oldPod := older.(*api.Pod)
	newPod := newer.(*api.Pod)
	if len(oldPod.Spec.InitContainers) == 0 {
		return nil
	}
	if len(oldPod.Spec.InitContainers) != len(newPod.Spec.InitContainers) {
		return fmt.Errorf("init container list changed")
	}
	if oldPod.UID != newPod.UID {
		return fmt.Errorf("two different pods exist in the condition: %s vs %s", oldPod.UID, newPod.UID)
	}
	if err := initContainersInvariants(oldPod); err != nil {
		return err
	}
	if err := initContainersInvariants(newPod); err != nil {
		return err
	}
	oldInit, _, _ := podInitialized(oldPod)
	newInit, _, _ := podInitialized(newPod)
	if oldInit && !newInit {
		// TODO: we may in the future enable resetting PodInitialized = false if the kubelet needs to restart it
		// from scratch
		return fmt.Errorf("pod cannot be initialized and then regress to not being initialized")
	}
	return nil
}

func podInitialized(pod *api.Pod) (ok bool, failed bool, err error) {
	allInit := true
	initFailed := false
	for _, s := range pod.Status.InitContainerStatuses {
		switch {
		case initFailed && s.State.Waiting == nil:
			return allInit, initFailed, fmt.Errorf("container %s is after a failed container but isn't waiting", s.Name)
		case allInit && s.State.Waiting == nil:
			return allInit, initFailed, fmt.Errorf("container %s is after an initializing container but isn't waiting", s.Name)
		case s.State.Terminated == nil:
			allInit = false
		case s.State.Terminated.ExitCode != 0:
			allInit = false
			initFailed = true
		case !s.Ready:
			return allInit, initFailed, fmt.Errorf("container %s initialized but isn't marked as ready", s.Name)
		}
	}
	return allInit, initFailed, nil
}

func initContainersInvariants(pod *api.Pod) error {
	allInit, initFailed, err := podInitialized(pod)
	if err != nil {
		return err
	}
	if !allInit || initFailed {
		for _, s := range pod.Status.ContainerStatuses {
			if s.State.Waiting == nil || s.RestartCount != 0 {
				return fmt.Errorf("container %s is not waiting but initialization not complete", s.Name)
			}
			if s.State.Waiting.Reason != "PodInitializing" {
				return fmt.Errorf("container %s should have reason PodInitializing: %s", s.Name, s.State.Waiting.Reason)
			}
		}
	}
	_, c := api.GetPodCondition(&pod.Status, api.PodInitialized)
	if c == nil {
		return fmt.Errorf("pod does not have initialized condition")
	}
	if c.LastTransitionTime.IsZero() {
		return fmt.Errorf("PodInitialized condition should always have a transition time")
	}
	switch {
	case c.Status == api.ConditionUnknown:
		return fmt.Errorf("PodInitialized condition should never be Unknown")
	case c.Status == api.ConditionTrue && (initFailed || !allInit):
		return fmt.Errorf("PodInitialized condition was True but all not all containers initialized")
	case c.Status == api.ConditionFalse && (!initFailed && allInit):
		return fmt.Errorf("PodInitialized condition was False but all containers initialized")
	}
	return nil
}

type InvariantFunc func(older, newer runtime.Object) error

func CheckInvariants(events []watch.Event, fns ...InvariantFunc) error {
	errs := sets.NewString()
	for i := range events {
		j := i + 1
		if j >= len(events) {
			continue
		}
		for _, fn := range fns {
			if err := fn(events[i].Object, events[j].Object); err != nil {
				errs.Insert(err.Error())
			}
		}
	}
	if errs.Len() > 0 {
		return fmt.Errorf("invariants violated:\n* %s", strings.Join(errs.List(), "\n* "))
	}
	return nil
}

// Waits default amount of time (PodStartTimeout) for the specified pod to become running.
// Returns an error if timeout occurs first, or pod goes in to failed state.
func WaitForPodRunningInNamespace(c *client.Client, pod *api.Pod) error {
	// this short-cicuit is needed for cases when we pass a list of pods instead
	// of newly created pod (eg. VerifyPods) which means we are getting already
	// running pod for which waiting does not make sense and will always fail
	if pod.Status.Phase == api.PodRunning {
		return nil
	}
	return waitTimeoutForPodRunningInNamespace(c, pod.Name, pod.Namespace, pod.ResourceVersion, PodStartTimeout)
}

// Waits default amount of time (PodStartTimeout) for the specified pod to become running.
// Returns an error if timeout occurs first, or pod goes in to failed state.
func WaitForPodNameRunningInNamespace(c *client.Client, podName, namespace string) error {
	return waitTimeoutForPodRunningInNamespace(c, podName, namespace, "", PodStartTimeout)
}

// Waits an extended amount of time (slowPodStartTimeout) for the specified pod to become running.
// The resourceVersion is used when Watching object changes, it tells since when we care
// about changes to the pod. Returns an error if timeout occurs first, or pod goes in to failed state.
func waitForPodRunningInNamespaceSlow(c *client.Client, podName, namespace, resourceVersion string) error {
	return waitTimeoutForPodRunningInNamespace(c, podName, namespace, resourceVersion, slowPodStartTimeout)
}

func waitTimeoutForPodRunningInNamespace(c *client.Client, podName, namespace, resourceVersion string, timeout time.Duration) error {
	w, err := c.Pods(namespace).Watch(api.SingleObject(api.ObjectMeta{Name: podName, ResourceVersion: resourceVersion}))
	if err != nil {
		return err
	}
	_, err = watch.Until(timeout, w, client.PodRunning)
	return err
}

// Waits default amount of time (podNoLongerRunningTimeout) for the specified pod to stop running.
// Returns an error if timeout occurs first.
func WaitForPodNoLongerRunningInNamespace(c *client.Client, podName, namespace, resourceVersion string) error {
	return waitTimeoutForPodNoLongerRunningInNamespace(c, podName, namespace, resourceVersion, podNoLongerRunningTimeout)
}

func waitTimeoutForPodNoLongerRunningInNamespace(c *client.Client, podName, namespace, resourceVersion string, timeout time.Duration) error {
	w, err := c.Pods(namespace).Watch(api.SingleObject(api.ObjectMeta{Name: podName, ResourceVersion: resourceVersion}))
	if err != nil {
		return err
	}
	_, err = watch.Until(timeout, w, client.PodCompleted)
	return err
}

func waitTimeoutForPodReadyInNamespace(c *client.Client, podName, namespace, resourceVersion string, timeout time.Duration) error {
	w, err := c.Pods(namespace).Watch(api.SingleObject(api.ObjectMeta{Name: podName, ResourceVersion: resourceVersion}))
	if err != nil {
		return err
	}
	_, err = watch.Until(timeout, w, client.PodRunningAndReady)
	return err
}

// WaitForPodNotPending returns an error if it took too long for the pod to go out of pending state.
// The resourceVersion is used when Watching object changes, it tells since when we care
// about changes to the pod.
func WaitForPodNotPending(c *client.Client, ns, podName, resourceVersion string) error {
	w, err := c.Pods(ns).Watch(api.SingleObject(api.ObjectMeta{Name: podName, ResourceVersion: resourceVersion}))
	if err != nil {
		return err
	}
	_, err = watch.Until(PodStartTimeout, w, client.PodNotPending)
	return err
}

// waitForPodTerminatedInNamespace returns an error if it took too long for the pod
// to terminate or if the pod terminated with an unexpected reason.
func waitForPodTerminatedInNamespace(c *client.Client, podName, reason, namespace string) error {
	return waitForPodCondition(c, namespace, podName, "terminated due to deadline exceeded", PodStartTimeout, func(pod *api.Pod) (bool, error) {
		if pod.Status.Phase == api.PodFailed {
			if pod.Status.Reason == reason {
				return true, nil
			} else {
				return true, fmt.Errorf("Expected pod %v in namespace %v to be terminated with reason %v, got reason: %v", podName, namespace, reason, pod.Status.Reason)
			}
		}

		return false, nil
	})
}

// waitForPodSuccessInNamespaceTimeout returns nil if the pod reached state success, or an error if it reached failure or ran too long.
func waitForPodSuccessInNamespaceTimeout(c *client.Client, podName string, contName string, namespace string, timeout time.Duration) error {
	return waitForPodCondition(c, namespace, podName, "success or failure", timeout, func(pod *api.Pod) (bool, error) {
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

// WaitForPodSuccessInNamespace returns nil if the pod reached state success, or an error if it reached failure or until podStartupTimeout.
func WaitForPodSuccessInNamespace(c *client.Client, podName string, contName string, namespace string) error {
	return waitForPodSuccessInNamespaceTimeout(c, podName, contName, namespace, PodStartTimeout)
}

// WaitForPodSuccessInNamespaceSlow returns nil if the pod reached state success, or an error if it reached failure or until slowPodStartupTimeout.
func WaitForPodSuccessInNamespaceSlow(c *client.Client, podName string, contName string, namespace string) error {
	return waitForPodSuccessInNamespaceTimeout(c, podName, contName, namespace, slowPodStartTimeout)
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

// WaitForRCToStabilize waits till the RC has a matching generation/replica count between spec and status.
func WaitForRCToStabilize(c *client.Client, ns, name string, timeout time.Duration) error {
	options := api.ListOptions{FieldSelector: fields.Set{
		"metadata.name":      name,
		"metadata.namespace": ns,
	}.AsSelector()}
	w, err := c.ReplicationControllers(ns).Watch(options)
	if err != nil {
		return err
	}
	_, err = watch.Until(timeout, w, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Deleted:
			return false, apierrs.NewNotFound(unversioned.GroupResource{Resource: "replicationcontrollers"}, "")
		}
		switch rc := event.Object.(type) {
		case *api.ReplicationController:
			if rc.Name == name && rc.Namespace == ns &&
				rc.Generation <= rc.Status.ObservedGeneration &&
				rc.Spec.Replicas == rc.Status.Replicas {
				return true, nil
			}
			Logf("Waiting for rc %s to stabilize, generation %v observed generation %v spec.replicas %d status.replicas %d",
				name, rc.Generation, rc.Status.ObservedGeneration, rc.Spec.Replicas, rc.Status.Replicas)
		}
		return false, nil
	})
	return err
}

func WaitForPodToDisappear(c *client.Client, ns, podName string, label labels.Selector, interval, timeout time.Duration) error {
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

// WaitForRCPodToDisappear returns nil if the pod from the given replication controller (described by rcName) no longer exists.
// In case of failure or too long waiting time, an error is returned.
func WaitForRCPodToDisappear(c *client.Client, ns, rcName, podName string) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": rcName}))
	// NodeController evicts pod after 5 minutes, so we need timeout greater than that.
	// Additionally, there can be non-zero grace period, so we are setting 10 minutes
	// to be on the safe size.
	return WaitForPodToDisappear(c, ns, podName, label, 20*time.Second, 10*time.Minute)
}

// WaitForService waits until the service appears (exist == true), or disappears (exist == false)
func WaitForService(c *client.Client, namespace, name string, exist bool, interval, timeout time.Duration) error {
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

//WaitForServiceEndpointsNum waits until the amount of endpoints that implement service to expectNum.
func WaitForServiceEndpointsNum(c *client.Client, namespace, serviceName string, expectNum int, interval, timeout time.Duration) error {
	return wait.Poll(interval, timeout, func() (bool, error) {
		Logf("Waiting for amount of service:%s endpoints to be %d", serviceName, expectNum)
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

// WaitForReplicationController waits until the RC appears (exist == true), or disappears (exist == false)
func WaitForReplicationController(c *client.Client, namespace, name string, exist bool, interval, timeout time.Duration) error {
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

func WaitForEndpoint(c *client.Client, ns, name string) error {
	for t := time.Now(); time.Since(t) < EndpointRegisterTimeout; time.Sleep(Poll) {
		endpoint, err := c.Endpoints(ns).Get(name)
		Expect(err).NotTo(HaveOccurred())
		if len(endpoint.Subsets) == 0 || len(endpoint.Subsets[0].Addresses) == 0 {
			Logf("Endpoint %s/%s is not ready yet", ns, name)
			continue
		} else {
			return nil
		}
	}
	return fmt.Errorf("Failed to get endpoints for %s/%s", ns, name)
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

func PodProxyResponseChecker(c *client.Client, ns string, label labels.Selector, controllerName string, respondName bool, pods *api.PodList) podProxyResponseChecker {
	return podProxyResponseChecker{c, ns, label, controllerName, respondName, pods}
}

// CheckAllResponses issues GETs to all pods in the context and verify they
// reply with their own pod name.
func (r podProxyResponseChecker) CheckAllResponses() (done bool, err error) {
	successes := 0
	options := api.ListOptions{LabelSelector: r.label}
	currentPods, err := r.c.Pods(r.ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	for i, pod := range r.pods.Items {
		// Check that the replica list remains unchanged, otherwise we have problems.
		if !isElementOf(pod.UID, currentPods) {
			return false, fmt.Errorf("pod with UID %s is no longer a member of the replica set.  Must have been restarted for some reason.  Current replica set: %v", pod.UID, currentPods)
		}
		subResourceProxyAvailable, err := ServerVersionGTE(SubResourcePodProxyVersion, r.c)
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

// ServerVersionGTE returns true if v is greater than or equal to the server
// version.
//
// TODO(18726): This should be incorporated into client.VersionInterface.
func ServerVersionGTE(v semver.Version, c discovery.ServerVersionInterface) (bool, error) {
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

// KubectlVersionGTE returns true if the kubectl version is greater than or
// equal to v.
func KubectlVersionGTE(v semver.Version) (bool, error) {
	kv, err := KubectlVersion()
	if err != nil {
		return false, err
	}
	return kv.GTE(v), nil
}

// KubectlVersion gets the version of kubectl that's currently being used (see
// --kubectl-path in e2e.go to use an alternate kubectl).
func KubectlVersion() (semver.Version, error) {
	output := RunKubectlOrDie("version", "--client")
	matches := gitVersionRegexp.FindStringSubmatch(output)
	if len(matches) != 2 {
		return semver.Version{}, fmt.Errorf("Could not find kubectl version in output %v", output)
	}
	// Don't use the full match, as it contains "GitVersion:\"" and a
	// trailing "\"".  Just use the submatch.
	return version.Parse(matches[1])
}

func PodsResponding(c *client.Client, ns, name string, wantName bool, pods *api.PodList) error {
	By("trying to dial each unique pod")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	return wait.PollImmediate(Poll, podRespondingTimeout, PodProxyResponseChecker(c, ns, label, name, wantName, pods).CheckAllResponses)
}

func PodsCreated(c *client.Client, ns, name string, replicas int32) (*api.PodList, error) {
	timeout := 2 * time.Minute
	// List the pods, making sure we observe all the replicas.
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		options := api.ListOptions{LabelSelector: label}
		pods, err := c.Pods(ns).List(options)
		if err != nil {
			return nil, err
		}

		created := []api.Pod{}
		for _, pod := range pods.Items {
			if pod.DeletionTimestamp != nil {
				continue
			}
			created = append(created, pod)
		}
		Logf("Pod name %s: Found %d pods out of %d", name, len(created), replicas)

		if int32(len(created)) == replicas {
			pods.Items = created
			return pods, nil
		}
	}
	return nil, fmt.Errorf("Pod name %s: Gave up waiting %v for %d pods to come up", name, timeout, replicas)
}

func podsRunning(c *client.Client, pods *api.PodList) []error {
	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	By("ensuring each pod is running")
	e := []error{}
	error_chan := make(chan error)

	for _, pod := range pods.Items {
		go func(p api.Pod) {
			error_chan <- WaitForPodRunningInNamespace(c, &p)
		}(pod)
	}

	for range pods.Items {
		err := <-error_chan
		if err != nil {
			e = append(e, err)
		}
	}

	return e
}

func VerifyPods(c *client.Client, ns, name string, wantName bool, replicas int32) error {
	pods, err := PodsCreated(c, ns, name, replicas)
	if err != nil {
		return err
	}
	e := podsRunning(c, pods)
	if len(e) > 0 {
		return fmt.Errorf("failed to wait for pods running: %v", e)
	}
	err = PodsResponding(c, ns, name, wantName, pods)
	if err != nil {
		return fmt.Errorf("failed to wait for pods responding: %v", err)
	}
	return nil
}

func ServiceResponding(c *client.Client, ns, name string) error {
	By(fmt.Sprintf("trying to dial the service %s.%s via the proxy", ns, name))

	return wait.PollImmediate(Poll, ServiceRespondingTimeout, func() (done bool, err error) {
		proxyRequest, errProxy := GetServicesProxyRequest(c, c.Get())
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

func restclientConfig(kubeContext string) (*clientcmdapi.Config, error) {
	Logf(">>> kubeConfig: %s\n", TestContext.KubeConfig)
	if TestContext.KubeConfig == "" {
		return nil, fmt.Errorf("KubeConfig must be specified to load client config")
	}
	c, err := clientcmd.LoadFromFile(TestContext.KubeConfig)
	if err != nil {
		return nil, fmt.Errorf("error loading KubeConfig: %v", err.Error())
	}
	if kubeContext != "" {
		Logf(">>> kubeContext: %s\n", kubeContext)
		c.CurrentContext = kubeContext
	}
	return c, nil
}

type ClientConfigGetter func() (*restclient.Config, error)

func LoadConfig() (*restclient.Config, error) {
	if TestContext.NodeName != "" {
		// This is a node e2e test, apply the node e2e configuration
		return &restclient.Config{Host: TestContext.Host}, nil
	}
	c, err := restclientConfig(TestContext.KubeContext)
	if err != nil {
		return nil, err
	}

	return clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: TestContext.Host}}).ClientConfig()
}

func LoadFederatedConfig() (*restclient.Config, error) {
	c, err := restclientConfig(federatedKubeContext)
	if err != nil {
		return nil, fmt.Errorf("error creating federation client config: %v", err.Error())
	}
	cfg, err := clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{}).ClientConfig()
	if cfg != nil {
		//TODO(colhom): this is only here because https://github.com/kubernetes/kubernetes/issues/25422
		cfg.NegotiatedSerializer = api.Codecs
	}
	if err != nil {
		return cfg, fmt.Errorf("error creating federation client config: %v", err.Error())
	}
	return cfg, nil
}

func loadClientFromConfig(config *restclient.Config) (*client.Client, error) {
	c, err := client.New(config)
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	if c.Client.Timeout == 0 {
		c.Client.Timeout = SingleCallTimeout
	}
	return c, nil
}

func setTimeouts(cs ...*http.Client) {
	for _, client := range cs {
		if client.Timeout == 0 {
			client.Timeout = SingleCallTimeout
		}
	}
}

func LoadFederationClientset_1_4() (*federation_release_1_4.Clientset, error) {
	config, err := LoadFederatedConfig()
	if err != nil {
		return nil, err
	}

	c, err := federation_release_1_4.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("error creating federation clientset: %v", err.Error())
	}
	// Set timeout for each client in the set.
	setTimeouts(c.DiscoveryClient.Client, c.FederationClient.Client, c.CoreClient.Client)
	return c, nil
}

func LoadClient() (*client.Client, error) {
	config, err := LoadConfig()
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

func ExpectNoError(err error, explain ...interface{}) {
	if err != nil {
		Logf("Unexpected error occurred: %v", err)
	}
	ExpectWithOffset(1, err).NotTo(HaveOccurred(), explain...)
}

// Stops everything from filePath from namespace ns and checks if everything matching selectors from the given namespace is correctly stopped.
func Cleanup(filePath, ns string, selectors ...string) {
	By("using delete to clean up resources")
	var nsArg string
	if ns != "" {
		nsArg = fmt.Sprintf("--namespace=%s", ns)
	}
	RunKubectlOrDie("delete", "--grace-period=0", "-f", filePath, nsArg)
	AssertCleanup(ns, selectors...)
}

// Asserts that cleanup of a namespace wrt selectors occurred.
func AssertCleanup(ns string, selectors ...string) {
	var nsArg string
	if ns != "" {
		nsArg = fmt.Sprintf("--namespace=%s", ns)
	}
	for _, selector := range selectors {
		resources := RunKubectlOrDie("get", "rc,svc", "-l", selector, "--no-headers", nsArg)
		if resources != "" {
			Failf("Resources left running after stop:\n%s", resources)
		}
		pods := RunKubectlOrDie("get", "pods", "-l", selector, nsArg, "-o", "go-template={{ range .items }}{{ if not .metadata.deletionTimestamp }}{{ .metadata.name }}{{ \"\\n\" }}{{ end }}{{ end }}")
		if pods != "" {
			Failf("Pods left unterminated after stop:\n%s", pods)
		}
	}
}

// validatorFn is the function which is individual tests will implement.
// we may want it to return more than just an error, at some point.
type validatorFn func(c *client.Client, podID string) error

// ValidateController is a generic mechanism for testing RC's that are running.
// It takes a container name, a test name, and a validator function which is plugged in by a specific test.
// "containername": this is grepped for.
// "containerImage" : this is the name of the image we expect to be launched.  Not to confuse w/ images (kitten.jpg)  which are validated.
// "testname":  which gets bubbled up to the logging/failure messages if errors happen.
// "validator" function: This function is given a podID and a client, and it can do some specific validations that way.
func ValidateController(c *client.Client, containerImage string, replicas int, containername string, testname string, validator validatorFn, ns string) {
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
	for start := time.Now(); time.Since(start) < PodStartTimeout; time.Sleep(5 * time.Second) {
		getPodsOutput := RunKubectlOrDie("get", "pods", "-o", "template", getPodsTemplate, "-l", testname, fmt.Sprintf("--namespace=%v", ns))
		pods := strings.Fields(getPodsOutput)
		if numPods := len(pods); numPods != replicas {
			By(fmt.Sprintf("Replicas for %s: expected=%d actual=%d", testname, replicas, numPods))
			continue
		}
		var runningPods []string
		for _, podID := range pods {
			running := RunKubectlOrDie("get", "pods", podID, "-o", "template", getContainerStateTemplate, fmt.Sprintf("--namespace=%v", ns))
			if running != "true" {
				Logf("%s is created but not running", podID)
				continue waitLoop
			}

			currentImage := RunKubectlOrDie("get", "pods", podID, "-o", "template", getImageTemplate, fmt.Sprintf("--namespace=%v", ns))
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
	Failf("Timed out after %v seconds waiting for %s pods to reach valid state", PodStartTimeout.Seconds(), testname)
}

// KubectlCmd runs the kubectl executable through the wrapper script.
func KubectlCmd(args ...string) *exec.Cmd {
	defaultArgs := []string{}

	// Reference a --server option so tests can run anywhere.
	if TestContext.Host != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.FlagAPIServer+"="+TestContext.Host)
	}
	if TestContext.KubeConfig != "" {
		defaultArgs = append(defaultArgs, "--"+clientcmd.RecommendedConfigPathFlag+"="+TestContext.KubeConfig)

		// Reference the KubeContext
		if TestContext.KubeContext != "" {
			defaultArgs = append(defaultArgs, "--"+clientcmd.FlagContext+"="+TestContext.KubeContext)
		}

	} else {
		if TestContext.CertDir != "" {
			defaultArgs = append(defaultArgs,
				fmt.Sprintf("--certificate-authority=%s", filepath.Join(TestContext.CertDir, "ca.crt")),
				fmt.Sprintf("--client-certificate=%s", filepath.Join(TestContext.CertDir, "kubecfg.crt")),
				fmt.Sprintf("--client-key=%s", filepath.Join(TestContext.CertDir, "kubecfg.key")))
		}
	}
	kubectlArgs := append(defaultArgs, args...)

	//We allow users to specify path to kubectl, so you can test either "kubectl" or "cluster/kubectl.sh"
	//and so on.
	cmd := exec.Command(TestContext.KubectlPath, kubectlArgs...)

	//caller will invoke this and wait on it.
	return cmd
}

// kubectlBuilder is used to build, customize and execute a kubectl Command.
// Add more functions to customize the builder as needed.
type kubectlBuilder struct {
	cmd     *exec.Cmd
	timeout <-chan time.Time
}

func NewKubectlCommand(args ...string) *kubectlBuilder {
	b := new(kubectlBuilder)
	b.cmd = KubectlCmd(args...)
	return b
}

func (b *kubectlBuilder) WithEnv(env []string) *kubectlBuilder {
	b.cmd.Env = env
	return b
}

func (b *kubectlBuilder) WithTimeout(t <-chan time.Time) *kubectlBuilder {
	b.timeout = t
	return b
}

func (b kubectlBuilder) WithStdinData(data string) *kubectlBuilder {
	b.cmd.Stdin = strings.NewReader(data)
	return &b
}

func (b kubectlBuilder) WithStdinReader(reader io.Reader) *kubectlBuilder {
	b.cmd.Stdin = reader
	return &b
}

func (b kubectlBuilder) ExecOrDie() string {
	str, err := b.Exec()
	Logf("stdout: %q", str)
	// In case of i/o timeout error, try talking to the apiserver again after 2s before dying.
	// Note that we're still dying after retrying so that we can get visibility to triage it further.
	if isTimeout(err) {
		Logf("Hit i/o timeout error, talking to the server 2s later to see if it's temporary.")
		time.Sleep(2 * time.Second)
		retryStr, retryErr := RunKubectl("version")
		Logf("stdout: %q", retryStr)
		Logf("err: %v", retryErr)
	}
	Expect(err).NotTo(HaveOccurred())
	return str
}

func isTimeout(err error) bool {
	switch err := err.(type) {
	case net.Error:
		if err.Timeout() {
			return true
		}
	case *url.Error:
		if err, ok := err.Err.(net.Error); ok && err.Timeout() {
			return true
		}
	}
	return false
}

func (b kubectlBuilder) Exec() (string, error) {
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
	Logf("stderr: %q", stderr.String())
	return stdout.String(), nil
}

// RunKubectlOrDie is a convenience wrapper over kubectlBuilder
func RunKubectlOrDie(args ...string) string {
	return NewKubectlCommand(args...).ExecOrDie()
}

// RunKubectl is a convenience wrapper over kubectlBuilder
func RunKubectl(args ...string) (string, error) {
	return NewKubectlCommand(args...).Exec()
}

// RunKubectlOrDieInput is a convenience wrapper over kubectlBuilder that takes input to stdin
func RunKubectlOrDieInput(data string, args ...string) string {
	return NewKubectlCommand(args...).WithStdinData(data).ExecOrDie()
}

func StartCmdAndStreamOutput(cmd *exec.Cmd) (stdout, stderr io.ReadCloser, err error) {
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
func TryKill(cmd *exec.Cmd) {
	if err := cmd.Process.Kill(); err != nil {
		Logf("ERROR failed to kill command %v! The process may leak", cmd)
	}
}

// testContainerOutputMatcher runs the given pod in the given namespace and waits
// for all of the containers in the podSpec to move into the 'Success' status, and tests
// the specified container log against the given expected output using the given matcher.
func (f *Framework) testContainerOutputMatcher(scenarioName string,
	pod *api.Pod,
	containerIndex int,
	expectedOutput []string,
	matcher func(string, ...interface{}) gomegatypes.GomegaMatcher) {
	By(fmt.Sprintf("Creating a pod to test %v", scenarioName))
	podClient := f.PodClient()
	ns := f.Namespace.Name

	defer podClient.Delete(pod.Name, api.NewDeleteOptions(0))
	podClient.Create(pod)

	// Wait for client pod to complete.
	var containerName string
	for id, container := range pod.Spec.Containers {
		ExpectNoError(WaitForPodSuccessInNamespace(f.Client, pod.Name, container.Name, ns))
		if id == containerIndex {
			containerName = container.Name
		}
	}
	if containerName == "" {
		Failf("Invalid container index: %d", containerIndex)
	}

	// Grab its logs.  Get host first.
	podStatus, err := podClient.Get(pod.Name)
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
		logs, err = GetPodLogs(f.Client, ns, pod.Name, containerName)
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
// namespace lifecycle for handling Cleanup).
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
			Replicas: int32(config.Replicas),
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

// RunReplicaSet launches (and verifies correctness) of a ReplicaSet
// and waits until all the pods it launches to reach the "Running" state.
// It's the caller's responsibility to clean up externally (i.e. use the
// namespace lifecycle for handling Cleanup).
func RunReplicaSet(config ReplicaSetConfig) error {
	err := config.create()
	if err != nil {
		return err
	}
	return config.start()
}

func (config *ReplicaSetConfig) create() error {
	By(fmt.Sprintf("creating replicaset %s in namespace %s", config.Name, config.Namespace))
	rs := &extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name: config.Name,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: int32(config.Replicas),
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

	config.applyTo(&rs.Spec.Template)

	_, err := config.Client.ReplicaSets(config.Namespace).Create(rs)
	if err != nil {
		return fmt.Errorf("Error creating replica set: %v", err)
	}
	Logf("Created replica set with name: %v, namespace: %v, replica count: %v", rs.Name, config.Namespace, rs.Spec.Replicas)
	return nil
}

// RunRC Launches (and verifies correctness) of a Replication Controller
// and will wait for all pods it spawns to become "Running".
// It's the caller's responsibility to clean up externally (i.e. use the
// namespace lifecycle for handling Cleanup).
func RunRC(config RCConfig) error {
	err := config.create()
	if err != nil {
		return err
	}
	return config.start()
}

func (config *RCConfig) create() error {
	By(fmt.Sprintf("creating replication controller %s in namespace %s", config.Name, config.Namespace))
	dnsDefault := api.DNSDefault
	if config.DNSPolicy == nil {
		config.DNSPolicy = &dnsDefault
	}
	rc := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: config.Name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: int32(config.Replicas),
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
					DNSPolicy:    *config.DNSPolicy,
					NodeSelector: config.NodeSelector,
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
	if config.NodeSelector != nil {
		template.Spec.NodeSelector = make(map[string]string)
		for k, v := range config.NodeSelector {
			template.Spec.NodeSelector[k] = v
		}
	}
	if config.Ports != nil {
		for k, v := range config.Ports {
			c := &template.Spec.Containers[0]
			c.Ports = append(c.Ports, api.ContainerPort{Name: k, ContainerPort: int32(v)})
		}
	}
	if config.HostPorts != nil {
		for k, v := range config.HostPorts {
			c := &template.Spec.Containers[0]
			c.Ports = append(c.Ports, api.ContainerPort{Name: k, ContainerPort: int32(v), HostPort: int32(v)})
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
	if len(config.Volumes) > 0 {
		template.Spec.Volumes = config.Volumes
	}
	if len(config.VolumeMounts) > 0 {
		template.Spec.Containers[0].VolumeMounts = config.VolumeMounts
	}
}

type RCStartupStatus struct {
	Expected              int
	Terminating           int
	Running               int
	RunningButNotReady    int
	Waiting               int
	Pending               int
	Unknown               int
	Inactive              int
	FailedContainers      int
	Created               []*api.Pod
	ContainerRestartNodes sets.String
}

func (s *RCStartupStatus) Print(name string) {
	Logf("%v Pods: %d out of %d created, %d running, %d pending, %d waiting, %d inactive, %d terminating, %d unknown, %d runningButNotReady ",
		name, len(s.Created), s.Expected, s.Running, s.Pending, s.Waiting, s.Inactive, s.Terminating, s.Unknown, s.RunningButNotReady)
}

func ComputeRCStartupStatus(pods []*api.Pod, expected int) RCStartupStatus {
	startupStatus := RCStartupStatus{
		Expected:              expected,
		Created:               make([]*api.Pod, 0, expected),
		ContainerRestartNodes: sets.NewString(),
	}
	for _, p := range pods {
		if p.DeletionTimestamp != nil {
			startupStatus.Terminating++
			continue
		}
		startupStatus.Created = append(startupStatus.Created, p)
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
				startupStatus.Running++
			} else {
				startupStatus.RunningButNotReady++
			}
			for _, v := range FailedContainers(p) {
				startupStatus.FailedContainers = startupStatus.FailedContainers + v.Restarts
				startupStatus.ContainerRestartNodes.Insert(p.Spec.NodeName)
			}
		} else if p.Status.Phase == api.PodPending {
			if p.Spec.NodeName == "" {
				startupStatus.Waiting++
			} else {
				startupStatus.Pending++
			}
		} else if p.Status.Phase == api.PodSucceeded || p.Status.Phase == api.PodFailed {
			startupStatus.Inactive++
		} else if p.Status.Phase == api.PodUnknown {
			startupStatus.Unknown++
		}
	}
	return startupStatus
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

	PodStore := NewPodStore(config.Client, config.Namespace, label, fields.Everything())
	defer PodStore.Stop()

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

		pods := PodStore.List()
		startupStatus := ComputeRCStartupStatus(pods, config.Replicas)

		pods = startupStatus.Created
		if config.CreatedPods != nil {
			*config.CreatedPods = pods
		}
		if !config.Silent {
			startupStatus.Print(config.Name)
		}

		promPushRunningPending(startupStatus.Running, startupStatus.Pending)

		if config.PodStatusFile != nil {
			fmt.Fprintf(config.PodStatusFile, "%d, running, %d, pending, %d, waiting, %d, inactive, %d, unknown, %d, runningButNotReady\n", startupStatus.Running, startupStatus.Pending, startupStatus.Waiting, startupStatus.Inactive, startupStatus.Unknown, startupStatus.RunningButNotReady)
		}

		if startupStatus.FailedContainers > maxContainerFailures {
			DumpNodeDebugInfo(config.Client, startupStatus.ContainerRestartNodes.List())
			// Get the logs from the failed containers to help diagnose what caused them to fail
			LogFailedContainers(config.Client, config.Namespace)
			return fmt.Errorf("%d containers failed which is more than allowed %d", startupStatus.FailedContainers, maxContainerFailures)
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

		if len(pods) > len(oldPods) || startupStatus.Running > oldRunning {
			lastChange = time.Now()
		}
		oldPods = pods
		oldRunning = startupStatus.Running

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

// Simplified version of RunRC, that does not create RC, but creates plain Pods.
// Optionally waits for pods to start running (if waitForRunning == true).
// The number of replicas must be non-zero.
func StartPods(c *client.Client, replicas int, namespace string, podNamePrefix string, pod api.Pod, waitForRunning bool) {
	// no pod to start
	if replicas < 1 {
		panic("StartPods: number of replicas must be non-zero")
	}
	startPodsID := string(uuid.NewUUID()) // So that we can label and find them
	for i := 0; i < replicas; i++ {
		podName := fmt.Sprintf("%v-%v", podNamePrefix, i)
		pod.ObjectMeta.Name = podName
		pod.ObjectMeta.Labels["name"] = podName
		pod.ObjectMeta.Labels["startPodsID"] = startPodsID
		pod.Spec.Containers[0].Name = podName
		_, err := c.Pods(namespace).Create(&pod)
		ExpectNoError(err)
	}
	Logf("Waiting for running...")
	if waitForRunning {
		label := labels.SelectorFromSet(labels.Set(map[string]string{"startPodsID": startPodsID}))
		err := WaitForPodsWithLabelRunning(c, namespace, label)
		ExpectNoError(err, "Error waiting for %d pods to be running - probably a timeout", replicas)
	}
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
	DumpNodeDebugInfo(c, badNodes.List())
}

func DumpAllNamespaceInfo(c *client.Client, namespace string) {
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
	// Note that we don't wait for any Cleanup to propagate, which means
	// that if you delete a bunch of pods right before ending your test,
	// you may or may not see the killing/deletion/Cleanup events.

	// If cluster is large, then the following logs are basically useless, because:
	// 1. it takes tens of minutes or hours to grab all of them
	// 2. there are so many of them that working with them are mostly impossible
	// So we dump them only if the cluster is relatively small.
	maxNodesForDump := 20
	if nodes, err := c.Nodes().List(api.ListOptions{}); err == nil {
		if len(nodes.Items) <= maxNodesForDump {
			dumpAllPodInfo(c)
			dumpAllNodeInfo(c)
		} else {
			Logf("skipping dumping cluster info - cluster too large")
		}
	} else {
		Logf("unable to fetch node list: %v", err)
	}
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
	DumpNodeDebugInfo(c, names)
}

func DumpNodeDebugInfo(c *client.Client, nodeNames []string) {
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
			Logf("%v started at %v (%d+%d container statuses recorded)", p.Name, p.Status.StartTime, len(p.Status.InitContainerStatuses), len(p.Status.ContainerStatuses))
			for _, c := range p.Status.InitContainerStatuses {
				Logf("\tInit container %v ready: %v, restart count %v",
					c.Name, c.Ready, c.RestartCount)
			}
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

// waitListSchedulableNodesOrDie is a wrapper around listing nodes supporting retries.
func waitListSchedulableNodesOrDie(c *client.Client) *api.NodeList {
	var nodes *api.NodeList
	var err error
	if wait.PollImmediate(Poll, SingleCallTimeout, func() (bool, error) {
		nodes, err = c.Nodes().List(api.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector()})
		return err == nil, nil
	}) != nil {
		ExpectNoError(err, "Timed out while listing nodes for e2e cluster.")
	}
	return nodes
}

// Node is schedulable if:
// 1) doesn't have "unschedulable" field set
// 2) it's Ready condition is set to true
// 3) doesn't have NetworkUnavailable condition set to true
func isNodeSchedulable(node *api.Node) bool {
	nodeReady := IsNodeConditionSetAsExpected(node, api.NodeReady, true)
	networkReady := IsNodeConditionUnset(node, api.NodeNetworkUnavailable) ||
		IsNodeConditionSetAsExpectedSilent(node, api.NodeNetworkUnavailable, false)
	return !node.Spec.Unschedulable && nodeReady && networkReady
}

// GetReadySchedulableNodesOrDie addresses the common use case of getting nodes you can do work on.
// 1) Needs to be schedulable.
// 2) Needs to be ready.
// If EITHER 1 or 2 is not true, most tests will want to ignore the node entirely.
func GetReadySchedulableNodesOrDie(c *client.Client) (nodes *api.NodeList) {
	nodes = waitListSchedulableNodesOrDie(c)
	// previous tests may have cause failures of some nodes. Let's skip
	// 'Not Ready' nodes, just in case (there is no need to fail the test).
	FilterNodes(nodes, func(node api.Node) bool {
		return isNodeSchedulable(&node)
	})
	return nodes
}

func WaitForAllNodesSchedulable(c *client.Client) error {
	return wait.PollImmediate(30*time.Second, 4*time.Hour, func() (bool, error) {
		opts := api.ListOptions{
			ResourceVersion: "0",
			FieldSelector:   fields.Set{"spec.unschedulable": "false"}.AsSelector(),
		}
		nodes, err := c.Nodes().List(opts)
		if err != nil {
			Logf("Unexpected error listing nodes: %v", err)
			// Ignore the error here - it will be retried.
			return false, nil
		}
		schedulable := 0
		for _, node := range nodes.Items {
			if isNodeSchedulable(&node) {
				schedulable++
			}
		}
		if schedulable != len(nodes.Items) {
			Logf("%d/%d nodes schedulable (polling after 30s)", schedulable, len(nodes.Items))
			return false, nil
		}
		return true, nil
	})
}

func AddOrUpdateLabelOnNode(c *client.Client, nodeName string, labelKey string, labelValue string) {
	patch := fmt.Sprintf(`{"metadata":{"labels":{"%s":"%s"}}}`, labelKey, labelValue)
	var err error
	for attempt := 0; attempt < UpdateRetries; attempt++ {
		err = c.Patch(api.MergePatchType).Resource("nodes").Name(nodeName).Body([]byte(patch)).Do().Error()
		if err != nil {
			if !apierrs.IsConflict(err) {
				ExpectNoError(err)
			} else {
				Logf("Conflict when trying to add a label %v:%v to %v", labelKey, labelValue, nodeName)
			}
		} else {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	ExpectNoError(err)
}

func ExpectNodeHasLabel(c *client.Client, nodeName string, labelKey string, labelValue string) {
	By("verifying the node has the label " + labelKey + " " + labelValue)
	node, err := c.Nodes().Get(nodeName)
	ExpectNoError(err)
	Expect(node.Labels[labelKey]).To(Equal(labelValue))
}

// RemoveLabelOffNode is for cleaning up labels temporarily added to node,
// won't fail if target label doesn't exist or has been removed.
func RemoveLabelOffNode(c *client.Client, nodeName string, labelKey string) {
	By("removing the label " + labelKey + " off the node " + nodeName)
	var nodeUpdated *api.Node
	var node *api.Node
	var err error
	for attempt := 0; attempt < UpdateRetries; attempt++ {
		node, err = c.Nodes().Get(nodeName)
		ExpectNoError(err)
		if node.Labels == nil || len(node.Labels[labelKey]) == 0 {
			return
		}
		delete(node.Labels, labelKey)
		nodeUpdated, err = c.Nodes().Update(node)
		if err != nil {
			if !apierrs.IsConflict(err) {
				ExpectNoError(err)
			} else {
				Logf("Conflict when trying to remove a label %v from %v", labelKey, nodeName)
			}
		} else {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	ExpectNoError(err)

	By("verifying the node doesn't have the label " + labelKey)
	if nodeUpdated.Labels != nil && len(nodeUpdated.Labels[labelKey]) != 0 {
		Failf("Failed removing label " + labelKey + " of the node " + nodeName)
	}
}

func AddOrUpdateTaintOnNode(c *client.Client, nodeName string, taint api.Taint) {
	node, err := c.Nodes().Get(nodeName)
	ExpectNoError(err)

	nodeTaints, err := api.GetTaintsFromNodeAnnotations(node.Annotations)
	ExpectNoError(err)

	var newTaints []api.Taint
	updated := false
	for _, existingTaint := range nodeTaints {
		if existingTaint.Key == taint.Key {
			newTaints = append(newTaints, taint)
			updated = true
			continue
		}

		newTaints = append(newTaints, existingTaint)
	}

	if !updated {
		newTaints = append(newTaints, taint)
	}

	taintsData, err := json.Marshal(newTaints)
	ExpectNoError(err)

	if node.Annotations == nil {
		node.Annotations = make(map[string]string)
	}
	node.Annotations[api.TaintsAnnotationKey] = string(taintsData)
	_, err = c.Nodes().Update(node)
	ExpectNoError(err)
}

func taintExists(taints []api.Taint, taintKey string) bool {
	for _, taint := range taints {
		if taint.Key == taintKey {
			return true
		}
	}
	return false
}

func ExpectNodeHasTaint(c *client.Client, nodeName string, taintKey string) {
	By("verifying the node has the taint " + taintKey)
	node, err := c.Nodes().Get(nodeName)
	ExpectNoError(err)

	nodeTaints, err := api.GetTaintsFromNodeAnnotations(node.Annotations)
	ExpectNoError(err)

	if len(nodeTaints) == 0 || !taintExists(nodeTaints, taintKey) {
		Failf("Failed to find taint %s on node %s", taintKey, nodeName)
	}
}

func deleteTaintByKey(taints []api.Taint, taintKey string) ([]api.Taint, error) {
	newTaints := []api.Taint{}
	found := false
	for _, taint := range taints {
		if taint.Key == taintKey {
			found = true
			continue
		}
		newTaints = append(newTaints, taint)
	}

	if !found {
		return nil, fmt.Errorf("taint key=\"%s\" not found.", taintKey)
	}
	return newTaints, nil
}

// RemoveTaintOffNode is for cleaning up taints temporarily added to node,
// won't fail if target taint doesn't exist or has been removed.
func RemoveTaintOffNode(c *client.Client, nodeName string, taintKey string) {
	By("removing the taint " + taintKey + " off the node " + nodeName)
	node, err := c.Nodes().Get(nodeName)
	ExpectNoError(err)

	nodeTaints, err := api.GetTaintsFromNodeAnnotations(node.Annotations)
	ExpectNoError(err)
	if len(nodeTaints) == 0 {
		return
	}

	if !taintExists(nodeTaints, taintKey) {
		return
	}

	newTaints, err := deleteTaintByKey(nodeTaints, taintKey)
	ExpectNoError(err)

	taintsData, err := json.Marshal(newTaints)
	ExpectNoError(err)
	node.Annotations[api.TaintsAnnotationKey] = string(taintsData)
	nodeUpdated, err := c.Nodes().Update(node)
	ExpectNoError(err)

	By("verifying the node doesn't have the taint " + taintKey)
	taintsGot, err := api.GetTaintsFromNodeAnnotations(nodeUpdated.Annotations)
	ExpectNoError(err)
	if taintExists(taintsGot, taintKey) {
		Failf("Failed removing taint " + taintKey + " of the node " + nodeName)
	}
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
	return WaitForRCPodsRunning(c, ns, name)
}

// Wait up to 10 minutes for pods to become Running. Assume that the pods of the
// rc are labels with {"name":rcName}.
func WaitForRCPodsRunning(c *client.Client, ns, rcName string) error {
	selector := labels.SelectorFromSet(labels.Set(map[string]string{"name": rcName}))
	err := WaitForPodsWithLabelRunning(c, ns, selector)
	if err != nil {
		return fmt.Errorf("Error while waiting for replication controller %s pods to be running: %v", rcName, err)
	}
	return nil
}

// Wait up to 10 minutes for all matching pods to become Running and at least one
// matching pod exists.
func WaitForPodsWithLabelRunning(c *client.Client, ns string, label labels.Selector) error {
	running := false
	PodStore := NewPodStore(c, ns, label, fields.Everything())
	defer PodStore.Stop()
waitLoop:
	for start := time.Now(); time.Since(start) < 10*time.Minute; time.Sleep(5 * time.Second) {
		pods := PodStore.List()
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

// Returns true if all the specified pods are scheduled, else returns false.
func podsWithLabelScheduled(c *client.Client, ns string, label labels.Selector) (bool, error) {
	PodStore := NewPodStore(c, ns, label, fields.Everything())
	defer PodStore.Stop()
	pods := PodStore.List()
	if len(pods) == 0 {
		return false, nil
	}
	for _, pod := range pods {
		if pod.Spec.NodeName == "" {
			return false, nil
		}
	}
	return true, nil
}

// Wait for all matching pods to become scheduled and at least one
// matching pod exists.  Return the list of matching pods.
func WaitForPodsWithLabelScheduled(c *client.Client, ns string, label labels.Selector) (pods *api.PodList, err error) {
	err = wait.PollImmediate(Poll, podScheduledBeforeTimeout,
		func() (bool, error) {
			pods, err = WaitForPodsWithLabel(c, ns, label)
			if err != nil {
				return false, err
			}
			for _, pod := range pods.Items {
				if pod.Spec.NodeName == "" {
					return false, nil
				}
			}
			return true, nil
		})
	return pods, err
}

// Wait up to PodListTimeout for getting pods with certain label
func WaitForPodsWithLabel(c *client.Client, ns string, label labels.Selector) (pods *api.PodList, err error) {
	for t := time.Now(); time.Since(t) < PodListTimeout; time.Sleep(Poll) {
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

// DeleteRCAndPods a Replication Controller and all pods it spawned
func DeleteRCAndPods(c *client.Client, ns, name string) error {
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
	ps, err := podStoreForRC(c, rc)
	if err != nil {
		return err
	}
	defer ps.Stop()
	startTime := time.Now()
	err = reaper.Stop(ns, name, 0, api.NewDeleteOptions(0))
	if apierrs.IsNotFound(err) {
		Logf("RC %s was already deleted: %v", name, err)
		return nil
	}
	deleteRCTime := time.Now().Sub(startTime)
	Logf("Deleting RC %s took: %v", name, deleteRCTime)
	if err != nil {
		return fmt.Errorf("error while stopping RC: %s: %v", name, err)
	}
	err = waitForPodsInactive(ps, 10*time.Millisecond, 10*time.Minute)
	if err != nil {
		return fmt.Errorf("error while waiting for pods to become inactive %s: %v", name, err)
	}
	terminatePodTime := time.Now().Sub(startTime) - deleteRCTime
	Logf("Terminating RC %s pods took: %v", name, terminatePodTime)
	// this is to relieve namespace controller's pressure when deleting the
	// namespace after a test.
	err = waitForPodsGone(ps, 10*time.Second, 10*time.Minute)
	if err != nil {
		return fmt.Errorf("error while waiting for pods gone %s: %v", name, err)
	}
	return nil
}

// DeleteRCAndWaitForGC deletes only the Replication Controller and waits for GC to delete the pods.
func DeleteRCAndWaitForGC(c *client.Client, ns, name string) error {
	By(fmt.Sprintf("deleting replication controller %s in namespace %s, will wait for the garbage collector to delete the pods", name, ns))
	rc, err := c.ReplicationControllers(ns).Get(name)
	if err != nil {
		if apierrs.IsNotFound(err) {
			Logf("RC %s was already deleted: %v", name, err)
			return nil
		}
		return err
	}
	ps, err := podStoreForRC(c, rc)
	if err != nil {
		return err
	}
	defer ps.Stop()
	startTime := time.Now()
	falseVar := false
	deleteOption := &api.DeleteOptions{OrphanDependents: &falseVar}
	err = c.ReplicationControllers(ns).Delete(name, deleteOption)
	if err != nil && apierrs.IsNotFound(err) {
		Logf("RC %s was already deleted: %v", name, err)
		return nil
	}
	if err != nil {
		return err
	}
	deleteRCTime := time.Now().Sub(startTime)
	Logf("Deleting RC %s took: %v", name, deleteRCTime)
	err = waitForPodsInactive(ps, 10*time.Millisecond, 10*time.Minute)
	if err != nil {
		return fmt.Errorf("error while waiting for pods to become inactive %s: %v", name, err)
	}
	terminatePodTime := time.Now().Sub(startTime) - deleteRCTime
	Logf("Terminating RC %s pods took: %v", name, terminatePodTime)
	err = waitForPodsGone(ps, 10*time.Second, 10*time.Minute)
	if err != nil {
		return fmt.Errorf("error while waiting for pods gone %s: %v", name, err)
	}
	return nil
}

// podStoreForRC creates a PodStore that monitors pods belong to the rc. It
// waits until the reflector does a List() before returning.
func podStoreForRC(c *client.Client, rc *api.ReplicationController) (*PodStore, error) {
	labels := labels.SelectorFromSet(rc.Spec.Selector)
	ps := NewPodStore(c, rc.Namespace, labels, fields.Everything())
	err := wait.Poll(1*time.Second, 1*time.Minute, func() (bool, error) {
		if len(ps.reflector.LastSyncResourceVersion()) != 0 {
			return true, nil
		}
		return false, nil
	})
	return ps, err
}

// waitForPodsInactive waits until there are no active pods left in the PodStore.
// This is to make a fair comparison of deletion time between DeleteRCAndPods
// and DeleteRCAndWaitForGC, because the RC controller decreases status.replicas
// when the pod is inactvie.
func waitForPodsInactive(ps *PodStore, interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		pods := ps.List()
		for _, pod := range pods {
			if controller.IsPodActive(*pod) {
				return false, nil
			}
		}
		return true, nil
	})
}

// waitForPodsGone waits until there are no pods left in the PodStore.
func waitForPodsGone(ps *PodStore, interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		if pods := ps.List(); len(pods) == 0 {
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
	return wait.PollImmediate(Poll, 2*time.Minute, func() (bool, error) {
		selector, err := unversioned.LabelSelectorAsSelector(rs.Spec.Selector)
		ExpectNoError(err)
		options := api.ListOptions{LabelSelector: selector}
		if pods, err := c.Pods(rs.Namespace).List(options); err == nil && len(pods.Items) == 0 {
			return true, nil
		}
		return false, nil
	})
}

// Waits for the deployment status to become valid (i.e. max unavailable and max surge aren't violated anymore).
// Note that the status should stay valid at all times unless shortly after a scaling event or the deployment is just created.
// To verify that the deployment status is valid and wait for the rollout to finish, use WaitForDeploymentStatus instead.
func WaitForDeploymentStatusValid(c clientset.Interface, d *extensions.Deployment) error {
	var (
		oldRSs, allOldRSs, allRSs []*extensions.ReplicaSet
		newRS                     *extensions.ReplicaSet
		deployment                *extensions.Deployment
		reason                    string
	)

	err := wait.Poll(Poll, 2*time.Minute, func() (bool, error) {
		var err error
		deployment, err = c.Extensions().Deployments(d.Namespace).Get(d.Name)
		if err != nil {
			return false, err
		}
		oldRSs, allOldRSs, newRS, err = deploymentutil.GetAllReplicaSets(deployment, c)
		if err != nil {
			return false, err
		}
		if newRS == nil {
			// New RC hasn't been created yet.
			reason = "new replica set hasn't been created yet"
			Logf(reason)
			return false, nil
		}
		allRSs = append(oldRSs, newRS)
		// The old/new ReplicaSets need to contain the pod-template-hash label
		for i := range allRSs {
			if !labelsutil.SelectorHasLabel(allRSs[i].Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey) {
				reason = "all replica sets need to contain the pod-template-hash label"
				Logf(reason)
				return false, nil
			}
		}
		totalCreated := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
		totalAvailable, err := deploymentutil.GetAvailablePodsForDeployment(c, deployment)
		if err != nil {
			return false, err
		}
		maxCreated := deployment.Spec.Replicas + deploymentutil.MaxSurge(*deployment)
		if totalCreated > maxCreated {
			reason = fmt.Sprintf("total pods created: %d, more than the max allowed: %d", totalCreated, maxCreated)
			Logf(reason)
			return false, nil
		}
		minAvailable := deploymentutil.MinAvailable(deployment)
		if totalAvailable < minAvailable {
			reason = fmt.Sprintf("total pods available: %d, less than the min required: %d", totalAvailable, minAvailable)
			Logf(reason)
			return false, nil
		}
		return true, nil
	})

	if err == wait.ErrWaitTimeout {
		logReplicaSetsOfDeployment(deployment, allOldRSs, newRS)
		logPodsOfDeployment(c, deployment)
		err = fmt.Errorf("%s", reason)
	}
	if err != nil {
		return fmt.Errorf("error waiting for deployment %q status to match expectation: %v", d.Name, err)
	}
	return nil
}

// Waits for the deployment to reach desired state.
// Returns an error if the deployment's rolling update strategy (max unavailable or max surge) is broken at any times.
func WaitForDeploymentStatus(c clientset.Interface, d *extensions.Deployment) error {
	var (
		oldRSs, allOldRSs, allRSs []*extensions.ReplicaSet
		newRS                     *extensions.ReplicaSet
		deployment                *extensions.Deployment
	)

	err := wait.Poll(Poll, 5*time.Minute, func() (bool, error) {
		var err error
		deployment, err = c.Extensions().Deployments(d.Namespace).Get(d.Name)
		if err != nil {
			return false, err
		}
		oldRSs, allOldRSs, newRS, err = deploymentutil.GetAllReplicaSets(deployment, c)
		if err != nil {
			return false, err
		}
		if newRS == nil {
			// New RS hasn't been created yet.
			return false, nil
		}
		allRSs = append(oldRSs, newRS)
		// The old/new ReplicaSets need to contain the pod-template-hash label
		for i := range allRSs {
			if !labelsutil.SelectorHasLabel(allRSs[i].Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey) {
				return false, nil
			}
		}
		totalCreated := deploymentutil.GetReplicaCountForReplicaSets(allRSs)
		totalAvailable, err := deploymentutil.GetAvailablePodsForDeployment(c, deployment)
		if err != nil {
			return false, err
		}
		maxCreated := deployment.Spec.Replicas + deploymentutil.MaxSurge(*deployment)
		if totalCreated > maxCreated {
			logReplicaSetsOfDeployment(deployment, allOldRSs, newRS)
			logPodsOfDeployment(c, deployment)
			return false, fmt.Errorf("total pods created: %d, more than the max allowed: %d", totalCreated, maxCreated)
		}
		minAvailable := deploymentutil.MinAvailable(deployment)
		if totalAvailable < minAvailable {
			logReplicaSetsOfDeployment(deployment, allOldRSs, newRS)
			logPodsOfDeployment(c, deployment)
			return false, fmt.Errorf("total pods available: %d, less than the min required: %d", totalAvailable, minAvailable)
		}

		// When the deployment status and its underlying resources reach the desired state, we're done
		if deployment.Status.Replicas == deployment.Spec.Replicas &&
			deployment.Status.UpdatedReplicas == deployment.Spec.Replicas &&
			deploymentutil.GetReplicaCountForReplicaSets(oldRSs) == 0 &&
			deploymentutil.GetReplicaCountForReplicaSets([]*extensions.ReplicaSet{newRS}) == deployment.Spec.Replicas {
			return true, nil
		}
		return false, nil
	})

	if err == wait.ErrWaitTimeout {
		logReplicaSetsOfDeployment(deployment, allOldRSs, newRS)
		logPodsOfDeployment(c, deployment)
	}
	if err != nil {
		return fmt.Errorf("error waiting for deployment %q status to match expectation: %v", d.Name, err)
	}
	return nil
}

// WaitForDeploymentUpdatedReplicasLTE waits for given deployment to be observed by the controller and has at least a number of updatedReplicas
func WaitForDeploymentUpdatedReplicasLTE(c clientset.Interface, ns, deploymentName string, minUpdatedReplicas int, desiredGeneration int64) error {
	err := wait.Poll(Poll, 5*time.Minute, func() (bool, error) {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		if err != nil {
			return false, err
		}
		if deployment.Status.ObservedGeneration >= desiredGeneration && deployment.Status.UpdatedReplicas >= int32(minUpdatedReplicas) {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		return fmt.Errorf("error waiting for deployment %s to have at least %d updpatedReplicas: %v", deploymentName, minUpdatedReplicas, err)
	}
	return nil
}

// WaitForDeploymentRollbackCleared waits for given deployment either started rolling back or doesn't need to rollback.
// Note that rollback should be cleared shortly, so we only wait for 1 minute here to fail early.
func WaitForDeploymentRollbackCleared(c clientset.Interface, ns, deploymentName string) error {
	err := wait.Poll(Poll, 1*time.Minute, func() (bool, error) {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		if err != nil {
			return false, err
		}
		// Rollback not set or is kicked off
		if deployment.Spec.RollbackTo == nil {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		return fmt.Errorf("error waiting for deployment %s rollbackTo to be cleared: %v", deploymentName, err)
	}
	return nil
}

// WaitForDeploymentRevisionAndImage waits for the deployment's and its new RS's revision and container image to match the given revision and image.
// Note that deployment revision and its new RS revision should be updated shortly, so we only wait for 1 minute here to fail early.
func WaitForDeploymentRevisionAndImage(c clientset.Interface, ns, deploymentName string, revision, image string) error {
	var deployment *extensions.Deployment
	var newRS *extensions.ReplicaSet
	err := wait.Poll(Poll, 1*time.Minute, func() (bool, error) {
		var err error
		deployment, err = c.Extensions().Deployments(ns).Get(deploymentName)
		if err != nil {
			return false, err
		}
		// The new ReplicaSet needs to be non-nil and contain the pod-template-hash label
		newRS, err = deploymentutil.GetNewReplicaSet(deployment, c)
		if err != nil || newRS == nil || !labelsutil.SelectorHasLabel(newRS.Spec.Selector, extensions.DefaultDeploymentUniqueLabelKey) {
			return false, err
		}
		// Check revision of this deployment, and of the new replica set of this deployment
		if deployment.Annotations == nil || deployment.Annotations[deploymentutil.RevisionAnnotation] != revision ||
			newRS.Annotations == nil || newRS.Annotations[deploymentutil.RevisionAnnotation] != revision ||
			deployment.Spec.Template.Spec.Containers[0].Image != image || newRS.Spec.Template.Spec.Containers[0].Image != image {
			return false, nil
		}
		return true, nil
	})
	if err == wait.ErrWaitTimeout {
		logReplicaSetsOfDeployment(deployment, nil, newRS)
	}
	if newRS == nil {
		return fmt.Errorf("deployment %s failed to create new RS: %v", deploymentName, err)
	}
	if err != nil {
		return fmt.Errorf("error waiting for deployment %s (got %s / %s) and new RS %s (got %s / %s) revision and image to match expectation (expected %s / %s): %v", deploymentName, deployment.Annotations[deploymentutil.RevisionAnnotation], deployment.Spec.Template.Spec.Containers[0].Image, newRS.Name, newRS.Annotations[deploymentutil.RevisionAnnotation], newRS.Spec.Template.Spec.Containers[0].Image, revision, image, err)
	}
	return nil
}

// CheckNewRSAnnotations check if the new RS's annotation is as expected
func CheckNewRSAnnotations(c clientset.Interface, ns, deploymentName string, expectedAnnotations map[string]string) error {
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	if err != nil {
		return err
	}
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	if err != nil {
		return err
	}
	for k, v := range expectedAnnotations {
		// Skip checking revision annotations
		if k != deploymentutil.RevisionAnnotation && v != newRS.Annotations[k] {
			return fmt.Errorf("Expected new RS annotations = %+v, got %+v", expectedAnnotations, newRS.Annotations)
		}
	}
	return nil
}

func WaitForPodsReady(c *clientset.Clientset, ns, name string, minReadySeconds int) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	options := api.ListOptions{LabelSelector: label}
	return wait.Poll(Poll, 5*time.Minute, func() (bool, error) {
		pods, err := c.Pods(ns).List(options)
		if err != nil {
			return false, nil
		}
		for _, pod := range pods.Items {
			if !deploymentutil.IsPodAvailable(&pod, int32(minReadySeconds), time.Now()) {
				return false, nil
			}
		}
		return true, nil
	})
}

// Waits for the deployment to clean up old rcs.
func WaitForDeploymentOldRSsNum(c *clientset.Clientset, ns, deploymentName string, desiredRSNum int) error {
	return wait.Poll(Poll, 5*time.Minute, func() (bool, error) {
		deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
		if err != nil {
			return false, err
		}
		_, oldRSs, err := deploymentutil.GetOldReplicaSets(deployment, c)
		if err != nil {
			return false, err
		}
		return len(oldRSs) == desiredRSNum, nil
	})
}

func logReplicaSetsOfDeployment(deployment *extensions.Deployment, allOldRSs []*extensions.ReplicaSet, newRS *extensions.ReplicaSet) {
	Logf("Deployment: %+v. Selector = %+v", *deployment, deployment.Spec.Selector)
	for i := range allOldRSs {
		Logf("All old ReplicaSets (%d/%d) of deployment %s: %+v. Selector = %+v", i+1, len(allOldRSs), deployment.Name, *allOldRSs[i], allOldRSs[i].Spec.Selector)
	}
	if newRS != nil {
		Logf("New ReplicaSet of deployment %s: %+v. Selector = %+v", deployment.Name, *newRS, newRS.Spec.Selector)
	} else {
		Logf("New ReplicaSet of deployment %s is nil.", deployment.Name)
	}
}

func WaitForObservedDeployment(c *clientset.Clientset, ns, deploymentName string, desiredGeneration int64) error {
	return deploymentutil.WaitForObservedDeployment(func() (*extensions.Deployment, error) { return c.Extensions().Deployments(ns).Get(deploymentName) }, desiredGeneration, Poll, 1*time.Minute)
}

func logPodsOfDeployment(c clientset.Interface, deployment *extensions.Deployment) {
	minReadySeconds := deployment.Spec.MinReadySeconds
	podList, err := deploymentutil.ListPods(deployment,
		func(namespace string, options api.ListOptions) (*api.PodList, error) {
			return c.Core().Pods(namespace).List(options)
		})
	if err != nil {
		Logf("Failed to list pods of deployment %s: %v", deployment.Name, err)
		return
	}
	if err == nil {
		for _, pod := range podList.Items {
			availability := "not available"
			if deploymentutil.IsPodAvailable(&pod, minReadySeconds, time.Now()) {
				availability = "available"
			}
			Logf("Pod %s is %s: %+v", pod.Name, availability, pod)
		}
	}
}

// Waits for the number of events on the given object to reach a desired count.
func WaitForEvents(c *client.Client, ns string, objOrRef runtime.Object, desiredEventsCount int) error {
	return wait.Poll(Poll, 5*time.Minute, func() (bool, error) {
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
func WaitForPartialEvents(c *client.Client, ns string, objOrRef runtime.Object, atLeastEventsCount int) error {
	return wait.Poll(Poll, 5*time.Minute, func() (bool, error) {
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

type updateDeploymentFunc func(d *extensions.Deployment)

func UpdateDeploymentWithRetries(c *clientset.Clientset, namespace, name string, applyUpdate updateDeploymentFunc) (deployment *extensions.Deployment, err error) {
	deployments := c.Extensions().Deployments(namespace)
	err = wait.Poll(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		if deployment, err = deployments.Get(name); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(deployment)
		if deployment, err = deployments.Update(deployment); err == nil {
			Logf("Updating deployment %s", name)
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
				state.Restarts = int(status.RestartCount)
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

// NodeSSHHosts returns SSH-able host names for all schedulable nodes - this excludes master node.
// It returns an error if it can't find an external IP for every node, though it still returns all
// hosts that it found in that case.
func NodeSSHHosts(c *client.Client) ([]string, error) {
	nodelist := waitListSchedulableNodesOrDie(c)

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
	signer, err := GetSigner(provider)
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

func IssueSSHCommand(cmd, provider string, node *api.Node) error {
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
	return RunKubectl("exec", fmt.Sprintf("--namespace=%v", ns), name, "--", "/bin/sh", "-c", cmd)
}

// RunHostCmdOrDie calls RunHostCmd and dies on error.
func RunHostCmdOrDie(ns, name, cmd string) string {
	stdout, err := RunHostCmd(ns, name, cmd)
	Logf("stdout: %v", stdout)
	ExpectNoError(err)
	return stdout
}

// LaunchHostExecPod launches a hostexec pod in the given namespace and waits
// until it's Running
func LaunchHostExecPod(client *client.Client, ns, name string) *api.Pod {
	hostExecPod := NewHostExecPodSpec(ns, name)
	pod, err := client.Pods(ns).Create(hostExecPod)
	ExpectNoError(err)
	err = WaitForPodRunningInNamespace(client, pod)
	ExpectNoError(err)
	return pod
}

// GetSigner returns an ssh.Signer for the provider ("gce", etc.) that can be
// used to SSH to their nodes.
func GetSigner(provider string) (ssh.Signer, error) {
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
		return nil, fmt.Errorf("GetSigner(...) not implemented for %s", provider)
	}
	key := filepath.Join(keydir, keyfile)

	return sshutil.MakePrivateKeySignerFromFile(key)
}

// CheckPodsRunningReady returns whether all pods whose names are listed in
// podNames in namespace ns are running and ready, using c and waiting at most
// timeout.
func CheckPodsRunningReady(c *client.Client, ns string, podNames []string, timeout time.Duration) bool {
	return CheckPodsCondition(c, ns, podNames, timeout, PodRunningReady, "running and ready")
}

// CheckPodsRunningReadyOrSucceeded returns whether all pods whose names are
// listed in podNames in namespace ns are running and ready, or succeeded; use
// c and waiting at most timeout.
func CheckPodsRunningReadyOrSucceeded(c *client.Client, ns string, podNames []string, timeout time.Duration) bool {
	return CheckPodsCondition(c, ns, podNames, timeout, PodRunningReadyOrSucceeded, "running and ready, or succeeded")
}

// CheckPodsCondition returns whether all pods whose names are listed in podNames
// in namespace ns are in the condition, using c and waiting at most timeout.
func CheckPodsCondition(c *client.Client, ns string, podNames []string, timeout time.Duration, condition podCondition, desc string) bool {
	np := len(podNames)
	Logf("Waiting up to %v for %d pods to be %s: %s", timeout, np, desc, podNames)
	result := make(chan bool, len(podNames))
	for ix := range podNames {
		// Launch off pod readiness checkers.
		go func(name string) {
			err := waitForPodCondition(c, ns, name, desc, timeout, condition)
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

// WaitForNodeToBeReady returns whether node name is ready within timeout.
func WaitForNodeToBeReady(c *client.Client, name string, timeout time.Duration) bool {
	return WaitForNodeToBe(c, name, api.NodeReady, true, timeout)
}

// WaitForNodeToBeNotReady returns whether node name is not ready (i.e. the
// readiness condition is anything but ready, e.g false or unknown) within
// timeout.
func WaitForNodeToBeNotReady(c *client.Client, name string, timeout time.Duration) bool {
	return WaitForNodeToBe(c, name, api.NodeReady, false, timeout)
}

func isNodeConditionSetAsExpected(node *api.Node, conditionType api.NodeConditionType, wantTrue, silent bool) bool {
	// Check the node readiness condition (logging all).
	for _, cond := range node.Status.Conditions {
		// Ensure that the condition type and the status matches as desired.
		if cond.Type == conditionType {
			if (cond.Status == api.ConditionTrue) == wantTrue {
				return true
			} else {
				if !silent {
					Logf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
						conditionType, node.Name, cond.Status == api.ConditionTrue, wantTrue, cond.Reason, cond.Message)
				}
				return false
			}
		}
	}
	if !silent {
		Logf("Couldn't find condition %v on node %v", conditionType, node.Name)
	}
	return false
}

func IsNodeConditionSetAsExpected(node *api.Node, conditionType api.NodeConditionType, wantTrue bool) bool {
	return isNodeConditionSetAsExpected(node, conditionType, wantTrue, false)
}

func IsNodeConditionSetAsExpectedSilent(node *api.Node, conditionType api.NodeConditionType, wantTrue bool) bool {
	return isNodeConditionSetAsExpected(node, conditionType, wantTrue, true)
}

func IsNodeConditionUnset(node *api.Node, conditionType api.NodeConditionType) bool {
	for _, cond := range node.Status.Conditions {
		if cond.Type == conditionType {
			return false
		}
	}
	return true
}

// WaitForNodeToBe returns whether node "name's" condition state matches wantTrue
// within timeout. If wantTrue is true, it will ensure the node condition status
// is ConditionTrue; if it's false, it ensures the node condition is in any state
// other than ConditionTrue (e.g. not true or unknown).
func WaitForNodeToBe(c *client.Client, name string, conditionType api.NodeConditionType, wantTrue bool, timeout time.Duration) bool {
	Logf("Waiting up to %v for node %s condition %s to be %t", timeout, name, conditionType, wantTrue)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		node, err := c.Nodes().Get(name)
		if err != nil {
			Logf("Couldn't get node %s", name)
			continue
		}

		if IsNodeConditionSetAsExpected(node, conditionType, wantTrue) {
			return true
		}
	}
	Logf("Node %s didn't reach desired %s condition status (%t) within %v", name, conditionType, wantTrue, timeout)
	return false
}

// checks whether all registered nodes are ready
func AllNodesReady(c *client.Client, timeout time.Duration) error {
	Logf("Waiting up to %v for all nodes to be ready", timeout)

	var notReady []api.Node
	err := wait.PollImmediate(Poll, timeout, func() (bool, error) {
		notReady = nil
		// It should be OK to list unschedulable Nodes here.
		nodes, err := c.Nodes().List(api.ListOptions{})
		if err != nil {
			return false, err
		}
		for _, node := range nodes.Items {
			if !IsNodeConditionSetAsExpected(&node, api.NodeReady, true) {
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
func FilterNodes(nodeList *api.NodeList, fn func(node api.Node) bool) {
	var l []api.Node

	for _, node := range nodeList.Items {
		if fn(node) {
			l = append(l, node)
		}
	}
	nodeList.Items = l
}

// ParseKVLines parses output that looks like lines containing "<key>: <val>"
// and returns <val> if <key> is found. Otherwise, it returns the empty string.
func ParseKVLines(output, key string) string {
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

func RestartKubeProxy(host string) error {
	// TODO: Make it work for all providers.
	if !ProviderIs("gce", "gke", "aws") {
		return fmt.Errorf("unsupported provider: %s", TestContext.Provider)
	}
	// kubelet will restart the kube-proxy since it's running in a static pod
	result, err := SSH("sudo pkill kube-proxy", host, TestContext.Provider)
	if err != nil || result.Code != 0 {
		LogSSHResult(result)
		return fmt.Errorf("couldn't restart kube-proxy: %v", err)
	}
	// wait for kube-proxy to come back up
	err = wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
		result, err := SSH("sudo /bin/sh -c 'pgrep kube-proxy | wc -l'", host, TestContext.Provider)
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

func RestartApiserver(c *client.Client) error {
	// TODO: Make it work for all providers.
	if !ProviderIs("gce", "gke", "aws") {
		return fmt.Errorf("unsupported provider: %s", TestContext.Provider)
	}
	if ProviderIs("gce", "aws") {
		return sshRestartMaster()
	}
	// GKE doesn't allow ssh access, so use a same-version master
	// upgrade to teardown/recreate master.
	v, err := c.ServerVersion()
	if err != nil {
		return err
	}
	return masterUpgradeGKE(v.GitVersion[1:]) // strip leading 'v'
}

func sshRestartMaster() error {
	if !ProviderIs("gce", "aws") {
		return fmt.Errorf("unsupported provider: %s", TestContext.Provider)
	}
	var command string
	if ProviderIs("gce") {
		command = "sudo docker ps | grep /kube-apiserver | cut -d ' ' -f 1 | xargs sudo docker kill"
	} else {
		command = "sudo /etc/init.d/kube-apiserver restart"
	}
	result, err := SSH(command, GetMasterHost()+":22", TestContext.Provider)
	if err != nil || result.Code != 0 {
		LogSSHResult(result)
		return fmt.Errorf("couldn't restart apiserver: %v", err)
	}
	return nil
}

func WaitForApiserverUp(c *client.Client) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		body, err := c.Get().AbsPath("/healthz").Do().Raw()
		if err == nil && string(body) == "ok" {
			return nil
		}
	}
	return fmt.Errorf("waiting for apiserver timed out")
}

// WaitForClusterSize waits until the cluster has desired size and there is no not-ready nodes in it.
// By cluster size we mean number of Nodes excluding Master Node.
func WaitForClusterSize(c *client.Client, size int, timeout time.Duration) error {
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
		FilterNodes(nodes, func(node api.Node) bool {
			return IsNodeConditionSetAsExpected(&node, api.NodeReady, true)
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

// GetHostExternalAddress gets the node for a pod and returns the first External
// address. Returns an error if the node the pod is on doesn't have an External
// address.
func GetHostExternalAddress(client *client.Client, p *api.Pod) (externalAddress string, err error) {
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
	return &http.Response{}, nil
}

// headersForConfig extracts any http client logic necessary for the provided
// config.
func headersForConfig(c *restclient.Config) (http.Header, error) {
	extract := &extractRT{}
	rt, err := restclient.HTTPWrappersForConfig(c, extract)
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
func OpenWebSocketForURL(url *url.URL, config *restclient.Config, protocols []string) (*websocket.Conn, error) {
	tlsConfig, err := restclient.TLSConfigFor(config)
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

// WaitForIngressAddress waits for the Ingress to acquire an address.
func WaitForIngressAddress(c *client.Client, ns, ingName string, timeout time.Duration) (string, error) {
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
func LookForStringInLog(ns, podName, container, expectedString string, timeout time.Duration) (result string, err error) {
	return LookForString(expectedString, timeout, func() string {
		return RunKubectlOrDie("logs", podName, container, fmt.Sprintf("--namespace=%v", ns))
	})
}

// Looks for the given string in a file in a specific pod container
func LookForStringInFile(ns, podName, container, file, expectedString string, timeout time.Duration) (result string, err error) {
	return LookForString(expectedString, timeout, func() string {
		return RunKubectlOrDie("exec", podName, "-c", container, fmt.Sprintf("--namespace=%v", ns), "--", "cat", file)
	})
}

// Looks for the given string in the output of a command executed in a specific pod container
func LookForStringInPodExec(ns, podName string, command []string, expectedString string, timeout time.Duration) (result string, err error) {
	return LookForString(expectedString, timeout, func() string {
		// use the first container
		args := []string{"exec", podName, fmt.Sprintf("--namespace=%v", ns), "--"}
		args = append(args, command...)
		return RunKubectlOrDie(args...)
	})
}

// Looks for the given string in the output of fn, repeatedly calling fn until
// the timeout is reached or the string is found. Returns last log and possibly
// error if the string was not found.
func LookForString(expectedString string, timeout time.Duration, fn func() string) (result string, err error) {
	for t := time.Now(); time.Since(t) < timeout; time.Sleep(Poll) {
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
		if p.Port == int32(svcPort) {
			if p.NodePort != 0 {
				return int(p.NodePort), nil
			}
		}
	}
	return 0, fmt.Errorf(
		"No node port found for service %v, port %v", name, svcPort)
}

// GetNodePortURL returns the url to a nodeport Service.
func GetNodePortURL(client *client.Client, ns, name string, svcPort int) (string, error) {
	nodePort, err := getSvcNodePort(client, ns, name, svcPort)
	if err != nil {
		return "", err
	}
	// This list of nodes must not include the master, which is marked
	// unschedulable, since the master doesn't run kube-proxy. Without
	// kube-proxy NodePorts won't work.
	var nodes *api.NodeList
	if wait.PollImmediate(Poll, SingleCallTimeout, func() (bool, error) {
		nodes, err = client.Nodes().List(api.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector()})
		return err == nil, nil
	}) != nil {
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

// ScaleRCByLabels scales an RC via ns/label lookup. If replicas == 0 it waits till
// none are running, otherwise it does what a synchronous scale operation would do.
func ScaleRCByLabels(client *client.Client, ns string, l map[string]string, replicas uint) error {
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
			ps, err := podStoreForRC(client, rc)
			if err != nil {
				return err
			}
			defer ps.Stop()
			if err = waitForPodsGone(ps, 10*time.Second, 10*time.Minute); err != nil {
				return fmt.Errorf("error while waiting for pods gone %s: %v", name, err)
			}
		} else {
			if err := WaitForPodsWithLabelRunning(
				client, ns, labels.SelectorFromSet(labels.Set(rc.Spec.Selector))); err != nil {
				return err
			}
		}
	}
	return nil
}

// TODO(random-liu): Change this to be a member function of the framework.
func GetPodLogs(c *client.Client, namespace, podName, containerName string) (string, error) {
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
	if TestContext.Provider == "gce" || TestContext.Provider == "gke" {
		return ensureGCELoadBalancerResourcesDeleted(ip, portRange)
	}
	return nil
}

func ensureGCELoadBalancerResourcesDeleted(ip, portRange string) error {
	gceCloud, ok := TestContext.CloudConfig.Provider.(*gcecloud.GCECloud)
	if !ok {
		return fmt.Errorf("failed to convert CloudConfig.Provider to GCECloud: %#v", TestContext.CloudConfig.Provider)
	}
	project := TestContext.CloudConfig.ProjectID
	region, err := gcecloud.GetGCERegion(TestContext.CloudConfig.Zone)
	if err != nil {
		return fmt.Errorf("could not get region for zone %q: %v", TestContext.CloudConfig.Zone, err)
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
//	defer UnblockNetwork(from, to)
//	BlockNetwork(from, to)
//	...
// }
//
func BlockNetwork(from string, to string) {
	Logf("block network traffic from %s to %s", from, to)
	iptablesRule := fmt.Sprintf("OUTPUT --destination %s --jump REJECT", to)
	dropCmd := fmt.Sprintf("sudo iptables --insert %s", iptablesRule)
	if result, err := SSH(dropCmd, from, TestContext.Provider); result.Code != 0 || err != nil {
		LogSSHResult(result)
		Failf("Unexpected error: %v", err)
	}
}

func UnblockNetwork(from string, to string) {
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
		result, err := SSH(undropCmd, from, TestContext.Provider)
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

func isElementOf(podUID types.UID, pods *api.PodList) bool {
	for _, pod := range pods.Items {
		if pod.UID == podUID {
			return true
		}
	}
	return false
}

func CheckRSHashLabel(rs *extensions.ReplicaSet) error {
	if len(rs.Labels[extensions.DefaultDeploymentUniqueLabelKey]) == 0 ||
		len(rs.Spec.Selector.MatchLabels[extensions.DefaultDeploymentUniqueLabelKey]) == 0 ||
		len(rs.Spec.Template.Labels[extensions.DefaultDeploymentUniqueLabelKey]) == 0 {
		return fmt.Errorf("unexpected RS missing required pod-hash-template: %+v, selector = %+v, template = %+v", rs, rs.Spec.Selector, rs.Spec.Template)
	}
	return nil
}

func CheckPodHashLabel(pods *api.PodList) error {
	invalidPod := ""
	for _, pod := range pods.Items {
		if len(pod.Labels[extensions.DefaultDeploymentUniqueLabelKey]) == 0 {
			if len(invalidPod) == 0 {
				invalidPod = "unexpected pods missing required pod-hash-template:"
			}
			invalidPod = fmt.Sprintf("%s %+v;", invalidPod, pod)
		}
	}
	if len(invalidPod) > 0 {
		return fmt.Errorf("%s", invalidPod)
	}
	return nil
}

// timeout for proxy requests.
const proxyTimeout = 2 * time.Minute

// NodeProxyRequest performs a get on a node proxy endpoint given the nodename and rest client.
func NodeProxyRequest(c *client.Client, node, endpoint string) (restclient.Result, error) {
	// proxy tends to hang in some cases when Node is not ready. Add an artificial timeout for this call.
	// This will leak a goroutine if proxy hangs. #22165
	subResourceProxyAvailable, err := ServerVersionGTE(subResourceServiceAndNodeProxyVersion, c)
	if err != nil {
		return restclient.Result{}, err
	}
	var result restclient.Result
	finished := make(chan struct{})
	go func() {
		if subResourceProxyAvailable {
			result = c.Get().
				Resource("nodes").
				SubResource("proxy").
				Name(fmt.Sprintf("%v:%v", node, ports.KubeletPort)).
				Suffix(endpoint).
				Do()

		} else {
			result = c.Get().
				Prefix("proxy").
				Resource("nodes").
				Name(fmt.Sprintf("%v:%v", node, ports.KubeletPort)).
				Suffix(endpoint).
				Do()
		}
		finished <- struct{}{}
	}()
	select {
	case <-finished:
		return result, nil
	case <-time.After(proxyTimeout):
		return restclient.Result{}, nil
	}
}

// GetKubeletPods retrieves the list of pods on the kubelet
func GetKubeletPods(c *client.Client, node string) (*api.PodList, error) {
	return getKubeletPods(c, node, "pods")
}

// GetKubeletRunningPods retrieves the list of running pods on the kubelet. The pods
// includes necessary information (e.g., UID, name, namespace for
// pods/containers), but do not contain the full spec.
func GetKubeletRunningPods(c *client.Client, node string) (*api.PodList, error) {
	return getKubeletPods(c, node, "runningpods")
}

func getKubeletPods(c *client.Client, node, resource string) (*api.PodList, error) {
	result := &api.PodList{}
	client, err := NodeProxyRequest(c, node, resource)
	if err != nil {
		return &api.PodList{}, err
	}
	if err = client.Into(result); err != nil {
		return &api.PodList{}, err
	}
	return result, nil
}

// LaunchWebserverPod launches a pod serving http on port 8080 to act
// as the target for networking connectivity checks.  The ip address
// of the created pod will be returned if the pod is launched
// successfully.
func LaunchWebserverPod(f *Framework, podName, nodeName string) (ip string) {
	containerName := fmt.Sprintf("%s-container", podName)
	port := 8080
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: podName,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  containerName,
					Image: "gcr.io/google_containers/porter:cd5cb5791ebaa8641955f0e8c2a9bed669b1eaab",
					Env:   []api.EnvVar{{Name: fmt.Sprintf("SERVE_PORT_%d", port), Value: "foo"}},
					Ports: []api.ContainerPort{{ContainerPort: int32(port)}},
				},
			},
			NodeName:      nodeName,
			RestartPolicy: api.RestartPolicyNever,
		},
	}
	podClient := f.Client.Pods(f.Namespace.Name)
	_, err := podClient.Create(pod)
	ExpectNoError(err)
	ExpectNoError(f.WaitForPodRunning(podName))
	createdPod, err := podClient.Get(podName)
	ExpectNoError(err)
	ip = fmt.Sprintf("%s:%d", createdPod.Status.PodIP, port)
	Logf("Target pod IP:port is %s", ip)
	return
}

// CheckConnectivityToHost launches a pod running wget on the
// specified node to test connectivity to the specified host.  An
// error will be returned if the host is not reachable from the pod.
func CheckConnectivityToHost(f *Framework, nodeName, podName, host string, timeout int) error {
	contName := fmt.Sprintf("%s-container", podName)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Name: podName,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    contName,
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"wget", fmt.Sprintf("--timeout=%d", timeout), "-s", host},
				},
			},
			NodeName:      nodeName,
			RestartPolicy: api.RestartPolicyNever,
		},
	}
	podClient := f.Client.Pods(f.Namespace.Name)
	_, err := podClient.Create(pod)
	if err != nil {
		return err
	}
	defer podClient.Delete(podName, nil)
	return WaitForPodSuccessInNamespace(f.Client, podName, contName, f.Namespace.Name)
}

// CoreDump SSHs to the master and all nodes and dumps their logs into dir.
// It shells out to cluster/log-dump.sh to accomplish this.
func CoreDump(dir string) {
	cmd := exec.Command(path.Join(TestContext.RepoRoot, "cluster", "log-dump.sh"), dir)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		Logf("Error running cluster/log-dump.sh: %v", err)
	}
}

func UpdatePodWithRetries(client *client.Client, ns, name string, update func(*api.Pod)) (*api.Pod, error) {
	for i := 0; i < 3; i++ {
		pod, err := client.Pods(ns).Get(name)
		if err != nil {
			return nil, fmt.Errorf("Failed to get pod %q: %v", name, err)
		}
		update(pod)
		pod, err = client.Pods(ns).Update(pod)
		if err == nil {
			return pod, nil
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			return nil, fmt.Errorf("Failed to update pod %q: %v", name, err)
		}
	}
	return nil, fmt.Errorf("Too many retries updating Pod %q", name)
}

func GetPodsInNamespace(c *client.Client, ns string, ignoreLabels map[string]string) ([]*api.Pod, error) {
	pods, err := c.Pods(ns).List(api.ListOptions{})
	if err != nil {
		return []*api.Pod{}, err
	}
	ignoreSelector := labels.SelectorFromSet(ignoreLabels)
	filtered := []*api.Pod{}
	for _, p := range pods.Items {
		if len(ignoreLabels) != 0 && ignoreSelector.Matches(labels.Set(p.Labels)) {
			continue
		}
		filtered = append(filtered, &p)
	}
	return filtered, nil
}

// RunCmd runs cmd using args and returns its stdout and stderr. It also outputs
// cmd's stdout and stderr to their respective OS streams.
func RunCmd(command string, args ...string) (string, string, error) {
	Logf("Running %s %v", command, args)
	var bout, berr bytes.Buffer
	cmd := exec.Command(command, args...)
	// We also output to the OS stdout/stderr to aid in debugging in case cmd
	// hangs and never returns before the test gets killed.
	//
	// This creates some ugly output because gcloud doesn't always provide
	// newlines.
	cmd.Stdout = io.MultiWriter(os.Stdout, &bout)
	cmd.Stderr = io.MultiWriter(os.Stderr, &berr)
	err := cmd.Run()
	stdout, stderr := bout.String(), berr.String()
	if err != nil {
		return "", "", fmt.Errorf("error running %s %v; got error %v, stdout %q, stderr %q",
			command, args, err, stdout, stderr)
	}
	return stdout, stderr, nil
}

// retryCmd runs cmd using args and retries it for up to SingleCallTimeout if
// it returns an error. It returns stdout and stderr.
func retryCmd(command string, args ...string) (string, string, error) {
	var err error
	stdout, stderr := "", ""
	wait.Poll(Poll, SingleCallTimeout, func() (bool, error) {
		stdout, stderr, err = RunCmd(command, args...)
		if err != nil {
			Logf("Got %v", err)
			return false, nil
		}
		return true, nil
	})
	return stdout, stderr, err
}

// GetPodsScheduled returns a number of currently scheduled and not scheduled Pods.
func GetPodsScheduled(masterNodes sets.String, pods *api.PodList) (scheduledPods, notScheduledPods []api.Pod) {
	for _, pod := range pods.Items {
		if !masterNodes.Has(pod.Spec.NodeName) {
			if pod.Spec.NodeName != "" {
				_, scheduledCondition := api.GetPodCondition(&pod.Status, api.PodScheduled)
				Expect(scheduledCondition != nil).To(Equal(true))
				Expect(scheduledCondition.Status).To(Equal(api.ConditionTrue))
				scheduledPods = append(scheduledPods, pod)
			} else {
				_, scheduledCondition := api.GetPodCondition(&pod.Status, api.PodScheduled)
				Expect(scheduledCondition != nil).To(Equal(true))
				Expect(scheduledCondition.Status).To(Equal(api.ConditionFalse))
				if scheduledCondition.Reason == "Unschedulable" {

					notScheduledPods = append(notScheduledPods, pod)
				}
			}
		}
	}
	return
}

// WaitForStableCluster waits until all existing pods are scheduled and returns their amount.
func WaitForStableCluster(c *client.Client, masterNodes sets.String) int {
	timeout := 10 * time.Minute
	startTime := time.Now()

	allPods, err := c.Pods(api.NamespaceAll).List(api.ListOptions{})
	ExpectNoError(err)
	// API server returns also Pods that succeeded. We need to filter them out.
	currentPods := make([]api.Pod, 0, len(allPods.Items))
	for _, pod := range allPods.Items {
		if pod.Status.Phase != api.PodSucceeded && pod.Status.Phase != api.PodFailed {
			currentPods = append(currentPods, pod)
		}

	}
	allPods.Items = currentPods
	scheduledPods, currentlyNotScheduledPods := GetPodsScheduled(masterNodes, allPods)
	for len(currentlyNotScheduledPods) != 0 {
		time.Sleep(2 * time.Second)

		allPods, err := c.Pods(api.NamespaceAll).List(api.ListOptions{})
		ExpectNoError(err)
		scheduledPods, currentlyNotScheduledPods = GetPodsScheduled(masterNodes, allPods)

		if startTime.Add(timeout).Before(time.Now()) {
			Failf("Timed out after %v waiting for stable cluster.", timeout)
			break
		}
	}
	return len(scheduledPods)
}

// GetMasterAndWorkerNodesOrDie will return a list masters and schedulable worker nodes
func GetMasterAndWorkerNodesOrDie(c *client.Client) (sets.String, *api.NodeList) {
	nodes := &api.NodeList{}
	masters := sets.NewString()
	all, _ := c.Nodes().List(api.ListOptions{})
	for _, n := range all.Items {
		if system.IsMasterNode(&n) {
			masters.Insert(n.Name)
		} else if isNodeSchedulable(&n) {
			nodes.Items = append(nodes.Items, n)
		}
	}
	return masters, nodes
}
