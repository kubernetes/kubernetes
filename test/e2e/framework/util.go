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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"text/tabwriter"
	"time"

	"github.com/golang/glog"
	"golang.org/x/crypto/ssh"
	"golang.org/x/net/websocket"
	"google.golang.org/api/googleapi"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	gomegatypes "github.com/onsi/gomega/types"

	apps "k8s.io/api/apps/v1"
	batch "k8s.io/api/batch/v1"
	"k8s.io/api/core/v1"
	extensions "k8s.io/api/extensions/v1beta1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/uuid"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apimachinery/pkg/util/wait"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apimachinery/pkg/watch"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	appsinternal "k8s.io/kubernetes/pkg/apis/apps"
	batchinternal "k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/conditions"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/azure"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/controller"
	nodectlr "k8s.io/kubernetes/pkg/controller/nodelifecycle"
	"k8s.io/kubernetes/pkg/controller/service"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	sshutil "k8s.io/kubernetes/pkg/ssh"
	"k8s.io/kubernetes/pkg/util/system"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
	"k8s.io/kubernetes/test/e2e/framework/ginkgowrapper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	uexec "k8s.io/utils/exec"
)

const (
	// How long to wait for the pod to be listable
	PodListTimeout = time.Minute
	// Initial pod start can be delayed O(minutes) by slow docker pulls
	// TODO: Make this 30 seconds once #4566 is resolved.
	PodStartTimeout = 5 * time.Minute

	// Same as `PodStartTimeout` to wait for the pod to be started, but shorter.
	// Use it case by case when we are sure pod start will not be delayed
	// minutes by slow docker pulls or something else.
	PodStartShortTimeout = 1 * time.Minute

	// How long to wait for a pod to be deleted
	PodDeleteTimeout = 5 * time.Minute

	// If there are any orphaned namespaces to clean up, this test is running
	// on a long lived cluster. A long wait here is preferably to spurious test
	// failures caused by leaked resources from a previous test run.
	NamespaceCleanupTimeout = 15 * time.Minute

	// Some pods can take much longer to get ready due to volume attach/detach latency.
	slowPodStartTimeout = 15 * time.Minute

	// How long to wait for a service endpoint to be resolvable.
	ServiceStartTimeout = 3 * time.Minute

	// How often to Poll pods, nodes and claims.
	Poll = 2 * time.Second

	pollShortTimeout = 1 * time.Minute
	pollLongTimeout  = 5 * time.Minute

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

	podRespondingTimeout     = 15 * time.Minute
	ServiceRespondingTimeout = 2 * time.Minute
	EndpointRegisterTimeout  = time.Minute

	// How long claims have to become dynamically provisioned
	ClaimProvisionTimeout = 5 * time.Minute

	// Same as `ClaimProvisionTimeout` to wait for claim to be dynamically provisioned, but shorter.
	// Use it case by case when we are sure this timeout is enough.
	ClaimProvisionShortTimeout = 1 * time.Minute

	// How long claims have to become bound
	ClaimBindingTimeout = 3 * time.Minute

	// How long claims have to become deleted
	ClaimDeletingTimeout = 3 * time.Minute

	// How long PVs have to beome reclaimed
	PVReclaimingTimeout = 3 * time.Minute

	// How long PVs have to become bound
	PVBindingTimeout = 3 * time.Minute

	// How long PVs have to become deleted
	PVDeletingTimeout = 3 * time.Minute

	// How long a node is allowed to become "Ready" after it is restarted before
	// the test is considered failed.
	RestartNodeReadyAgainTimeout = 5 * time.Minute

	// How long a pod is allowed to become "running" and "ready" after a node
	// restart before test is considered failed.
	RestartPodReadyAgainTimeout = 5 * time.Minute

	// Number of objects that gc can delete in a second.
	// GC issues 2 requestes for single delete.
	gcThroughput = 10

	// Minimal number of nodes for the cluster to be considered large.
	largeClusterThreshold = 100

	// TODO(justinsb): Avoid hardcoding this.
	awsMasterIP = "172.20.0.9"

	// ssh port
	sshPort = "22"
)

var (
	BusyBoxImage = imageutils.GetE2EImage(imageutils.BusyBox)

	// For parsing Kubectl version for version-skewed testing.
	gitVersionRegexp = regexp.MustCompile("GitVersion:\"(v.+?)\"")

	// Slice of regexps for names of pods that have to be running to consider a Node "healthy"
	requiredPerNodePods = []*regexp.Regexp{
		regexp.MustCompile(".*kube-proxy.*"),
		regexp.MustCompile(".*fluentd-elasticsearch.*"),
		regexp.MustCompile(".*node-problem-detector.*"),
	}

	// Serve hostname image name
	ServeHostnameImage = imageutils.GetE2EImage(imageutils.ServeHostname)
)

type Address struct {
	internalIP string
	externalIP string
	hostname   string
}

// GetServerArchitecture fetches the architecture of the cluster's apiserver.
func GetServerArchitecture(c clientset.Interface) string {
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

func GetServicesProxyRequest(c clientset.Interface, request *restclient.Request) (*restclient.Request, error) {
	return request.Resource("services").SubResource("proxy"), nil
}

// unique identifier of the e2e run
var RunId = uuid.NewUUID()

type CreateTestingNSFn func(baseName string, c clientset.Interface, labels map[string]string) (*v1.Namespace, error)

type ContainerFailures struct {
	status   *v1.ContainerStateTerminated
	Restarts int
}

func GetMasterHost() string {
	masterUrl, err := url.Parse(TestContext.Host)
	ExpectNoError(err)
	return masterUrl.Hostname()
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
	FailfWithOffset(1, format, args...)
}

// FailfWithOffset calls "Fail" and logs the error at "offset" levels above its caller
// (for example, for call chain f -> g -> FailfWithOffset(1, ...) error would be logged for "f").
func FailfWithOffset(offset int, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	log("INFO", msg)
	ginkgowrapper.Fail(nowStamp()+": "+msg, 1+offset)
}

func Skipf(format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	log("INFO", msg)
	ginkgowrapper.Skip(nowStamp() + ": " + msg)
}

func SkipUnlessNodeCountIsAtLeast(minNodeCount int) {
	if TestContext.CloudConfig.NumNodes < minNodeCount {
		Skipf("Requires at least %d nodes (not %d)", minNodeCount, TestContext.CloudConfig.NumNodes)
	}
}

func SkipUnlessNodeCountIsAtMost(maxNodeCount int) {
	if TestContext.CloudConfig.NumNodes > maxNodeCount {
		Skipf("Requires at most %d nodes (not %d)", maxNodeCount, TestContext.CloudConfig.NumNodes)
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

func SkipUnlessLocalEphemeralStorageEnabled() {
	if !utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) {
		Skipf("Only supported when %v feature is enabled", features.LocalStorageCapacityIsolation)
	}
}

func SkipUnlessSSHKeyPresent() {
	if _, err := GetSigner(TestContext.Provider); err != nil {
		Skipf("No SSH Key for provider %s: '%v'", TestContext.Provider, err)
	}
}

func SkipUnlessProviderIs(supportedProviders ...string) {
	if !ProviderIs(supportedProviders...) {
		Skipf("Only supported for providers %v (not %s)", supportedProviders, TestContext.Provider)
	}
}

func SkipUnlessMultizone(c clientset.Interface) {
	zones, err := GetClusterZones(c)
	if err != nil {
		Skipf("Error listing cluster zones")
	}
	if zones.Len() <= 1 {
		Skipf("Requires more than one zone")
	}
}

func SkipIfMultizone(c clientset.Interface) {
	zones, err := GetClusterZones(c)
	if err != nil {
		Skipf("Error listing cluster zones")
	}
	if zones.Len() > 1 {
		Skipf("Requires more than one zone")
	}
}

func SkipUnlessClusterMonitoringModeIs(supportedMonitoring ...string) {
	if !ClusterMonitoringModeIs(supportedMonitoring...) {
		Skipf("Only next monitoring modes are supported %v (not %s)", supportedMonitoring, TestContext.ClusterMonitoringMode)
	}
}

func SkipUnlessPrometheusMonitoringIsEnabled(supportedMonitoring ...string) {
	if !TestContext.EnablePrometheusMonitoring {
		Skipf("Skipped because prometheus monitoring is not enabled")
	}
}

func SkipUnlessMasterOSDistroIs(supportedMasterOsDistros ...string) {
	if !MasterOSDistroIs(supportedMasterOsDistros...) {
		Skipf("Only supported for master OS distro %v (not %s)", supportedMasterOsDistros, TestContext.MasterOSDistro)
	}
}

func SkipUnlessNodeOSDistroIs(supportedNodeOsDistros ...string) {
	if !NodeOSDistroIs(supportedNodeOsDistros...) {
		Skipf("Only supported for node OS distro %v (not %s)", supportedNodeOsDistros, TestContext.NodeOSDistro)
	}
}

func SkipUnlessSecretExistsAfterWait(c clientset.Interface, name, namespace string, timeout time.Duration) {
	Logf("Waiting for secret %v in namespace %v to exist in duration %v", name, namespace, timeout)
	start := time.Now()
	if wait.PollImmediate(15*time.Second, timeout, func() (bool, error) {
		_, err := c.CoreV1().Secrets(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			Logf("Secret %v in namespace %v still does not exist after duration %v", name, namespace, time.Since(start))
			return false, nil
		}
		return true, nil
	}) != nil {
		Skipf("Secret %v in namespace %v did not exist after timeout of %v", name, namespace, timeout)
	}
	Logf("Secret %v in namespace %v found after duration %v", name, namespace, time.Since(start))
}

func SkipIfContainerRuntimeIs(runtimes ...string) {
	for _, runtime := range runtimes {
		if runtime == TestContext.ContainerRuntime {
			Skipf("Not supported under container runtime %s", runtime)
		}
	}
}

func RunIfContainerRuntimeIs(runtimes ...string) {
	for _, runtime := range runtimes {
		if runtime == TestContext.ContainerRuntime {
			return
		}
	}
	Skipf("Skipped because container runtime %q is not in %s", TestContext.ContainerRuntime, runtimes)
}

func RunIfSystemSpecNameIs(names ...string) {
	for _, name := range names {
		if name == TestContext.SystemSpecName {
			return
		}
	}
	Skipf("Skipped because system spec name %q is not in %v", TestContext.SystemSpecName, names)
}

func ProviderIs(providers ...string) bool {
	for _, provider := range providers {
		if strings.ToLower(provider) == strings.ToLower(TestContext.Provider) {
			return true
		}
	}
	return false
}

func ClusterMonitoringModeIs(monitoringModes ...string) bool {
	for _, mode := range monitoringModes {
		if strings.ToLower(mode) == strings.ToLower(TestContext.ClusterMonitoringMode) {
			return true
		}
	}
	return false
}

func MasterOSDistroIs(supportedMasterOsDistros ...string) bool {
	for _, distro := range supportedMasterOsDistros {
		if strings.ToLower(distro) == strings.ToLower(TestContext.MasterOSDistro) {
			return true
		}
	}
	return false
}

func NodeOSDistroIs(supportedNodeOsDistros ...string) bool {
	for _, distro := range supportedNodeOsDistros {
		if strings.ToLower(distro) == strings.ToLower(TestContext.NodeOSDistro) {
			return true
		}
	}
	return false
}

func ProxyMode(f *Framework) (string, error) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "kube-proxy-mode-detector",
			Namespace: f.Namespace.Name,
		},
		Spec: v1.PodSpec{
			HostNetwork: true,
			Containers: []v1.Container{
				{
					Name:    "detector",
					Image:   imageutils.GetE2EImage(imageutils.Net),
					Command: []string{"/bin/sleep", "3600"},
				},
			},
		},
	}
	f.PodClient().CreateSync(pod)
	defer f.PodClient().DeleteSync(pod.Name, &metav1.DeleteOptions{}, DefaultPodDeletionTimeout)

	cmd := "curl -q -s --connect-timeout 1 http://localhost:10249/proxyMode"
	stdout, err := RunHostCmd(pod.Namespace, pod.Name, cmd)
	if err != nil {
		return "", err
	}
	Logf("ProxyMode: %s", stdout)
	return stdout, nil
}

func SkipUnlessServerVersionGTE(v *utilversion.Version, c discovery.ServerVersionInterface) {
	gte, err := ServerVersionGTE(v, c)
	if err != nil {
		Failf("Failed to get server version: %v", err)
	}
	if !gte {
		Skipf("Not supported for server versions before %q", v)
	}
}

func SkipIfMissingResource(dynamicClient dynamic.Interface, gvr schema.GroupVersionResource, namespace string) {
	resourceClient := dynamicClient.Resource(gvr).Namespace(namespace)
	_, err := resourceClient.List(metav1.ListOptions{})
	if err != nil {
		// not all resources support list, so we ignore those
		if apierrs.IsMethodNotSupported(err) || apierrs.IsNotFound(err) || apierrs.IsForbidden(err) {
			Skipf("Could not find %s resource, skipping test: %#v", gvr, err)
		}
		Failf("Unexpected error getting %v: %v", gvr, err)
	}
}

// ProvidersWithSSH are those providers where each node is accessible with SSH
var ProvidersWithSSH = []string{"gce", "gke", "aws", "local"}

type podCondition func(pod *v1.Pod) (bool, error)

// logPodStates logs basic info of provided pods for debugging.
func logPodStates(pods []v1.Pod) {
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

// errorBadPodsStates create error message of basic info of bad pods for debugging.
func errorBadPodsStates(badPods []v1.Pod, desiredPods int, ns, desiredState string, timeout time.Duration) string {
	errStr := fmt.Sprintf("%d / %d pods in namespace %q are NOT in %s state in %v\n", len(badPods), desiredPods, ns, desiredState, timeout)
	// Print bad pods info only if there are fewer than 10 bad pods
	if len(badPods) > 10 {
		return errStr + "There are too many bad pods. Please check log for details."
	}

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
	return errStr + buf.String()
}

// WaitForPodsSuccess waits till all labels matching the given selector enter
// the Success state. The caller is expected to only invoke this method once the
// pods have been created.
func WaitForPodsSuccess(c clientset.Interface, ns string, successPodLabels map[string]string, timeout time.Duration) error {
	successPodSelector := labels.SelectorFromSet(successPodLabels)
	start, badPods, desiredPods := time.Now(), []v1.Pod{}, 0

	if wait.PollImmediate(30*time.Second, timeout, func() (bool, error) {
		podList, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{LabelSelector: successPodSelector.String()})
		if err != nil {
			Logf("Error getting pods in namespace %q: %v", ns, err)
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		if len(podList.Items) == 0 {
			Logf("Waiting for pods to enter Success, but no pods in %q match label %v", ns, successPodLabels)
			return true, nil
		}
		badPods = []v1.Pod{}
		desiredPods = len(podList.Items)
		for _, pod := range podList.Items {
			if pod.Status.Phase != v1.PodSucceeded {
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
		LogPodsWithLabels(c, ns, successPodLabels, Logf)
		return errors.New(errorBadPodsStates(badPods, desiredPods, ns, "SUCCESS", timeout))

	}
	return nil
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
func WaitForPodsRunningReady(c clientset.Interface, ns string, minPods, allowedNotReadyPods int32, timeout time.Duration, ignoreLabels map[string]string) error {
	ignoreSelector := labels.SelectorFromSet(map[string]string{})
	start := time.Now()
	Logf("Waiting up to %v for all pods (need at least %d) in namespace '%s' to be running and ready",
		timeout, minPods, ns)
	wg := sync.WaitGroup{}
	wg.Add(1)
	var ignoreNotReady bool
	badPods := []v1.Pod{}
	desiredPods := 0
	notReady := int32(0)

	if wait.PollImmediate(Poll, timeout, func() (bool, error) {
		// We get the new list of pods, replication controllers, and
		// replica sets in every iteration because more pods come
		// online during startup and we want to ensure they are also
		// checked.
		replicas, replicaOk := int32(0), int32(0)

		rcList, err := c.CoreV1().ReplicationControllers(ns).List(metav1.ListOptions{})
		if err != nil {
			Logf("Error getting replication controllers in namespace '%s': %v", ns, err)
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		for _, rc := range rcList.Items {
			replicas += *rc.Spec.Replicas
			replicaOk += rc.Status.ReadyReplicas
		}

		rsList, err := c.ExtensionsV1beta1().ReplicaSets(ns).List(metav1.ListOptions{})
		if err != nil {
			Logf("Error getting replication sets in namespace %q: %v", ns, err)
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		for _, rs := range rsList.Items {
			replicas += *rs.Spec.Replicas
			replicaOk += rs.Status.ReadyReplicas
		}

		podList, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{})
		if err != nil {
			Logf("Error getting pods in namespace '%s': %v", ns, err)
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
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
				Logf("The status of Pod %s is Succeeded, skipping waiting", pod.ObjectMeta.Name)
				// it doesn't make sense to wait for this pod
				continue
			case pod.Status.Phase != v1.PodFailed:
				Logf("The status of Pod %s is %s (Ready = false), waiting for it to be either Running (with Ready = true) or Failed", pod.ObjectMeta.Name, pod.Status.Phase)
				notReady++
				badPods = append(badPods, pod)
			default:
				if metav1.GetControllerOf(&pod) == nil {
					Logf("Pod %s is Failed, but it's not controlled by a controller", pod.ObjectMeta.Name)
					badPods = append(badPods, pod)
				}
				//ignore failed pods that are controlled by some controller
			}
		}

		Logf("%d / %d pods in namespace '%s' are running and ready (%d seconds elapsed)",
			nOk, len(podList.Items), ns, int(time.Since(start).Seconds()))
		Logf("expected %d pod replicas in namespace '%s', %d are Running and Ready.", replicas, ns, replicaOk)

		if replicaOk == replicas && nOk >= minPods && len(badPods) == 0 {
			return true, nil
		}
		ignoreNotReady = (notReady <= allowedNotReadyPods)
		logPodStates(badPods)
		return false, nil
	}) != nil {
		if !ignoreNotReady {
			return errors.New(errorBadPodsStates(badPods, desiredPods, ns, "RUNNING and READY", timeout))
		}
		Logf("Number of not-ready pods (%d) is below the allowed threshold (%d).", notReady, allowedNotReadyPods)
	}
	return nil
}

// WaitForDaemonSets for all daemonsets in the given namespace to be ready
// (defined as all but 'allowedNotReadyNodes' pods associated with that
// daemonset are ready).
func WaitForDaemonSets(c clientset.Interface, ns string, allowedNotReadyNodes int32, timeout time.Duration) error {
	start := time.Now()
	Logf("Waiting up to %v for all daemonsets in namespace '%s' to start",
		timeout, ns)

	return wait.PollImmediate(Poll, timeout, func() (bool, error) {
		dsList, err := c.AppsV1().DaemonSets(ns).List(metav1.ListOptions{})
		if err != nil {
			Logf("Error getting daemonsets in namespace: '%s': %v", ns, err)
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		var notReadyDaemonSets []string
		for _, ds := range dsList.Items {
			Logf("%d / %d pods ready in namespace '%s' in daemonset '%s' (%d seconds elapsed)", ds.Status.NumberReady, ds.Status.DesiredNumberScheduled, ns, ds.ObjectMeta.Name, int(time.Since(start).Seconds()))
			if ds.Status.DesiredNumberScheduled-ds.Status.NumberReady > allowedNotReadyNodes {
				notReadyDaemonSets = append(notReadyDaemonSets, ds.ObjectMeta.Name)
			}
		}

		if len(notReadyDaemonSets) > 0 {
			Logf("there are not ready daemonsets: %v", notReadyDaemonSets)
			return false, nil
		}

		return true, nil
	})
}

func kubectlLogPod(c clientset.Interface, pod v1.Pod, containerNameSubstr string, logFunc func(ftm string, args ...interface{})) {
	for _, container := range pod.Spec.Containers {
		if strings.Contains(container.Name, containerNameSubstr) {
			// Contains() matches all strings if substr is empty
			logs, err := GetPodLogs(c, pod.Namespace, pod.Name, container.Name)
			if err != nil {
				logs, err = getPreviousPodLogs(c, pod.Namespace, pod.Name, container.Name)
				if err != nil {
					logFunc("Failed to get logs of pod %v, container %v, err: %v", pod.Name, container.Name, err)
				}
			}
			logFunc("Logs of %v/%v:%v on node %v", pod.Namespace, pod.Name, container.Name, pod.Spec.NodeName)
			logFunc("%s : STARTLOG\n%s\nENDLOG for container %v:%v:%v", containerNameSubstr, logs, pod.Namespace, pod.Name, container.Name)
		}
	}
}

func LogFailedContainers(c clientset.Interface, ns string, logFunc func(ftm string, args ...interface{})) {
	podList, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{})
	if err != nil {
		logFunc("Error getting pods in namespace '%s': %v", ns, err)
		return
	}
	logFunc("Running kubectl logs on non-ready containers in %v", ns)
	for _, pod := range podList.Items {
		if res, err := testutils.PodRunningReady(&pod); !res || err != nil {
			kubectlLogPod(c, pod, "", Logf)
		}
	}
}

func LogPodsWithLabels(c clientset.Interface, ns string, match map[string]string, logFunc func(ftm string, args ...interface{})) {
	podList, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{LabelSelector: labels.SelectorFromSet(match).String()})
	if err != nil {
		logFunc("Error getting pods in namespace %q: %v", ns, err)
		return
	}
	logFunc("Running kubectl logs on pods with labels %v in %v", match, ns)
	for _, pod := range podList.Items {
		kubectlLogPod(c, pod, "", logFunc)
	}
}

func LogContainersInPodsWithLabels(c clientset.Interface, ns string, match map[string]string, containerSubstr string, logFunc func(ftm string, args ...interface{})) {
	podList, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{LabelSelector: labels.SelectorFromSet(match).String()})
	if err != nil {
		Logf("Error getting pods in namespace %q: %v", ns, err)
		return
	}
	for _, pod := range podList.Items {
		kubectlLogPod(c, pod, containerSubstr, logFunc)
	}
}

// DeleteNamespaces deletes all namespaces that match the given delete and skip filters.
// Filter is by simple strings.Contains; first skip filter, then delete filter.
// Returns the list of deleted namespaces or an error.
func DeleteNamespaces(c clientset.Interface, deleteFilter, skipFilter []string) ([]string, error) {
	By("Deleting namespaces")
	nsList, err := c.CoreV1().Namespaces().List(metav1.ListOptions{})
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
			Expect(c.CoreV1().Namespaces().Delete(nsName, nil)).To(Succeed())
			Logf("namespace : %v api call to delete is complete ", nsName)
		}(item.Name)
	}
	wg.Wait()
	return deleted, nil
}

func WaitForNamespacesDeleted(c clientset.Interface, namespaces []string, timeout time.Duration) error {
	By("Waiting for namespaces to vanish")
	nsMap := map[string]bool{}
	for _, ns := range namespaces {
		nsMap[ns] = true
	}
	//Now POLL until all namespaces have been eradicated.
	return wait.Poll(2*time.Second, timeout,
		func() (bool, error) {
			nsList, err := c.CoreV1().Namespaces().List(metav1.ListOptions{})
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

func waitForServiceAccountInNamespace(c clientset.Interface, ns, serviceAccountName string, timeout time.Duration) error {
	w, err := c.CoreV1().ServiceAccounts(ns).Watch(metav1.SingleObject(metav1.ObjectMeta{Name: serviceAccountName}))
	if err != nil {
		return err
	}
	ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), timeout)
	defer cancel()
	_, err = watchtools.UntilWithoutRetry(ctx, w, conditions.ServiceAccountHasSecrets)
	return err
}

func WaitForPodCondition(c clientset.Interface, ns, podName, desc string, timeout time.Duration, condition podCondition) error {
	Logf("Waiting up to %v for pod %q in namespace %q to be %q", timeout, podName, ns, desc)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pod, err := c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{})
		if err != nil {
			if apierrs.IsNotFound(err) {
				Logf("Pod %q in namespace %q not found. Error: %v", podName, ns, err)
				return err
			}
			Logf("Get pod %q in namespace %q failed, ignoring for %v. Error: %v", podName, ns, Poll, err)
			continue
		}
		// log now so that current pod info is reported before calling `condition()`
		Logf("Pod %q: Phase=%q, Reason=%q, readiness=%t. Elapsed: %v",
			podName, pod.Status.Phase, pod.Status.Reason, podutil.IsPodReady(pod), time.Since(start))
		if done, err := condition(pod); done {
			if err == nil {
				Logf("Pod %q satisfied condition %q", podName, desc)
			}
			return err
		}
	}
	return fmt.Errorf("Gave up after waiting %v for pod %q to be %q", timeout, podName, desc)
}

// WaitForMatchPodsCondition finds match pods based on the input ListOptions.
// waits and checks if all match pods are in the given podCondition
func WaitForMatchPodsCondition(c clientset.Interface, opts metav1.ListOptions, desc string, timeout time.Duration, condition podCondition) error {
	Logf("Waiting up to %v for matching pods' status to be %s", timeout, desc)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(opts)
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
		Logf("%d pods are not %s: %v", len(conditionNotMatch), desc, conditionNotMatch)
	}
	return fmt.Errorf("gave up waiting for matching pods to be '%s' after %v", desc, timeout)
}

// WaitForDefaultServiceAccountInNamespace waits for the default service account to be provisioned
// the default service account is what is associated with pods when they do not specify a service account
// as a result, pods are not able to be provisioned in a namespace until the service account is provisioned
func WaitForDefaultServiceAccountInNamespace(c clientset.Interface, namespace string) error {
	return waitForServiceAccountInNamespace(c, namespace, "default", ServiceAccountProvisionTimeout)
}

// WaitForPersistentVolumePhase waits for a PersistentVolume to be in a specific phase or until timeout occurs, whichever comes first.
func WaitForPersistentVolumePhase(phase v1.PersistentVolumePhase, c clientset.Interface, pvName string, Poll, timeout time.Duration) error {
	Logf("Waiting up to %v for PersistentVolume %s to have phase %s", timeout, pvName, phase)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pv, err := c.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
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

// WaitForStatefulSetReplicasReady waits for all replicas of a StatefulSet to become ready or until timeout occurs, whichever comes first.
func WaitForStatefulSetReplicasReady(statefulSetName, ns string, c clientset.Interface, Poll, timeout time.Duration) error {
	Logf("Waiting up to %v for StatefulSet %s to have all replicas ready", timeout, statefulSetName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		sts, err := c.AppsV1().StatefulSets(ns).Get(statefulSetName, metav1.GetOptions{})
		if err != nil {
			Logf("Get StatefulSet %s failed, ignoring for %v: %v", statefulSetName, Poll, err)
			continue
		} else {
			if sts.Status.ReadyReplicas == *sts.Spec.Replicas {
				Logf("All %d replicas of StatefulSet %s are ready. (%v)", sts.Status.ReadyReplicas, statefulSetName, time.Since(start))
				return nil
			} else {
				Logf("StatefulSet %s found but there are %d ready replicas and %d total replicas.", statefulSetName, sts.Status.ReadyReplicas, *sts.Spec.Replicas)
			}
		}
	}
	return fmt.Errorf("StatefulSet %s still has unready pods within %v", statefulSetName, timeout)
}

// WaitForPersistentVolumeDeleted waits for a PersistentVolume to get deleted or until timeout occurs, whichever comes first.
func WaitForPersistentVolumeDeleted(c clientset.Interface, pvName string, Poll, timeout time.Duration) error {
	Logf("Waiting up to %v for PersistentVolume %s to get deleted", timeout, pvName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pv, err := c.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
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
func WaitForPersistentVolumeClaimPhase(phase v1.PersistentVolumeClaimPhase, c clientset.Interface, ns string, pvcName string, Poll, timeout time.Duration) error {
	Logf("Waiting up to %v for PersistentVolumeClaim %s to have phase %s", timeout, pvcName, phase)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pvc, err := c.CoreV1().PersistentVolumeClaims(ns).Get(pvcName, metav1.GetOptions{})
		if err != nil {
			Logf("Failed to get claim %q, retrying in %v. Error: %v", pvcName, Poll, err)
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
func CreateTestingNS(baseName string, c clientset.Interface, labels map[string]string) (*v1.Namespace, error) {
	if labels == nil {
		labels = map[string]string{}
	}
	labels["e2e-run"] = string(RunId)

	namespaceObj := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: fmt.Sprintf("e2e-tests-%v-", baseName),
			Namespace:    "",
			Labels:       labels,
		},
		Status: v1.NamespaceStatus{},
	}
	// Be robust about making the namespace creation call.
	var got *v1.Namespace
	if err := wait.PollImmediate(Poll, 30*time.Second, func() (bool, error) {
		var err error
		got, err = c.CoreV1().Namespaces().Create(namespaceObj)
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
			// Even if we fail to create serviceAccount in the namespace,
			// we have successfully create a namespace.
			// So, return the created namespace.
			return got, err
		}
	}
	return got, nil
}

// CheckTestingNSDeletedExcept checks whether all e2e based existing namespaces are in the Terminating state
// and waits until they are finally deleted. It ignores namespace skip.
func CheckTestingNSDeletedExcept(c clientset.Interface, skip string) error {
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
		namespaces, err := c.CoreV1().Namespaces().List(metav1.ListOptions{})
		if err != nil {
			Logf("Listing namespaces failed: %v", err)
			continue
		}
		terminating := 0
		for _, ns := range namespaces.Items {
			if strings.HasPrefix(ns.ObjectMeta.Name, "e2e-tests-") && ns.ObjectMeta.Name != skip {
				if ns.Status.Phase == v1.NamespaceActive {
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
func deleteNS(c clientset.Interface, dynamicClient dynamic.Interface, namespace string, timeout time.Duration) error {
	startTime := time.Now()
	if err := c.CoreV1().Namespaces().Delete(namespace, nil); err != nil {
		return err
	}

	// wait for namespace to delete or timeout.
	err := wait.PollImmediate(2*time.Second, timeout, func() (bool, error) {
		if _, err := c.CoreV1().Namespaces().Get(namespace, metav1.GetOptions{}); err != nil {
			if apierrs.IsNotFound(err) {
				return true, nil
			}
			Logf("Error while waiting for namespace to be terminated: %v", err)
			return false, nil
		}
		return false, nil
	})

	// verify there is no more remaining content in the namespace
	remainingContent, cerr := hasRemainingContent(c, dynamicClient, namespace)
	if cerr != nil {
		return cerr
	}

	// if content remains, let's dump information about the namespace, and system for flake debugging.
	remainingPods := 0
	missingTimestamp := 0
	if remainingContent {
		// log information about namespace, and set of namespaces in api server to help flake detection
		logNamespace(c, namespace)
		logNamespaces(c, namespace)

		// if we can, check if there were pods remaining with no timestamp.
		remainingPods, missingTimestamp, _ = countRemainingPods(c, namespace)
	}

	// a timeout waiting for namespace deletion happened!
	if err != nil {
		// some content remains in the namespace
		if remainingContent {
			// pods remain
			if remainingPods > 0 {
				if missingTimestamp != 0 {
					// pods remained, but were not undergoing deletion (namespace controller is probably culprit)
					return fmt.Errorf("namespace %v was not deleted with limit: %v, pods remaining: %v, pods missing deletion timestamp: %v", namespace, err, remainingPods, missingTimestamp)
				}
				// but they were all undergoing deletion (kubelet is probably culprit, check NodeLost)
				return fmt.Errorf("namespace %v was not deleted with limit: %v, pods remaining: %v", namespace, err, remainingPods)
			}
			// other content remains (namespace controller is probably screwed up)
			return fmt.Errorf("namespace %v was not deleted with limit: %v, namespaced content other than pods remain", namespace, err)
		}
		// no remaining content, but namespace was not deleted (namespace controller is probably wedged)
		return fmt.Errorf("namespace %v was not deleted with limit: %v, namespace is empty but is not yet removed", namespace, err)
	}
	Logf("namespace %v deletion completed in %s", namespace, time.Since(startTime))
	return nil
}

// logNamespaces logs the number of namespaces by phase
// namespace is the namespace the test was operating against that failed to delete so it can be grepped in logs
func logNamespaces(c clientset.Interface, namespace string) {
	namespaceList, err := c.CoreV1().Namespaces().List(metav1.ListOptions{})
	if err != nil {
		Logf("namespace: %v, unable to list namespaces: %v", namespace, err)
		return
	}

	numActive := 0
	numTerminating := 0
	for _, namespace := range namespaceList.Items {
		if namespace.Status.Phase == v1.NamespaceActive {
			numActive++
		} else {
			numTerminating++
		}
	}
	Logf("namespace: %v, total namespaces: %v, active: %v, terminating: %v", namespace, len(namespaceList.Items), numActive, numTerminating)
}

// logNamespace logs detail about a namespace
func logNamespace(c clientset.Interface, namespace string) {
	ns, err := c.CoreV1().Namespaces().Get(namespace, metav1.GetOptions{})
	if err != nil {
		if apierrs.IsNotFound(err) {
			Logf("namespace: %v no longer exists", namespace)
			return
		}
		Logf("namespace: %v, unable to get namespace due to error: %v", namespace, err)
		return
	}
	Logf("namespace: %v, DeletionTimetamp: %v, Finalizers: %v, Phase: %v", ns.Name, ns.DeletionTimestamp, ns.Spec.Finalizers, ns.Status.Phase)
}

// countRemainingPods queries the server to count number of remaining pods, and number of pods that had a missing deletion timestamp.
func countRemainingPods(c clientset.Interface, namespace string) (int, int, error) {
	// check for remaining pods
	pods, err := c.CoreV1().Pods(namespace).List(metav1.ListOptions{})
	if err != nil {
		return 0, 0, err
	}

	// nothing remains!
	if len(pods.Items) == 0 {
		return 0, 0, nil
	}

	// stuff remains, log about it
	logPodStates(pods.Items)

	// check if there were any pods with missing deletion timestamp
	numPods := len(pods.Items)
	missingTimestamp := 0
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp == nil {
			missingTimestamp++
		}
	}
	return numPods, missingTimestamp, nil
}

// isDynamicDiscoveryError returns true if the error is a group discovery error
// only for groups expected to be created/deleted dynamically during e2e tests
func isDynamicDiscoveryError(err error) bool {
	if !discovery.IsGroupDiscoveryFailedError(err) {
		return false
	}
	discoveryErr := err.(*discovery.ErrGroupDiscoveryFailed)
	for gv := range discoveryErr.Groups {
		switch gv.Group {
		case "mygroup.example.com":
			// custom_resource_definition
			// garbage_collector
		case "wardle.k8s.io":
			// aggregator
		case "metrics.k8s.io":
			// aggregated metrics server add-on, no persisted resources
		default:
			Logf("discovery error for unexpected group: %#v", gv)
			return false
		}
	}
	return true
}

// hasRemainingContent checks if there is remaining content in the namespace via API discovery
func hasRemainingContent(c clientset.Interface, dynamicClient dynamic.Interface, namespace string) (bool, error) {
	// some tests generate their own framework.Client rather than the default
	// TODO: ensure every test call has a configured dynamicClient
	if dynamicClient == nil {
		return false, nil
	}

	// find out what content is supported on the server
	// Since extension apiserver is not always available, e.g. metrics server sometimes goes down,
	// add retry here.
	resources, err := waitForServerPreferredNamespacedResources(c.Discovery(), 30*time.Second)
	if err != nil {
		return false, err
	}
	groupVersionResources, err := discovery.GroupVersionResources(resources)
	if err != nil {
		return false, err
	}

	// TODO: temporary hack for https://github.com/kubernetes/kubernetes/issues/31798
	ignoredResources := sets.NewString("bindings")

	contentRemaining := false

	// dump how many of resource type is on the server in a log.
	for gvr := range groupVersionResources {
		// get a client for this group version...
		dynamicClient := dynamicClient.Resource(gvr).Namespace(namespace)
		if err != nil {
			// not all resource types support list, so some errors here are normal depending on the resource type.
			Logf("namespace: %s, unable to get client - gvr: %v, error: %v", namespace, gvr, err)
			continue
		}
		// get the api resource
		apiResource := metav1.APIResource{Name: gvr.Resource, Namespaced: true}
		if ignoredResources.Has(gvr.Resource) {
			Logf("namespace: %s, resource: %s, ignored listing per whitelist", namespace, apiResource.Name)
			continue
		}
		unstructuredList, err := dynamicClient.List(metav1.ListOptions{})
		if err != nil {
			// not all resources support list, so we ignore those
			if apierrs.IsMethodNotSupported(err) || apierrs.IsNotFound(err) || apierrs.IsForbidden(err) {
				continue
			}
			// skip unavailable servers
			if apierrs.IsServiceUnavailable(err) {
				continue
			}
			return false, err
		}
		if len(unstructuredList.Items) > 0 {
			Logf("namespace: %s, resource: %s, items remaining: %v", namespace, apiResource.Name, len(unstructuredList.Items))
			contentRemaining = true
		}
	}
	return contentRemaining, nil
}

func ContainerInitInvariant(older, newer runtime.Object) error {
	oldPod := older.(*v1.Pod)
	newPod := newer.(*v1.Pod)
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

func podInitialized(pod *v1.Pod) (ok bool, failed bool, err error) {
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

func initContainersInvariants(pod *v1.Pod) error {
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
	_, c := podutil.GetPodCondition(&pod.Status, v1.PodInitialized)
	if c == nil {
		return fmt.Errorf("pod does not have initialized condition")
	}
	if c.LastTransitionTime.IsZero() {
		return fmt.Errorf("PodInitialized condition should always have a transition time")
	}
	switch {
	case c.Status == v1.ConditionUnknown:
		return fmt.Errorf("PodInitialized condition should never be Unknown")
	case c.Status == v1.ConditionTrue && (initFailed || !allInit):
		return fmt.Errorf("PodInitialized condition was True but all not all containers initialized")
	case c.Status == v1.ConditionFalse && (!initFailed && allInit):
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
func WaitForPodRunningInNamespace(c clientset.Interface, pod *v1.Pod) error {
	if pod.Status.Phase == v1.PodRunning {
		return nil
	}
	return WaitTimeoutForPodRunningInNamespace(c, pod.Name, pod.Namespace, PodStartTimeout)
}

// Waits default amount of time (PodStartTimeout) for the specified pod to become running.
// Returns an error if timeout occurs first, or pod goes in to failed state.
func WaitForPodNameRunningInNamespace(c clientset.Interface, podName, namespace string) error {
	return WaitTimeoutForPodRunningInNamespace(c, podName, namespace, PodStartTimeout)
}

// Waits an extended amount of time (slowPodStartTimeout) for the specified pod to become running.
// The resourceVersion is used when Watching object changes, it tells since when we care
// about changes to the pod. Returns an error if timeout occurs first, or pod goes in to failed state.
func waitForPodRunningInNamespaceSlow(c clientset.Interface, podName, namespace string) error {
	return WaitTimeoutForPodRunningInNamespace(c, podName, namespace, slowPodStartTimeout)
}

func WaitTimeoutForPodRunningInNamespace(c clientset.Interface, podName, namespace string, timeout time.Duration) error {
	return wait.PollImmediate(Poll, timeout, podRunning(c, podName, namespace))
}

func podRunning(c clientset.Interface, podName, namespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		switch pod.Status.Phase {
		case v1.PodRunning:
			return true, nil
		case v1.PodFailed, v1.PodSucceeded:
			return false, conditions.ErrPodCompleted
		}
		return false, nil
	}
}

// Waits default amount of time (DefaultPodDeletionTimeout) for the specified pod to stop running.
// Returns an error if timeout occurs first.
func WaitForPodNoLongerRunningInNamespace(c clientset.Interface, podName, namespace string) error {
	return WaitTimeoutForPodNoLongerRunningInNamespace(c, podName, namespace, DefaultPodDeletionTimeout)
}

func WaitTimeoutForPodNoLongerRunningInNamespace(c clientset.Interface, podName, namespace string, timeout time.Duration) error {
	return wait.PollImmediate(Poll, timeout, podCompleted(c, podName, namespace))
}

func podCompleted(c clientset.Interface, podName, namespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(podName, metav1.GetOptions{})
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

func waitTimeoutForPodReadyInNamespace(c clientset.Interface, podName, namespace string, timeout time.Duration) error {
	return wait.PollImmediate(Poll, timeout, podRunningAndReady(c, podName, namespace))
}

func podRunningAndReady(c clientset.Interface, podName, namespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		switch pod.Status.Phase {
		case v1.PodFailed, v1.PodSucceeded:
			return false, conditions.ErrPodCompleted
		case v1.PodRunning:
			return podutil.IsPodReady(pod), nil
		}
		return false, nil
	}
}

// WaitForPodNotPending returns an error if it took too long for the pod to go out of pending state.
// The resourceVersion is used when Watching object changes, it tells since when we care
// about changes to the pod.
func WaitForPodNotPending(c clientset.Interface, ns, podName string) error {
	return wait.PollImmediate(Poll, PodStartTimeout, podNotPending(c, podName, ns))
}

func podNotPending(c clientset.Interface, podName, namespace string) wait.ConditionFunc {
	return func() (bool, error) {
		pod, err := c.CoreV1().Pods(namespace).Get(podName, metav1.GetOptions{})
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

// waitForPodTerminatedInNamespace returns an error if it takes too long for the pod to terminate,
// if the pod Get api returns an error (IsNotFound or other), or if the pod failed (and thus did not
// terminate) with an unexpected reason. Typically called to test that the passed-in pod is fully
// terminated (reason==""), but may be called to detect if a pod did *not* terminate according to
// the supplied reason.
func waitForPodTerminatedInNamespace(c clientset.Interface, podName, reason, namespace string) error {
	return WaitForPodCondition(c, namespace, podName, "terminated due to deadline exceeded", PodStartTimeout, func(pod *v1.Pod) (bool, error) {
		// Only consider Failed pods. Successful pods will be deleted and detected in
		// waitForPodCondition's Get call returning `IsNotFound`
		if pod.Status.Phase == v1.PodFailed {
			if pod.Status.Reason == reason { // short-circuit waitForPodCondition's loop
				return true, nil
			} else {
				return true, fmt.Errorf("Expected pod %q in namespace %q to be terminated with reason %q, got reason: %q", podName, namespace, reason, pod.Status.Reason)
			}
		}
		return false, nil
	})
}

// waitForPodNotFoundInNamespace returns an error if it takes too long for the pod to fully terminate.
// Unlike `waitForPodTerminatedInNamespace`, the pod's Phase and Reason are ignored. If the pod Get
// api returns IsNotFound then the wait stops and nil is returned. If the Get api returns an error other
// than "not found" then that error is returned and the wait stops.
func waitForPodNotFoundInNamespace(c clientset.Interface, podName, ns string, timeout time.Duration) error {
	return wait.PollImmediate(Poll, timeout, func() (bool, error) {
		_, err := c.CoreV1().Pods(ns).Get(podName, metav1.GetOptions{})
		if apierrs.IsNotFound(err) {
			return true, nil // done
		}
		if err != nil {
			return true, err // stop wait with error
		}
		return false, nil
	})
}

// waitForPodSuccessInNamespaceTimeout returns nil if the pod reached state success, or an error if it reached failure or ran too long.
func waitForPodSuccessInNamespaceTimeout(c clientset.Interface, podName string, namespace string, timeout time.Duration) error {
	return WaitForPodCondition(c, namespace, podName, "success or failure", timeout, func(pod *v1.Pod) (bool, error) {
		if pod.Spec.RestartPolicy == v1.RestartPolicyAlways {
			return true, fmt.Errorf("pod %q will never terminate with a succeeded state since its restart policy is Always", podName)
		}
		switch pod.Status.Phase {
		case v1.PodSucceeded:
			By("Saw pod success")
			return true, nil
		case v1.PodFailed:
			return true, fmt.Errorf("pod %q failed with status: %+v", podName, pod.Status)
		default:
			return false, nil
		}
	})
}

// WaitForPodSuccessInNamespace returns nil if the pod reached state success, or an error if it reached failure or until podStartupTimeout.
func WaitForPodSuccessInNamespace(c clientset.Interface, podName string, namespace string) error {
	return waitForPodSuccessInNamespaceTimeout(c, podName, namespace, PodStartTimeout)
}

// WaitForPodSuccessInNamespaceSlow returns nil if the pod reached state success, or an error if it reached failure or until slowPodStartupTimeout.
func WaitForPodSuccessInNamespaceSlow(c clientset.Interface, podName string, namespace string) error {
	return waitForPodSuccessInNamespaceTimeout(c, podName, namespace, slowPodStartTimeout)
}

// WaitForRCToStabilize waits till the RC has a matching generation/replica count between spec and status.
func WaitForRCToStabilize(c clientset.Interface, ns, name string, timeout time.Duration) error {
	options := metav1.ListOptions{FieldSelector: fields.Set{
		"metadata.name":      name,
		"metadata.namespace": ns,
	}.AsSelector().String()}
	w, err := c.CoreV1().ReplicationControllers(ns).Watch(options)
	if err != nil {
		return err
	}
	ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), timeout)
	defer cancel()
	_, err = watchtools.UntilWithoutRetry(ctx, w, func(event watch.Event) (bool, error) {
		switch event.Type {
		case watch.Deleted:
			return false, apierrs.NewNotFound(schema.GroupResource{Resource: "replicationcontrollers"}, "")
		}
		switch rc := event.Object.(type) {
		case *v1.ReplicationController:
			if rc.Name == name && rc.Namespace == ns &&
				rc.Generation <= rc.Status.ObservedGeneration &&
				*(rc.Spec.Replicas) == rc.Status.Replicas {
				return true, nil
			}
			Logf("Waiting for rc %s to stabilize, generation %v observed generation %v spec.replicas %d status.replicas %d",
				name, rc.Generation, rc.Status.ObservedGeneration, *(rc.Spec.Replicas), rc.Status.Replicas)
		}
		return false, nil
	})
	return err
}

func WaitForPodToDisappear(c clientset.Interface, ns, podName string, label labels.Selector, interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		Logf("Waiting for pod %s to disappear", podName)
		options := metav1.ListOptions{LabelSelector: label.String()}
		pods, err := c.CoreV1().Pods(ns).List(options)
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		found := false
		for _, pod := range pods.Items {
			if pod.Name == podName {
				Logf("Pod %s still exists", podName)
				found = true
				break
			}
		}
		if !found {
			Logf("Pod %s no longer exists", podName)
			return true, nil
		}
		return false, nil
	})
}

// WaitForPodNameUnschedulableInNamespace returns an error if it takes too long for the pod to become Pending
// and have condition Status equal to Unschedulable,
// if the pod Get api returns an error (IsNotFound or other), or if the pod failed with an unexpected reason.
// Typically called to test that the passed-in pod is Pending and Unschedulable.
func WaitForPodNameUnschedulableInNamespace(c clientset.Interface, podName, namespace string) error {
	return WaitForPodCondition(c, namespace, podName, "Unschedulable", PodStartTimeout, func(pod *v1.Pod) (bool, error) {
		// Only consider Failed pods. Successful pods will be deleted and detected in
		// waitForPodCondition's Get call returning `IsNotFound`
		if pod.Status.Phase == v1.PodPending {
			for _, cond := range pod.Status.Conditions {
				if cond.Type == v1.PodScheduled && cond.Status == v1.ConditionFalse && cond.Reason == "Unschedulable" {
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

// WaitForService waits until the service appears (exist == true), or disappears (exist == false)
func WaitForService(c clientset.Interface, namespace, name string, exist bool, interval, timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		_, err := c.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
		switch {
		case err == nil:
			Logf("Service %s in namespace %s found.", name, namespace)
			return exist, nil
		case apierrs.IsNotFound(err):
			Logf("Service %s in namespace %s disappeared.", name, namespace)
			return !exist, nil
		case !testutils.IsRetryableAPIError(err):
			Logf("Non-retryable failure while getting service.")
			return false, err
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

// WaitForServiceWithSelector waits until any service with given selector appears (exist == true), or disappears (exist == false)
func WaitForServiceWithSelector(c clientset.Interface, namespace string, selector labels.Selector, exist bool, interval,
	timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		services, err := c.CoreV1().Services(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
		switch {
		case len(services.Items) != 0:
			Logf("Service with %s in namespace %s found.", selector.String(), namespace)
			return exist, nil
		case len(services.Items) == 0:
			Logf("Service with %s in namespace %s disappeared.", selector.String(), namespace)
			return !exist, nil
		case !testutils.IsRetryableAPIError(err):
			Logf("Non-retryable failure while listing service.")
			return false, err
		default:
			Logf("List service with %s in namespace %s failed: %v", selector.String(), namespace, err)
			return false, nil
		}
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for service with %s in namespace %s %s: %v", selector.String(), namespace, stateMsg[exist], err)
	}
	return nil
}

//WaitForServiceEndpointsNum waits until the amount of endpoints that implement service to expectNum.
func WaitForServiceEndpointsNum(c clientset.Interface, namespace, serviceName string, expectNum int, interval, timeout time.Duration) error {
	return wait.Poll(interval, timeout, func() (bool, error) {
		Logf("Waiting for amount of service:%s endpoints to be %d", serviceName, expectNum)
		list, err := c.CoreV1().Endpoints(namespace).List(metav1.ListOptions{})
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

func countEndpointsNum(e *v1.Endpoints) int {
	num := 0
	for _, sub := range e.Subsets {
		num += len(sub.Addresses)
	}
	return num
}

func WaitForEndpoint(c clientset.Interface, ns, name string) error {
	for t := time.Now(); time.Since(t) < EndpointRegisterTimeout; time.Sleep(Poll) {
		endpoint, err := c.CoreV1().Endpoints(ns).Get(name, metav1.GetOptions{})
		if apierrs.IsNotFound(err) {
			Logf("Endpoint %s/%s is not ready yet", ns, name)
			continue
		}
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
// proxy) and verifying that they answer with there own pod name.
type podProxyResponseChecker struct {
	c              clientset.Interface
	ns             string
	label          labels.Selector
	controllerName string
	respondName    bool // Whether the pod should respond with its own name.
	pods           *v1.PodList
}

func PodProxyResponseChecker(c clientset.Interface, ns string, label labels.Selector, controllerName string, respondName bool, pods *v1.PodList) podProxyResponseChecker {
	return podProxyResponseChecker{c, ns, label, controllerName, respondName, pods}
}

// CheckAllResponses issues GETs to all pods in the context and verify they
// reply with their own pod name.
func (r podProxyResponseChecker) CheckAllResponses() (done bool, err error) {
	successes := 0
	options := metav1.ListOptions{LabelSelector: r.label.String()}
	currentPods, err := r.c.CoreV1().Pods(r.ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	for i, pod := range r.pods.Items {
		// Check that the replica list remains unchanged, otherwise we have problems.
		if !isElementOf(pod.UID, currentPods) {
			return false, fmt.Errorf("pod with UID %s is no longer a member of the replica set.  Must have been restarted for some reason.  Current replica set: %v", pod.UID, currentPods)
		}

		ctx, cancel := context.WithTimeout(context.Background(), SingleCallTimeout)
		defer cancel()

		body, err := r.c.CoreV1().RESTClient().Get().
			Context(ctx).
			Namespace(r.ns).
			Resource("pods").
			SubResource("proxy").
			Name(string(pod.Name)).
			Do().
			Raw()

		if err != nil {
			if ctx.Err() != nil {
				// We may encounter errors here because of a race between the pod readiness and apiserver
				// proxy. So, we log the error and retry if this occurs.
				Logf("Controller %s: Failed to Get from replica %d [%s]: %v\n pod status: %#v", r.controllerName, i+1, pod.Name, err, pod.Status)
				return false, nil
			}
			Logf("Controller %s: Failed to GET from replica %d [%s]: %v\npod status: %#v", r.controllerName, i+1, pod.Name, err, pod.Status)
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
func ServerVersionGTE(v *utilversion.Version, c discovery.ServerVersionInterface) (bool, error) {
	serverVersion, err := c.ServerVersion()
	if err != nil {
		return false, fmt.Errorf("Unable to get server version: %v", err)
	}
	sv, err := utilversion.ParseSemantic(serverVersion.GitVersion)
	if err != nil {
		return false, fmt.Errorf("Unable to parse server version %q: %v", serverVersion.GitVersion, err)
	}
	return sv.AtLeast(v), nil
}

func SkipUnlessKubectlVersionGTE(v *utilversion.Version) {
	gte, err := KubectlVersionGTE(v)
	if err != nil {
		Failf("Failed to get kubectl version: %v", err)
	}
	if !gte {
		Skipf("Not supported for kubectl versions before %q", v)
	}
}

// KubectlVersionGTE returns true if the kubectl version is greater than or
// equal to v.
func KubectlVersionGTE(v *utilversion.Version) (bool, error) {
	kv, err := KubectlVersion()
	if err != nil {
		return false, err
	}
	return kv.AtLeast(v), nil
}

// KubectlVersion gets the version of kubectl that's currently being used (see
// --kubectl-path in e2e.go to use an alternate kubectl).
func KubectlVersion() (*utilversion.Version, error) {
	output := RunKubectlOrDie("version", "--client")
	matches := gitVersionRegexp.FindStringSubmatch(output)
	if len(matches) != 2 {
		return nil, fmt.Errorf("Could not find kubectl version in output %v", output)
	}
	// Don't use the full match, as it contains "GitVersion:\"" and a
	// trailing "\"".  Just use the submatch.
	return utilversion.ParseSemantic(matches[1])
}

func PodsResponding(c clientset.Interface, ns, name string, wantName bool, pods *v1.PodList) error {
	By("trying to dial each unique pod")
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	return wait.PollImmediate(Poll, podRespondingTimeout, PodProxyResponseChecker(c, ns, label, name, wantName, pods).CheckAllResponses)
}

func PodsCreated(c clientset.Interface, ns, name string, replicas int32) (*v1.PodList, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	return PodsCreatedByLabel(c, ns, name, replicas, label)
}

func PodsCreatedByLabel(c clientset.Interface, ns, name string, replicas int32, label labels.Selector) (*v1.PodList, error) {
	timeout := 2 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		options := metav1.ListOptions{LabelSelector: label.String()}

		// List the pods, making sure we observe all the replicas.
		pods, err := c.CoreV1().Pods(ns).List(options)
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
		Logf("Pod name %s: Found %d pods out of %d", name, len(created), replicas)

		if int32(len(created)) == replicas {
			pods.Items = created
			return pods, nil
		}
	}
	return nil, fmt.Errorf("Pod name %s: Gave up waiting %v for %d pods to come up", name, timeout, replicas)
}

func podsRunning(c clientset.Interface, pods *v1.PodList) []error {
	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	By("ensuring each pod is running")
	e := []error{}
	error_chan := make(chan error)

	for _, pod := range pods.Items {
		go func(p v1.Pod) {
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

func VerifyPods(c clientset.Interface, ns, name string, wantName bool, replicas int32) error {
	return podRunningMaybeResponding(c, ns, name, wantName, replicas, true)
}

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

func ServiceResponding(c clientset.Interface, ns, name string) error {
	By(fmt.Sprintf("trying to dial the service %s.%s via the proxy", ns, name))

	return wait.PollImmediate(Poll, ServiceRespondingTimeout, func() (done bool, err error) {
		proxyRequest, errProxy := GetServicesProxyRequest(c, c.CoreV1().RESTClient().Get())
		if errProxy != nil {
			Logf("Failed to get services proxy request: %v:", errProxy)
			return false, nil
		}

		ctx, cancel := context.WithTimeout(context.Background(), SingleCallTimeout)
		defer cancel()

		body, err := proxyRequest.Namespace(ns).
			Context(ctx).
			Name(name).
			Do().
			Raw()
		if err != nil {
			if ctx.Err() != nil {
				Failf("Failed to GET from service %s: %v", name, err)
				return true, err
			}
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

func RestclientConfig(kubeContext string) (*clientcmdapi.Config, error) {
	Logf(">>> kubeConfig: %s", TestContext.KubeConfig)
	if TestContext.KubeConfig == "" {
		return nil, fmt.Errorf("KubeConfig must be specified to load client config")
	}
	c, err := clientcmd.LoadFromFile(TestContext.KubeConfig)
	if err != nil {
		return nil, fmt.Errorf("error loading KubeConfig: %v", err.Error())
	}
	if kubeContext != "" {
		Logf(">>> kubeContext: %s", kubeContext)
		c.CurrentContext = kubeContext
	}
	return c, nil
}

type ClientConfigGetter func() (*restclient.Config, error)

func LoadConfig() (*restclient.Config, error) {
	if TestContext.NodeE2E {
		// This is a node e2e test, apply the node e2e configuration
		return &restclient.Config{Host: TestContext.Host}, nil
	}
	c, err := RestclientConfig(TestContext.KubeContext)
	if err != nil {
		if TestContext.KubeConfig == "" {
			return restclient.InClusterConfig()
		} else {
			return nil, err
		}
	}

	return clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: TestContext.Host}}).ClientConfig()
}
func LoadInternalClientset() (*internalclientset.Clientset, error) {
	config, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	return internalclientset.NewForConfig(config)
}

func LoadClientset() (*clientset.Clientset, error) {
	config, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	return clientset.NewForConfig(config)
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
	ExpectNoErrorWithOffset(1, err, explain...)
}

// ExpectNoErrorWithOffset checks if "err" is set, and if so, fails assertion while logging the error at "offset" levels above its caller
// (for example, for call chain f -> g -> ExpectNoErrorWithOffset(1, ...) error would be logged for "f").
func ExpectNoErrorWithOffset(offset int, err error, explain ...interface{}) {
	if err != nil {
		Logf("Unexpected error occurred: %v", err)
	}
	ExpectWithOffset(1+offset, err).NotTo(HaveOccurred(), explain...)
}

func ExpectNoErrorWithRetries(fn func() error, maxRetries int, explain ...interface{}) {
	var err error
	for i := 0; i < maxRetries; i++ {
		err = fn()
		if err == nil {
			return
		}
		Logf("(Attempt %d of %d) Unexpected error occurred: %v", i+1, maxRetries, err)
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

	var e error
	verifyCleanupFunc := func() (bool, error) {
		e = nil
		for _, selector := range selectors {
			resources := RunKubectlOrDie("get", "rc,svc", "-l", selector, "--no-headers", nsArg)
			if resources != "" {
				e = fmt.Errorf("Resources left running after stop:\n%s", resources)
				return false, nil
			}
			pods := RunKubectlOrDie("get", "pods", "-l", selector, nsArg, "-o", "go-template={{ range .items }}{{ if not .metadata.deletionTimestamp }}{{ .metadata.name }}{{ \"\\n\" }}{{ end }}{{ end }}")
			if pods != "" {
				e = fmt.Errorf("Pods left unterminated after stop:\n%s", pods)
				return false, nil
			}
		}
		return true, nil
	}
	err := wait.PollImmediate(500*time.Millisecond, 1*time.Minute, verifyCleanupFunc)
	if err != nil {
		Failf(e.Error())
	}
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
		return "", fmt.Errorf("error starting %v:\nCommand stdout:\n%v\nstderr:\n%v\nerror:\n%v\n", cmd, cmd.Stdout, cmd.Stderr, err)
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Wait()
	}()
	select {
	case err := <-errCh:
		if err != nil {
			var rc int = 127
			if ee, ok := err.(*exec.ExitError); ok {
				rc = int(ee.Sys().(syscall.WaitStatus).ExitStatus())
				Logf("rc: %d", rc)
			}
			return "", uexec.CodeExitError{
				Err:  fmt.Errorf("error running %v:\nCommand stdout:\n%v\nstderr:\n%v\nerror:\n%v\n", cmd, cmd.Stdout, cmd.Stderr, err),
				Code: rc,
			}
		}
	case <-b.timeout:
		b.cmd.Process.Kill()
		return "", fmt.Errorf("timed out waiting for command %v:\nCommand stdout:\n%v\nstderr:\n%v\n", cmd, cmd.Stdout, cmd.Stderr)
	}
	Logf("stderr: %q", stderr.String())
	Logf("stdout: %q", stdout.String())
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

// RunKubemciWithKubeconfig is a convenience wrapper over RunKubemciCmd
func RunKubemciWithKubeconfig(args ...string) (string, error) {
	if TestContext.KubeConfig != "" {
		args = append(args, "--"+clientcmd.RecommendedConfigPathFlag+"="+TestContext.KubeConfig)
	}
	return RunKubemciCmd(args...)
}

// RunKubemciCmd is a convenience wrapper over kubectlBuilder to run kubemci.
// It assumes that kubemci exists in PATH.
func RunKubemciCmd(args ...string) (string, error) {
	// kubemci is assumed to be in PATH.
	kubemci := "kubemci"
	b := new(kubectlBuilder)
	args = append(args, "--gcp-project="+TestContext.CloudConfig.ProjectID)

	b.cmd = exec.Command(kubemci, args...)
	return b.Exec()
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
	pod *v1.Pod,
	containerIndex int,
	expectedOutput []string,
	matcher func(string, ...interface{}) gomegatypes.GomegaMatcher) {
	By(fmt.Sprintf("Creating a pod to test %v", scenarioName))
	if containerIndex < 0 || containerIndex >= len(pod.Spec.Containers) {
		Failf("Invalid container index: %d", containerIndex)
	}
	ExpectNoError(f.MatchContainerOutput(pod, pod.Spec.Containers[containerIndex].Name, expectedOutput, matcher))
}

// MatchContainerOutput creates a pod and waits for all it's containers to exit with success.
// It then tests that the matcher with each expectedOutput matches the output of the specified container.
func (f *Framework) MatchContainerOutput(
	pod *v1.Pod,
	containerName string,
	expectedOutput []string,
	matcher func(string, ...interface{}) gomegatypes.GomegaMatcher) error {
	ns := pod.ObjectMeta.Namespace
	if ns == "" {
		ns = f.Namespace.Name
	}
	podClient := f.PodClientNS(ns)

	createdPod := podClient.Create(pod)
	defer func() {
		By("delete the pod")
		podClient.DeleteSync(createdPod.Name, &metav1.DeleteOptions{}, DefaultPodDeletionTimeout)
	}()

	// Wait for client pod to complete.
	podErr := WaitForPodSuccessInNamespace(f.ClientSet, createdPod.Name, ns)

	// Grab its logs.  Get host first.
	podStatus, err := podClient.Get(createdPod.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get pod status: %v", err)
	}

	if podErr != nil {
		// Pod failed. Dump all logs from all containers to see what's wrong
		for _, container := range podStatus.Spec.Containers {
			logs, err := GetPodLogs(f.ClientSet, ns, podStatus.Name, container.Name)
			if err != nil {
				Logf("Failed to get logs from node %q pod %q container %q: %v",
					podStatus.Spec.NodeName, podStatus.Name, container.Name, err)
				continue
			}
			Logf("Output of node %q pod %q container %q: %s", podStatus.Spec.NodeName, podStatus.Name, container.Name, logs)
		}
		return fmt.Errorf("expected pod %q success: %v", createdPod.Name, podErr)
	}

	Logf("Trying to get logs from node %s pod %s container %s: %v",
		podStatus.Spec.NodeName, podStatus.Name, containerName, err)

	// Sometimes the actual containers take a second to get started, try to get logs for 60s
	logs, err := GetPodLogs(f.ClientSet, ns, podStatus.Name, containerName)
	if err != nil {
		Logf("Failed to get logs from node %q pod %q container %q. %v",
			podStatus.Spec.NodeName, podStatus.Name, containerName, err)
		return fmt.Errorf("failed to get logs from %s for %s: %v", podStatus.Name, containerName, err)
	}

	for _, expected := range expectedOutput {
		m := matcher(expected)
		matches, err := m.Match(logs)
		if err != nil {
			return fmt.Errorf("expected %q in container output: %v", expected, err)
		} else if !matches {
			return fmt.Errorf("expected %q in container output: %s", expected, m.FailureMessage(logs))
		}
	}

	return nil
}

type EventsLister func(opts metav1.ListOptions, ns string) (*v1.EventList, error)

func DumpEventsInNamespace(eventsLister EventsLister, namespace string) {
	By(fmt.Sprintf("Collecting events from namespace %q.", namespace))
	events, err := eventsLister(metav1.ListOptions{}, namespace)
	Expect(err).NotTo(HaveOccurred())

	By(fmt.Sprintf("Found %d events.", len(events.Items)))
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
}

func DumpAllNamespaceInfo(c clientset.Interface, namespace string) {
	DumpEventsInNamespace(func(opts metav1.ListOptions, ns string) (*v1.EventList, error) {
		return c.CoreV1().Events(ns).List(opts)
	}, namespace)

	// If cluster is large, then the following logs are basically useless, because:
	// 1. it takes tens of minutes or hours to grab all of them
	// 2. there are so many of them that working with them are mostly impossible
	// So we dump them only if the cluster is relatively small.
	maxNodesForDump := 20
	if nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{}); err == nil {
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
type byFirstTimestamp []v1.Event

func (o byFirstTimestamp) Len() int      { return len(o) }
func (o byFirstTimestamp) Swap(i, j int) { o[i], o[j] = o[j], o[i] }

func (o byFirstTimestamp) Less(i, j int) bool {
	if o[i].FirstTimestamp.Equal(&o[j].FirstTimestamp) {
		return o[i].InvolvedObject.Name < o[j].InvolvedObject.Name
	}
	return o[i].FirstTimestamp.Before(&o[j].FirstTimestamp)
}

func dumpAllPodInfo(c clientset.Interface) {
	pods, err := c.CoreV1().Pods("").List(metav1.ListOptions{})
	if err != nil {
		Logf("unable to fetch pod debug info: %v", err)
	}
	logPodStates(pods.Items)
}

func dumpAllNodeInfo(c clientset.Interface) {
	// It should be OK to list unschedulable Nodes here.
	nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		Logf("unable to fetch node list: %v", err)
		return
	}
	names := make([]string, len(nodes.Items))
	for ix := range nodes.Items {
		names[ix] = nodes.Items[ix].Name
	}
	DumpNodeDebugInfo(c, names, Logf)
}

func DumpNodeDebugInfo(c clientset.Interface, nodeNames []string, logFunc func(fmt string, args ...interface{})) {
	for _, n := range nodeNames {
		logFunc("\nLogging node info for node %v", n)
		node, err := c.CoreV1().Nodes().Get(n, metav1.GetOptions{})
		if err != nil {
			logFunc("Error getting node info %v", err)
		}
		logFunc("Node Info: %v", node)

		logFunc("\nLogging kubelet events for node %v", n)
		for _, e := range getNodeEvents(c, n) {
			logFunc("source %v type %v message %v reason %v first ts %v last ts %v, involved obj %+v",
				e.Source, e.Type, e.Message, e.Reason, e.FirstTimestamp, e.LastTimestamp, e.InvolvedObject)
		}
		logFunc("\nLogging pods the kubelet thinks is on node %v", n)
		podList, err := GetKubeletPods(c, n)
		if err != nil {
			logFunc("Unable to retrieve kubelet pods for node %v: %v", n, err)
			continue
		}
		for _, p := range podList.Items {
			logFunc("%v started at %v (%d+%d container statuses recorded)", p.Name, p.Status.StartTime, len(p.Status.InitContainerStatuses), len(p.Status.ContainerStatuses))
			for _, c := range p.Status.InitContainerStatuses {
				logFunc("\tInit container %v ready: %v, restart count %v",
					c.Name, c.Ready, c.RestartCount)
			}
			for _, c := range p.Status.ContainerStatuses {
				logFunc("\tContainer %v ready: %v, restart count %v",
					c.Name, c.Ready, c.RestartCount)
			}
		}
		HighLatencyKubeletOperations(c, 10*time.Second, n, logFunc)
		// TODO: Log node resource info
	}
}

// logNodeEvents logs kubelet events from the given node. This includes kubelet
// restart and node unhealthy events. Note that listing events like this will mess
// with latency metrics, beware of calling it during a test.
func getNodeEvents(c clientset.Interface, nodeName string) []v1.Event {
	selector := fields.Set{
		"involvedObject.kind":      "Node",
		"involvedObject.name":      nodeName,
		"involvedObject.namespace": metav1.NamespaceAll,
		"source":                   "kubelet",
	}.AsSelector().String()
	options := metav1.ListOptions{FieldSelector: selector}
	events, err := c.CoreV1().Events(metav1.NamespaceSystem).List(options)
	if err != nil {
		Logf("Unexpected error retrieving node events %v", err)
		return []v1.Event{}
	}
	return events.Items
}

// waitListSchedulableNodes is a wrapper around listing nodes supporting retries.
func waitListSchedulableNodes(c clientset.Interface) (*v1.NodeList, error) {
	var nodes *v1.NodeList
	var err error
	if wait.PollImmediate(Poll, SingleCallTimeout, func() (bool, error) {
		nodes, err = c.CoreV1().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		return true, nil
	}) != nil {
		return nodes, err
	}
	return nodes, nil
}

// waitListSchedulableNodesOrDie is a wrapper around listing nodes supporting retries.
func waitListSchedulableNodesOrDie(c clientset.Interface) *v1.NodeList {
	nodes, err := waitListSchedulableNodes(c)
	if err != nil {
		ExpectNoError(err, "Non-retryable failure or timed out while listing nodes for e2e cluster.")
	}
	return nodes
}

// Node is schedulable if:
// 1) doesn't have "unschedulable" field set
// 2) it's Ready condition is set to true
// 3) doesn't have NetworkUnavailable condition set to true
func isNodeSchedulable(node *v1.Node) bool {
	nodeReady := IsNodeConditionSetAsExpected(node, v1.NodeReady, true)
	networkReady := IsNodeConditionUnset(node, v1.NodeNetworkUnavailable) ||
		IsNodeConditionSetAsExpectedSilent(node, v1.NodeNetworkUnavailable, false)
	return !node.Spec.Unschedulable && nodeReady && networkReady
}

// Test whether a fake pod can be scheduled on "node", given its current taints.
func isNodeUntainted(node *v1.Node) bool {
	fakePod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "fake-not-scheduled",
			Namespace: "fake-not-scheduled",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "fake-not-scheduled",
					Image: "fake-not-scheduled",
				},
			},
		},
	}
	nodeInfo := schedulercache.NewNodeInfo()
	nodeInfo.SetNode(node)
	fit, _, err := predicates.PodToleratesNodeTaints(fakePod, nil, nodeInfo)
	if err != nil {
		Failf("Can't test predicates for node %s: %v", node.Name, err)
		return false
	}
	return fit
}

// GetReadySchedulableNodesOrDie addresses the common use case of getting nodes you can do work on.
// 1) Needs to be schedulable.
// 2) Needs to be ready.
// If EITHER 1 or 2 is not true, most tests will want to ignore the node entirely.
func GetReadySchedulableNodesOrDie(c clientset.Interface) (nodes *v1.NodeList) {
	nodes = waitListSchedulableNodesOrDie(c)
	// previous tests may have cause failures of some nodes. Let's skip
	// 'Not Ready' nodes, just in case (there is no need to fail the test).
	FilterNodes(nodes, func(node v1.Node) bool {
		return isNodeSchedulable(&node) && isNodeUntainted(&node)
	})
	return nodes
}

// GetReadyNodesIncludingTaintedOrDie returns all ready nodes, even those which are tainted.
// There are cases when we care about tainted nodes
// E.g. in tests related to nodes with gpu we care about nodes despite
// presence of nvidia.com/gpu=present:NoSchedule taint
func GetReadyNodesIncludingTaintedOrDie(c clientset.Interface) (nodes *v1.NodeList) {
	nodes = waitListSchedulableNodesOrDie(c)
	FilterNodes(nodes, func(node v1.Node) bool {
		return isNodeSchedulable(&node)
	})
	return nodes
}

// WaitForAllNodesSchedulable waits up to timeout for all
// (but TestContext.AllowedNotReadyNodes) to become scheduable.
func WaitForAllNodesSchedulable(c clientset.Interface, timeout time.Duration) error {
	Logf("Waiting up to %v for all (but %d) nodes to be schedulable", timeout, TestContext.AllowedNotReadyNodes)

	var notSchedulable []*v1.Node
	attempt := 0
	return wait.PollImmediate(30*time.Second, timeout, func() (bool, error) {
		attempt++
		notSchedulable = nil
		opts := metav1.ListOptions{
			ResourceVersion: "0",
			FieldSelector:   fields.Set{"spec.unschedulable": "false"}.AsSelector().String(),
		}
		nodes, err := c.CoreV1().Nodes().List(opts)
		if err != nil {
			Logf("Unexpected error listing nodes: %v", err)
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		for i := range nodes.Items {
			node := &nodes.Items[i]
			if _, hasMasterRoleLabel := node.ObjectMeta.Labels[service.LabelNodeRoleMaster]; hasMasterRoleLabel {
				// Kops clusters have masters with spec.unscheduable = false and
				// node-role.kubernetes.io/master NoSchedule taint.
				// Don't wait for them.
				continue
			}
			if !isNodeSchedulable(node) || !isNodeUntainted(node) {
				notSchedulable = append(notSchedulable, node)
			}
		}
		// Framework allows for <TestContext.AllowedNotReadyNodes> nodes to be non-ready,
		// to make it possible e.g. for incorrect deployment of some small percentage
		// of nodes (which we allow in cluster validation). Some nodes that are not
		// provisioned correctly at startup will never become ready (e.g. when something
		// won't install correctly), so we can't expect them to be ready at any point.
		//
		// However, we only allow non-ready nodes with some specific reasons.
		if len(notSchedulable) > 0 {
			// In large clusters, log them only every 10th pass.
			if len(nodes.Items) >= largeClusterThreshold && attempt%10 == 0 {
				Logf("Unschedulable nodes:")
				for i := range notSchedulable {
					Logf("-> %s Ready=%t Network=%t Taints=%v",
						notSchedulable[i].Name,
						IsNodeConditionSetAsExpectedSilent(notSchedulable[i], v1.NodeReady, true),
						IsNodeConditionSetAsExpectedSilent(notSchedulable[i], v1.NodeNetworkUnavailable, false),
						notSchedulable[i].Spec.Taints)
				}
				Logf("================================")
			}
		}
		return len(notSchedulable) <= TestContext.AllowedNotReadyNodes, nil
	})
}

func GetPodSecretUpdateTimeout(c clientset.Interface) time.Duration {
	// With SecretManager(ConfigMapManager), we may have to wait up to full sync period +
	// TTL of secret(configmap) to elapse before the Kubelet projects the update into the
	// volume and the container picks it up.
	// So this timeout is based on default Kubelet sync period (1 minute) + maximum TTL for
	// secret(configmap) that's based on cluster size + additional time as a fudge factor.
	secretTTL, err := GetNodeTTLAnnotationValue(c)
	if err != nil {
		Logf("Couldn't get node TTL annotation (using default value of 0): %v", err)
	}
	podLogTimeout := 240*time.Second + secretTTL
	return podLogTimeout
}

func GetNodeTTLAnnotationValue(c clientset.Interface) (time.Duration, error) {
	nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
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

func AddOrUpdateLabelOnNode(c clientset.Interface, nodeName string, labelKey, labelValue string) {
	ExpectNoError(testutils.AddLabelsToNode(c, nodeName, map[string]string{labelKey: labelValue}))
}

func AddOrUpdateLabelOnNodeAndReturnOldValue(c clientset.Interface, nodeName string, labelKey, labelValue string) string {
	var oldValue string
	node, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	ExpectNoError(err)
	oldValue = node.Labels[labelKey]
	ExpectNoError(testutils.AddLabelsToNode(c, nodeName, map[string]string{labelKey: labelValue}))
	return oldValue
}

func ExpectNodeHasLabel(c clientset.Interface, nodeName string, labelKey string, labelValue string) {
	By("verifying the node has the label " + labelKey + " " + labelValue)
	node, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	ExpectNoError(err)
	Expect(node.Labels[labelKey]).To(Equal(labelValue))
}

func RemoveTaintOffNode(c clientset.Interface, nodeName string, taint v1.Taint) {
	ExpectNoError(controller.RemoveTaintOffNode(c, nodeName, nil, &taint))
	VerifyThatTaintIsGone(c, nodeName, &taint)
}

func AddOrUpdateTaintOnNode(c clientset.Interface, nodeName string, taint v1.Taint) {
	ExpectNoError(controller.AddOrUpdateTaintOnNode(c, nodeName, &taint))
}

// RemoveLabelOffNode is for cleaning up labels temporarily added to node,
// won't fail if target label doesn't exist or has been removed.
func RemoveLabelOffNode(c clientset.Interface, nodeName string, labelKey string) {
	By("removing the label " + labelKey + " off the node " + nodeName)
	ExpectNoError(testutils.RemoveLabelOffNode(c, nodeName, []string{labelKey}))

	By("verifying the node doesn't have the label " + labelKey)
	ExpectNoError(testutils.VerifyLabelsRemoved(c, nodeName, []string{labelKey}))
}

func VerifyThatTaintIsGone(c clientset.Interface, nodeName string, taint *v1.Taint) {
	By("verifying the node doesn't have the taint " + taint.ToString())
	nodeUpdated, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	ExpectNoError(err)
	if taintutils.TaintExists(nodeUpdated.Spec.Taints, taint) {
		Failf("Failed removing taint " + taint.ToString() + " of the node " + nodeName)
	}
}

func ExpectNodeHasTaint(c clientset.Interface, nodeName string, taint *v1.Taint) {
	By("verifying the node has the taint " + taint.ToString())
	if has, err := NodeHasTaint(c, nodeName, taint); !has {
		ExpectNoError(err)
		Failf("Failed to find taint %s on node %s", taint.ToString(), nodeName)
	}
}

func NodeHasTaint(c clientset.Interface, nodeName string, taint *v1.Taint) (bool, error) {
	node, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	if err != nil {
		return false, err
	}

	nodeTaints := node.Spec.Taints

	if len(nodeTaints) == 0 || !taintutils.TaintExists(nodeTaints, taint) {
		return false, nil
	}
	return true, nil
}

//AddOrUpdateAvoidPodOnNode adds avoidPods annotations to node, will override if it exists
func AddOrUpdateAvoidPodOnNode(c clientset.Interface, nodeName string, avoidPods v1.AvoidPods) {
	err := wait.PollImmediate(Poll, SingleCallTimeout, func() (bool, error) {
		node, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}

		taintsData, err := json.Marshal(avoidPods)
		ExpectNoError(err)

		if node.Annotations == nil {
			node.Annotations = make(map[string]string)
		}
		node.Annotations[v1.PreferAvoidPodsAnnotationKey] = string(taintsData)
		_, err = c.CoreV1().Nodes().Update(node)
		if err != nil {
			if !apierrs.IsConflict(err) {
				ExpectNoError(err)
			} else {
				Logf("Conflict when trying to add/update avoidPonds %v to %v", avoidPods, nodeName)
			}
		}
		return true, nil
	})
	ExpectNoError(err)
}

//RemoveAnnotationOffNode removes AvoidPods annotations from the node. It does not fail if no such annotation exists.
func RemoveAvoidPodsOffNode(c clientset.Interface, nodeName string) {
	err := wait.PollImmediate(Poll, SingleCallTimeout, func() (bool, error) {
		node, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}

		if node.Annotations == nil {
			return true, nil
		}
		delete(node.Annotations, v1.PreferAvoidPodsAnnotationKey)
		_, err = c.CoreV1().Nodes().Update(node)
		if err != nil {
			if !apierrs.IsConflict(err) {
				ExpectNoError(err)
			} else {
				Logf("Conflict when trying to remove avoidPods to %v", nodeName)
			}
		}
		return true, nil
	})
	ExpectNoError(err)
}

func ScaleResource(
	clientset clientset.Interface,
	scalesGetter scaleclient.ScalesGetter,
	ns, name string,
	size uint,
	wait bool,
	kind schema.GroupKind,
	gr schema.GroupResource,
) error {
	By(fmt.Sprintf("Scaling %v %s in namespace %s to %d", kind, name, ns, size))
	if err := testutils.ScaleResourceWithRetries(scalesGetter, ns, name, size, gr); err != nil {
		return fmt.Errorf("error while scaling RC %s to %d replicas: %v", name, size, err)
	}
	if !wait {
		return nil
	}
	return WaitForControlledPodsRunning(clientset, ns, name, kind)
}

// Wait up to 10 minutes for pods to become Running.
func WaitForControlledPodsRunning(c clientset.Interface, ns, name string, kind schema.GroupKind) error {
	rtObject, err := getRuntimeObjectForKind(c, kind, ns, name)
	if err != nil {
		return err
	}
	selector, err := getSelectorFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}
	replicas, err := getReplicasFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}
	err = testutils.WaitForEnoughPodsWithLabelRunning(c, ns, selector, int(replicas))
	if err != nil {
		return fmt.Errorf("Error while waiting for replication controller %s pods to be running: %v", name, err)
	}
	return nil
}

// Wait up to PodListTimeout for getting pods of the specified controller name and return them.
func WaitForControlledPods(c clientset.Interface, ns, name string, kind schema.GroupKind) (pods *v1.PodList, err error) {
	rtObject, err := getRuntimeObjectForKind(c, kind, ns, name)
	if err != nil {
		return nil, err
	}
	selector, err := getSelectorFromRuntimeObject(rtObject)
	if err != nil {
		return nil, err
	}
	return WaitForPodsWithLabel(c, ns, selector)
}

// Returns true if all the specified pods are scheduled, else returns false.
func podsWithLabelScheduled(c clientset.Interface, ns string, label labels.Selector) (bool, error) {
	ps, err := testutils.NewPodStore(c, ns, label, fields.Everything())
	if err != nil {
		return false, err
	}
	defer ps.Stop()
	pods := ps.List()
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
func WaitForPodsWithLabelScheduled(c clientset.Interface, ns string, label labels.Selector) (pods *v1.PodList, err error) {
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
func WaitForPodsWithLabel(c clientset.Interface, ns string, label labels.Selector) (pods *v1.PodList, err error) {
	for t := time.Now(); time.Since(t) < PodListTimeout; time.Sleep(Poll) {
		options := metav1.ListOptions{LabelSelector: label.String()}
		pods, err = c.CoreV1().Pods(ns).List(options)
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				continue
			}
			return
		}
		if len(pods.Items) > 0 {
			break
		}
	}
	if pods == nil || len(pods.Items) == 0 {
		err = fmt.Errorf("Timeout while waiting for pods with label %v", label)
	}
	return
}

// Wait for exact amount of matching pods to become running and ready.
// Return the list of matching pods.
func WaitForPodsWithLabelRunningReady(c clientset.Interface, ns string, label labels.Selector, num int, timeout time.Duration) (pods *v1.PodList, err error) {
	var current int
	err = wait.Poll(Poll, timeout,
		func() (bool, error) {
			pods, err := WaitForPodsWithLabel(c, ns, label)
			if err != nil {
				Logf("Failed to list pods: %v", err)
				if testutils.IsRetryableAPIError(err) {
					return false, nil
				}
				return false, err
			}
			current = 0
			for _, pod := range pods.Items {
				if flag, err := testutils.PodRunningReady(&pod); err == nil && flag == true {
					current++
				}
			}
			if current != num {
				Logf("Got %v pods running and ready, expect: %v", current, num)
				return false, nil
			}
			return true, nil
		})
	return pods, err
}

func getRuntimeObjectForKind(c clientset.Interface, kind schema.GroupKind, ns, name string) (runtime.Object, error) {
	switch kind {
	case api.Kind("ReplicationController"):
		return c.CoreV1().ReplicationControllers(ns).Get(name, metav1.GetOptions{})
	case extensionsinternal.Kind("ReplicaSet"), appsinternal.Kind("ReplicaSet"):
		return c.ExtensionsV1beta1().ReplicaSets(ns).Get(name, metav1.GetOptions{})
	case extensionsinternal.Kind("Deployment"), appsinternal.Kind("Deployment"):
		return c.ExtensionsV1beta1().Deployments(ns).Get(name, metav1.GetOptions{})
	case extensionsinternal.Kind("DaemonSet"):
		return c.ExtensionsV1beta1().DaemonSets(ns).Get(name, metav1.GetOptions{})
	case batchinternal.Kind("Job"):
		return c.BatchV1().Jobs(ns).Get(name, metav1.GetOptions{})
	default:
		return nil, fmt.Errorf("Unsupported kind when getting runtime object: %v", kind)
	}
}

func getSelectorFromRuntimeObject(obj runtime.Object) (labels.Selector, error) {
	switch typed := obj.(type) {
	case *v1.ReplicationController:
		return labels.SelectorFromSet(typed.Spec.Selector), nil
	case *extensions.ReplicaSet:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	case *extensions.Deployment:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	case *extensions.DaemonSet:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	case *batch.Job:
		return metav1.LabelSelectorAsSelector(typed.Spec.Selector)
	default:
		return nil, fmt.Errorf("Unsupported kind when getting selector: %v", obj)
	}
}

func getReplicasFromRuntimeObject(obj runtime.Object) (int32, error) {
	switch typed := obj.(type) {
	case *v1.ReplicationController:
		if typed.Spec.Replicas != nil {
			return *typed.Spec.Replicas, nil
		}
		return 0, nil
	case *extensions.ReplicaSet:
		if typed.Spec.Replicas != nil {
			return *typed.Spec.Replicas, nil
		}
		return 0, nil
	case *extensions.Deployment:
		if typed.Spec.Replicas != nil {
			return *typed.Spec.Replicas, nil
		}
		return 0, nil
	case *extensions.DaemonSet:
		return 0, nil
	case *batch.Job:
		// TODO: currently we use pause pods so that's OK. When we'll want to switch to Pods
		// that actually finish we need a better way to do this.
		if typed.Spec.Parallelism != nil {
			return *typed.Spec.Parallelism, nil
		}
		return 0, nil
	default:
		return -1, fmt.Errorf("Unsupported kind when getting number of replicas: %v", obj)
	}
}

// DeleteResourceAndWaitForGC deletes only given resource and waits for GC to delete the pods.
func DeleteResourceAndWaitForGC(c clientset.Interface, kind schema.GroupKind, ns, name string) error {
	By(fmt.Sprintf("deleting %v %s in namespace %s, will wait for the garbage collector to delete the pods", kind, name, ns))

	rtObject, err := getRuntimeObjectForKind(c, kind, ns, name)
	if err != nil {
		if apierrs.IsNotFound(err) {
			Logf("%v %s not found: %v", kind, name, err)
			return nil
		}
		return err
	}
	selector, err := getSelectorFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}
	replicas, err := getReplicasFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}

	ps, err := testutils.NewPodStore(c, ns, selector, fields.Everything())
	if err != nil {
		return err
	}

	defer ps.Stop()
	falseVar := false
	deleteOption := &metav1.DeleteOptions{OrphanDependents: &falseVar}
	startTime := time.Now()
	if err := testutils.DeleteResourceWithRetries(c, kind, ns, name, deleteOption); err != nil {
		return err
	}
	deleteTime := time.Since(startTime)
	Logf("Deleting %v %s took: %v", kind, name, deleteTime)

	var interval, timeout time.Duration
	switch {
	case replicas < 100:
		interval = 100 * time.Millisecond
	case replicas < 1000:
		interval = 1 * time.Second
	default:
		interval = 10 * time.Second
	}
	if replicas < 5000 {
		timeout = 10 * time.Minute
	} else {
		timeout = time.Duration(replicas/gcThroughput) * time.Second
		// gcThroughput is pretty strict now, add a bit more to it
		timeout = timeout + 3*time.Minute
	}

	err = waitForPodsInactive(ps, interval, timeout)
	if err != nil {
		return fmt.Errorf("error while waiting for pods to become inactive %s: %v", name, err)
	}
	terminatePodTime := time.Since(startTime) - deleteTime
	Logf("Terminating %v %s pods took: %v", kind, name, terminatePodTime)

	err = waitForPodsGone(ps, interval, 10*time.Minute)
	if err != nil {
		return fmt.Errorf("error while waiting for pods gone %s: %v", name, err)
	}
	return nil
}

// waitForPodsInactive waits until there are no active pods left in the PodStore.
// This is to make a fair comparison of deletion time between DeleteRCAndPods
// and DeleteRCAndWaitForGC, because the RC controller decreases status.replicas
// when the pod is inactvie.
func waitForPodsInactive(ps *testutils.PodStore, interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		pods := ps.List()
		for _, pod := range pods {
			if controller.IsPodActive(pod) {
				return false, nil
			}
		}
		return true, nil
	})
}

// waitForPodsGone waits until there are no pods left in the PodStore.
func waitForPodsGone(ps *testutils.PodStore, interval, timeout time.Duration) error {
	return wait.PollImmediate(interval, timeout, func() (bool, error) {
		if pods := ps.List(); len(pods) == 0 {
			return true, nil
		}
		return false, nil
	})
}

func WaitForPodsReady(c clientset.Interface, ns, name string, minReadySeconds int) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	options := metav1.ListOptions{LabelSelector: label.String()}
	return wait.Poll(Poll, 5*time.Minute, func() (bool, error) {
		pods, err := c.CoreV1().Pods(ns).List(options)
		if err != nil {
			return false, nil
		}
		for _, pod := range pods.Items {
			if !podutil.IsPodAvailable(&pod, int32(minReadySeconds), metav1.Now()) {
				return false, nil
			}
		}
		return true, nil
	})
}

// Waits for the number of events on the given object to reach a desired count.
func WaitForEvents(c clientset.Interface, ns string, objOrRef runtime.Object, desiredEventsCount int) error {
	return wait.Poll(Poll, 5*time.Minute, func() (bool, error) {
		events, err := c.CoreV1().Events(ns).Search(legacyscheme.Scheme, objOrRef)
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
func WaitForPartialEvents(c clientset.Interface, ns string, objOrRef runtime.Object, atLeastEventsCount int) error {
	return wait.Poll(Poll, 5*time.Minute, func() (bool, error) {
		events, err := c.CoreV1().Events(ns).Search(legacyscheme.Scheme, objOrRef)
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

type updateDSFunc func(*apps.DaemonSet)

func UpdateDaemonSetWithRetries(c clientset.Interface, namespace, name string, applyUpdate updateDSFunc) (ds *apps.DaemonSet, err error) {
	daemonsets := c.AppsV1().DaemonSets(namespace)
	var updateErr error
	pollErr := wait.PollImmediate(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		if ds, err = daemonsets.Get(name, metav1.GetOptions{}); err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(ds)
		if ds, err = daemonsets.Update(ds); err == nil {
			Logf("Updating DaemonSet %s", name)
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("couldn't apply the provided updated to DaemonSet %q: %v", name, updateErr)
	}
	return ds, pollErr
}

// NodeAddresses returns the first address of the given type of each node.
func NodeAddresses(nodelist *v1.NodeList, addrType v1.NodeAddressType) []string {
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
func NodeSSHHosts(c clientset.Interface) ([]string, error) {
	nodelist := waitListSchedulableNodesOrDie(c)

	// TODO(roberthbailey): Use the "preferred" address for the node, once such a thing is defined (#2462).
	hosts := NodeAddresses(nodelist, v1.NodeExternalIP)

	// Error if any node didn't have an external IP.
	if len(hosts) != len(nodelist.Items) {
		return hosts, fmt.Errorf(
			"only found %d external IPs on nodes, but found %d nodes. Nodelist: %v",
			len(hosts), len(nodelist.Items), nodelist)
	}

	sshHosts := make([]string, 0, len(hosts))
	for _, h := range hosts {
		sshHosts = append(sshHosts, net.JoinHostPort(h, sshPort))
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

// NodeExec execs the given cmd on node via SSH. Note that the nodeName is an sshable name,
// eg: the name returned by framework.GetMasterHost(). This is also not guaranteed to work across
// cloud providers since it involves ssh.
func NodeExec(nodeName, cmd string) (SSHResult, error) {
	return SSH(cmd, net.JoinHostPort(nodeName, sshPort), TestContext.Provider)
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

func IssueSSHCommandWithResult(cmd, provider string, node *v1.Node) (*SSHResult, error) {
	Logf("Getting external IP address for %s", node.Name)
	host := ""
	for _, a := range node.Status.Addresses {
		if a.Type == v1.NodeExternalIP {
			host = net.JoinHostPort(a.Address, sshPort)
			break
		}
	}

	if host == "" {
		// No external IPs were found, let's try to use internal as plan B
		for _, a := range node.Status.Addresses {
			if a.Type == v1.NodeInternalIP {
				host = net.JoinHostPort(a.Address, sshPort)
				break
			}
		}
	}

	if host == "" {
		return nil, fmt.Errorf("couldn't find any IP address for node %s", node.Name)
	}

	Logf("SSH %q on %s(%s)", cmd, node.Name, host)
	result, err := SSH(cmd, host, provider)
	LogSSHResult(result)

	if result.Code != 0 || err != nil {
		return nil, fmt.Errorf("failed running %q: %v (exit code %d)",
			cmd, err, result.Code)
	}

	return &result, nil
}

func IssueSSHCommand(cmd, provider string, node *v1.Node) error {
	_, err := IssueSSHCommandWithResult(cmd, provider, node)
	if err != nil {
		return err
	}
	return nil
}

// NewHostExecPodSpec returns the pod spec of hostexec pod
func NewHostExecPodSpec(ns, name string) *v1.Pod {
	immediate := int64(0)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "hostexec",
					Image:           imageutils.GetE2EImage(imageutils.Hostexec),
					ImagePullPolicy: v1.PullIfNotPresent,
				},
			},
			HostNetwork:                   true,
			SecurityContext:               &v1.PodSecurityContext{},
			TerminationGracePeriodSeconds: &immediate,
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

// RunHostCmdWithRetries calls RunHostCmd and retries all errors
// until it succeeds or the specified timeout expires.
// This can be used with idempotent commands to deflake transient Node issues.
func RunHostCmdWithRetries(ns, name, cmd string, interval, timeout time.Duration) (string, error) {
	start := time.Now()
	for {
		out, err := RunHostCmd(ns, name, cmd)
		if err == nil {
			return out, nil
		}
		if elapsed := time.Since(start); elapsed > timeout {
			return out, fmt.Errorf("RunHostCmd still failed after %v: %v", elapsed, err)
		}
		Logf("Waiting %v to retry failed RunHostCmd: %v", interval, err)
		time.Sleep(interval)
	}
}

// LaunchHostExecPod launches a hostexec pod in the given namespace and waits
// until it's Running
func LaunchHostExecPod(client clientset.Interface, ns, name string) *v1.Pod {
	hostExecPod := NewHostExecPodSpec(ns, name)
	pod, err := client.CoreV1().Pods(ns).Create(hostExecPod)
	ExpectNoError(err)
	err = WaitForPodRunningInNamespace(client, pod)
	ExpectNoError(err)
	return pod
}

// newExecPodSpec returns the pod spec of exec pod
func newExecPodSpec(ns, generateName string) *v1.Pod {
	immediate := int64(0)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: generateName,
			Namespace:    ns,
		},
		Spec: v1.PodSpec{
			TerminationGracePeriodSeconds: &immediate,
			Containers: []v1.Container{
				{
					Name:    "exec",
					Image:   BusyBoxImage,
					Command: []string{"sh", "-c", "trap exit TERM; while true; do sleep 5; done"},
				},
			},
		},
	}
	return pod
}

// CreateExecPodOrFail creates a simple busybox pod in a sleep loop used as a
// vessel for kubectl exec commands.
// Returns the name of the created pod.
func CreateExecPodOrFail(client clientset.Interface, ns, generateName string, tweak func(*v1.Pod)) string {
	Logf("Creating new exec pod")
	execPod := newExecPodSpec(ns, generateName)
	if tweak != nil {
		tweak(execPod)
	}
	created, err := client.CoreV1().Pods(ns).Create(execPod)
	Expect(err).NotTo(HaveOccurred())
	err = wait.PollImmediate(Poll, 5*time.Minute, func() (bool, error) {
		retrievedPod, err := client.CoreV1().Pods(execPod.Namespace).Get(created.Name, metav1.GetOptions{})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		return retrievedPod.Status.Phase == v1.PodRunning, nil
	})
	Expect(err).NotTo(HaveOccurred())
	return created.Name
}

func CreatePodOrFail(c clientset.Interface, ns, name string, labels map[string]string, containerPorts []v1.ContainerPort) {
	By(fmt.Sprintf("Creating pod %s in namespace %s", name, ns))
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: imageutils.GetPauseImageName(),
					Ports: containerPorts,
					// Add a dummy environment variable to work around a docker issue.
					// https://github.com/docker/docker/issues/14203
					Env: []v1.EnvVar{{Name: "FOO", Value: " "}},
				},
			},
		},
	}
	_, err := c.CoreV1().Pods(ns).Create(pod)
	Expect(err).NotTo(HaveOccurred())
}

func DeletePodOrFail(c clientset.Interface, ns, name string) {
	By(fmt.Sprintf("Deleting pod %s in namespace %s", name, ns))
	err := c.CoreV1().Pods(ns).Delete(name, nil)
	Expect(err).NotTo(HaveOccurred())
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
	key := ""
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
	case "local", "vsphere":
		keyfile = os.Getenv("LOCAL_SSH_KEY") // maybe?
		if len(keyfile) == 0 {
			keyfile = "id_rsa"
		}
	case "skeleton":
		keyfile = os.Getenv("KUBE_SSH_KEY")
		if len(keyfile) == 0 {
			keyfile = "id_rsa"
		}
	default:
		return nil, fmt.Errorf("GetSigner(...) not implemented for %s", provider)
	}

	if len(key) == 0 {
		key = filepath.Join(keydir, keyfile)
	}

	return sshutil.MakePrivateKeySignerFromFile(key)
}

// CheckPodsRunningReady returns whether all pods whose names are listed in
// podNames in namespace ns are running and ready, using c and waiting at most
// timeout.
func CheckPodsRunningReady(c clientset.Interface, ns string, podNames []string, timeout time.Duration) bool {
	return CheckPodsCondition(c, ns, podNames, timeout, testutils.PodRunningReady, "running and ready")
}

// CheckPodsRunningReadyOrSucceeded returns whether all pods whose names are
// listed in podNames in namespace ns are running and ready, or succeeded; use
// c and waiting at most timeout.
func CheckPodsRunningReadyOrSucceeded(c clientset.Interface, ns string, podNames []string, timeout time.Duration) bool {
	return CheckPodsCondition(c, ns, podNames, timeout, testutils.PodRunningReadyOrSucceeded, "running and ready, or succeeded")
}

// CheckPodsCondition returns whether all pods whose names are listed in podNames
// in namespace ns are in the condition, using c and waiting at most timeout.
func CheckPodsCondition(c clientset.Interface, ns string, podNames []string, timeout time.Duration, condition podCondition, desc string) bool {
	np := len(podNames)
	Logf("Waiting up to %v for %d pods to be %s: %s", timeout, np, desc, podNames)
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
			Logf("Pod %[1]s failed to be %[2]s.", res.podName, desc)
			success = false
		}
	}
	Logf("Wanted all %d pods to be %s. Result: %t. Pods: %v", np, desc, success, podNames)
	return success
}

// WaitForNodeToBeReady returns whether node name is ready within timeout.
func WaitForNodeToBeReady(c clientset.Interface, name string, timeout time.Duration) bool {
	return WaitForNodeToBe(c, name, v1.NodeReady, true, timeout)
}

// WaitForNodeToBeNotReady returns whether node name is not ready (i.e. the
// readiness condition is anything but ready, e.g false or unknown) within
// timeout.
func WaitForNodeToBeNotReady(c clientset.Interface, name string, timeout time.Duration) bool {
	return WaitForNodeToBe(c, name, v1.NodeReady, false, timeout)
}

func isNodeConditionSetAsExpected(node *v1.Node, conditionType v1.NodeConditionType, wantTrue, silent bool) bool {
	// Check the node readiness condition (logging all).
	for _, cond := range node.Status.Conditions {
		// Ensure that the condition type and the status matches as desired.
		if cond.Type == conditionType {
			// For NodeReady condition we need to check Taints as well
			if cond.Type == v1.NodeReady {
				hasNodeControllerTaints := false
				// For NodeReady we need to check if Taints are gone as well
				taints := node.Spec.Taints
				for _, taint := range taints {
					if taint.MatchTaint(nodectlr.UnreachableTaintTemplate) || taint.MatchTaint(nodectlr.NotReadyTaintTemplate) {
						hasNodeControllerTaints = true
						break
					}
				}
				if wantTrue {
					if (cond.Status == v1.ConditionTrue) && !hasNodeControllerTaints {
						return true
					} else {
						msg := ""
						if !hasNodeControllerTaints {
							msg = fmt.Sprintf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
								conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
						} else {
							msg = fmt.Sprintf("Condition %s of node %s is %v, but Node is tainted by NodeController with %v. Failure",
								conditionType, node.Name, cond.Status == v1.ConditionTrue, taints)
						}
						if !silent {
							Logf(msg)
						}
						return false
					}
				} else {
					// TODO: check if the Node is tainted once we enable NC notReady/unreachable taints by default
					if cond.Status != v1.ConditionTrue {
						return true
					}
					if !silent {
						Logf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
							conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
					}
					return false
				}
			}
			if (wantTrue && (cond.Status == v1.ConditionTrue)) || (!wantTrue && (cond.Status != v1.ConditionTrue)) {
				return true
			} else {
				if !silent {
					Logf("Condition %s of node %s is %v instead of %t. Reason: %v, message: %v",
						conditionType, node.Name, cond.Status == v1.ConditionTrue, wantTrue, cond.Reason, cond.Message)
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

func IsNodeConditionSetAsExpected(node *v1.Node, conditionType v1.NodeConditionType, wantTrue bool) bool {
	return isNodeConditionSetAsExpected(node, conditionType, wantTrue, false)
}

func IsNodeConditionSetAsExpectedSilent(node *v1.Node, conditionType v1.NodeConditionType, wantTrue bool) bool {
	return isNodeConditionSetAsExpected(node, conditionType, wantTrue, true)
}

func IsNodeConditionUnset(node *v1.Node, conditionType v1.NodeConditionType) bool {
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
func WaitForNodeToBe(c clientset.Interface, name string, conditionType v1.NodeConditionType, wantTrue bool, timeout time.Duration) bool {
	Logf("Waiting up to %v for node %s condition %s to be %t", timeout, name, conditionType, wantTrue)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		node, err := c.CoreV1().Nodes().Get(name, metav1.GetOptions{})
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

// Checks whether all registered nodes are ready.
// TODO: we should change the AllNodesReady call in AfterEach to WaitForAllNodesHealthy,
// and figure out how to do it in a configurable way, as we can't expect all setups to run
// default test add-ons.
func AllNodesReady(c clientset.Interface, timeout time.Duration) error {
	Logf("Waiting up to %v for all (but %d) nodes to be ready", timeout, TestContext.AllowedNotReadyNodes)

	var notReady []*v1.Node
	err := wait.PollImmediate(Poll, timeout, func() (bool, error) {
		notReady = nil
		// It should be OK to list unschedulable Nodes here.
		nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		for i := range nodes.Items {
			node := &nodes.Items[i]
			if !IsNodeConditionSetAsExpected(node, v1.NodeReady, true) {
				notReady = append(notReady, node)
			}
		}
		// Framework allows for <TestContext.AllowedNotReadyNodes> nodes to be non-ready,
		// to make it possible e.g. for incorrect deployment of some small percentage
		// of nodes (which we allow in cluster validation). Some nodes that are not
		// provisioned correctly at startup will never become ready (e.g. when something
		// won't install correctly), so we can't expect them to be ready at any point.
		return len(notReady) <= TestContext.AllowedNotReadyNodes, nil
	})

	if err != nil && err != wait.ErrWaitTimeout {
		return err
	}

	if len(notReady) > TestContext.AllowedNotReadyNodes {
		msg := ""
		for _, node := range notReady {
			msg = fmt.Sprintf("%s, %s", msg, node.Name)
		}
		return fmt.Errorf("Not ready nodes: %#v", msg)
	}
	return nil
}

// checks whether all registered nodes are ready and all required Pods are running on them.
func WaitForAllNodesHealthy(c clientset.Interface, timeout time.Duration) error {
	Logf("Waiting up to %v for all nodes to be ready", timeout)

	var notReady []v1.Node
	var missingPodsPerNode map[string][]string
	err := wait.PollImmediate(Poll, timeout, func() (bool, error) {
		notReady = nil
		// It should be OK to list unschedulable Nodes here.
		nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{ResourceVersion: "0"})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		for _, node := range nodes.Items {
			if !IsNodeConditionSetAsExpected(&node, v1.NodeReady, true) {
				notReady = append(notReady, node)
			}
		}
		pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{ResourceVersion: "0"})
		if err != nil {
			return false, err
		}

		systemPodsPerNode := make(map[string][]string)
		for _, pod := range pods.Items {
			if pod.Namespace == metav1.NamespaceSystem && pod.Status.Phase == v1.PodRunning {
				if pod.Spec.NodeName != "" {
					systemPodsPerNode[pod.Spec.NodeName] = append(systemPodsPerNode[pod.Spec.NodeName], pod.Name)
				}
			}
		}
		missingPodsPerNode = make(map[string][]string)
		for _, node := range nodes.Items {
			if !system.IsMasterNode(node.Name) {
				for _, requiredPod := range requiredPerNodePods {
					foundRequired := false
					for _, presentPod := range systemPodsPerNode[node.Name] {
						if requiredPod.MatchString(presentPod) {
							foundRequired = true
							break
						}
					}
					if !foundRequired {
						missingPodsPerNode[node.Name] = append(missingPodsPerNode[node.Name], requiredPod.String())
					}
				}
			}
		}
		return len(notReady) == 0 && len(missingPodsPerNode) == 0, nil
	})

	if err != nil && err != wait.ErrWaitTimeout {
		return err
	}

	if len(notReady) > 0 {
		return fmt.Errorf("Not ready nodes: %v", notReady)
	}
	if len(missingPodsPerNode) > 0 {
		return fmt.Errorf("Not running system Pods: %v", missingPodsPerNode)
	}
	return nil

}

// Filters nodes in NodeList in place, removing nodes that do not
// satisfy the given condition
// TODO: consider merging with pkg/client/cache.NodeLister
func FilterNodes(nodeList *v1.NodeList, fn func(node v1.Node) bool) {
	var l []v1.Node

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
	Logf("Killing kube-proxy on node %v", host)
	result, err := SSH("sudo pkill kube-proxy", host, TestContext.Provider)
	if err != nil || result.Code != 0 {
		LogSSHResult(result)
		return fmt.Errorf("couldn't restart kube-proxy: %v", err)
	}
	// wait for kube-proxy to come back up
	sshCmd := "sudo /bin/sh -c 'pgrep kube-proxy | wc -l'"
	err = wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
		Logf("Waiting for kubeproxy to come back up with %v on %v", sshCmd, host)
		result, err := SSH(sshCmd, host, TestContext.Provider)
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

func RestartKubelet(host string) error {
	// TODO: Make it work for all providers and distros.
	supportedProviders := []string{"gce", "aws", "vsphere"}
	if !ProviderIs(supportedProviders...) {
		return fmt.Errorf("unsupported provider: %s, supported providers are: %v", TestContext.Provider, supportedProviders)
	}
	if ProviderIs("gce") && !NodeOSDistroIs("debian", "gci") {
		return fmt.Errorf("unsupported node OS distro: %s", TestContext.NodeOSDistro)
	}
	var cmd string

	if ProviderIs("gce") && NodeOSDistroIs("debian") {
		cmd = "sudo /etc/init.d/kubelet restart"
	} else if ProviderIs("vsphere") {
		var sudoPresent bool
		sshResult, err := SSH("sudo --version", host, TestContext.Provider)
		if err != nil {
			return fmt.Errorf("Unable to ssh to host %s with error %v", host, err)
		}
		if !strings.Contains(sshResult.Stderr, "command not found") {
			sudoPresent = true
		}
		sshResult, err = SSH("systemctl --version", host, TestContext.Provider)
		if !strings.Contains(sshResult.Stderr, "command not found") {
			cmd = "systemctl restart kubelet"
		} else {
			cmd = "service kubelet restart"
		}
		if sudoPresent {
			cmd = fmt.Sprintf("sudo %s", cmd)
		}
	} else {
		cmd = "sudo systemctl restart kubelet"
	}
	Logf("Restarting kubelet via ssh on host %s with command %s", host, cmd)
	result, err := SSH(cmd, host, TestContext.Provider)
	if err != nil || result.Code != 0 {
		LogSSHResult(result)
		return fmt.Errorf("couldn't restart kubelet: %v", err)
	}
	return nil
}

func WaitForKubeletUp(host string) error {
	cmd := "curl http://localhost:" + strconv.Itoa(ports.KubeletReadOnlyPort) + "/healthz"
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		result, err := SSH(cmd, host, TestContext.Provider)
		if err != nil || result.Code != 0 {
			LogSSHResult(result)
		}
		if result.Stdout == "ok" {
			return nil
		}
	}
	return fmt.Errorf("waiting for kubelet timed out")
}

func RestartApiserver(cs clientset.Interface) error {
	// TODO: Make it work for all providers.
	if !ProviderIs("gce", "gke", "aws") {
		return fmt.Errorf("unsupported provider: %s", TestContext.Provider)
	}
	if ProviderIs("gce", "aws") {
		initialRestartCount, err := getApiserverRestartCount(cs)
		if err != nil {
			return fmt.Errorf("failed to get apiserver's restart count: %v", err)
		}
		if err := sshRestartMaster(); err != nil {
			return fmt.Errorf("failed to restart apiserver: %v", err)
		}
		return waitForApiserverRestarted(cs, initialRestartCount)
	}
	// GKE doesn't allow ssh access, so use a same-version master
	// upgrade to teardown/recreate master.
	v, err := cs.Discovery().ServerVersion()
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
		command = "pidof kube-apiserver | xargs sudo kill"
	} else {
		command = "sudo /etc/init.d/kube-apiserver restart"
	}
	Logf("Restarting master via ssh, running: %v", command)
	result, err := SSH(command, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
	if err != nil || result.Code != 0 {
		LogSSHResult(result)
		return fmt.Errorf("couldn't restart apiserver: %v", err)
	}
	return nil
}

func WaitForApiserverUp(c clientset.Interface) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		body, err := c.CoreV1().RESTClient().Get().AbsPath("/healthz").Do().Raw()
		if err == nil && string(body) == "ok" {
			return nil
		}
	}
	return fmt.Errorf("waiting for apiserver timed out")
}

// waitForApiserverRestarted waits until apiserver's restart count increased.
func waitForApiserverRestarted(c clientset.Interface, initialRestartCount int32) error {
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		restartCount, err := getApiserverRestartCount(c)
		if err != nil {
			Logf("Failed to get apiserver's restart count: %v", err)
			continue
		}
		if restartCount > initialRestartCount {
			Logf("Apiserver has restarted.")
			return nil
		}
		Logf("Waiting for apiserver restart count to increase")
	}
	return fmt.Errorf("timed out waiting for apiserver to be restarted")
}

func getApiserverRestartCount(c clientset.Interface) (int32, error) {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"component": "kube-apiserver"}))
	listOpts := metav1.ListOptions{LabelSelector: label.String()}
	pods, err := c.CoreV1().Pods(metav1.NamespaceSystem).List(listOpts)
	if err != nil {
		return -1, err
	}
	if len(pods.Items) != 1 {
		return -1, fmt.Errorf("unexpected number of apiserver pod: %d", len(pods.Items))
	}
	for _, s := range pods.Items[0].Status.ContainerStatuses {
		if s.Name != "kube-apiserver" {
			continue
		}
		return s.RestartCount, nil
	}
	return -1, fmt.Errorf("failed to find kube-apiserver container in pod")
}

func RestartControllerManager() error {
	// TODO: Make it work for all providers and distros.
	if !ProviderIs("gce", "aws") {
		return fmt.Errorf("unsupported provider: %s", TestContext.Provider)
	}
	if ProviderIs("gce") && !MasterOSDistroIs("gci") {
		return fmt.Errorf("unsupported master OS distro: %s", TestContext.MasterOSDistro)
	}
	cmd := "pidof kube-controller-manager | xargs sudo kill"
	Logf("Restarting controller-manager via ssh, running: %v", cmd)
	result, err := SSH(cmd, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
	if err != nil || result.Code != 0 {
		LogSSHResult(result)
		return fmt.Errorf("couldn't restart controller-manager: %v", err)
	}
	return nil
}

func WaitForControllerManagerUp() error {
	cmd := "curl http://localhost:" + strconv.Itoa(ports.InsecureKubeControllerManagerPort) + "/healthz"
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		result, err := SSH(cmd, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
		if err != nil || result.Code != 0 {
			LogSSHResult(result)
		}
		if result.Stdout == "ok" {
			return nil
		}
	}
	return fmt.Errorf("waiting for controller-manager timed out")
}

// CheckForControllerManagerHealthy checks that the controller manager does not crash within "duration"
func CheckForControllerManagerHealthy(duration time.Duration) error {
	var PID string
	cmd := "pidof kube-controller-manager"
	for start := time.Now(); time.Since(start) < duration; time.Sleep(5 * time.Second) {
		result, err := SSH(cmd, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
		if err != nil {
			// We don't necessarily know that it crashed, pipe could just be broken
			LogSSHResult(result)
			return fmt.Errorf("master unreachable after %v", time.Since(start))
		} else if result.Code != 0 {
			LogSSHResult(result)
			return fmt.Errorf("SSH result code not 0. actually: %v after %v", result.Code, time.Since(start))
		} else if result.Stdout != PID {
			if PID == "" {
				PID = result.Stdout
			} else {
				//its dead
				return fmt.Errorf("controller manager crashed, old PID: %s, new PID: %s", PID, result.Stdout)
			}
		} else {
			Logf("kube-controller-manager still healthy after %v", time.Since(start))
		}
	}
	return nil
}

// NumberOfRegisteredNodes returns number of registered Nodes excluding Master Node.
func NumberOfRegisteredNodes(c clientset.Interface) (int, error) {
	nodes, err := waitListSchedulableNodes(c)
	if err != nil {
		Logf("Failed to list nodes: %v", err)
		return 0, err
	}
	return len(nodes.Items), nil
}

// NumberOfReadyNodes returns number of ready Nodes excluding Master Node.
func NumberOfReadyNodes(c clientset.Interface) (int, error) {
	nodes, err := waitListSchedulableNodes(c)
	if err != nil {
		Logf("Failed to list nodes: %v", err)
		return 0, err
	}

	// Filter out not-ready nodes.
	FilterNodes(nodes, func(node v1.Node) bool {
		return IsNodeConditionSetAsExpected(&node, v1.NodeReady, true)
	})
	return len(nodes.Items), nil
}

// CheckNodesReady waits up to timeout for cluster to has desired size and
// there is no not-ready nodes in it. By cluster size we mean number of Nodes
// excluding Master Node.
func CheckNodesReady(c clientset.Interface, size int, timeout time.Duration) ([]v1.Node, error) {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		nodes, err := waitListSchedulableNodes(c)
		if err != nil {
			Logf("Failed to list nodes: %v", err)
			continue
		}
		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		FilterNodes(nodes, func(node v1.Node) bool {
			nodeReady := IsNodeConditionSetAsExpected(&node, v1.NodeReady, true)
			networkReady := IsNodeConditionUnset(&node, v1.NodeNetworkUnavailable) || IsNodeConditionSetAsExpected(&node, v1.NodeNetworkUnavailable, false)
			return nodeReady && networkReady
		})
		numReady := len(nodes.Items)

		if numNodes == size && numReady == size {
			Logf("Cluster has reached the desired number of ready nodes %d", size)
			return nodes.Items, nil
		}
		Logf("Waiting for ready nodes %d, current ready %d, not ready nodes %d", size, numReady, numNodes-numReady)
	}
	return nil, fmt.Errorf("timeout waiting %v for number of ready nodes to be %d", timeout, size)
}

// WaitForReadyNodes waits up to timeout for cluster to has desired size and
// there is no not-ready nodes in it. By cluster size we mean number of Nodes
// excluding Master Node.
func WaitForReadyNodes(c clientset.Interface, size int, timeout time.Duration) error {
	_, err := CheckNodesReady(c, size, timeout)
	return err
}

func GenerateMasterRegexp(prefix string) string {
	return prefix + "(-...)?"
}

// waitForMasters waits until the cluster has the desired number of ready masters in it.
func WaitForMasters(masterPrefix string, c clientset.Interface, size int, timeout time.Duration) error {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
		if err != nil {
			Logf("Failed to list nodes: %v", err)
			continue
		}

		// Filter out nodes that are not master replicas
		FilterNodes(nodes, func(node v1.Node) bool {
			res, err := regexp.Match(GenerateMasterRegexp(masterPrefix), ([]byte)(node.Name))
			if err != nil {
				Logf("Failed to match regexp to node name: %v", err)
				return false
			}
			return res
		})

		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		FilterNodes(nodes, func(node v1.Node) bool {
			return IsNodeConditionSetAsExpected(&node, v1.NodeReady, true)
		})

		numReady := len(nodes.Items)

		if numNodes == size && numReady == size {
			Logf("Cluster has reached the desired number of masters %d", size)
			return nil
		}
		Logf("Waiting for the number of masters %d, current %d, not ready master nodes %d", size, numNodes, numNodes-numReady)
	}
	return fmt.Errorf("timeout waiting %v for the number of masters to be %d", timeout, size)
}

// GetHostExternalAddress gets the node for a pod and returns the first External
// address. Returns an error if the node the pod is on doesn't have an External
// address.
func GetHostExternalAddress(client clientset.Interface, p *v1.Pod) (externalAddress string, err error) {
	node, err := client.CoreV1().Nodes().Get(p.Spec.NodeName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	for _, address := range node.Status.Addresses {
		if address.Type == v1.NodeExternalIP {
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
func getSvcNodePort(client clientset.Interface, ns, name string, svcPort int) (int, error) {
	svc, err := client.CoreV1().Services(ns).Get(name, metav1.GetOptions{})
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
func GetNodePortURL(client clientset.Interface, ns, name string, svcPort int) (string, error) {
	nodePort, err := getSvcNodePort(client, ns, name, svcPort)
	if err != nil {
		return "", err
	}
	// This list of nodes must not include the master, which is marked
	// unschedulable, since the master doesn't run kube-proxy. Without
	// kube-proxy NodePorts won't work.
	var nodes *v1.NodeList
	if wait.PollImmediate(Poll, SingleCallTimeout, func() (bool, error) {
		nodes, err = client.CoreV1().Nodes().List(metav1.ListOptions{FieldSelector: fields.Set{
			"spec.unschedulable": "false",
		}.AsSelector().String()})
		if err != nil {
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		return true, nil
	}) != nil {
		return "", err
	}
	if len(nodes.Items) == 0 {
		return "", fmt.Errorf("Unable to list nodes in cluster.")
	}
	for _, node := range nodes.Items {
		for _, address := range node.Status.Addresses {
			if address.Type == v1.NodeExternalIP {
				if address.Address != "" {
					return fmt.Sprintf("http://%v:%v", address.Address, nodePort), nil
				}
			}
		}
	}
	return "", fmt.Errorf("Failed to find external address for service %v", name)
}

// TODO(random-liu): Change this to be a member function of the framework.
func GetPodLogs(c clientset.Interface, namespace, podName, containerName string) (string, error) {
	return getPodLogsInternal(c, namespace, podName, containerName, false)
}

func getPreviousPodLogs(c clientset.Interface, namespace, podName, containerName string) (string, error) {
	return getPodLogsInternal(c, namespace, podName, containerName, true)
}

// utility function for gomega Eventually
func getPodLogsInternal(c clientset.Interface, namespace, podName, containerName string, previous bool) (string, error) {
	logs, err := c.CoreV1().RESTClient().Get().
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

func GetGCECloud() (*gcecloud.GCECloud, error) {
	gceCloud, ok := TestContext.CloudConfig.Provider.(*gcecloud.GCECloud)
	if !ok {
		return nil, fmt.Errorf("failed to convert CloudConfig.Provider to GCECloud: %#v", TestContext.CloudConfig.Provider)
	}
	return gceCloud, nil
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
	gceCloud, err := GetGCECloud()
	if err != nil {
		return err
	}
	project := TestContext.CloudConfig.ProjectID
	region, err := gcecloud.GetGCERegion(TestContext.CloudConfig.Zone)
	if err != nil {
		return fmt.Errorf("could not get region for zone %q: %v", TestContext.CloudConfig.Zone, err)
	}

	return wait.Poll(10*time.Second, 5*time.Minute, func() (bool, error) {
		service := gceCloud.ComputeServices().GA
		list, err := service.ForwardingRules.List(project, region).Do()
		if err != nil {
			return false, err
		}
		for _, item := range list.Items {
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

func isElementOf(podUID types.UID, pods *v1.PodList) bool {
	for _, pod := range pods.Items {
		if pod.UID == podUID {
			return true
		}
	}
	return false
}

// timeout for proxy requests.
const proxyTimeout = 2 * time.Minute

// NodeProxyRequest performs a get on a node proxy endpoint given the nodename and rest client.
func NodeProxyRequest(c clientset.Interface, node, endpoint string, port int) (restclient.Result, error) {
	// proxy tends to hang in some cases when Node is not ready. Add an artificial timeout for this call.
	// This will leak a goroutine if proxy hangs. #22165
	var result restclient.Result
	finished := make(chan struct{})
	go func() {
		result = c.CoreV1().RESTClient().Get().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", node, port)).
			Suffix(endpoint).
			Do()

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
func GetKubeletPods(c clientset.Interface, node string) (*v1.PodList, error) {
	return getKubeletPods(c, node, "pods")
}

// GetKubeletRunningPods retrieves the list of running pods on the kubelet. The pods
// includes necessary information (e.g., UID, name, namespace for
// pods/containers), but do not contain the full spec.
func GetKubeletRunningPods(c clientset.Interface, node string) (*v1.PodList, error) {
	return getKubeletPods(c, node, "runningpods")
}

func getKubeletPods(c clientset.Interface, node, resource string) (*v1.PodList, error) {
	result := &v1.PodList{}
	client, err := NodeProxyRequest(c, node, resource, ports.KubeletPort)
	if err != nil {
		return &v1.PodList{}, err
	}
	if err = client.Into(result); err != nil {
		return &v1.PodList{}, err
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
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  containerName,
					Image: imageutils.GetE2EImage(imageutils.Porter),
					Env:   []v1.EnvVar{{Name: fmt.Sprintf("SERVE_PORT_%d", port), Value: "foo"}},
					Ports: []v1.ContainerPort{{ContainerPort: int32(port)}},
				},
			},
			NodeName:      nodeName,
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	podClient := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
	_, err := podClient.Create(pod)
	ExpectNoError(err)
	ExpectNoError(f.WaitForPodRunning(podName))
	createdPod, err := podClient.Get(podName, metav1.GetOptions{})
	ExpectNoError(err)
	ip = net.JoinHostPort(createdPod.Status.PodIP, strconv.Itoa(port))
	Logf("Target pod IP:port is %s", ip)
	return
}

type PingCommand string

const (
	IPv4PingCommand PingCommand = "ping"
	IPv6PingCommand PingCommand = "ping6"
)

// CheckConnectivityToHost launches a pod to test connectivity to the specified
// host. An error will be returned if the host is not reachable from the pod.
//
// An empty nodeName will use the schedule to choose where the pod is executed.
func CheckConnectivityToHost(f *Framework, nodeName, podName, host string, pingCmd PingCommand, timeout int) error {
	contName := fmt.Sprintf("%s-container", podName)

	command := []string{
		string(pingCmd),
		"-c", "3", // send 3 pings
		"-W", "2", // wait at most 2 seconds for a reply
		"-w", strconv.Itoa(timeout),
		host,
	}

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    contName,
					Image:   BusyBoxImage,
					Command: command,
				},
			},
			NodeName:      nodeName,
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	podClient := f.ClientSet.CoreV1().Pods(f.Namespace.Name)
	_, err := podClient.Create(pod)
	if err != nil {
		return err
	}
	err = WaitForPodSuccessInNamespace(f.ClientSet, podName, f.Namespace.Name)

	if err != nil {
		logs, logErr := GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, contName)
		if logErr != nil {
			Logf("Warning: Failed to get logs from pod %q: %v", pod.Name, logErr)
		} else {
			Logf("pod %s/%s logs:\n%s", f.Namespace.Name, pod.Name, logs)
		}
	}

	return err
}

// CoreDump SSHs to the master and all nodes and dumps their logs into dir.
// It shells out to cluster/log-dump/log-dump.sh to accomplish this.
func CoreDump(dir string) {
	if TestContext.DisableLogDump {
		Logf("Skipping dumping logs from cluster")
		return
	}
	var cmd *exec.Cmd
	if TestContext.LogexporterGCSPath != "" {
		Logf("Dumping logs from nodes to GCS directly at path: %s", TestContext.LogexporterGCSPath)
		cmd = exec.Command(path.Join(TestContext.RepoRoot, "cluster", "log-dump", "log-dump.sh"), dir, TestContext.LogexporterGCSPath)
	} else {
		Logf("Dumping logs locally to: %s", dir)
		cmd = exec.Command(path.Join(TestContext.RepoRoot, "cluster", "log-dump", "log-dump.sh"), dir)
	}
	cmd.Env = append(os.Environ(), fmt.Sprintf("LOG_DUMP_SYSTEMD_SERVICES=%s", parseSystemdServices(TestContext.SystemdServices)))
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		Logf("Error running cluster/log-dump/log-dump.sh: %v", err)
	}
}

// parseSystemdServices converts services separator from comma to space.
func parseSystemdServices(services string) string {
	return strings.TrimSpace(strings.Replace(services, ",", " ", -1))
}

func UpdatePodWithRetries(client clientset.Interface, ns, name string, update func(*v1.Pod)) (*v1.Pod, error) {
	for i := 0; i < 3; i++ {
		pod, err := client.CoreV1().Pods(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("Failed to get pod %q: %v", name, err)
		}
		update(pod)
		pod, err = client.CoreV1().Pods(ns).Update(pod)
		if err == nil {
			return pod, nil
		}
		if !apierrs.IsConflict(err) && !apierrs.IsServerTimeout(err) {
			return nil, fmt.Errorf("Failed to update pod %q: %v", name, err)
		}
	}
	return nil, fmt.Errorf("Too many retries updating Pod %q", name)
}

func GetPodsInNamespace(c clientset.Interface, ns string, ignoreLabels map[string]string) ([]*v1.Pod, error) {
	pods, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{})
	if err != nil {
		return []*v1.Pod{}, err
	}
	ignoreSelector := labels.SelectorFromSet(ignoreLabels)
	filtered := []*v1.Pod{}
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
	return RunCmdEnv(nil, command, args...)
}

// RunCmdEnv runs cmd with the provided environment and args and
// returns its stdout and stderr. It also outputs cmd's stdout and
// stderr to their respective OS streams.
func RunCmdEnv(env []string, command string, args ...string) (string, string, error) {
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
	cmd.Env = env
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
func GetPodsScheduled(masterNodes sets.String, pods *v1.PodList) (scheduledPods, notScheduledPods []v1.Pod) {
	for _, pod := range pods.Items {
		if !masterNodes.Has(pod.Spec.NodeName) {
			if pod.Spec.NodeName != "" {
				_, scheduledCondition := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
				Expect(scheduledCondition != nil).To(Equal(true))
				Expect(scheduledCondition.Status).To(Equal(v1.ConditionTrue))
				scheduledPods = append(scheduledPods, pod)
			} else {
				_, scheduledCondition := podutil.GetPodCondition(&pod.Status, v1.PodScheduled)
				Expect(scheduledCondition != nil).To(Equal(true))
				Expect(scheduledCondition.Status).To(Equal(v1.ConditionFalse))
				if scheduledCondition.Reason == "Unschedulable" {

					notScheduledPods = append(notScheduledPods, pod)
				}
			}
		}
	}
	return
}

// WaitForStableCluster waits until all existing pods are scheduled and returns their amount.
func WaitForStableCluster(c clientset.Interface, masterNodes sets.String) int {
	timeout := 10 * time.Minute
	startTime := time.Now()

	allPods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{})
	ExpectNoError(err)
	// API server returns also Pods that succeeded. We need to filter them out.
	currentPods := make([]v1.Pod, 0, len(allPods.Items))
	for _, pod := range allPods.Items {
		if pod.Status.Phase != v1.PodSucceeded && pod.Status.Phase != v1.PodFailed {
			currentPods = append(currentPods, pod)
		}

	}
	allPods.Items = currentPods
	scheduledPods, currentlyNotScheduledPods := GetPodsScheduled(masterNodes, allPods)
	for len(currentlyNotScheduledPods) != 0 {
		time.Sleep(2 * time.Second)

		allPods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{})
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
func GetMasterAndWorkerNodesOrDie(c clientset.Interface) (sets.String, *v1.NodeList) {
	nodes := &v1.NodeList{}
	masters := sets.NewString()
	all, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	ExpectNoError(err)
	for _, n := range all.Items {
		if system.IsMasterNode(n.Name) {
			masters.Insert(n.Name)
		} else if isNodeSchedulable(&n) && isNodeUntainted(&n) {
			nodes.Items = append(nodes.Items, n)
		}
	}
	return masters, nodes
}

func ListNamespaceEvents(c clientset.Interface, ns string) error {
	ls, err := c.CoreV1().Events(ns).List(metav1.ListOptions{})
	if err != nil {
		return err
	}
	for _, event := range ls.Items {
		glog.Infof("Event(%#v): type: '%v' reason: '%v' %v", event.InvolvedObject, event.Type, event.Reason, event.Message)
	}
	return nil
}

// E2ETestNodePreparer implements testutils.TestNodePreparer interface, which is used
// to create/modify Nodes before running a test.
type E2ETestNodePreparer struct {
	client clientset.Interface
	// Specifies how many nodes should be modified using the given strategy.
	// Only one strategy can be applied to a single Node, so there needs to
	// be at least <sum_of_keys> Nodes in the cluster.
	countToStrategy       []testutils.CountToStrategy
	nodeToAppliedStrategy map[string]testutils.PrepareNodeStrategy
}

func NewE2ETestNodePreparer(client clientset.Interface, countToStrategy []testutils.CountToStrategy) testutils.TestNodePreparer {
	return &E2ETestNodePreparer{
		client:                client,
		countToStrategy:       countToStrategy,
		nodeToAppliedStrategy: make(map[string]testutils.PrepareNodeStrategy),
	}
}

func (p *E2ETestNodePreparer) PrepareNodes() error {
	nodes := GetReadySchedulableNodesOrDie(p.client)
	numTemplates := 0
	for _, v := range p.countToStrategy {
		numTemplates += v.Count
	}
	if numTemplates > len(nodes.Items) {
		return fmt.Errorf("Can't prepare Nodes. Got more templates than existing Nodes.")
	}
	index := 0
	sum := 0
	for _, v := range p.countToStrategy {
		sum += v.Count
		for ; index < sum; index++ {
			if err := testutils.DoPrepareNode(p.client, &nodes.Items[index], v.Strategy); err != nil {
				glog.Errorf("Aborting node preparation: %v", err)
				return err
			}
			p.nodeToAppliedStrategy[nodes.Items[index].Name] = v.Strategy
		}
	}
	return nil
}

func (p *E2ETestNodePreparer) CleanupNodes() error {
	var encounteredError error
	nodes := GetReadySchedulableNodesOrDie(p.client)
	for i := range nodes.Items {
		var err error
		name := nodes.Items[i].Name
		strategy, found := p.nodeToAppliedStrategy[name]
		if found {
			if err = testutils.DoCleanupNode(p.client, name, strategy); err != nil {
				glog.Errorf("Skipping cleanup of Node: failed update of %v: %v", name, err)
				encounteredError = err
			}
		}
	}
	return encounteredError
}

func GetClusterID(c clientset.Interface) (string, error) {
	cm, err := c.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(gcecloud.UIDConfigMapName, metav1.GetOptions{})
	if err != nil || cm == nil {
		return "", fmt.Errorf("error getting cluster ID: %v", err)
	}
	clusterID, clusterIDExists := cm.Data[gcecloud.UIDCluster]
	providerID, providerIDExists := cm.Data[gcecloud.UIDProvider]
	if !clusterIDExists {
		return "", fmt.Errorf("cluster ID not set")
	}
	if providerIDExists {
		return providerID, nil
	}
	return clusterID, nil
}

// CleanupGCEResources cleans up GCE Service Type=LoadBalancer resources with
// the given name. The name is usually the UUID of the Service prefixed with an
// alpha-numeric character ('a') to work around cloudprovider rules.
func CleanupGCEResources(c clientset.Interface, loadBalancerName, region, zone string) (retErr error) {
	gceCloud, err := GetGCECloud()
	if err != nil {
		return err
	}
	if region == "" {
		// Attempt to parse region from zone if no region is given.
		region, err = gcecloud.GetGCERegion(zone)
		if err != nil {
			return fmt.Errorf("error parsing GCE/GKE region from zone %q: %v", zone, err)
		}
	}
	if err := gceCloud.DeleteFirewall(gcecloud.MakeFirewallName(loadBalancerName)); err != nil &&
		!IsGoogleAPIHTTPErrorCode(err, http.StatusNotFound) {
		retErr = err
	}
	if err := gceCloud.DeleteRegionForwardingRule(loadBalancerName, region); err != nil &&
		!IsGoogleAPIHTTPErrorCode(err, http.StatusNotFound) {
		retErr = fmt.Errorf("%v\n%v", retErr, err)

	}
	if err := gceCloud.DeleteRegionAddress(loadBalancerName, region); err != nil &&
		!IsGoogleAPIHTTPErrorCode(err, http.StatusNotFound) {
		retErr = fmt.Errorf("%v\n%v", retErr, err)
	}
	clusterID, err := GetClusterID(c)
	if err != nil {
		retErr = fmt.Errorf("%v\n%v", retErr, err)
		return
	}
	hcNames := []string{gcecloud.MakeNodesHealthCheckName(clusterID)}
	hc, getErr := gceCloud.GetHttpHealthCheck(loadBalancerName)
	if getErr != nil && !IsGoogleAPIHTTPErrorCode(getErr, http.StatusNotFound) {
		retErr = fmt.Errorf("%v\n%v", retErr, getErr)
		return
	}
	if hc != nil {
		hcNames = append(hcNames, hc.Name)
	}
	if err := gceCloud.DeleteExternalTargetPoolAndChecks(&v1.Service{}, loadBalancerName, region, clusterID, hcNames...); err != nil &&
		!IsGoogleAPIHTTPErrorCode(err, http.StatusNotFound) {
		retErr = fmt.Errorf("%v\n%v", retErr, err)
	}
	return
}

// IsHTTPErrorCode returns true if the error is a google api
// error matching the corresponding HTTP error code.
func IsGoogleAPIHTTPErrorCode(err error, code int) bool {
	apiErr, ok := err.(*googleapi.Error)
	return ok && apiErr.Code == code
}

// getMaster populates the externalIP, internalIP and hostname fields of the master.
// If any of these is unavailable, it is set to "".
func getMaster(c clientset.Interface) Address {
	master := Address{}

	// Populate the internal IP.
	eps, err := c.CoreV1().Endpoints(metav1.NamespaceDefault).Get("kubernetes", metav1.GetOptions{})
	if err != nil {
		Failf("Failed to get kubernetes endpoints: %v", err)
	}
	if len(eps.Subsets) != 1 || len(eps.Subsets[0].Addresses) != 1 {
		Failf("There are more than 1 endpoints for kubernetes service: %+v", eps)
	}
	master.internalIP = eps.Subsets[0].Addresses[0].IP

	// Populate the external IP/hostname.
	url, err := url.Parse(TestContext.Host)
	if err != nil {
		Failf("Failed to parse hostname: %v", err)
	}
	if net.ParseIP(url.Host) != nil {
		// TODO: Check that it is external IP (not having a reserved IP address as per RFC1918).
		master.externalIP = url.Host
	} else {
		master.hostname = url.Host
	}

	return master
}

// GetMasterAddress returns the hostname/external IP/internal IP as appropriate for e2e tests on a particular provider
// which is the address of the interface used for communication with the kubelet.
func GetMasterAddress(c clientset.Interface) string {
	master := getMaster(c)
	switch TestContext.Provider {
	case "gce", "gke":
		return master.externalIP
	case "aws":
		return awsMasterIP
	default:
		Failf("This test is not supported for provider %s and should be disabled", TestContext.Provider)
	}
	return ""
}

// GetNodeExternalIP returns node external IP concatenated with port 22 for ssh
// e.g. 1.2.3.4:22
func GetNodeExternalIP(node *v1.Node) (string, error) {
	Logf("Getting external IP address for %s", node.Name)
	host := ""
	for _, a := range node.Status.Addresses {
		if a.Type == v1.NodeExternalIP {
			host = net.JoinHostPort(a.Address, sshPort)
			break
		}
	}
	if host == "" {
		return "", fmt.Errorf("Couldn't get the external IP of host %s with addresses %v", node.Name, node.Status.Addresses)
	}
	return host, nil
}

// GetNodeInternalIP returns node internal IP
func GetNodeInternalIP(node *v1.Node) (string, error) {
	host := ""
	for _, address := range node.Status.Addresses {
		if address.Type == v1.NodeInternalIP {
			if address.Address != "" {
				host = net.JoinHostPort(address.Address, sshPort)
				break
			}
		}
	}
	if host == "" {
		return "", fmt.Errorf("Couldn't get the external IP of host %s with addresses %v", node.Name, node.Status.Addresses)
	}
	return host, nil
}

// SimpleGET executes a get on the given url, returns error if non-200 returned.
func SimpleGET(c *http.Client, url, host string) (string, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", err
	}
	req.Host = host
	res, err := c.Do(req)
	if err != nil {
		return "", err
	}
	defer res.Body.Close()
	rawBody, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return "", err
	}
	body := string(rawBody)
	if res.StatusCode != http.StatusOK {
		err = fmt.Errorf(
			"GET returned http error %v", res.StatusCode)
	}
	return body, err
}

// PollURL polls till the url responds with a healthy http code. If
// expectUnreachable is true, it breaks on first non-healthy http code instead.
func PollURL(route, host string, timeout time.Duration, interval time.Duration, httpClient *http.Client, expectUnreachable bool) error {
	var lastBody string
	pollErr := wait.PollImmediate(interval, timeout, func() (bool, error) {
		var err error
		lastBody, err = SimpleGET(httpClient, route, host)
		if err != nil {
			Logf("host %v path %v: %v unreachable", host, route, err)
			return expectUnreachable, nil
		}
		Logf("host %v path %v: reached", host, route)
		return !expectUnreachable, nil
	})
	if pollErr != nil {
		return fmt.Errorf("Failed to execute a successful GET within %v, Last response body for %v, host %v:\n%v\n\n%v\n",
			timeout, route, host, lastBody, pollErr)
	}
	return nil
}

func DescribeIng(ns string) {
	Logf("\nOutput of kubectl describe ing:\n")
	desc, _ := RunKubectl(
		"describe", "ing", fmt.Sprintf("--namespace=%v", ns))
	Logf(desc)
}

// NewTestPod returns a pod that has the specified requests and limits
func (f *Framework) NewTestPod(name string, requests v1.ResourceList, limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: imageutils.GetPauseImageName(),
					Resources: v1.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
		},
	}
}

// create empty file at given path on the pod.
func CreateEmptyFileOnPod(namespace string, podName string, filePath string) error {
	_, err := RunKubectl("exec", fmt.Sprintf("--namespace=%s", namespace), podName, "--", "/bin/sh", "-c", fmt.Sprintf("touch %s", filePath))
	return err
}

// GetAzureCloud returns azure cloud provider
func GetAzureCloud() (*azure.Cloud, error) {
	cloud, ok := TestContext.CloudConfig.Provider.(*azure.Cloud)
	if !ok {
		return nil, fmt.Errorf("failed to convert CloudConfig.Provider to Azure: %#v", TestContext.CloudConfig.Provider)
	}
	return cloud, nil
}

func PrintSummaries(summaries []TestDataSummary, testBaseName string) {
	now := time.Now()
	for i := range summaries {
		Logf("Printing summary: %v", summaries[i].SummaryKind())
		switch TestContext.OutputPrintType {
		case "hr":
			if TestContext.ReportDir == "" {
				Logf(summaries[i].PrintHumanReadable())
			} else {
				// TODO: learn to extract test name and append it to the kind instead of timestamp.
				filePath := path.Join(TestContext.ReportDir, summaries[i].SummaryKind()+"_"+testBaseName+"_"+now.Format(time.RFC3339)+".txt")
				if err := ioutil.WriteFile(filePath, []byte(summaries[i].PrintHumanReadable()), 0644); err != nil {
					Logf("Failed to write file %v with test performance data: %v", filePath, err)
				}
			}
		case "json":
			fallthrough
		default:
			if TestContext.OutputPrintType != "json" {
				Logf("Unknown output type: %v. Printing JSON", TestContext.OutputPrintType)
			}
			if TestContext.ReportDir == "" {
				Logf("%v JSON\n%v", summaries[i].SummaryKind(), summaries[i].PrintJSON())
				Logf("Finished")
			} else {
				// TODO: learn to extract test name and append it to the kind instead of timestamp.
				filePath := path.Join(TestContext.ReportDir, summaries[i].SummaryKind()+"_"+testBaseName+"_"+now.Format(time.RFC3339)+".json")
				Logf("Writing to %s", filePath)
				if err := ioutil.WriteFile(filePath, []byte(summaries[i].PrintJSON()), 0644); err != nil {
					Logf("Failed to write file %v with test performance data: %v", filePath, err)
				}
			}
		}
	}
}

func DumpDebugInfo(c clientset.Interface, ns string) {
	sl, _ := c.CoreV1().Pods(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	for _, s := range sl.Items {
		desc, _ := RunKubectl("describe", "po", s.Name, fmt.Sprintf("--namespace=%v", ns))
		Logf("\nOutput of kubectl describe %v:\n%v", s.Name, desc)

		l, _ := RunKubectl("logs", s.Name, fmt.Sprintf("--namespace=%v", ns), "--tail=100")
		Logf("\nLast 100 log lines of %v:\n%v", s.Name, l)
	}
}

// DsFromManifest reads a .json/yaml file and returns the daemonset in it.
func DsFromManifest(url string) (*apps.DaemonSet, error) {
	var controller apps.DaemonSet
	Logf("Parsing ds from %v", url)

	var response *http.Response
	var err error

	for i := 1; i <= 5; i++ {
		response, err = http.Get(url)
		if err == nil && response.StatusCode == 200 {
			break
		}
		time.Sleep(time.Duration(i) * time.Second)
	}

	if err != nil {
		return nil, fmt.Errorf("failed to get url: %v", err)
	}
	if response.StatusCode != 200 {
		return nil, fmt.Errorf("invalid http response status: %v", response.StatusCode)
	}
	defer response.Body.Close()

	data, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read html response body: %v", err)
	}

	json, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse data to json: %v", err)
	}

	err = runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), json, &controller)
	if err != nil {
		return nil, fmt.Errorf("failed to decode DaemonSet spec: %v", err)
	}
	return &controller, nil
}

// waitForServerPreferredNamespacedResources waits until server preferred namespaced resources could be successfully discovered.
// TODO: Fix https://github.com/kubernetes/kubernetes/issues/55768 and remove the following retry.
func waitForServerPreferredNamespacedResources(d discovery.DiscoveryInterface, timeout time.Duration) ([]*metav1.APIResourceList, error) {
	Logf("Waiting up to %v for server preferred namespaced resources to be successfully discovered", timeout)
	var resources []*metav1.APIResourceList
	if err := wait.PollImmediate(Poll, timeout, func() (bool, error) {
		var err error
		resources, err = d.ServerPreferredNamespacedResources()
		if err == nil || isDynamicDiscoveryError(err) {
			return true, nil
		}
		if !discovery.IsGroupDiscoveryFailedError(err) {
			return false, err
		}
		Logf("Error discoverying server preferred namespaced resources: %v, retrying in %v.", err, Poll)
		return false, nil
	}); err != nil {
		return nil, err
	}
	return resources, nil
}

// WaitForPersistentVolumeClaimDeleted waits for a PersistentVolumeClaim to be removed from the system until timeout occurs, whichever comes first.
func WaitForPersistentVolumeClaimDeleted(c clientset.Interface, ns string, pvcName string, Poll, timeout time.Duration) error {
	Logf("Waiting up to %v for PersistentVolumeClaim %s to be removed", timeout, pvcName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		_, err := c.CoreV1().PersistentVolumeClaims(ns).Get(pvcName, metav1.GetOptions{})
		if err != nil {
			if apierrs.IsNotFound(err) {
				Logf("Claim %q in namespace %q doesn't exist in the system", pvcName, ns)
				return nil
			}
			Logf("Failed to get claim %q in namespace %q, retrying in %v. Error: %v", pvcName, ns, Poll, err)
		}
	}
	return fmt.Errorf("PersistentVolumeClaim %s is not removed from the system within %v", pvcName, timeout)
}

func GetClusterZones(c clientset.Interface) (sets.String, error) {
	nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("Error getting nodes while attempting to list cluster zones: %v", err)
	}

	// collect values of zone label from all nodes
	zones := sets.NewString()
	for _, node := range nodes.Items {
		if zone, found := node.Labels[kubeletapis.LabelZoneFailureDomain]; found {
			zones.Insert(zone)
		}
	}
	return zones, nil
}
