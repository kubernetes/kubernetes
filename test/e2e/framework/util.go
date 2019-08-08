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

	"golang.org/x/net/websocket"
	"k8s.io/klog"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	gomegatypes "github.com/onsi/gomega/types"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
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
	"k8s.io/client-go/kubernetes/scheme"
	restclient "k8s.io/client-go/rest"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	watchtools "k8s.io/client-go/tools/watch"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/client/conditions"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/service"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	taintutils "k8s.io/kubernetes/pkg/util/taints"
	"k8s.io/kubernetes/test/e2e/framework/ginkgowrapper"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eresource "k8s.io/kubernetes/test/e2e/framework/resource"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	"k8s.io/kubernetes/test/e2e/manifest"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	uexec "k8s.io/utils/exec"
	utilnet "k8s.io/utils/net"
)

const (
	// PodListTimeout is how long to wait for the pod to be listable.
	PodListTimeout = time.Minute
	// PodStartTimeout is how long to wait for the pod to be started.
	// Initial pod start can be delayed O(minutes) by slow docker pulls.
	// TODO: Make this 30 seconds once #4566 is resolved.
	PodStartTimeout = 5 * time.Minute

	// PodStartShortTimeout is same as `PodStartTimeout` to wait for the pod to be started, but shorter.
	// Use it case by case when we are sure pod start will not be delayed.
	// minutes by slow docker pulls or something else.
	PodStartShortTimeout = 2 * time.Minute

	// PodDeleteTimeout is how long to wait for a pod to be deleted.
	PodDeleteTimeout = 5 * time.Minute

	// PodEventTimeout is how much we wait for a pod event to occur.
	PodEventTimeout = 2 * time.Minute

	// NamespaceCleanupTimeout is how long to wait for the namespace to be deleted.
	// If there are any orphaned namespaces to clean up, this test is running
	// on a long lived cluster. A long wait here is preferably to spurious test
	// failures caused by leaked resources from a previous test run.
	NamespaceCleanupTimeout = 15 * time.Minute

	// Some pods can take much longer to get ready due to volume attach/detach latency.
	slowPodStartTimeout = 15 * time.Minute

	// ServiceStartTimeout is how long to wait for a service endpoint to be resolvable.
	ServiceStartTimeout = 3 * time.Minute

	// Poll is how often to Poll pods, nodes and claims.
	Poll = 2 * time.Second

	// PollShortTimeout is the short timeout value in polling.
	PollShortTimeout = 1 * time.Minute

	// ServiceAccountProvisionTimeout is how long to wait for a service account to be provisioned.
	// service accounts are provisioned after namespace creation
	// a service account is required to support pod creation in a namespace as part of admission control
	ServiceAccountProvisionTimeout = 2 * time.Minute

	// SingleCallTimeout is how long to try single API calls (like 'get' or 'list'). Used to prevent
	// transient failures from failing tests.
	// TODO: client should not apply this timeout to Watch calls. Increased from 30s until that is fixed.
	SingleCallTimeout = 5 * time.Minute

	// NodeReadyInitialTimeout is how long nodes have to be "ready" when a test begins. They should already
	// be "ready" before the test starts, so this is small.
	NodeReadyInitialTimeout = 20 * time.Second

	// PodReadyBeforeTimeout is how long pods have to be "ready" when a test begins.
	PodReadyBeforeTimeout = 5 * time.Minute

	// How long pods have to become scheduled onto nodes
	podScheduledBeforeTimeout = PodListTimeout + (20 * time.Second)

	podRespondingTimeout = 15 * time.Minute
	// ClaimProvisionTimeout is how long claims have to become dynamically provisioned.
	ClaimProvisionTimeout = 5 * time.Minute

	// ClaimProvisionShortTimeout is same as `ClaimProvisionTimeout` to wait for claim to be dynamically provisioned, but shorter.
	// Use it case by case when we are sure this timeout is enough.
	ClaimProvisionShortTimeout = 1 * time.Minute

	// ClaimBindingTimeout is how long claims have to become bound.
	ClaimBindingTimeout = 3 * time.Minute

	// ClaimDeletingTimeout is How long claims have to become deleted.
	ClaimDeletingTimeout = 3 * time.Minute

	// PVReclaimingTimeout is how long PVs have to beome reclaimed.
	PVReclaimingTimeout = 3 * time.Minute

	// PVBindingTimeout is how long PVs have to become bound.
	PVBindingTimeout = 3 * time.Minute

	// PVDeletingTimeout is how long PVs have to become deleted.
	PVDeletingTimeout = 3 * time.Minute

	// RecreateNodeReadyAgainTimeout is how long a node is allowed to become "Ready" after it is recreated before
	// the test is considered failed.
	RecreateNodeReadyAgainTimeout = 10 * time.Minute

	// RestartNodeReadyAgainTimeout is how long a node is allowed to become "Ready" after it is restarted before
	// the test is considered failed.
	RestartNodeReadyAgainTimeout = 5 * time.Minute

	// RestartPodReadyAgainTimeout is how long a pod is allowed to become "running" and "ready" after a node
	// restart before test is considered failed.
	RestartPodReadyAgainTimeout = 5 * time.Minute

	// SnapshotCreateTimeout is how long for snapshot to create snapshotContent.
	SnapshotCreateTimeout = 5 * time.Minute

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
	// BusyBoxImage is the image URI of BusyBox.
	BusyBoxImage = imageutils.GetE2EImage(imageutils.BusyBox)

	// For parsing Kubectl version for version-skewed testing.
	gitVersionRegexp = regexp.MustCompile("GitVersion:\"(v.+?)\"")

	// Slice of regexps for names of pods that have to be running to consider a Node "healthy"
	requiredPerNodePods = []*regexp.Regexp{
		regexp.MustCompile(".*kube-proxy.*"),
		regexp.MustCompile(".*fluentd-elasticsearch.*"),
		regexp.MustCompile(".*node-problem-detector.*"),
	}

	// ServeHostnameImage is a serve hostname image name.
	ServeHostnameImage = imageutils.GetE2EImage(imageutils.Agnhost)
)

// RunID is a unique identifier of the e2e run.
// Beware that this ID is not the same for all tests in the e2e run, because each Ginkgo node creates it separately.
var RunID = uuid.NewUUID()

// CreateTestingNSFn is a func that is responsible for creating namespace used for executing e2e tests.
type CreateTestingNSFn func(baseName string, c clientset.Interface, labels map[string]string) (*v1.Namespace, error)

// GetMasterHost returns a hostname of a master.
func GetMasterHost() string {
	masterURL, err := url.Parse(TestContext.Host)
	ExpectNoError(err)
	return masterURL.Hostname()
}

func nowStamp() string {
	return time.Now().Format(time.StampMilli)
}

func log(level string, format string, args ...interface{}) {
	fmt.Fprintf(ginkgo.GinkgoWriter, nowStamp()+": "+level+": "+format+"\n", args...)
}

func skipInternalf(caller int, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	log("INFO", msg)
	ginkgowrapper.Skip(msg, caller+1)
}

// Skipf skips with information about why the test is being skipped.
func Skipf(format string, args ...interface{}) {
	skipInternalf(1, format, args...)
}

// SkipUnlessNodeCountIsAtLeast skips if the number of nodes is less than the minNodeCount.
func SkipUnlessNodeCountIsAtLeast(minNodeCount int) {
	if TestContext.CloudConfig.NumNodes < minNodeCount {
		skipInternalf(1, "Requires at least %d nodes (not %d)", minNodeCount, TestContext.CloudConfig.NumNodes)
	}
}

// SkipUnlessNodeCountIsAtMost skips if the number of nodes is greater than the maxNodeCount.
func SkipUnlessNodeCountIsAtMost(maxNodeCount int) {
	if TestContext.CloudConfig.NumNodes > maxNodeCount {
		skipInternalf(1, "Requires at most %d nodes (not %d)", maxNodeCount, TestContext.CloudConfig.NumNodes)
	}
}

// SkipUnlessAtLeast skips if the value is less than the minValue.
func SkipUnlessAtLeast(value int, minValue int, message string) {
	if value < minValue {
		skipInternalf(1, message)
	}
}

// SkipIfProviderIs skips if the provider is included in the unsupportedProviders.
func SkipIfProviderIs(unsupportedProviders ...string) {
	if ProviderIs(unsupportedProviders...) {
		skipInternalf(1, "Not supported for providers %v (found %s)", unsupportedProviders, TestContext.Provider)
	}
}

// SkipUnlessLocalEphemeralStorageEnabled skips if the LocalStorageCapacityIsolation is not enabled.
func SkipUnlessLocalEphemeralStorageEnabled() {
	if !utilfeature.DefaultFeatureGate.Enabled(features.LocalStorageCapacityIsolation) {
		skipInternalf(1, "Only supported when %v feature is enabled", features.LocalStorageCapacityIsolation)
	}
}

// SkipUnlessSSHKeyPresent skips if no SSH key is found.
func SkipUnlessSSHKeyPresent() {
	if _, err := e2essh.GetSigner(TestContext.Provider); err != nil {
		skipInternalf(1, "No SSH Key for provider %s: '%v'", TestContext.Provider, err)
	}
}

// SkipUnlessProviderIs skips if the provider is not included in the supportedProviders.
func SkipUnlessProviderIs(supportedProviders ...string) {
	if !ProviderIs(supportedProviders...) {
		skipInternalf(1, "Only supported for providers %v (not %s)", supportedProviders, TestContext.Provider)
	}
}

// SkipUnlessMultizone skips if the cluster does not have multizone.
func SkipUnlessMultizone(c clientset.Interface) {
	zones, err := GetClusterZones(c)
	if err != nil {
		skipInternalf(1, "Error listing cluster zones")
	}
	if zones.Len() <= 1 {
		skipInternalf(1, "Requires more than one zone")
	}
}

// SkipIfMultizone skips if the cluster has multizone.
func SkipIfMultizone(c clientset.Interface) {
	zones, err := GetClusterZones(c)
	if err != nil {
		skipInternalf(1, "Error listing cluster zones")
	}
	if zones.Len() > 1 {
		skipInternalf(1, "Requires at most one zone")
	}
}

// SkipUnlessPrometheusMonitoringIsEnabled skips if the prometheus monitoring is not enabled.
func SkipUnlessPrometheusMonitoringIsEnabled(supportedMonitoring ...string) {
	if !TestContext.EnablePrometheusMonitoring {
		skipInternalf(1, "Skipped because prometheus monitoring is not enabled")
	}
}

// SkipUnlessMasterOSDistroIs skips if the master OS distro is not included in the supportedMasterOsDistros.
func SkipUnlessMasterOSDistroIs(supportedMasterOsDistros ...string) {
	if !MasterOSDistroIs(supportedMasterOsDistros...) {
		skipInternalf(1, "Only supported for master OS distro %v (not %s)", supportedMasterOsDistros, TestContext.MasterOSDistro)
	}
}

// SkipUnlessNodeOSDistroIs skips if the node OS distro is not included in the supportedNodeOsDistros.
func SkipUnlessNodeOSDistroIs(supportedNodeOsDistros ...string) {
	if !NodeOSDistroIs(supportedNodeOsDistros...) {
		skipInternalf(1, "Only supported for node OS distro %v (not %s)", supportedNodeOsDistros, TestContext.NodeOSDistro)
	}
}

// SkipUnlessTaintBasedEvictionsEnabled skips if the TaintBasedEvictions is not enabled.
func SkipUnlessTaintBasedEvictionsEnabled() {
	if !utilfeature.DefaultFeatureGate.Enabled(features.TaintBasedEvictions) {
		skipInternalf(1, "Only supported when %v feature is enabled", features.TaintBasedEvictions)
	}
}

// SkipIfContainerRuntimeIs skips if the container runtime is included in the runtimes.
func SkipIfContainerRuntimeIs(runtimes ...string) {
	for _, containerRuntime := range runtimes {
		if containerRuntime == TestContext.ContainerRuntime {
			skipInternalf(1, "Not supported under container runtime %s", containerRuntime)
		}
	}
}

// RunIfContainerRuntimeIs runs if the container runtime is included in the runtimes.
func RunIfContainerRuntimeIs(runtimes ...string) {
	for _, containerRuntime := range runtimes {
		if containerRuntime == TestContext.ContainerRuntime {
			return
		}
	}
	skipInternalf(1, "Skipped because container runtime %q is not in %s", TestContext.ContainerRuntime, runtimes)
}

// RunIfSystemSpecNameIs runs if the system spec name is included in the names.
func RunIfSystemSpecNameIs(names ...string) {
	for _, name := range names {
		if name == TestContext.SystemSpecName {
			return
		}
	}
	skipInternalf(1, "Skipped because system spec name %q is not in %v", TestContext.SystemSpecName, names)
}

// Run a test container to try and contact the Kubernetes api-server from a pod, wait for it
// to flip to Ready, log its output and delete it.
func runKubernetesServiceTestContainer(c clientset.Interface, ns string) {
	path := "test/images/clusterapi-tester/pod.yaml"
	e2elog.Logf("Parsing pod from %v", path)
	p, err := manifest.PodFromManifest(path)
	if err != nil {
		e2elog.Logf("Failed to parse clusterapi-tester from manifest %v: %v", path, err)
		return
	}
	p.Namespace = ns
	if _, err := c.CoreV1().Pods(ns).Create(p); err != nil {
		e2elog.Logf("Failed to create %v: %v", p.Name, err)
		return
	}
	defer func() {
		if err := c.CoreV1().Pods(ns).Delete(p.Name, nil); err != nil {
			e2elog.Logf("Failed to delete pod %v: %v", p.Name, err)
		}
	}()
	timeout := 5 * time.Minute
	if err := e2epod.WaitForPodCondition(c, ns, p.Name, "clusterapi-tester", timeout, testutils.PodRunningReady); err != nil {
		e2elog.Logf("Pod %v took longer than %v to enter running/ready: %v", p.Name, timeout, err)
		return
	}
	logs, err := e2epod.GetPodLogs(c, ns, p.Name, p.Spec.Containers[0].Name)
	if err != nil {
		e2elog.Logf("Failed to retrieve logs from %v: %v", p.Name, err)
	} else {
		e2elog.Logf("Output of clusterapi-tester:\n%v", logs)
	}
}

// getDefaultClusterIPFamily obtains the default IP family of the cluster
// using the Cluster IP address of the kubernetes service created in the default namespace
// This unequivocally identifies the default IP family because services are single family
func getDefaultClusterIPFamily(c clientset.Interface) string {
	// Get the ClusterIP of the kubernetes service created in the default namespace
	svc, err := c.CoreV1().Services(metav1.NamespaceDefault).Get("kubernetes", metav1.GetOptions{})
	if err != nil {
		e2elog.Failf("Failed to get kubernetes service ClusterIP: %v", err)
	}

	if utilnet.IsIPv6String(svc.Spec.ClusterIP) {
		return "ipv6"
	}
	return "ipv4"
}

// ProviderIs returns true if the provider is included is the providers. Otherwise false.
func ProviderIs(providers ...string) bool {
	for _, provider := range providers {
		if strings.ToLower(provider) == strings.ToLower(TestContext.Provider) {
			return true
		}
	}
	return false
}

// MasterOSDistroIs returns true if the master OS distro is included in the supportedMasterOsDistros. Otherwise false.
func MasterOSDistroIs(supportedMasterOsDistros ...string) bool {
	for _, distro := range supportedMasterOsDistros {
		if strings.ToLower(distro) == strings.ToLower(TestContext.MasterOSDistro) {
			return true
		}
	}
	return false
}

// NodeOSDistroIs returns true if the node OS distro is included in the supportedNodeOsDistros. Otherwise false.
func NodeOSDistroIs(supportedNodeOsDistros ...string) bool {
	for _, distro := range supportedNodeOsDistros {
		if strings.ToLower(distro) == strings.ToLower(TestContext.NodeOSDistro) {
			return true
		}
	}
	return false
}

// ProxyMode returns a proxyMode of a kube-proxy.
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
					Image:   imageutils.GetE2EImage(imageutils.Agnhost),
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
	e2elog.Logf("ProxyMode: %s", stdout)
	return stdout, nil
}

// SkipUnlessServerVersionGTE skips if the server version is less than v.
func SkipUnlessServerVersionGTE(v *utilversion.Version, c discovery.ServerVersionInterface) {
	gte, err := ServerVersionGTE(v, c)
	if err != nil {
		e2elog.Failf("Failed to get server version: %v", err)
	}
	if !gte {
		skipInternalf(1, "Not supported for server versions before %q", v)
	}
}

// SkipIfMissingResource skips if the gvr resource is missing.
func SkipIfMissingResource(dynamicClient dynamic.Interface, gvr schema.GroupVersionResource, namespace string) {
	resourceClient := dynamicClient.Resource(gvr).Namespace(namespace)
	_, err := resourceClient.List(metav1.ListOptions{})
	if err != nil {
		// not all resources support list, so we ignore those
		if apierrs.IsMethodNotSupported(err) || apierrs.IsNotFound(err) || apierrs.IsForbidden(err) {
			skipInternalf(1, "Could not find %s resource, skipping test: %#v", gvr, err)
		}
		e2elog.Failf("Unexpected error getting %v: %v", gvr, err)
	}
}

// ProvidersWithSSH are those providers where each node is accessible with SSH
var ProvidersWithSSH = []string{"gce", "gke", "aws", "local"}

type podCondition func(pod *v1.Pod) (bool, error)

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

// WaitForDaemonSets for all daemonsets in the given namespace to be ready
// (defined as all but 'allowedNotReadyNodes' pods associated with that
// daemonset are ready).
func WaitForDaemonSets(c clientset.Interface, ns string, allowedNotReadyNodes int32, timeout time.Duration) error {
	start := time.Now()
	e2elog.Logf("Waiting up to %v for all daemonsets in namespace '%s' to start",
		timeout, ns)

	return wait.PollImmediate(Poll, timeout, func() (bool, error) {
		dsList, err := c.AppsV1().DaemonSets(ns).List(metav1.ListOptions{})
		if err != nil {
			e2elog.Logf("Error getting daemonsets in namespace: '%s': %v", ns, err)
			if testutils.IsRetryableAPIError(err) {
				return false, nil
			}
			return false, err
		}
		var notReadyDaemonSets []string
		for _, ds := range dsList.Items {
			e2elog.Logf("%d / %d pods ready in namespace '%s' in daemonset '%s' (%d seconds elapsed)", ds.Status.NumberReady, ds.Status.DesiredNumberScheduled, ns, ds.ObjectMeta.Name, int(time.Since(start).Seconds()))
			if ds.Status.DesiredNumberScheduled-ds.Status.NumberReady > allowedNotReadyNodes {
				notReadyDaemonSets = append(notReadyDaemonSets, ds.ObjectMeta.Name)
			}
		}

		if len(notReadyDaemonSets) > 0 {
			e2elog.Logf("there are not ready daemonsets: %v", notReadyDaemonSets)
			return false, nil
		}

		return true, nil
	})
}

func kubectlLogPod(c clientset.Interface, pod v1.Pod, containerNameSubstr string, logFunc func(ftm string, args ...interface{})) {
	for _, container := range pod.Spec.Containers {
		if strings.Contains(container.Name, containerNameSubstr) {
			// Contains() matches all strings if substr is empty
			logs, err := e2epod.GetPodLogs(c, pod.Namespace, pod.Name, container.Name)
			if err != nil {
				logs, err = e2epod.GetPreviousPodLogs(c, pod.Namespace, pod.Name, container.Name)
				if err != nil {
					logFunc("Failed to get logs of pod %v, container %v, err: %v", pod.Name, container.Name, err)
				}
			}
			logFunc("Logs of %v/%v:%v on node %v", pod.Namespace, pod.Name, container.Name, pod.Spec.NodeName)
			logFunc("%s : STARTLOG\n%s\nENDLOG for container %v:%v:%v", containerNameSubstr, logs, pod.Namespace, pod.Name, container.Name)
		}
	}
}

// LogFailedContainers runs `kubectl logs` on a failed containers.
func LogFailedContainers(c clientset.Interface, ns string, logFunc func(ftm string, args ...interface{})) {
	podList, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{})
	if err != nil {
		logFunc("Error getting pods in namespace '%s': %v", ns, err)
		return
	}
	logFunc("Running kubectl logs on non-ready containers in %v", ns)
	for _, pod := range podList.Items {
		if res, err := testutils.PodRunningReady(&pod); !res || err != nil {
			kubectlLogPod(c, pod, "", e2elog.Logf)
		}
	}
}

// DeleteNamespaces deletes all namespaces that match the given delete and skip filters.
// Filter is by simple strings.Contains; first skip filter, then delete filter.
// Returns the list of deleted namespaces or an error.
func DeleteNamespaces(c clientset.Interface, deleteFilter, skipFilter []string) ([]string, error) {
	ginkgo.By("Deleting namespaces")
	nsList, err := c.CoreV1().Namespaces().List(metav1.ListOptions{})
	ExpectNoError(err, "Failed to get namespace list")
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
			defer ginkgo.GinkgoRecover()
			gomega.Expect(c.CoreV1().Namespaces().Delete(nsName, nil)).To(gomega.Succeed())
			e2elog.Logf("namespace : %v api call to delete is complete ", nsName)
		}(item.Name)
	}
	wg.Wait()
	return deleted, nil
}

// WaitForNamespacesDeleted waits for the namespaces to be deleted.
func WaitForNamespacesDeleted(c clientset.Interface, namespaces []string, timeout time.Duration) error {
	ginkgo.By("Waiting for namespaces to vanish")
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

// WaitForDefaultServiceAccountInNamespace waits for the default service account to be provisioned
// the default service account is what is associated with pods when they do not specify a service account
// as a result, pods are not able to be provisioned in a namespace until the service account is provisioned
func WaitForDefaultServiceAccountInNamespace(c clientset.Interface, namespace string) error {
	return waitForServiceAccountInNamespace(c, namespace, "default", ServiceAccountProvisionTimeout)
}

// WaitForPersistentVolumePhase waits for a PersistentVolume to be in a specific phase or until timeout occurs, whichever comes first.
func WaitForPersistentVolumePhase(phase v1.PersistentVolumePhase, c clientset.Interface, pvName string, Poll, timeout time.Duration) error {
	e2elog.Logf("Waiting up to %v for PersistentVolume %s to have phase %s", timeout, pvName, phase)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pv, err := c.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
		if err != nil {
			e2elog.Logf("Get persistent volume %s in failed, ignoring for %v: %v", pvName, Poll, err)
			continue
		}
		if pv.Status.Phase == phase {
			e2elog.Logf("PersistentVolume %s found and phase=%s (%v)", pvName, phase, time.Since(start))
			return nil
		}
		e2elog.Logf("PersistentVolume %s found but phase is %s instead of %s.", pvName, pv.Status.Phase, phase)
	}
	return fmt.Errorf("PersistentVolume %s not in phase %s within %v", pvName, phase, timeout)
}

// WaitForStatefulSetReplicasReady waits for all replicas of a StatefulSet to become ready or until timeout occurs, whichever comes first.
func WaitForStatefulSetReplicasReady(statefulSetName, ns string, c clientset.Interface, Poll, timeout time.Duration) error {
	e2elog.Logf("Waiting up to %v for StatefulSet %s to have all replicas ready", timeout, statefulSetName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		sts, err := c.AppsV1().StatefulSets(ns).Get(statefulSetName, metav1.GetOptions{})
		if err != nil {
			e2elog.Logf("Get StatefulSet %s failed, ignoring for %v: %v", statefulSetName, Poll, err)
			continue
		}
		if sts.Status.ReadyReplicas == *sts.Spec.Replicas {
			e2elog.Logf("All %d replicas of StatefulSet %s are ready. (%v)", sts.Status.ReadyReplicas, statefulSetName, time.Since(start))
			return nil
		}
		e2elog.Logf("StatefulSet %s found but there are %d ready replicas and %d total replicas.", statefulSetName, sts.Status.ReadyReplicas, *sts.Spec.Replicas)
	}
	return fmt.Errorf("StatefulSet %s still has unready pods within %v", statefulSetName, timeout)
}

// WaitForPersistentVolumeDeleted waits for a PersistentVolume to get deleted or until timeout occurs, whichever comes first.
func WaitForPersistentVolumeDeleted(c clientset.Interface, pvName string, Poll, timeout time.Duration) error {
	e2elog.Logf("Waiting up to %v for PersistentVolume %s to get deleted", timeout, pvName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		pv, err := c.CoreV1().PersistentVolumes().Get(pvName, metav1.GetOptions{})
		if err == nil {
			e2elog.Logf("PersistentVolume %s found and phase=%s (%v)", pvName, pv.Status.Phase, time.Since(start))
			continue
		}
		if apierrs.IsNotFound(err) {
			e2elog.Logf("PersistentVolume %s was removed", pvName)
			return nil
		}
		e2elog.Logf("Get persistent volume %s in failed, ignoring for %v: %v", pvName, Poll, err)
	}
	return fmt.Errorf("PersistentVolume %s still exists within %v", pvName, timeout)
}

// WaitForPersistentVolumeClaimPhase waits for a PersistentVolumeClaim to be in a specific phase or until timeout occurs, whichever comes first.
func WaitForPersistentVolumeClaimPhase(phase v1.PersistentVolumeClaimPhase, c clientset.Interface, ns string, pvcName string, Poll, timeout time.Duration) error {
	return WaitForPersistentVolumeClaimsPhase(phase, c, ns, []string{pvcName}, Poll, timeout, true)
}

// WaitForPersistentVolumeClaimsPhase waits for any (if matchAny is true) or all (if matchAny is false) PersistentVolumeClaims
// to be in a specific phase or until timeout occurs, whichever comes first.
func WaitForPersistentVolumeClaimsPhase(phase v1.PersistentVolumeClaimPhase, c clientset.Interface, ns string, pvcNames []string, Poll, timeout time.Duration, matchAny bool) error {
	if len(pvcNames) == 0 {
		return fmt.Errorf("Incorrect parameter: Need at least one PVC to track. Found 0")
	}
	e2elog.Logf("Waiting up to %v for PersistentVolumeClaims %v to have phase %s", timeout, pvcNames, phase)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		phaseFoundInAllClaims := true
		for _, pvcName := range pvcNames {
			pvc, err := c.CoreV1().PersistentVolumeClaims(ns).Get(pvcName, metav1.GetOptions{})
			if err != nil {
				e2elog.Logf("Failed to get claim %q, retrying in %v. Error: %v", pvcName, Poll, err)
				continue
			}
			if pvc.Status.Phase == phase {
				e2elog.Logf("PersistentVolumeClaim %s found and phase=%s (%v)", pvcName, phase, time.Since(start))
				if matchAny {
					return nil
				}
			} else {
				e2elog.Logf("PersistentVolumeClaim %s found but phase is %s instead of %s.", pvcName, pvc.Status.Phase, phase)
				phaseFoundInAllClaims = false
			}
		}
		if phaseFoundInAllClaims {
			return nil
		}
	}
	return fmt.Errorf("PersistentVolumeClaims %v not all in phase %s within %v", pvcNames, phase, timeout)
}

// findAvailableNamespaceName random namespace name starting with baseName.
func findAvailableNamespaceName(baseName string, c clientset.Interface) (string, error) {
	var name string
	err := wait.PollImmediate(Poll, 30*time.Second, func() (bool, error) {
		name = fmt.Sprintf("%v-%v", baseName, RandomSuffix())
		_, err := c.CoreV1().Namespaces().Get(name, metav1.GetOptions{})
		if err == nil {
			// Already taken
			return false, nil
		}
		if apierrs.IsNotFound(err) {
			return true, nil
		}
		e2elog.Logf("Unexpected error while getting namespace: %v", err)
		return false, nil
	})
	return name, err
}

// CreateTestingNS should be used by every test, note that we append a common prefix to the provided test name.
// Please see NewFramework instead of using this directly.
func CreateTestingNS(baseName string, c clientset.Interface, labels map[string]string) (*v1.Namespace, error) {
	if labels == nil {
		labels = map[string]string{}
	}
	labels["e2e-run"] = string(RunID)

	// We don't use ObjectMeta.GenerateName feature, as in case of API call
	// failure we don't know whether the namespace was created and what is its
	// name.
	name, err := findAvailableNamespaceName(baseName, c)
	if err != nil {
		return nil, err
	}

	namespaceObj := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "",
			Labels:    labels,
		},
		Status: v1.NamespaceStatus{},
	}
	// Be robust about making the namespace creation call.
	var got *v1.Namespace
	if err := wait.PollImmediate(Poll, 30*time.Second, func() (bool, error) {
		var err error
		got, err = c.CoreV1().Namespaces().Create(namespaceObj)
		if err != nil {
			e2elog.Logf("Unexpected error while creating namespace: %v", err)
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

	e2elog.Logf("Waiting for terminating namespaces to be deleted...")
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(15 * time.Second) {
		namespaces, err := c.CoreV1().Namespaces().List(metav1.ListOptions{})
		if err != nil {
			e2elog.Logf("Listing namespaces failed: %v", err)
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
			e2elog.Logf("Error while waiting for namespace to be terminated: %v", err)
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
		remainingPods, missingTimestamp, _ = e2epod.CountRemainingPods(c, namespace)
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
	e2elog.Logf("namespace %v deletion completed in %s", namespace, time.Since(startTime))
	return nil
}

// logNamespaces logs the number of namespaces by phase
// namespace is the namespace the test was operating against that failed to delete so it can be grepped in logs
func logNamespaces(c clientset.Interface, namespace string) {
	namespaceList, err := c.CoreV1().Namespaces().List(metav1.ListOptions{})
	if err != nil {
		e2elog.Logf("namespace: %v, unable to list namespaces: %v", namespace, err)
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
	e2elog.Logf("namespace: %v, total namespaces: %v, active: %v, terminating: %v", namespace, len(namespaceList.Items), numActive, numTerminating)
}

// logNamespace logs detail about a namespace
func logNamespace(c clientset.Interface, namespace string) {
	ns, err := c.CoreV1().Namespaces().Get(namespace, metav1.GetOptions{})
	if err != nil {
		if apierrs.IsNotFound(err) {
			e2elog.Logf("namespace: %v no longer exists", namespace)
			return
		}
		e2elog.Logf("namespace: %v, unable to get namespace due to error: %v", namespace, err)
		return
	}
	e2elog.Logf("namespace: %v, DeletionTimetamp: %v, Finalizers: %v, Phase: %v", ns.Name, ns.DeletionTimestamp, ns.Spec.Finalizers, ns.Status.Phase)
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
			e2elog.Logf("discovery error for unexpected group: %#v", gv)
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
	resources = discovery.FilteredBy(discovery.SupportsAllVerbs{Verbs: []string{"list", "delete"}}, resources)
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
			e2elog.Logf("namespace: %s, unable to get client - gvr: %v, error: %v", namespace, gvr, err)
			continue
		}
		// get the api resource
		apiResource := metav1.APIResource{Name: gvr.Resource, Namespaced: true}
		if ignoredResources.Has(gvr.Resource) {
			e2elog.Logf("namespace: %s, resource: %s, ignored listing per whitelist", namespace, apiResource.Name)
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
			e2elog.Logf("namespace: %s, resource: %s, items remaining: %v", namespace, apiResource.Name, len(unstructuredList.Items))
			contentRemaining = true
		}
	}
	return contentRemaining, nil
}

// ContainerInitInvariant checks for an init containers are initialized and invariant on both older and newer.
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
	oldInit, _, _ := e2epod.Initialized(oldPod)
	newInit, _, _ := e2epod.Initialized(newPod)
	if oldInit && !newInit {
		// TODO: we may in the future enable resetting Initialized = false if the kubelet needs to restart it
		// from scratch
		return fmt.Errorf("pod cannot be initialized and then regress to not being initialized")
	}
	return nil
}

func initContainersInvariants(pod *v1.Pod) error {
	allInit, initFailed, err := e2epod.Initialized(pod)
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

// InvariantFunc is a func that checks for invariant.
type InvariantFunc func(older, newer runtime.Object) error

// CheckInvariants checks for invariant of the each events.
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
			e2elog.Logf("Waiting for rc %s to stabilize, generation %v observed generation %v spec.replicas %d status.replicas %d",
				name, rc.Generation, rc.Status.ObservedGeneration, *(rc.Spec.Replicas), rc.Status.Replicas)
		}
		return false, nil
	})
	return err
}

// WaitForService waits until the service appears (exist == true), or disappears (exist == false)
func WaitForService(c clientset.Interface, namespace, name string, exist bool, interval, timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		_, err := c.CoreV1().Services(namespace).Get(name, metav1.GetOptions{})
		switch {
		case err == nil:
			e2elog.Logf("Service %s in namespace %s found.", name, namespace)
			return exist, nil
		case apierrs.IsNotFound(err):
			e2elog.Logf("Service %s in namespace %s disappeared.", name, namespace)
			return !exist, nil
		case !testutils.IsRetryableAPIError(err):
			e2elog.Logf("Non-retryable failure while getting service.")
			return false, err
		default:
			e2elog.Logf("Get service %s in namespace %s failed: %v", name, namespace, err)
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
			e2elog.Logf("Service with %s in namespace %s found.", selector.String(), namespace)
			return exist, nil
		case len(services.Items) == 0:
			e2elog.Logf("Service with %s in namespace %s disappeared.", selector.String(), namespace)
			return !exist, nil
		case !testutils.IsRetryableAPIError(err):
			e2elog.Logf("Non-retryable failure while listing service.")
			return false, err
		default:
			e2elog.Logf("List service with %s in namespace %s failed: %v", selector.String(), namespace, err)
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
		e2elog.Logf("Waiting for amount of service:%s endpoints to be %d", serviceName, expectNum)
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

// RestclientConfig returns a config holds the information needed to build connection to kubernetes clusters.
func RestclientConfig(kubeContext string) (*clientcmdapi.Config, error) {
	e2elog.Logf(">>> kubeConfig: %s", TestContext.KubeConfig)
	if TestContext.KubeConfig == "" {
		return nil, fmt.Errorf("KubeConfig must be specified to load client config")
	}
	c, err := clientcmd.LoadFromFile(TestContext.KubeConfig)
	if err != nil {
		return nil, fmt.Errorf("error loading KubeConfig: %v", err.Error())
	}
	if kubeContext != "" {
		e2elog.Logf(">>> kubeContext: %s", kubeContext)
		c.CurrentContext = kubeContext
	}
	return c, nil
}

// ClientConfigGetter is a func that returns getter to return a config.
type ClientConfigGetter func() (*restclient.Config, error)

// LoadConfig returns a config for a rest client.
func LoadConfig() (*restclient.Config, error) {
	if TestContext.NodeE2E {
		// This is a node e2e test, apply the node e2e configuration
		return &restclient.Config{Host: TestContext.Host}, nil
	}
	c, err := RestclientConfig(TestContext.KubeContext)
	if err != nil {
		if TestContext.KubeConfig == "" {
			return restclient.InClusterConfig()
		}
		return nil, err
	}

	return clientcmd.NewDefaultClientConfig(*c, &clientcmd.ConfigOverrides{ClusterInfo: clientcmdapi.Cluster{Server: TestContext.Host}}).ClientConfig()
}

// LoadClientset returns clientset for connecting to kubernetes clusters.
func LoadClientset() (*clientset.Clientset, error) {
	config, err := LoadConfig()
	if err != nil {
		return nil, fmt.Errorf("error creating client: %v", err.Error())
	}
	return clientset.NewForConfig(config)
}

// RandomSuffix provides a random string to append to pods,services,rcs.
// TODO: Allow service names to have the same form as names
//       for pods and replication controllers so we don't
//       need to use such a function and can instead
//       use the UUID utility function.
func RandomSuffix() string {
	return strconv.Itoa(rand.Intn(10000))
}

// ExpectEqual expects the specified two are the same, otherwise an exception raises
func ExpectEqual(actual interface{}, extra interface{}, explain ...interface{}) {
	gomega.Expect(actual).To(gomega.Equal(extra), explain...)
}

// ExpectNotEqual expects the specified two are not the same, otherwise an exception raises
func ExpectNotEqual(actual interface{}, extra interface{}, explain ...interface{}) {
	gomega.Expect(actual).NotTo(gomega.Equal(extra), explain...)
}

// ExpectError expects an error happens, otherwise an exception raises
func ExpectError(err error, explain ...interface{}) {
	gomega.Expect(err).To(gomega.HaveOccurred(), explain...)
}

// ExpectNoError checks if "err" is set, and if so, fails assertion while logging the error.
func ExpectNoError(err error, explain ...interface{}) {
	ExpectNoErrorWithOffset(1, err, explain...)
}

// ExpectNoErrorWithOffset checks if "err" is set, and if so, fails assertion while logging the error at "offset" levels above its caller
// (for example, for call chain f -> g -> ExpectNoErrorWithOffset(1, ...) error would be logged for "f").
func ExpectNoErrorWithOffset(offset int, err error, explain ...interface{}) {
	if err != nil {
		e2elog.Logf("Unexpected error occurred: %v", err)
	}
	gomega.ExpectWithOffset(1+offset, err).NotTo(gomega.HaveOccurred(), explain...)
}

// ExpectNoErrorWithRetries checks if an error occurs with the given retry count.
func ExpectNoErrorWithRetries(fn func() error, maxRetries int, explain ...interface{}) {
	var err error
	for i := 0; i < maxRetries; i++ {
		err = fn()
		if err == nil {
			return
		}
		e2elog.Logf("(Attempt %d of %d) Unexpected error occurred: %v", i+1, maxRetries, err)
	}
	gomega.ExpectWithOffset(1, err).NotTo(gomega.HaveOccurred(), explain...)
}

// Cleanup stops everything from filePath from namespace ns and checks if everything matching selectors from the given namespace is correctly stopped.
func Cleanup(filePath, ns string, selectors ...string) {
	ginkgo.By("using delete to clean up resources")
	var nsArg string
	if ns != "" {
		nsArg = fmt.Sprintf("--namespace=%s", ns)
	}
	RunKubectlOrDie("delete", "--grace-period=0", "-f", filePath, nsArg)
	AssertCleanup(ns, selectors...)
}

// AssertCleanup asserts that cleanup of a namespace wrt selectors occurred.
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
		e2elog.Failf(e.Error())
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

// LookForStringInPodExec looks for the given string in the output of a command
// executed in a specific pod container.
// TODO(alejandrox1): move to pod/ subpkg once kubectl methods are refactored.
func LookForStringInPodExec(ns, podName string, command []string, expectedString string, timeout time.Duration) (result string, err error) {
	return LookForString(expectedString, timeout, func() string {
		// use the first container
		args := []string{"exec", podName, fmt.Sprintf("--namespace=%v", ns), "--"}
		args = append(args, command...)
		return RunKubectlOrDie(args...)
	})
}

// LookForString looks for the given string in the output of fn, repeatedly calling fn until
// the timeout is reached or the string is found. Returns last log and possibly
// error if the string was not found.
// TODO(alejandrox1): move to pod/ subpkg once kubectl methods are refactored.
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

// KubectlBuilder is used to build, customize and execute a kubectl Command.
// Add more functions to customize the builder as needed.
type KubectlBuilder struct {
	cmd     *exec.Cmd
	timeout <-chan time.Time
}

// NewKubectlCommand returns a KubectlBuilder for running kubectl.
func NewKubectlCommand(args ...string) *KubectlBuilder {
	b := new(KubectlBuilder)
	b.cmd = KubectlCmd(args...)
	return b
}

// WithEnv sets the given environment and returns itself.
func (b *KubectlBuilder) WithEnv(env []string) *KubectlBuilder {
	b.cmd.Env = env
	return b
}

// WithTimeout sets the given timeout and returns itself.
func (b *KubectlBuilder) WithTimeout(t <-chan time.Time) *KubectlBuilder {
	b.timeout = t
	return b
}

// WithStdinData sets the given data to stdin and returns itself.
func (b KubectlBuilder) WithStdinData(data string) *KubectlBuilder {
	b.cmd.Stdin = strings.NewReader(data)
	return &b
}

// WithStdinReader sets the given reader and returns itself.
func (b KubectlBuilder) WithStdinReader(reader io.Reader) *KubectlBuilder {
	b.cmd.Stdin = reader
	return &b
}

// ExecOrDie runs the kubectl executable or dies if error occurs.
func (b KubectlBuilder) ExecOrDie() string {
	str, err := b.Exec()
	// In case of i/o timeout error, try talking to the apiserver again after 2s before dying.
	// Note that we're still dying after retrying so that we can get visibility to triage it further.
	if isTimeout(err) {
		e2elog.Logf("Hit i/o timeout error, talking to the server 2s later to see if it's temporary.")
		time.Sleep(2 * time.Second)
		retryStr, retryErr := RunKubectl("version")
		e2elog.Logf("stdout: %q", retryStr)
		e2elog.Logf("err: %v", retryErr)
	}
	ExpectNoError(err)
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

// Exec runs the kubectl executable.
func (b KubectlBuilder) Exec() (string, error) {
	var stdout, stderr bytes.Buffer
	cmd := b.cmd
	cmd.Stdout, cmd.Stderr = &stdout, &stderr

	e2elog.Logf("Running '%s %s'", cmd.Path, strings.Join(cmd.Args[1:], " ")) // skip arg[0] as it is printed separately
	if err := cmd.Start(); err != nil {
		return "", fmt.Errorf("error starting %v:\nCommand stdout:\n%v\nstderr:\n%v\nerror:\n%v", cmd, cmd.Stdout, cmd.Stderr, err)
	}
	errCh := make(chan error, 1)
	go func() {
		errCh <- cmd.Wait()
	}()
	select {
	case err := <-errCh:
		if err != nil {
			var rc = 127
			if ee, ok := err.(*exec.ExitError); ok {
				rc = int(ee.Sys().(syscall.WaitStatus).ExitStatus())
				e2elog.Logf("rc: %d", rc)
			}
			return "", uexec.CodeExitError{
				Err:  fmt.Errorf("error running %v:\nCommand stdout:\n%v\nstderr:\n%v\nerror:\n%v", cmd, cmd.Stdout, cmd.Stderr, err),
				Code: rc,
			}
		}
	case <-b.timeout:
		b.cmd.Process.Kill()
		return "", fmt.Errorf("timed out waiting for command %v:\nCommand stdout:\n%v\nstderr:\n%v", cmd, cmd.Stdout, cmd.Stderr)
	}
	e2elog.Logf("stderr: %q", stderr.String())
	e2elog.Logf("stdout: %q", stdout.String())
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

// RunKubectlInput is a convenience wrapper over kubectlBuilder that takes input to stdin
func RunKubectlInput(data string, args ...string) (string, error) {
	return NewKubectlCommand(args...).WithStdinData(data).Exec()
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
	b := new(KubectlBuilder)
	args = append(args, "--gcp-project="+TestContext.CloudConfig.ProjectID)

	b.cmd = exec.Command(kubemci, args...)
	return b.Exec()
}

// StartCmdAndStreamOutput returns stdout and stderr after starting the given cmd.
func StartCmdAndStreamOutput(cmd *exec.Cmd) (stdout, stderr io.ReadCloser, err error) {
	stdout, err = cmd.StdoutPipe()
	if err != nil {
		return
	}
	stderr, err = cmd.StderrPipe()
	if err != nil {
		return
	}
	e2elog.Logf("Asynchronously running '%s %s'", cmd.Path, strings.Join(cmd.Args, " "))
	err = cmd.Start()
	return
}

// TryKill is rough equivalent of ctrl+c for cleaning up processes. Intended to be run in defer.
func TryKill(cmd *exec.Cmd) {
	if err := cmd.Process.Kill(); err != nil {
		e2elog.Logf("ERROR failed to kill command %v! The process may leak", cmd)
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
	ginkgo.By(fmt.Sprintf("Creating a pod to test %v", scenarioName))
	if containerIndex < 0 || containerIndex >= len(pod.Spec.Containers) {
		e2elog.Failf("Invalid container index: %d", containerIndex)
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
		ginkgo.By("delete the pod")
		podClient.DeleteSync(createdPod.Name, &metav1.DeleteOptions{}, DefaultPodDeletionTimeout)
	}()

	// Wait for client pod to complete.
	podErr := e2epod.WaitForPodSuccessInNamespace(f.ClientSet, createdPod.Name, ns)

	// Grab its logs.  Get host first.
	podStatus, err := podClient.Get(createdPod.Name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get pod status: %v", err)
	}

	if podErr != nil {
		// Pod failed. Dump all logs from all containers to see what's wrong
		for _, container := range podStatus.Spec.Containers {
			logs, err := e2epod.GetPodLogs(f.ClientSet, ns, podStatus.Name, container.Name)
			if err != nil {
				e2elog.Logf("Failed to get logs from node %q pod %q container %q: %v",
					podStatus.Spec.NodeName, podStatus.Name, container.Name, err)
				continue
			}
			e2elog.Logf("Output of node %q pod %q container %q: %s", podStatus.Spec.NodeName, podStatus.Name, container.Name, logs)
		}
		return fmt.Errorf("expected pod %q success: %v", createdPod.Name, podErr)
	}

	e2elog.Logf("Trying to get logs from node %s pod %s container %s: %v",
		podStatus.Spec.NodeName, podStatus.Name, containerName, err)

	// Sometimes the actual containers take a second to get started, try to get logs for 60s
	logs, err := e2epod.GetPodLogs(f.ClientSet, ns, podStatus.Name, containerName)
	if err != nil {
		e2elog.Logf("Failed to get logs from node %q pod %q container %q. %v",
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

// EventsLister is a func that lists events.
type EventsLister func(opts metav1.ListOptions, ns string) (*v1.EventList, error)

// DumpEventsInNamespace dumps events in the given namespace.
func DumpEventsInNamespace(eventsLister EventsLister, namespace string) {
	ginkgo.By(fmt.Sprintf("Collecting events from namespace %q.", namespace))
	events, err := eventsLister(metav1.ListOptions{}, namespace)
	ExpectNoError(err, "failed to list events in namespace %q", namespace)

	ginkgo.By(fmt.Sprintf("Found %d events.", len(events.Items)))
	// Sort events by their first timestamp
	sortedEvents := events.Items
	if len(sortedEvents) > 1 {
		sort.Sort(byFirstTimestamp(sortedEvents))
	}
	for _, e := range sortedEvents {
		e2elog.Logf("At %v - event for %v: %v %v: %v", e.FirstTimestamp, e.InvolvedObject.Name, e.Source, e.Reason, e.Message)
	}
	// Note that we don't wait for any Cleanup to propagate, which means
	// that if you delete a bunch of pods right before ending your test,
	// you may or may not see the killing/deletion/Cleanup events.
}

// DumpAllNamespaceInfo dumps events, pods and nodes information in the given namespace.
func DumpAllNamespaceInfo(c clientset.Interface, namespace string) {
	DumpEventsInNamespace(func(opts metav1.ListOptions, ns string) (*v1.EventList, error) {
		return c.CoreV1().Events(ns).List(opts)
	}, namespace)

	e2epod.DumpAllPodInfoForNamespace(c, namespace)

	// If cluster is large, then the following logs are basically useless, because:
	// 1. it takes tens of minutes or hours to grab all of them
	// 2. there are so many of them that working with them are mostly impossible
	// So we dump them only if the cluster is relatively small.
	maxNodesForDump := TestContext.MaxNodesToGather
	if nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{}); err == nil {
		if len(nodes.Items) <= maxNodesForDump {
			dumpAllNodeInfo(c)
		} else {
			e2elog.Logf("skipping dumping cluster info - cluster too large")
		}
	} else {
		e2elog.Logf("unable to fetch node list: %v", err)
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

func dumpAllNodeInfo(c clientset.Interface) {
	// It should be OK to list unschedulable Nodes here.
	nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		e2elog.Logf("unable to fetch node list: %v", err)
		return
	}
	names := make([]string, len(nodes.Items))
	for ix := range nodes.Items {
		names[ix] = nodes.Items[ix].Name
	}
	DumpNodeDebugInfo(c, names, e2elog.Logf)
}

// DumpNodeDebugInfo dumps debug information of the given nodes.
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
		podList, err := e2ekubelet.GetKubeletPods(c, n)
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
		e2emetrics.HighLatencyKubeletOperations(c, 10*time.Second, n, logFunc)
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
		e2elog.Logf("Unexpected error retrieving node events %v", err)
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
	nodeReady := e2enode.IsConditionSetAsExpected(node, v1.NodeReady, true)
	networkReady := e2enode.IsConditionUnset(node, v1.NodeNetworkUnavailable) ||
		e2enode.IsConditionSetAsExpectedSilent(node, v1.NodeNetworkUnavailable, false)
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
	nodeInfo := schedulernodeinfo.NewNodeInfo()
	nodeInfo.SetNode(node)
	fit, _, err := predicates.PodToleratesNodeTaints(fakePod, nil, nodeInfo)
	if err != nil {
		e2elog.Failf("Can't test predicates for node %s: %v", node.Name, err)
		return false
	}
	return fit
}

// GetReadySchedulableNodesOrDie addresses the common use case of getting nodes you can do work on.
// 1) Needs to be schedulable.
// 2) Needs to be ready.
// If EITHER 1 or 2 is not true, most tests will want to ignore the node entirely.
// TODO: remove this function here when references point to e2enode.
func GetReadySchedulableNodesOrDie(c clientset.Interface) (nodes *v1.NodeList) {
	nodes = waitListSchedulableNodesOrDie(c)
	// previous tests may have cause failures of some nodes. Let's skip
	// 'Not Ready' nodes, just in case (there is no need to fail the test).
	e2enode.Filter(nodes, func(node v1.Node) bool {
		return isNodeSchedulable(&node) && isNodeUntainted(&node)
	})
	return nodes
}

// WaitForAllNodesSchedulable waits up to timeout for all
// (but TestContext.AllowedNotReadyNodes) to become scheduable.
func WaitForAllNodesSchedulable(c clientset.Interface, timeout time.Duration) error {
	e2elog.Logf("Waiting up to %v for all (but %d) nodes to be schedulable", timeout, TestContext.AllowedNotReadyNodes)

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
			e2elog.Logf("Unexpected error listing nodes: %v", err)
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
			if len(nodes.Items) < largeClusterThreshold || attempt%10 == 0 {
				e2elog.Logf("Unschedulable nodes:")
				for i := range notSchedulable {
					e2elog.Logf("-> %s Ready=%t Network=%t Taints=%v",
						notSchedulable[i].Name,
						e2enode.IsConditionSetAsExpectedSilent(notSchedulable[i], v1.NodeReady, true),
						e2enode.IsConditionSetAsExpectedSilent(notSchedulable[i], v1.NodeNetworkUnavailable, false),
						notSchedulable[i].Spec.Taints)
				}
				e2elog.Logf("================================")
			}
		}
		return len(notSchedulable) <= TestContext.AllowedNotReadyNodes, nil
	})
}

// GetPodSecretUpdateTimeout reuturns the timeout duration for updating pod secret.
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

// AddOrUpdateLabelOnNode adds the given label key and value to the given node or updates value.
func AddOrUpdateLabelOnNode(c clientset.Interface, nodeName string, labelKey, labelValue string) {
	ExpectNoError(testutils.AddLabelsToNode(c, nodeName, map[string]string{labelKey: labelValue}))
}

// AddOrUpdateLabelOnNodeAndReturnOldValue adds the given label key and value to the given node or updates value and returns the old label value.
func AddOrUpdateLabelOnNodeAndReturnOldValue(c clientset.Interface, nodeName string, labelKey, labelValue string) string {
	var oldValue string
	node, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	ExpectNoError(err)
	oldValue = node.Labels[labelKey]
	ExpectNoError(testutils.AddLabelsToNode(c, nodeName, map[string]string{labelKey: labelValue}))
	return oldValue
}

// ExpectNodeHasLabel expects that the given node has the given label pair.
func ExpectNodeHasLabel(c clientset.Interface, nodeName string, labelKey string, labelValue string) {
	ginkgo.By("verifying the node has the label " + labelKey + " " + labelValue)
	node, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	ExpectNoError(err)
	ExpectEqual(node.Labels[labelKey], labelValue)
}

// RemoveTaintOffNode removes the given taint from the given node.
func RemoveTaintOffNode(c clientset.Interface, nodeName string, taint v1.Taint) {
	ExpectNoError(controller.RemoveTaintOffNode(c, nodeName, nil, &taint))
	verifyThatTaintIsGone(c, nodeName, &taint)
}

// AddOrUpdateTaintOnNode adds the given taint to the given node or updates taint.
func AddOrUpdateTaintOnNode(c clientset.Interface, nodeName string, taint v1.Taint) {
	ExpectNoError(controller.AddOrUpdateTaintOnNode(c, nodeName, &taint))
}

// RemoveLabelOffNode is for cleaning up labels temporarily added to node,
// won't fail if target label doesn't exist or has been removed.
func RemoveLabelOffNode(c clientset.Interface, nodeName string, labelKey string) {
	ginkgo.By("removing the label " + labelKey + " off the node " + nodeName)
	ExpectNoError(testutils.RemoveLabelOffNode(c, nodeName, []string{labelKey}))

	ginkgo.By("verifying the node doesn't have the label " + labelKey)
	ExpectNoError(testutils.VerifyLabelsRemoved(c, nodeName, []string{labelKey}))
}

func verifyThatTaintIsGone(c clientset.Interface, nodeName string, taint *v1.Taint) {
	ginkgo.By("verifying the node doesn't have the taint " + taint.ToString())
	nodeUpdated, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	ExpectNoError(err)
	if taintutils.TaintExists(nodeUpdated.Spec.Taints, taint) {
		e2elog.Failf("Failed removing taint " + taint.ToString() + " of the node " + nodeName)
	}
}

// ExpectNodeHasTaint expects that the node has the given taint.
func ExpectNodeHasTaint(c clientset.Interface, nodeName string, taint *v1.Taint) {
	ginkgo.By("verifying the node has the taint " + taint.ToString())
	if has, err := NodeHasTaint(c, nodeName, taint); !has {
		ExpectNoError(err)
		e2elog.Failf("Failed to find taint %s on node %s", taint.ToString(), nodeName)
	}
}

// NodeHasTaint returns true if the node has the given taint, else returns false.
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

// AddOrUpdateAvoidPodOnNode adds avoidPods annotations to node, will override if it exists
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
				e2elog.Logf("Conflict when trying to add/update avoidPods %v to %v with error %v", avoidPods, nodeName, err)
			}
		}
		return true, nil
	})
	ExpectNoError(err)
}

// RemoveAvoidPodsOffNode removes AvoidPods annotations from the node. It does not fail if no such annotation exists.
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
				e2elog.Logf("Conflict when trying to remove avoidPods to %v", nodeName)
			}
		}
		return true, nil
	})
	ExpectNoError(err)
}

// ScaleResource scales resource to the given size.
func ScaleResource(
	clientset clientset.Interface,
	scalesGetter scaleclient.ScalesGetter,
	ns, name string,
	size uint,
	wait bool,
	kind schema.GroupKind,
	gr schema.GroupResource,
) error {
	ginkgo.By(fmt.Sprintf("Scaling %v %s in namespace %s to %d", kind, name, ns, size))
	if err := testutils.ScaleResourceWithRetries(scalesGetter, ns, name, size, gr); err != nil {
		return fmt.Errorf("error while scaling RC %s to %d replicas: %v", name, size, err)
	}
	if !wait {
		return nil
	}
	return e2epod.WaitForControlledPodsRunning(clientset, ns, name, kind)
}

// DeleteResourceAndWaitForGC deletes only given resource and waits for GC to delete the pods.
func DeleteResourceAndWaitForGC(c clientset.Interface, kind schema.GroupKind, ns, name string) error {
	ginkgo.By(fmt.Sprintf("deleting %v %s in namespace %s, will wait for the garbage collector to delete the pods", kind, name, ns))

	rtObject, err := e2eresource.GetRuntimeObjectForKind(c, kind, ns, name)
	if err != nil {
		if apierrs.IsNotFound(err) {
			e2elog.Logf("%v %s not found: %v", kind, name, err)
			return nil
		}
		return err
	}
	selector, err := e2eresource.GetSelectorFromRuntimeObject(rtObject)
	if err != nil {
		return err
	}
	replicas, err := e2eresource.GetReplicasFromRuntimeObject(rtObject)
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
	e2elog.Logf("Deleting %v %s took: %v", kind, name, deleteTime)

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

	err = e2epod.WaitForPodsInactive(ps, interval, timeout)
	if err != nil {
		return fmt.Errorf("error while waiting for pods to become inactive %s: %v", name, err)
	}
	terminatePodTime := time.Since(startTime) - deleteTime
	e2elog.Logf("Terminating %v %s pods took: %v", kind, name, terminatePodTime)

	// In gce, at any point, small percentage of nodes can disappear for
	// ~10 minutes due to hostError. 20 minutes should be long enough to
	// restart VM in that case and delete the pod.
	err = e2epod.WaitForPodsGone(ps, interval, 20*time.Minute)
	if err != nil {
		return fmt.Errorf("error while waiting for pods gone %s: %v", name, err)
	}
	return nil
}

type updateDSFunc func(*appsv1.DaemonSet)

// UpdateDaemonSetWithRetries updates daemonsets with the given applyUpdate func
// until it succeeds or a timeout expires.
func UpdateDaemonSetWithRetries(c clientset.Interface, namespace, name string, applyUpdate updateDSFunc) (ds *appsv1.DaemonSet, err error) {
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
			e2elog.Logf("Updating DaemonSet %s", name)
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

// RunHostCmd runs the given cmd in the context of the given pod using `kubectl exec`
// inside of a shell.
func RunHostCmd(ns, name, cmd string) (string, error) {
	return RunKubectl("exec", fmt.Sprintf("--namespace=%v", ns), name, "--", "/bin/sh", "-x", "-c", cmd)
}

// RunHostCmdOrDie calls RunHostCmd and dies on error.
func RunHostCmdOrDie(ns, name, cmd string) string {
	stdout, err := RunHostCmd(ns, name, cmd)
	e2elog.Logf("stdout: %v", stdout)
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
		e2elog.Logf("Waiting %v to retry failed RunHostCmd: %v", interval, err)
		time.Sleep(interval)
	}
}

// AllNodesReady checks whether all registered nodes are ready.
// TODO: we should change the AllNodesReady call in AfterEach to WaitForAllNodesHealthy,
// and figure out how to do it in a configurable way, as we can't expect all setups to run
// default test add-ons.
func AllNodesReady(c clientset.Interface, timeout time.Duration) error {
	e2elog.Logf("Waiting up to %v for all (but %d) nodes to be ready", timeout, TestContext.AllowedNotReadyNodes)

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
			if !e2enode.IsConditionSetAsExpected(node, v1.NodeReady, true) {
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

// RestartKubeProxy restarts kube-proxy on the given host.
func RestartKubeProxy(host string) error {
	// TODO: Make it work for all providers.
	if !ProviderIs("gce", "gke", "aws") {
		return fmt.Errorf("unsupported provider for RestartKubeProxy: %s", TestContext.Provider)
	}
	// kubelet will restart the kube-proxy since it's running in a static pod
	e2elog.Logf("Killing kube-proxy on node %v", host)
	result, err := e2essh.SSH("sudo pkill kube-proxy", host, TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't restart kube-proxy: %v", err)
	}
	// wait for kube-proxy to come back up
	sshCmd := "sudo /bin/sh -c 'pgrep kube-proxy | wc -l'"
	err = wait.Poll(5*time.Second, 60*time.Second, func() (bool, error) {
		e2elog.Logf("Waiting for kubeproxy to come back up with %v on %v", sshCmd, host)
		result, err := e2essh.SSH(sshCmd, host, TestContext.Provider)
		if err != nil {
			return false, err
		}
		if result.Code != 0 {
			e2essh.LogResult(result)
			return false, fmt.Errorf("failed to run command, exited %d", result.Code)
		}
		if result.Stdout == "0\n" {
			return false, nil
		}
		e2elog.Logf("kube-proxy is back up.")
		return true, nil
	})
	if err != nil {
		return fmt.Errorf("kube-proxy didn't recover: %v", err)
	}
	return nil
}

// RestartKubelet restarts kubelet on the given host.
func RestartKubelet(host string) error {
	// TODO: Make it work for all providers and distros.
	supportedProviders := []string{"gce", "aws", "vsphere"}
	if !ProviderIs(supportedProviders...) {
		return fmt.Errorf("unsupported provider for RestartKubelet: %s, supported providers are: %v", TestContext.Provider, supportedProviders)
	}
	if ProviderIs("gce") && !NodeOSDistroIs("debian", "gci") {
		return fmt.Errorf("unsupported node OS distro: %s", TestContext.NodeOSDistro)
	}
	var cmd string

	if ProviderIs("gce") && NodeOSDistroIs("debian") {
		cmd = "sudo /etc/init.d/kubelet restart"
	} else if ProviderIs("vsphere") {
		var sudoPresent bool
		sshResult, err := e2essh.SSH("sudo --version", host, TestContext.Provider)
		if err != nil {
			return fmt.Errorf("Unable to ssh to host %s with error %v", host, err)
		}
		if !strings.Contains(sshResult.Stderr, "command not found") {
			sudoPresent = true
		}
		sshResult, err = e2essh.SSH("systemctl --version", host, TestContext.Provider)
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
	e2elog.Logf("Restarting kubelet via ssh on host %s with command %s", host, cmd)
	result, err := e2essh.SSH(cmd, host, TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't restart kubelet: %v", err)
	}
	return nil
}

// WaitForKubeletUp waits for the kubelet on the given host to be up.
func WaitForKubeletUp(host string) error {
	cmd := "curl http://localhost:" + strconv.Itoa(ports.KubeletReadOnlyPort) + "/healthz"
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		result, err := e2essh.SSH(cmd, host, TestContext.Provider)
		if err != nil || result.Code != 0 {
			e2essh.LogResult(result)
		}
		if result.Stdout == "ok" {
			return nil
		}
	}
	return fmt.Errorf("waiting for kubelet timed out")
}

// RestartApiserver restarts the kube-apiserver.
func RestartApiserver(cs clientset.Interface) error {
	// TODO: Make it work for all providers.
	if !ProviderIs("gce", "gke", "aws") {
		return fmt.Errorf("unsupported provider for RestartApiserver: %s", TestContext.Provider)
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
		return fmt.Errorf("unsupported provider for sshRestartMaster: %s", TestContext.Provider)
	}
	var command string
	if ProviderIs("gce") {
		command = "pidof kube-apiserver | xargs sudo kill"
	} else {
		command = "sudo /etc/init.d/kube-apiserver restart"
	}
	e2elog.Logf("Restarting master via ssh, running: %v", command)
	result, err := e2essh.SSH(command, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't restart apiserver: %v", err)
	}
	return nil
}

// WaitForApiserverUp waits for the kube-apiserver to be up.
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
			e2elog.Logf("Failed to get apiserver's restart count: %v", err)
			continue
		}
		if restartCount > initialRestartCount {
			e2elog.Logf("Apiserver has restarted.")
			return nil
		}
		e2elog.Logf("Waiting for apiserver restart count to increase")
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
	return -1, fmt.Errorf("Failed to find kube-apiserver container in pod")
}

// RestartControllerManager restarts the kube-controller-manager.
func RestartControllerManager() error {
	// TODO: Make it work for all providers and distros.
	if !ProviderIs("gce", "aws") {
		return fmt.Errorf("unsupported provider for RestartControllerManager: %s", TestContext.Provider)
	}
	if ProviderIs("gce") && !MasterOSDistroIs("gci") {
		return fmt.Errorf("unsupported master OS distro: %s", TestContext.MasterOSDistro)
	}
	cmd := "pidof kube-controller-manager | xargs sudo kill"
	e2elog.Logf("Restarting controller-manager via ssh, running: %v", cmd)
	result, err := e2essh.SSH(cmd, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
	if err != nil || result.Code != 0 {
		e2essh.LogResult(result)
		return fmt.Errorf("couldn't restart controller-manager: %v", err)
	}
	return nil
}

// WaitForControllerManagerUp waits for the kube-controller-manager to be up.
func WaitForControllerManagerUp() error {
	cmd := "curl http://localhost:" + strconv.Itoa(ports.InsecureKubeControllerManagerPort) + "/healthz"
	for start := time.Now(); time.Since(start) < time.Minute; time.Sleep(5 * time.Second) {
		result, err := e2essh.SSH(cmd, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
		if err != nil || result.Code != 0 {
			e2essh.LogResult(result)
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
		result, err := e2essh.SSH(cmd, net.JoinHostPort(GetMasterHost(), sshPort), TestContext.Provider)
		if err != nil {
			// We don't necessarily know that it crashed, pipe could just be broken
			e2essh.LogResult(result)
			return fmt.Errorf("master unreachable after %v", time.Since(start))
		} else if result.Code != 0 {
			e2essh.LogResult(result)
			return fmt.Errorf("SSH result code not 0. actually: %v after %v", result.Code, time.Since(start))
		} else if result.Stdout != PID {
			if PID == "" {
				PID = result.Stdout
			} else {
				//its dead
				return fmt.Errorf("controller manager crashed, old PID: %s, new PID: %s", PID, result.Stdout)
			}
		} else {
			e2elog.Logf("kube-controller-manager still healthy after %v", time.Since(start))
		}
	}
	return nil
}

// GenerateMasterRegexp returns a regex for matching master node name.
func GenerateMasterRegexp(prefix string) string {
	return prefix + "(-...)?"
}

// WaitForMasters waits until the cluster has the desired number of ready masters in it.
func WaitForMasters(masterPrefix string, c clientset.Interface, size int, timeout time.Duration) error {
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
		if err != nil {
			e2elog.Logf("Failed to list nodes: %v", err)
			continue
		}

		// Filter out nodes that are not master replicas
		e2enode.Filter(nodes, func(node v1.Node) bool {
			res, err := regexp.Match(GenerateMasterRegexp(masterPrefix), ([]byte)(node.Name))
			if err != nil {
				e2elog.Logf("Failed to match regexp to node name: %v", err)
				return false
			}
			return res
		})

		numNodes := len(nodes.Items)

		// Filter out not-ready nodes.
		e2enode.Filter(nodes, func(node v1.Node) bool {
			return e2enode.IsConditionSetAsExpected(&node, v1.NodeReady, true)
		})

		numReady := len(nodes.Items)

		if numNodes == size && numReady == size {
			e2elog.Logf("Cluster has reached the desired number of masters %d", size)
			return nil
		}
		e2elog.Logf("Waiting for the number of masters %d, current %d, not ready master nodes %d", size, numNodes, numNodes-numReady)
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

// GetHostAddress gets the node for a pod and returns the first
// address. Returns an error if the node the pod is on doesn't have an
// address.
func GetHostAddress(client clientset.Interface, p *v1.Pod) (string, error) {
	node, err := client.CoreV1().Nodes().Get(p.Spec.NodeName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	// Try externalAddress first
	for _, address := range node.Status.Addresses {
		if address.Type == v1.NodeExternalIP {
			if address.Address != "" {
				return address.Address, nil
			}
		}
	}
	// If no externalAddress found, try internalAddress
	for _, address := range node.Status.Addresses {
		if address.Type == v1.NodeInternalIP {
			if address.Address != "" {
				return address.Address, nil
			}
		}
	}

	// If not found, return error
	return "", fmt.Errorf("No address for pod %v on node %v",
		p.Name, p.Spec.NodeName)
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
func headersForConfig(c *restclient.Config, url *url.URL) (http.Header, error) {
	extract := &extractRT{}
	rt, err := restclient.HTTPWrappersForConfig(c, extract)
	if err != nil {
		return nil, err
	}
	if _, err := rt.RoundTrip(&http.Request{URL: url}); err != nil {
		return nil, err
	}
	return extract.Header, nil
}

// OpenWebSocketForURL constructs a websocket connection to the provided URL, using the client
// config, with the specified protocols.
func OpenWebSocketForURL(url *url.URL, config *restclient.Config, protocols []string) (*websocket.Conn, error) {
	tlsConfig, err := restclient.TLSConfigFor(config)
	if err != nil {
		return nil, fmt.Errorf("Failed to create tls config: %v", err)
	}
	if url.Scheme == "https" {
		url.Scheme = "wss"
	} else {
		url.Scheme = "ws"
	}
	headers, err := headersForConfig(config, url)
	if err != nil {
		return nil, fmt.Errorf("Failed to load http headers: %v", err)
	}
	cfg, err := websocket.NewConfig(url.String(), "http://localhost")
	if err != nil {
		return nil, fmt.Errorf("Failed to create websocket config: %v", err)
	}
	cfg.Header = headers
	cfg.TlsConfig = tlsConfig
	cfg.Protocol = protocols
	return websocket.DialConfig(cfg)
}

// LookForStringInLog looks for the given string in the log of a specific pod container
func LookForStringInLog(ns, podName, container, expectedString string, timeout time.Duration) (result string, err error) {
	return LookForString(expectedString, timeout, func() string {
		return RunKubectlOrDie("logs", podName, container, fmt.Sprintf("--namespace=%v", ns))
	})
}

// LookForStringInFile looks for the given string in a file in a specific pod container
func LookForStringInFile(ns, podName, container, file, expectedString string, timeout time.Duration) (result string, err error) {
	return LookForString(expectedString, timeout, func() string {
		return RunKubectlOrDie("exec", podName, "-c", container, fmt.Sprintf("--namespace=%v", ns), "--", "cat", file)
	})
}

// EnsureLoadBalancerResourcesDeleted ensures that cloud load balancer resources that were created
// are actually cleaned up.  Currently only implemented for GCE/GKE.
func EnsureLoadBalancerResourcesDeleted(ip, portRange string) error {
	return TestContext.CloudConfig.Provider.EnsureLoadBalancerResourcesDeleted(ip, portRange)
}

// BlockNetwork blocks network between the given from value and the given to value.
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
	e2elog.Logf("block network traffic from %s to %s", from, to)
	iptablesRule := fmt.Sprintf("OUTPUT --destination %s --jump REJECT", to)
	dropCmd := fmt.Sprintf("sudo iptables --insert %s", iptablesRule)
	if result, err := e2essh.SSH(dropCmd, from, TestContext.Provider); result.Code != 0 || err != nil {
		e2essh.LogResult(result)
		e2elog.Failf("Unexpected error: %v", err)
	}
}

// UnblockNetwork unblocks network between the given from value and the given to value.
func UnblockNetwork(from string, to string) {
	e2elog.Logf("Unblock network traffic from %s to %s", from, to)
	iptablesRule := fmt.Sprintf("OUTPUT --destination %s --jump REJECT", to)
	undropCmd := fmt.Sprintf("sudo iptables --delete %s", iptablesRule)
	// Undrop command may fail if the rule has never been created.
	// In such case we just lose 30 seconds, but the cluster is healthy.
	// But if the rule had been created and removing it failed, the node is broken and
	// not coming back. Subsequent tests will run or fewer nodes (some of the tests
	// may fail). Manual intervention is required in such case (recreating the
	// cluster solves the problem too).
	err := wait.Poll(time.Millisecond*100, time.Second*30, func() (bool, error) {
		result, err := e2essh.SSH(undropCmd, from, TestContext.Provider)
		if result.Code == 0 && err == nil {
			return true, nil
		}
		e2essh.LogResult(result)
		if err != nil {
			e2elog.Logf("Unexpected error: %v", err)
		}
		return false, nil
	})
	if err != nil {
		e2elog.Failf("Failed to remove the iptable REJECT rule. Manual intervention is "+
			"required on host %s: remove rule %s, if exists", from, iptablesRule)
	}
}

// PingCommand is the type to hold ping command.
type PingCommand string

const (
	// IPv4PingCommand is a ping command for IPv4.
	IPv4PingCommand PingCommand = "ping"
	// IPv6PingCommand is a ping command for IPv6.
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
	err = e2epod.WaitForPodSuccessInNamespace(f.ClientSet, podName, f.Namespace.Name)

	if err != nil {
		logs, logErr := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, contName)
		if logErr != nil {
			e2elog.Logf("Warning: Failed to get logs from pod %q: %v", pod.Name, logErr)
		} else {
			e2elog.Logf("pod %s/%s logs:\n%s", f.Namespace.Name, pod.Name, logs)
		}
	}

	return err
}

// CoreDump SSHs to the master and all nodes and dumps their logs into dir.
// It shells out to cluster/log-dump/log-dump.sh to accomplish this.
func CoreDump(dir string) {
	if TestContext.DisableLogDump {
		e2elog.Logf("Skipping dumping logs from cluster")
		return
	}
	var cmd *exec.Cmd
	if TestContext.LogexporterGCSPath != "" {
		e2elog.Logf("Dumping logs from nodes to GCS directly at path: %s", TestContext.LogexporterGCSPath)
		cmd = exec.Command(path.Join(TestContext.RepoRoot, "cluster", "log-dump", "log-dump.sh"), dir, TestContext.LogexporterGCSPath)
	} else {
		e2elog.Logf("Dumping logs locally to: %s", dir)
		cmd = exec.Command(path.Join(TestContext.RepoRoot, "cluster", "log-dump", "log-dump.sh"), dir)
	}
	cmd.Env = append(os.Environ(), fmt.Sprintf("LOG_DUMP_SYSTEMD_SERVICES=%s", parseSystemdServices(TestContext.SystemdServices)))
	cmd.Env = append(os.Environ(), fmt.Sprintf("LOG_DUMP_SYSTEMD_JOURNAL=%v", TestContext.DumpSystemdJournal))

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		e2elog.Logf("Error running cluster/log-dump/log-dump.sh: %v", err)
	}
}

// parseSystemdServices converts services separator from comma to space.
func parseSystemdServices(services string) string {
	return strings.TrimSpace(strings.Replace(services, ",", " ", -1))
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
	e2elog.Logf("Running %s %v", command, args)
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
			e2elog.Logf("Got %v", err)
			return false, nil
		}
		return true, nil
	})
	return stdout, stderr, err
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
	scheduledPods, currentlyNotScheduledPods := e2epod.GetPodsScheduled(masterNodes, allPods)
	for len(currentlyNotScheduledPods) != 0 {
		time.Sleep(2 * time.Second)

		allPods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{})
		ExpectNoError(err)
		scheduledPods, currentlyNotScheduledPods = e2epod.GetPodsScheduled(masterNodes, allPods)

		if startTime.Add(timeout).Before(time.Now()) {
			e2elog.Failf("Timed out after %v waiting for stable cluster.", timeout)
			break
		}
	}
	return len(scheduledPods)
}

// ListNamespaceEvents lists the events in the given namespace.
func ListNamespaceEvents(c clientset.Interface, ns string) error {
	ls, err := c.CoreV1().Events(ns).List(metav1.ListOptions{})
	if err != nil {
		return err
	}
	for _, event := range ls.Items {
		klog.Infof("Event(%#v): type: '%v' reason: '%v' %v", event.InvolvedObject, event.Type, event.Reason, event.Message)
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

// NewE2ETestNodePreparer returns a new instance of E2ETestNodePreparer.
func NewE2ETestNodePreparer(client clientset.Interface, countToStrategy []testutils.CountToStrategy) testutils.TestNodePreparer {
	return &E2ETestNodePreparer{
		client:                client,
		countToStrategy:       countToStrategy,
		nodeToAppliedStrategy: make(map[string]testutils.PrepareNodeStrategy),
	}
}

// PrepareNodes prepares nodes in the cluster.
func (p *E2ETestNodePreparer) PrepareNodes() error {
	nodes := GetReadySchedulableNodesOrDie(p.client)
	numTemplates := 0
	for _, v := range p.countToStrategy {
		numTemplates += v.Count
	}
	if numTemplates > len(nodes.Items) {
		return fmt.Errorf("Can't prepare Nodes. Got more templates than existing Nodes")
	}
	index := 0
	sum := 0
	for _, v := range p.countToStrategy {
		sum += v.Count
		for ; index < sum; index++ {
			if err := testutils.DoPrepareNode(p.client, &nodes.Items[index], v.Strategy); err != nil {
				klog.Errorf("Aborting node preparation: %v", err)
				return err
			}
			p.nodeToAppliedStrategy[nodes.Items[index].Name] = v.Strategy
		}
	}
	return nil
}

// CleanupNodes cleanups nodes in the cluster.
func (p *E2ETestNodePreparer) CleanupNodes() error {
	var encounteredError error
	nodes := GetReadySchedulableNodesOrDie(p.client)
	for i := range nodes.Items {
		var err error
		name := nodes.Items[i].Name
		strategy, found := p.nodeToAppliedStrategy[name]
		if found {
			if err = testutils.DoCleanupNode(p.client, name, strategy); err != nil {
				klog.Errorf("Skipping cleanup of Node: failed update of %v: %v", name, err)
				encounteredError = err
			}
		}
	}
	return encounteredError
}

// getMasterAddresses returns the externalIP, internalIP and hostname fields of the master.
// If any of these is unavailable, it is set to "".
func getMasterAddresses(c clientset.Interface) (string, string, string) {
	var externalIP, internalIP, hostname string

	// Populate the internal IP.
	eps, err := c.CoreV1().Endpoints(metav1.NamespaceDefault).Get("kubernetes", metav1.GetOptions{})
	if err != nil {
		e2elog.Failf("Failed to get kubernetes endpoints: %v", err)
	}
	if len(eps.Subsets) != 1 || len(eps.Subsets[0].Addresses) != 1 {
		e2elog.Failf("There are more than 1 endpoints for kubernetes service: %+v", eps)
	}
	internalIP = eps.Subsets[0].Addresses[0].IP

	// Populate the external IP/hostname.
	hostURL, err := url.Parse(TestContext.Host)
	if err != nil {
		e2elog.Failf("Failed to parse hostname: %v", err)
	}
	if net.ParseIP(hostURL.Host) != nil {
		externalIP = hostURL.Host
	} else {
		hostname = hostURL.Host
	}

	return externalIP, internalIP, hostname
}

// GetAllMasterAddresses returns all IP addresses on which the kubelet can reach the master.
// It may return internal and external IPs, even if we expect for
// e.g. internal IPs to be used (issue #56787), so that we can be
// sure to block the master fully during tests.
func GetAllMasterAddresses(c clientset.Interface) []string {
	externalIP, internalIP, _ := getMasterAddresses(c)

	ips := sets.NewString()
	switch TestContext.Provider {
	case "gce", "gke":
		if externalIP != "" {
			ips.Insert(externalIP)
		}
		if internalIP != "" {
			ips.Insert(internalIP)
		}
	case "aws":
		ips.Insert(awsMasterIP)
	default:
		e2elog.Failf("This test is not supported for provider %s and should be disabled", TestContext.Provider)
	}
	return ips.List()
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
			e2elog.Logf("host %v path %v: %v unreachable", host, route, err)
			return expectUnreachable, nil
		}
		e2elog.Logf("host %v path %v: reached", host, route)
		return !expectUnreachable, nil
	})
	if pollErr != nil {
		return fmt.Errorf("Failed to execute a successful GET within %v, Last response body for %v, host %v:\n%v\n\n%v",
			timeout, route, host, lastBody, pollErr)
	}
	return nil
}

// DescribeIng describes information of ingress by running kubectl describe ing.
func DescribeIng(ns string) {
	e2elog.Logf("\nOutput of kubectl describe ing:\n")
	desc, _ := RunKubectl(
		"describe", "ing", fmt.Sprintf("--namespace=%v", ns))
	e2elog.Logf(desc)
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

// NewAgnhostPod returns a pod that uses the agnhost image. The image's binary supports various subcommands
// that behave the same, no matter the underlying OS.
func (f *Framework) NewAgnhostPod(name string, args ...string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "agnhost",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  args,
				},
			},
		},
	}
}

// CreateEmptyFileOnPod creates empty file at given path on the pod.
// TODO(alejandrox1): move to subpkg pod once kubectl methods have been refactored.
func CreateEmptyFileOnPod(namespace string, podName string, filePath string) error {
	_, err := RunKubectl("exec", fmt.Sprintf("--namespace=%s", namespace), podName, "--", "/bin/sh", "-c", fmt.Sprintf("touch %s", filePath))
	return err
}

// PrintSummaries prints summaries of tests.
func PrintSummaries(summaries []TestDataSummary, testBaseName string) {
	now := time.Now()
	for i := range summaries {
		e2elog.Logf("Printing summary: %v", summaries[i].SummaryKind())
		switch TestContext.OutputPrintType {
		case "hr":
			if TestContext.ReportDir == "" {
				e2elog.Logf(summaries[i].PrintHumanReadable())
			} else {
				// TODO: learn to extract test name and append it to the kind instead of timestamp.
				filePath := path.Join(TestContext.ReportDir, summaries[i].SummaryKind()+"_"+testBaseName+"_"+now.Format(time.RFC3339)+".txt")
				if err := ioutil.WriteFile(filePath, []byte(summaries[i].PrintHumanReadable()), 0644); err != nil {
					e2elog.Logf("Failed to write file %v with test performance data: %v", filePath, err)
				}
			}
		case "json":
			fallthrough
		default:
			if TestContext.OutputPrintType != "json" {
				e2elog.Logf("Unknown output type: %v. Printing JSON", TestContext.OutputPrintType)
			}
			if TestContext.ReportDir == "" {
				e2elog.Logf("%v JSON\n%v", summaries[i].SummaryKind(), summaries[i].PrintJSON())
				e2elog.Logf("Finished")
			} else {
				// TODO: learn to extract test name and append it to the kind instead of timestamp.
				filePath := path.Join(TestContext.ReportDir, summaries[i].SummaryKind()+"_"+testBaseName+"_"+now.Format(time.RFC3339)+".json")
				e2elog.Logf("Writing to %s", filePath)
				if err := ioutil.WriteFile(filePath, []byte(summaries[i].PrintJSON()), 0644); err != nil {
					e2elog.Logf("Failed to write file %v with test performance data: %v", filePath, err)
				}
			}
		}
	}
}

// DumpDebugInfo dumps debug info of tests.
func DumpDebugInfo(c clientset.Interface, ns string) {
	sl, _ := c.CoreV1().Pods(ns).List(metav1.ListOptions{LabelSelector: labels.Everything().String()})
	for _, s := range sl.Items {
		desc, _ := RunKubectl("describe", "po", s.Name, fmt.Sprintf("--namespace=%v", ns))
		e2elog.Logf("\nOutput of kubectl describe %v:\n%v", s.Name, desc)

		l, _ := RunKubectl("logs", s.Name, fmt.Sprintf("--namespace=%v", ns), "--tail=100")
		e2elog.Logf("\nLast 100 log lines of %v:\n%v", s.Name, l)
	}
}

// DsFromManifest reads a .json/yaml file and returns the daemonset in it.
func DsFromManifest(url string) (*appsv1.DaemonSet, error) {
	var ds appsv1.DaemonSet
	e2elog.Logf("Parsing ds from %v", url)

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
		return nil, fmt.Errorf("Failed to get url: %v", err)
	}
	if response.StatusCode != 200 {
		return nil, fmt.Errorf("invalid http response status: %v", response.StatusCode)
	}
	defer response.Body.Close()

	data, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return nil, fmt.Errorf("Failed to read html response body: %v", err)
	}

	dataJSON, err := utilyaml.ToJSON(data)
	if err != nil {
		return nil, fmt.Errorf("Failed to parse data to json: %v", err)
	}

	err = runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), dataJSON, &ds)
	if err != nil {
		return nil, fmt.Errorf("Failed to decode DaemonSet spec: %v", err)
	}
	return &ds, nil
}

// waitForServerPreferredNamespacedResources waits until server preferred namespaced resources could be successfully discovered.
// TODO: Fix https://github.com/kubernetes/kubernetes/issues/55768 and remove the following retry.
func waitForServerPreferredNamespacedResources(d discovery.DiscoveryInterface, timeout time.Duration) ([]*metav1.APIResourceList, error) {
	e2elog.Logf("Waiting up to %v for server preferred namespaced resources to be successfully discovered", timeout)
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
		e2elog.Logf("Error discoverying server preferred namespaced resources: %v, retrying in %v.", err, Poll)
		return false, nil
	}); err != nil {
		return nil, err
	}
	return resources, nil
}

// WaitForPersistentVolumeClaimDeleted waits for a PersistentVolumeClaim to be removed from the system until timeout occurs, whichever comes first.
func WaitForPersistentVolumeClaimDeleted(c clientset.Interface, ns string, pvcName string, Poll, timeout time.Duration) error {
	e2elog.Logf("Waiting up to %v for PersistentVolumeClaim %s to be removed", timeout, pvcName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		_, err := c.CoreV1().PersistentVolumeClaims(ns).Get(pvcName, metav1.GetOptions{})
		if err != nil {
			if apierrs.IsNotFound(err) {
				e2elog.Logf("Claim %q in namespace %q doesn't exist in the system", pvcName, ns)
				return nil
			}
			e2elog.Logf("Failed to get claim %q in namespace %q, retrying in %v. Error: %v", pvcName, ns, Poll, err)
		}
	}
	return fmt.Errorf("PersistentVolumeClaim %s is not removed from the system within %v", pvcName, timeout)
}

// GetClusterZones returns the values of zone label collected from all nodes.
func GetClusterZones(c clientset.Interface) (sets.String, error) {
	nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("Error getting nodes while attempting to list cluster zones: %v", err)
	}

	// collect values of zone label from all nodes
	zones := sets.NewString()
	for _, node := range nodes.Items {
		if zone, found := node.Labels[v1.LabelZoneFailureDomain]; found {
			zones.Insert(zone)
		}
	}
	return zones, nil
}

// WaitForNodeHasTaintOrNot waits for a taint to be added/removed from the node until timeout occurs, whichever comes first.
func WaitForNodeHasTaintOrNot(c clientset.Interface, nodeName string, taint *v1.Taint, wantTrue bool, timeout time.Duration) error {
	if err := wait.PollImmediate(Poll, timeout, func() (bool, error) {
		has, err := NodeHasTaint(c, nodeName, taint)
		if err != nil {
			return false, fmt.Errorf("Failed to check taint %s on node %s or not", taint.ToString(), nodeName)
		}
		return has == wantTrue, nil
	}); err != nil {
		return fmt.Errorf("expect node %v to have taint = %v within %v: %v", nodeName, wantTrue, timeout, err)
	}
	return nil
}

// GetFileModeRegex returns a file mode related regex which should be matched by the mounttest pods' output.
// If the given mask is nil, then the regex will contain the default OS file modes, which are 0644 for Linux and 0775 for Windows.
func GetFileModeRegex(filePath string, mask *int32) string {
	var (
		linuxMask   int32
		windowsMask int32
	)
	if mask == nil {
		linuxMask = int32(0644)
		windowsMask = int32(0775)
	} else {
		linuxMask = *mask
		windowsMask = *mask
	}

	linuxOutput := fmt.Sprintf("mode of file \"%s\": %v", filePath, os.FileMode(linuxMask))
	windowsOutput := fmt.Sprintf("mode of Windows file \"%v\": %s", filePath, os.FileMode(windowsMask))

	return fmt.Sprintf("(%s|%s)", linuxOutput, windowsOutput)
}
