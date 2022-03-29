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

package e2enode

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/featuregate"
	internalapi "k8s.io/cri-api/pkg/apis"
	"k8s.io/klog/v2"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	kubeletpodresourcesv1alpha1 "k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/cluster/ports"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/cri/remote"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/util"

	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2enodekubelet "k8s.io/kubernetes/test/e2e_node/kubeletconfig"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var startServices = flag.Bool("start-services", true, "If true, start local node services")
var stopServices = flag.Bool("stop-services", true, "If true, stop local node services after running tests")
var busyboxImage = imageutils.GetE2EImage(imageutils.BusyBox)

const (
	// Kubelet internal cgroup name for node allocatable cgroup.
	defaultNodeAllocatableCgroup = "kubepods"
	// defaultPodResourcesPath is the path to the local endpoint serving the podresources GRPC service.
	defaultPodResourcesPath    = "/var/lib/kubelet/pod-resources"
	defaultPodResourcesTimeout = 10 * time.Second
	defaultPodResourcesMaxSize = 1024 * 1024 * 16 // 16 Mb
	// state files
	cpuManagerStateFile    = "/var/lib/kubelet/cpu_manager_state"
	memoryManagerStateFile = "/var/lib/kubelet/memory_manager_state"
)

var kubeletHealthCheckURL = fmt.Sprintf("http://127.0.0.1:%d/healthz", ports.KubeletHealthzPort)

func getNodeSummary() (*stats.Summary, error) {
	kubeletConfig, err := getCurrentKubeletConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to get current kubelet config")
	}
	req, err := http.NewRequest("GET", fmt.Sprintf("http://%s/stats/summary", net.JoinHostPort(kubeletConfig.Address, strconv.Itoa(int(kubeletConfig.ReadOnlyPort)))), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to build http request: %v", err)
	}
	req.Header.Add("Accept", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get /stats/summary: %v", err)
	}

	defer resp.Body.Close()
	contentsBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read /stats/summary: %+v", resp)
	}

	decoder := json.NewDecoder(strings.NewReader(string(contentsBytes)))
	summary := stats.Summary{}
	err = decoder.Decode(&summary)
	if err != nil {
		return nil, fmt.Errorf("failed to parse /stats/summary to go struct: %+v", resp)
	}
	return &summary, nil
}

func getV1alpha1NodeDevices() (*kubeletpodresourcesv1alpha1.ListPodResourcesResponse, error) {
	endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
	if err != nil {
		return nil, fmt.Errorf("Error getting local endpoint: %v", err)
	}
	client, conn, err := podresources.GetV1alpha1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
	if err != nil {
		return nil, fmt.Errorf("Error getting grpc client: %v", err)
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	resp, err := client.List(ctx, &kubeletpodresourcesv1alpha1.ListPodResourcesRequest{})
	if err != nil {
		return nil, fmt.Errorf("%v.Get(_) = _, %v", client, err)
	}
	return resp, nil
}

func getV1NodeDevices() (*kubeletpodresourcesv1.ListPodResourcesResponse, error) {
	endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
	if err != nil {
		return nil, fmt.Errorf("Error getting local endpoint: %v", err)
	}
	client, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
	if err != nil {
		return nil, fmt.Errorf("Error getting gRPC client: %v", err)
	}
	defer conn.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	resp, err := client.List(ctx, &kubeletpodresourcesv1.ListPodResourcesRequest{})
	if err != nil {
		return nil, fmt.Errorf("%v.Get(_) = _, %v", client, err)
	}
	return resp, nil
}

// Returns the current KubeletConfiguration
func getCurrentKubeletConfig() (*kubeletconfig.KubeletConfiguration, error) {
	// namespace only relevant if useProxy==true, so we don't bother
	return e2ekubelet.GetCurrentKubeletConfig(framework.TestContext.NodeName, "", false)
}

// Must be called within a Context. Allows the function to modify the KubeletConfiguration during the BeforeEach of the context.
// The change is reverted in the AfterEach of the context.
// Returns true on success.
func tempSetCurrentKubeletConfig(f *framework.Framework, updateFunction func(initialConfig *kubeletconfig.KubeletConfiguration)) {
	var oldCfg *kubeletconfig.KubeletConfiguration

	ginkgo.BeforeEach(func() {
		var err error
		oldCfg, err = getCurrentKubeletConfig()
		framework.ExpectNoError(err)

		newCfg := oldCfg.DeepCopy()
		updateFunction(newCfg)
		if apiequality.Semantic.DeepEqual(*newCfg, *oldCfg) {
			return
		}

		updateKubeletConfig(f, newCfg, true)
	})

	ginkgo.AfterEach(func() {
		if oldCfg != nil {
			// Update the Kubelet configuration.
			updateKubeletConfig(f, oldCfg, true)
		}
	})
}

func updateKubeletConfig(f *framework.Framework, kubeletConfig *kubeletconfig.KubeletConfiguration, deleteStateFiles bool) {
	// Update the Kubelet configuration.
	ginkgo.By("Stopping the kubelet")
	startKubelet := stopKubelet()

	// wait until the kubelet health check will fail
	gomega.Eventually(func() bool {
		return kubeletHealthCheck(kubeletHealthCheckURL)
	}, time.Minute, time.Second).Should(gomega.BeFalse())

	// Delete CPU and memory manager state files to be sure it will not prevent the kubelet restart
	if deleteStateFiles {
		deleteStateFile(cpuManagerStateFile)
		deleteStateFile(memoryManagerStateFile)
	}

	framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(kubeletConfig))

	ginkgo.By("Starting the kubelet")
	startKubelet()

	// wait until the kubelet health check will succeed
	gomega.Eventually(func() bool {
		return kubeletHealthCheck(kubeletHealthCheckURL)
	}, 2*time.Minute, 5*time.Second).Should(gomega.BeTrue())

	// Wait for the Kubelet to be ready.
	gomega.Eventually(func() bool {
		nodes, err := e2enode.TotalReady(f.ClientSet)
		framework.ExpectNoError(err)
		return nodes == 1
	}, time.Minute, time.Second).Should(gomega.BeTrue())
}

func deleteStateFile(stateFileName string) {
	err := exec.Command("/bin/sh", "-c", fmt.Sprintf("rm -f %s", stateFileName)).Run()
	framework.ExpectNoError(err, "failed to delete the state file")
}

// listNamespaceEvents lists the events in the given namespace.
func listNamespaceEvents(c clientset.Interface, ns string) error {
	ls, err := c.CoreV1().Events(ns).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return err
	}
	for _, event := range ls.Items {
		klog.Infof("Event(%#v): type: '%v' reason: '%v' %v", event.InvolvedObject, event.Type, event.Reason, event.Message)
	}
	return nil
}

func logPodEvents(f *framework.Framework) {
	framework.Logf("Summary of pod events during the test:")
	err := listNamespaceEvents(f.ClientSet, f.Namespace.Name)
	framework.ExpectNoError(err)
}

func logNodeEvents(f *framework.Framework) {
	framework.Logf("Summary of node events during the test:")
	err := listNamespaceEvents(f.ClientSet, "")
	framework.ExpectNoError(err)
}

func getLocalNode(f *framework.Framework) *v1.Node {
	nodeList, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
	framework.ExpectNoError(err)
	framework.ExpectEqual(len(nodeList.Items), 1, "Unexpected number of node objects for node e2e. Expects only one node.")
	return &nodeList.Items[0]
}

// getLocalTestNode fetches the node object describing the local worker node set up by the e2e_node infra, alongside with its ready state.
// getLocalTestNode is a variant of `getLocalNode` which reports but does not set any requirement about the node readiness state, letting
// the caller decide. The check is intentionally done like `getLocalNode` does.
// Note `getLocalNode` aborts (as in ginkgo.Expect) the test implicitly if the worker node is not ready.
func getLocalTestNode(f *framework.Framework) (*v1.Node, bool) {
	node, err := f.ClientSet.CoreV1().Nodes().Get(context.TODO(), framework.TestContext.NodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	ready := e2enode.IsNodeReady(node)
	schedulable := e2enode.IsNodeSchedulable(node)
	framework.Logf("node %q ready=%v schedulable=%v", node.Name, ready, schedulable)
	return node, ready && schedulable
}

// logKubeletLatencyMetrics logs KubeletLatencyMetrics computed from the Prometheus
// metrics exposed on the current node and identified by the metricNames.
// The Kubelet subsystem prefix is automatically prepended to these metric names.
func logKubeletLatencyMetrics(metricNames ...string) {
	metricSet := sets.NewString()
	for _, key := range metricNames {
		metricSet.Insert(kubeletmetrics.KubeletSubsystem + "_" + key)
	}
	metric, err := e2emetrics.GrabKubeletMetricsWithoutProxy(fmt.Sprintf("%s:%d", framework.TestContext.NodeName, ports.KubeletReadOnlyPort), "/metrics")
	if err != nil {
		framework.Logf("Error getting kubelet metrics: %v", err)
	} else {
		framework.Logf("Kubelet Metrics: %+v", e2emetrics.GetKubeletLatencyMetrics(metric, metricSet))
	}
}

// runCommand runs the cmd and returns the combined stdout and stderr, or an
// error if the command failed.
func runCommand(cmd ...string) (string, error) {
	output, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("failed to run %q: %s (%s)", strings.Join(cmd, " "), err, output)
	}
	return string(output), nil
}

// getCRIClient connects CRI and returns CRI runtime service clients and image service client.
func getCRIClient() (internalapi.RuntimeService, internalapi.ImageManagerService, error) {
	// connection timeout for CRI service connection
	const connectionTimeout = 2 * time.Minute
	runtimeEndpoint := framework.TestContext.ContainerRuntimeEndpoint
	r, err := remote.NewRemoteRuntimeService(runtimeEndpoint, connectionTimeout)
	if err != nil {
		return nil, nil, err
	}
	imageManagerEndpoint := runtimeEndpoint
	if framework.TestContext.ImageServiceEndpoint != "" {
		//ImageServiceEndpoint is the same as ContainerRuntimeEndpoint if not
		//explicitly specified
		imageManagerEndpoint = framework.TestContext.ImageServiceEndpoint
	}
	i, err := remote.NewRemoteImageService(imageManagerEndpoint, connectionTimeout)
	if err != nil {
		return nil, nil, err
	}
	return r, i, nil
}

// findKubeletServiceName searches the unit name among the services known to systemd.
// if the `running` parameter is true, restricts the search among currently running services;
// otherwise, also stopped, failed, exited (non-running in general) services are also considered.
// TODO: Find a uniform way to deal with systemctl/initctl/service operations. #34494
func findKubeletServiceName(running bool) string {
	cmdLine := []string{
		"systemctl", "list-units", "*kubelet*",
	}
	if running {
		cmdLine = append(cmdLine, "--state=running")
	}
	stdout, err := exec.Command("sudo", cmdLine...).CombinedOutput()
	framework.ExpectNoError(err)
	regex := regexp.MustCompile("(kubelet-\\w+)")
	matches := regex.FindStringSubmatch(string(stdout))
	framework.ExpectNotEqual(len(matches), 0, "Found more than one kubelet service running: %q", stdout)
	kubeletServiceName := matches[0]
	framework.Logf("Get running kubelet with systemctl: %v, %v", string(stdout), kubeletServiceName)
	return kubeletServiceName
}

// restartKubelet restarts the current kubelet service.
// the "current" kubelet service is the instance managed by the current e2e_node test run.
// If `running` is true, restarts only if the current kubelet is actually running. In some cases,
// the kubelet may have exited or can be stopped, typically because it was intentionally stopped
// earlier during a test, or, sometimes, because it just crashed.
// Warning: the "current" kubelet is poorly defined. The "current" kubelet is assumed to be the most
// recent kubelet service unit, IOW there is not a unique ID we use to bind explicitly a kubelet
// instance to a test run.
func restartKubelet(running bool) {
	kubeletServiceName := findKubeletServiceName(running)
	// reset the kubelet service start-limit-hit
	stdout, err := exec.Command("sudo", "systemctl", "reset-failed", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to reset kubelet start-limit-hit with systemctl: %v, %s", err, string(stdout))

	stdout, err = exec.Command("sudo", "systemctl", "restart", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to restart kubelet with systemctl: %v, %s", err, string(stdout))
}

// stopKubelet will kill the running kubelet, and returns a func that will restart the process again
func stopKubelet() func() {
	kubeletServiceName := findKubeletServiceName(true)

	// reset the kubelet service start-limit-hit
	stdout, err := exec.Command("sudo", "systemctl", "reset-failed", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to reset kubelet start-limit-hit with systemctl: %v, %s", err, string(stdout))

	stdout, err = exec.Command("sudo", "systemctl", "kill", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to stop kubelet with systemctl: %v, %s", err, string(stdout))

	return func() {
		// we should restart service, otherwise the transient service start will fail
		stdout, err := exec.Command("sudo", "systemctl", "restart", kubeletServiceName).CombinedOutput()
		framework.ExpectNoError(err, "Failed to restart kubelet with systemctl: %v, %v", err, stdout)
	}
}

// killKubelet sends a signal (SIGINT, SIGSTOP, SIGTERM...) to the running kubelet
func killKubelet(sig string) {
	kubeletServiceName := findKubeletServiceName(true)

	// reset the kubelet service start-limit-hit
	stdout, err := exec.Command("sudo", "systemctl", "reset-failed", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to reset kubelet start-limit-hit with systemctl: %v, %v", err, stdout)

	stdout, err = exec.Command("sudo", "systemctl", "kill", "-s", sig, kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to stop kubelet with systemctl: %v, %v", err, stdout)
}

func kubeletHealthCheck(url string) bool {
	insecureTransport := http.DefaultTransport.(*http.Transport).Clone()
	insecureTransport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	insecureHTTPClient := &http.Client{
		Transport: insecureTransport,
	}

	req, err := http.NewRequest("HEAD", url, nil)
	if err != nil {
		return false
	}
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", framework.TestContext.BearerToken))
	resp, err := insecureHTTPClient.Do(req)
	if err != nil {
		klog.Warningf("Health check on %q failed, error=%v", url, err)
	} else if resp.StatusCode != http.StatusOK {
		klog.Warningf("Health check on %q failed, status=%d", url, resp.StatusCode)
	}
	return err == nil && resp.StatusCode == http.StatusOK
}

func toCgroupFsName(cgroupName cm.CgroupName) string {
	if framework.TestContext.KubeletConfig.CgroupDriver == "systemd" {
		return cgroupName.ToSystemd()
	}
	return cgroupName.ToCgroupfs()
}

// reduceAllocatableMemoryUsageIfCgroupv1 uses memory.force_empty (https://lwn.net/Articles/432224/)
// to make the kernel reclaim memory in the allocatable cgroup
// the time to reduce pressure may be unbounded, but usually finishes within a second.
// memory.force_empty is no supported in cgroupv2.
func reduceAllocatableMemoryUsageIfCgroupv1() {
	if !IsCgroup2UnifiedMode() {
		cmd := fmt.Sprintf("echo 0 > /sys/fs/cgroup/memory/%s/memory.force_empty", toCgroupFsName(cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup)))
		_, err := exec.Command("sudo", "sh", "-c", cmd).CombinedOutput()
		framework.ExpectNoError(err)
	}
}

// Equivalent of featuregatetesting.SetFeatureGateDuringTest
// which can't be used here because we're not in a Testing context.
// This must be in a non-"_test" file to pass
// make verify WHAT=test-featuregates
func withFeatureGate(feature featuregate.Feature, desired bool) func() {
	current := utilfeature.DefaultFeatureGate.Enabled(feature)
	utilfeature.DefaultMutableFeatureGate.Set(fmt.Sprintf("%s=%v", string(feature), desired))
	return func() {
		utilfeature.DefaultMutableFeatureGate.Set(fmt.Sprintf("%s=%v", string(feature), current))
	}
}
