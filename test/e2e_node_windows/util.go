//go:build windows

/*
Copyright The Kubernetes Authors.

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

package e2enodewindows

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	internalapi "k8s.io/cri-api/pkg/apis"
	remote "k8s.io/cri-client/pkg"
	"k8s.io/klog/v2"
	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/cluster/ports"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var (
	startServices = flag.Bool("start-services", true, "If true, start local node services")
	stopServices  = flag.Bool("stop-services", true, "If true, stop local node services after running tests")
	busyboxImage  = imageutils.GetE2EImage(imageutils.BusyBox)

	// kubeletCfg is the kubelet configuration the test is running against.
	kubeletCfg *kubeletconfig.KubeletConfiguration

	kubeletHealthCheckURL = fmt.Sprintf("http://127.0.0.1:%d/healthz", ports.KubeletHealthzPort)
)

const (
	kubeletServiceName = "kubelet"

	// CPU/memory manager state files live in the kubelet root directory
	// (--root-dir, /var/lib/kubelet). They must be removed when the kubelet is
	// reconfigured in a way that invalidates the persisted checkpoint (e.g.
	// toggling the static CPU manager's strict-cpu-reservation option), otherwise
	// the kubelet refuses to start with "invalid state, please drain node and
	// remove policy state file".
	cpuManagerStateFile    = "/var/lib/kubelet/cpu_manager_state"
	memoryManagerStateFile = "/var/lib/kubelet/memory_manager_state"
)

// getKubeletServicePID returns the PID of the kubelet service, or 0 if it
// cannot be determined (e.g. the service is no longer running).
func getKubeletServicePID() int {
	cmdLine := []string{"sc.exe", "queryex", kubeletServiceName}

	stdout, err := exec.Command(cmdLine[0], cmdLine[1:]...).CombinedOutput()
	if err != nil {
		return 0
	}

	regex := regexp.MustCompile(`PID\s*:\s*(\d+)`)
	matches := regex.FindStringSubmatch(string(stdout))
	if len(matches) <= 1 {
		return 0
	}

	pid, err := strconv.Atoi(matches[1])
	if err != nil {
		return 0
	}
	return pid
}

// killProcessByPID kills the process with the given PID. A "process not found"
// outcome is treated as success — by the time we taskkill, the kubelet may
// already have exited on its own (e.g. in response to an earlier sc.exe stop).
func killProcessByPID(pid int) {
	if pid <= 0 {
		return
	}
	cmdLine := []string{"taskkill", "/F", "/PID", strconv.Itoa(pid)}

	stdout, err := exec.Command(cmdLine[0], cmdLine[1:]...).CombinedOutput()
	if err == nil {
		return
	}
	// taskkill returns non-zero with "not found" / "no running instance" when the
	// process has already exited. That's the state we wanted, so swallow it.
	out := strings.ToLower(string(stdout))
	if strings.Contains(out, "not found") || strings.Contains(out, "no running") {
		return
	}
	framework.ExpectNoError(err, "taskkill failed for PID %d: %s", pid, string(stdout))
}

// findKubeletServiceState searches for the state of the kubelet service.
func findKubeletServiceState() string {
	cmdLine := []string{"sc.exe", "query", kubeletServiceName}

	// Assume kubelet service has already been registered
	stdout, err := exec.Command(cmdLine[0], cmdLine[1:]...).CombinedOutput()
	framework.ExpectNoError(err)

	regex := regexp.MustCompile(`(?m)STATE\s*:\s*\d+\s+(\w+)`)
	matches := regex.FindStringSubmatch(string(stdout))
	gomega.Expect(len(matches)).To(gomega.BeNumerically(">", 1), "Found the matched state: %q", stdout)
	state := matches[1]

	return state
}

// stopKubeletService stops the kubelet Windows service and waits until SCM
// reports it as STOPPED. sc.exe stop is asynchronous: the service typically
// transitions RUNNING -> STOP_PENDING -> STOPPED. Issuing sc.exe start while
// SCM still considers the service running yields error 1056
// ("An instance of the service is already running"), so we must gate the next
// start on the SCM state, not on the HTTP health probe (which goes down well
// before SCM finishes the transition).
func stopKubeletService(ctx context.Context) {
	state := findKubeletServiceState()
	if strings.EqualFold(state, "STOPPED") {
		return
	}

	if strings.EqualFold(state, "RUNNING") {
		stdout, err := exec.CommandContext(ctx, "sc.exe", "stop", kubeletServiceName).CombinedOutput()
		framework.ExpectNoError(err, "Failed to stop kubelet service: %v, %s", err, string(stdout))
	}

	// Wait for SCM to report STOPPED. If it stays in STOP_PENDING for too long,
	// fall back to forcibly killing the kubelet process.
	const (
		stopTimeout = 30 * time.Second
		stopPoll    = 250 * time.Millisecond
	)
	deadline := time.Now().Add(stopTimeout)
	for time.Now().Before(deadline) {
		if strings.EqualFold(findKubeletServiceState(), "STOPPED") {
			return
		}
		time.Sleep(stopPoll)
	}

	// Stuck in STOP_PENDING — force-kill the process and re-check.
	killProcessByPID(getKubeletServicePID())
	gomega.Eventually(ctx, func() string {
		return strings.ToUpper(findKubeletServiceState())
	}, 10*time.Second, stopPoll).Should(gomega.Equal("STOPPED"), "kubelet service did not reach STOPPED state")
}

// startKubeletService starts the kubelet Windows service and waits for the
// kubelet HTTP health check to succeed.
func startKubeletService(ctx context.Context, f *framework.Framework) {
	stdout, err := exec.CommandContext(ctx, "sc.exe", "start", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to start kubelet service with sc.exe: %v, %s", err, string(stdout))
	waitForKubeletToStart(ctx, f)
}

// restartKubelet restarts the current kubelet service.
// the "current" kubelet service is the instance managed by the current e2e_node test run.
// If `running` is true, restarts only if the current kubelet is actually running. In some cases,
// the kubelet may have exited or can be stopped, typically because it was intentionally stopped
// earlier during a test, or, sometimes, because it just crashed.
// Warning: the "current" kubelet is poorly defined. The "current" kubelet is assumed to be the most
// recent kubelet service unit, IOW there is not a unique ID we use to bind explicitly a kubelet
// instance to a test run.
func restartKubelet(ctx context.Context, running bool) {
	stopKubeletService(ctx)

	stdout, err := exec.CommandContext(ctx, "sc.exe", "start", kubeletServiceName).CombinedOutput()
	framework.ExpectNoError(err, "Failed to restart kubelet service with sc.exe: %v, %s", err, string(stdout))
}

// mustStopKubelet will kill the running kubelet, and returns a func that will restart the process again
func mustStopKubelet(ctx context.Context, f *framework.Framework) func(ctx context.Context) {
	stopKubeletService(ctx)

	// Belt-and-braces: ensure the HTTP health endpoint is also down before
	// returning, since that is the surface the next test will probe.
	gomega.Eventually(ctx, func() bool {
		return e2enode.HealthCheck(kubeletHealthCheckURL)
	}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeFalseBecause("kubelet was expected to be stopped but it is still running"))

	return func(ctx context.Context) {
		startKubeletService(ctx, f)
	}
}

// TODO: add the windows part implementation
func stopContainerRuntime() error {
	return nil
}

func startContainerRuntime() error {
	return nil
}

// deleteStateFile deletes the state file with the filename.
func deleteStateFile(stateFileName string) {
	err := os.Remove(stateFileName)
	if err != nil && !os.IsNotExist(err) {
		framework.ExpectNoError(err, "failed to delete the state file %q", stateFileName)
	}
}

// systemValidation validates the system spec.
func systemValidation(systemSpecFile *string) {
	klog.Warningf("system spec validation is not supported on platform other than linux yet")
}

func getNodeSummary(ctx context.Context) (*stats.Summary, error) {
	kubeletConfig, err := getCurrentKubeletConfig(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to get current kubelet config")
	}
	req, err := http.NewRequestWithContext(ctx, "GET", fmt.Sprintf("http://%s/stats/summary", net.JoinHostPort(kubeletConfig.Address, strconv.Itoa(int(kubeletConfig.ReadOnlyPort)))), nil)
	if err != nil {
		return nil, fmt.Errorf("failed to build http request: %w", err)
	}
	req.Header.Add("Accept", "application/json")

	client := &http.Client{}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to get /stats/summary: %w", err)
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

func addAfterEachForCleaningUpPods(f *framework.Framework) {
	ginkgo.AfterEach(func(ctx context.Context) {
		ginkgo.By("Deleting any Pods created by the test in namespace: " + f.Namespace.Name)
		l, err := e2epod.NewPodClient(f).List(ctx, metav1.ListOptions{})
		framework.ExpectNoError(err)
		for _, p := range l.Items {
			if p.Namespace != f.Namespace.Name {
				continue
			}
			framework.Logf("Deleting pod: %s", p.Name)
			e2epod.NewPodClient(f).DeleteSync(ctx, p.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
		}
	})
}

func waitForKubeletToStart(ctx context.Context, f *framework.Framework) {
	// wait until the kubelet health check will succeed
	gomega.Eventually(ctx, func() bool {
		return e2enode.HealthCheck(kubeletHealthCheckURL)
	}, 2*time.Minute, 5*time.Second).Should(gomega.BeTrueBecause("expected kubelet to be in healthy state"))

	// Wait for the Kubelet to be ready.
	gomega.Eventually(ctx, func(ctx context.Context) error {
		nodes, err := e2enode.TotalReady(ctx, f.ClientSet)
		if err != nil {
			return fmt.Errorf("error getting ready nodes: %w", err)
		}
		if nodes != 1 {
			return fmt.Errorf("expected 1 ready node, got %d", nodes)
		}
		return nil
	}, time.Minute, time.Second).Should(gomega.Succeed())
}

// listNamespaceEvents lists the events in the given namespace.
func listNamespaceEvents(ctx context.Context, c clientset.Interface, ns string) error {
	ls, err := c.CoreV1().Events(ns).List(ctx, metav1.ListOptions{})
	if err != nil {
		return err
	}
	for _, event := range ls.Items {
		klog.Infof("Event(%#v): type: '%v' reason: '%v' %v", event.InvolvedObject, event.Type, event.Reason, event.Message)
	}
	return nil
}

func logPodEvents(ctx context.Context, f *framework.Framework) {
	framework.Logf("Summary of pod events during the test:")
	err := listNamespaceEvents(ctx, f.ClientSet, f.Namespace.Name)
	framework.ExpectNoError(err)
}

func logNodeEvents(ctx context.Context, f *framework.Framework) {
	framework.Logf("Summary of node events during the test:")
	err := listNamespaceEvents(ctx, f.ClientSet, "")
	framework.ExpectNoError(err)
}

func getLocalNode(ctx context.Context, f *framework.Framework) *v1.Node {
	nodeList, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
	framework.ExpectNoError(err)
	gomega.Expect(nodeList.Items).Should(gomega.HaveLen(1), "Unexpected number of node objects for node e2e. Expects only one node.")
	return &nodeList.Items[0]
}

// getLocalTestNode fetches the node object describing the local worker node set up by the e2e_node infra, alongside with its ready state.
func getLocalTestNode(ctx context.Context, f *framework.Framework) (*v1.Node, bool) {
	logger := klog.FromContext(ctx)
	node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	ready := e2enode.IsNodeReady(logger, node)
	schedulable := e2enode.IsNodeSchedulable(logger, node)
	framework.Logf("node %q ready=%v schedulable=%v", node.Name, ready, schedulable)
	return node, ready && schedulable
}

func getLocalNodeCPUDetails(ctx context.Context, f *framework.Framework) (cpuCapVal int64, cpuAllocVal int64, cpuResVal int64) {
	localNodeCap := getLocalNode(ctx, f).Status.Capacity
	cpuCap := localNodeCap[v1.ResourceCPU]
	localNodeAlloc := getLocalNode(ctx, f).Status.Allocatable
	cpuAlloc := localNodeAlloc[v1.ResourceCPU]
	cpuRes := cpuCap.DeepCopy()
	cpuRes.Sub(cpuAlloc)

	// RoundUp reserved CPUs to get only integer cores.
	cpuRes.RoundUp(0)

	return cpuCap.Value(), cpuCap.Value() - cpuRes.Value(), cpuRes.Value()
}

func nodeNameOrIP() string {
	return "localhost"
}

// logKubeletLatencyMetrics logs KubeletLatencyMetrics computed from the Prometheus
// metrics exposed on the current node and identified by the metricNames.
// The Kubelet subsystem prefix is automatically prepended to these metric names.
func logKubeletLatencyMetrics(ctx context.Context, metricNames ...string) {
	metricSet := sets.NewString()
	for _, key := range metricNames {
		metricSet.Insert(kubeletmetrics.KubeletSubsystem + "_" + key)
	}
	metric, err := e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, fmt.Sprintf("%s:%d", nodeNameOrIP(), ports.KubeletReadOnlyPort), "/metrics")
	if err != nil {
		framework.Logf("Error getting kubelet metrics: %v", err)
	} else {
		framework.Logf("Kubelet Metrics: %+v", e2emetrics.GetKubeletLatencyMetrics(metric, metricSet))
	}
}

// getCRIClient connects CRI and returns CRI runtime service clients and image service client.
func getCRIClient(ctx context.Context) (internalapi.RuntimeService, internalapi.ImageManagerService, error) {
	// connection timeout for CRI service connection
	const connectionTimeout = 2 * time.Minute
	runtimeEndpoint := framework.TestContext.ContainerRuntimeEndpoint
	useStreaming := utilfeature.DefaultFeatureGate.Enabled(features.CRIListStreaming)
	r, err := remote.NewRemoteRuntimeServiceBuilder().
		WithEndpoint(runtimeEndpoint).
		WithConnectionTimeout(connectionTimeout).
		WithUseStreaming(useStreaming).
		Build(ctx)
	if err != nil {
		return nil, nil, err
	}
	imageManagerEndpoint := runtimeEndpoint
	if framework.TestContext.ImageServiceEndpoint != "" {
		// ImageServiceEndpoint is the same as ContainerRuntimeEndpoint if not
		// explicitly specified.
		imageManagerEndpoint = framework.TestContext.ImageServiceEndpoint
	}
	i, err := remote.NewRemoteImageServiceBuilder().
		WithEndpoint(imageManagerEndpoint).
		WithConnectionTimeout(connectionTimeout).
		WithUseStreaming(useStreaming).
		Build(ctx)
	if err != nil {
		return nil, nil, err
	}
	return r, i, nil
}
