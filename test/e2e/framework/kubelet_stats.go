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
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"

	clientset "k8s.io/client-go/kubernetes"
	stats "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/test/e2e/framework/metrics"

	"github.com/prometheus/common/model"
)

// KubeletMetric stores metrics scraped from the kubelet server's /metric endpoint.
// TODO: Get some more structure around the metrics and this type
type KubeletLatencyMetric struct {
	// eg: list, info, create
	Operation string
	// eg: sync_pods, pod_worker
	Method string
	// 0 <= quantile <=1, e.g. 0.95 is 95%tile, 0.5 is median.
	Quantile float64
	Latency  time.Duration
}

// KubeletMetricByLatency implements sort.Interface for []KubeletMetric based on
// the latency field.
type KubeletLatencyMetrics []KubeletLatencyMetric

func (a KubeletLatencyMetrics) Len() int           { return len(a) }
func (a KubeletLatencyMetrics) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a KubeletLatencyMetrics) Less(i, j int) bool { return a[i].Latency > a[j].Latency }

// If a apiserver client is passed in, the function will try to get kubelet metrics from metrics grabber;
// or else, the function will try to get kubelet metrics directly from the node.
func getKubeletMetricsFromNode(c clientset.Interface, nodeName string) (metrics.KubeletMetrics, error) {
	if c == nil {
		return metrics.GrabKubeletMetricsWithoutProxy(nodeName)
	}
	grabber, err := metrics.NewMetricsGrabber(c, nil, true, false, false, false, false)
	if err != nil {
		return metrics.KubeletMetrics{}, err
	}
	return grabber.GrabFromKubelet(nodeName)
}

// getKubeletMetrics gets all metrics in kubelet subsystem from specified node and trims
// the subsystem prefix.
func getKubeletMetrics(c clientset.Interface, nodeName string) (metrics.KubeletMetrics, error) {
	ms, err := getKubeletMetricsFromNode(c, nodeName)
	if err != nil {
		return metrics.KubeletMetrics{}, err
	}

	kubeletMetrics := make(metrics.KubeletMetrics)
	for name, samples := range ms {
		const prefix = kubeletmetrics.KubeletSubsystem + "_"
		if !strings.HasPrefix(name, prefix) {
			// Not a kubelet metric.
			continue
		}
		method := strings.TrimPrefix(name, prefix)
		kubeletMetrics[method] = samples
	}
	return kubeletMetrics, nil
}

// GetKubeletLatencyMetrics gets all latency related kubelet metrics. Note that the KubeletMetrcis
// passed in should not contain subsystem prefix.
func GetKubeletLatencyMetrics(ms metrics.KubeletMetrics) KubeletLatencyMetrics {
	latencyMethods := sets.NewString(
		kubeletmetrics.PodWorkerLatencyKey,
		kubeletmetrics.PodWorkerStartLatencyKey,
		kubeletmetrics.PodStartLatencyKey,
		kubeletmetrics.CgroupManagerOperationsKey,
		kubeletmetrics.DockerOperationsLatencyKey,
		kubeletmetrics.PodWorkerStartLatencyKey,
		kubeletmetrics.PLEGRelistLatencyKey,
	)
	return GetKubeletMetrics(ms, latencyMethods)
}

func GetKubeletMetrics(ms metrics.KubeletMetrics, methods sets.String) KubeletLatencyMetrics {
	var latencyMetrics KubeletLatencyMetrics
	for method, samples := range ms {
		if !methods.Has(method) {
			continue
		}
		for _, sample := range samples {
			latency := sample.Value
			operation := string(sample.Metric["operation_type"])
			var quantile float64
			if val, ok := sample.Metric[model.QuantileLabel]; ok {
				var err error
				if quantile, err = strconv.ParseFloat(string(val), 64); err != nil {
					continue
				}
			}

			latencyMetrics = append(latencyMetrics, KubeletLatencyMetric{
				Operation: operation,
				Method:    method,
				Quantile:  quantile,
				Latency:   time.Duration(int64(latency)) * time.Microsecond,
			})
		}
	}
	return latencyMetrics
}

// RuntimeOperationMonitor is the tool getting and parsing docker operation metrics.
type RuntimeOperationMonitor struct {
	client          clientset.Interface
	nodesRuntimeOps map[string]NodeRuntimeOperationErrorRate
}

// NodeRuntimeOperationErrorRate is the runtime operation error rate on one node.
type NodeRuntimeOperationErrorRate map[string]*RuntimeOperationErrorRate

// RuntimeOperationErrorRate is the error rate of a specified runtime operation.
type RuntimeOperationErrorRate struct {
	TotalNumber float64
	ErrorRate   float64
	TimeoutRate float64
}

func NewRuntimeOperationMonitor(c clientset.Interface) *RuntimeOperationMonitor {
	m := &RuntimeOperationMonitor{
		client:          c,
		nodesRuntimeOps: make(map[string]NodeRuntimeOperationErrorRate),
	}
	nodes, err := m.client.Core().Nodes().List(metav1.ListOptions{})
	if err != nil {
		Failf("RuntimeOperationMonitor: unable to get list of nodes: %v", err)
	}
	for _, node := range nodes.Items {
		m.nodesRuntimeOps[node.Name] = make(NodeRuntimeOperationErrorRate)
	}
	// Initialize the runtime operation error rate
	m.GetRuntimeOperationErrorRate()
	return m
}

// GetRuntimeOperationErrorRate gets runtime operation records from kubelet metrics and calculate
// error rates of all runtime operations.
func (m *RuntimeOperationMonitor) GetRuntimeOperationErrorRate() map[string]NodeRuntimeOperationErrorRate {
	for node := range m.nodesRuntimeOps {
		nodeResult, err := getNodeRuntimeOperationErrorRate(m.client, node)
		if err != nil {
			Logf("GetRuntimeOperationErrorRate: unable to get kubelet metrics from node %q: %v", node, err)
			continue
		}
		m.nodesRuntimeOps[node] = nodeResult
	}
	return m.nodesRuntimeOps
}

// GetLatestRuntimeOperationErrorRate gets latest error rate and timeout rate from last observed RuntimeOperationErrorRate.
func (m *RuntimeOperationMonitor) GetLatestRuntimeOperationErrorRate() map[string]NodeRuntimeOperationErrorRate {
	result := make(map[string]NodeRuntimeOperationErrorRate)
	for node := range m.nodesRuntimeOps {
		result[node] = make(NodeRuntimeOperationErrorRate)
		oldNodeResult := m.nodesRuntimeOps[node]
		curNodeResult, err := getNodeRuntimeOperationErrorRate(m.client, node)
		if err != nil {
			Logf("GetLatestRuntimeOperationErrorRate: unable to get kubelet metrics from node %q: %v", node, err)
			continue
		}
		for op, cur := range curNodeResult {
			t := *cur
			if old, found := oldNodeResult[op]; found {
				t.ErrorRate = (t.ErrorRate*t.TotalNumber - old.ErrorRate*old.TotalNumber) / (t.TotalNumber - old.TotalNumber)
				t.TimeoutRate = (t.TimeoutRate*t.TotalNumber - old.TimeoutRate*old.TotalNumber) / (t.TotalNumber - old.TotalNumber)
				t.TotalNumber -= old.TotalNumber
			}
			result[node][op] = &t
		}
		m.nodesRuntimeOps[node] = curNodeResult
	}
	return result
}

// FormatRuntimeOperationErrorRate formats the runtime operation error rate to string.
func FormatRuntimeOperationErrorRate(nodesResult map[string]NodeRuntimeOperationErrorRate) string {
	lines := []string{}
	for node, nodeResult := range nodesResult {
		lines = append(lines, fmt.Sprintf("node %q runtime operation error rate:", node))
		for op, result := range nodeResult {
			line := fmt.Sprintf("operation %q: total - %.0f; error rate - %f; timeout rate - %f", op,
				result.TotalNumber, result.ErrorRate, result.TimeoutRate)
			lines = append(lines, line)
		}
		lines = append(lines, fmt.Sprintln())
	}
	return strings.Join(lines, "\n")
}

// getNodeRuntimeOperationErrorRate gets runtime operation error rate from specified node.
func getNodeRuntimeOperationErrorRate(c clientset.Interface, node string) (NodeRuntimeOperationErrorRate, error) {
	result := make(NodeRuntimeOperationErrorRate)
	ms, err := getKubeletMetrics(c, node)
	if err != nil {
		return result, err
	}
	// If no corresponding metrics are found, the returned samples will be empty. Then the following
	// loop will be skipped automatically.
	allOps := ms[kubeletmetrics.DockerOperationsKey]
	errOps := ms[kubeletmetrics.DockerOperationsErrorsKey]
	timeoutOps := ms[kubeletmetrics.DockerOperationsTimeoutKey]
	for _, sample := range allOps {
		operation := string(sample.Metric["operation_type"])
		result[operation] = &RuntimeOperationErrorRate{TotalNumber: float64(sample.Value)}
	}
	for _, sample := range errOps {
		operation := string(sample.Metric["operation_type"])
		// Should always find the corresponding item, just in case
		if _, found := result[operation]; found {
			result[operation].ErrorRate = float64(sample.Value) / result[operation].TotalNumber
		}
	}
	for _, sample := range timeoutOps {
		operation := string(sample.Metric["operation_type"])
		if _, found := result[operation]; found {
			result[operation].TimeoutRate = float64(sample.Value) / result[operation].TotalNumber
		}
	}
	return result, nil
}

// HighLatencyKubeletOperations logs and counts the high latency metrics exported by the kubelet server via /metrics.
func HighLatencyKubeletOperations(c clientset.Interface, threshold time.Duration, nodeName string, logFunc func(fmt string, args ...interface{})) (KubeletLatencyMetrics, error) {
	ms, err := getKubeletMetrics(c, nodeName)
	if err != nil {
		return KubeletLatencyMetrics{}, err
	}
	latencyMetrics := GetKubeletLatencyMetrics(ms)
	sort.Sort(latencyMetrics)
	var badMetrics KubeletLatencyMetrics
	logFunc("\nLatency metrics for node %v", nodeName)
	for _, m := range latencyMetrics {
		if m.Latency > threshold {
			badMetrics = append(badMetrics, m)
			Logf("%+v", m)
		}
	}
	return badMetrics, nil
}

// getStatsSummary contacts kubelet for the container information.
func getStatsSummary(c clientset.Interface, nodeName string) (*stats.Summary, error) {
	subResourceProxyAvailable, err := ServerVersionGTE(SubResourceServiceAndNodeProxyVersion, c.Discovery())
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), SingleCallTimeout)
	defer cancel()

	var data []byte
	if subResourceProxyAvailable {
		data, err = c.Core().RESTClient().Get().
			Context(ctx).
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", nodeName, ports.KubeletPort)).
			Suffix("stats/summary").
			Do().Raw()

	} else {
		data, err = c.Core().RESTClient().Get().
			Context(ctx).
			Prefix("proxy").
			Resource("nodes").
			Name(fmt.Sprintf("%v:%v", nodeName, ports.KubeletPort)).
			Suffix("stats/summary").
			Do().Raw()
	}
	if err != nil {
		return nil, err
	}

	summary := stats.Summary{}
	err = json.Unmarshal(data, &summary)
	if err != nil {
		return nil, err
	}
	return &summary, nil
}

func removeUint64Ptr(ptr *uint64) uint64 {
	if ptr == nil {
		return 0
	}
	return *ptr
}

// getOneTimeResourceUsageOnNode queries the node's /stats/summary endpoint
// and returns the resource usage of all containerNames for the past
// cpuInterval.
// The acceptable range of the interval is 2s~120s. Be warned that as the
// interval (and #containers) increases, the size of kubelet's response
// could be significant. E.g., the 60s interval stats for ~20 containers is
// ~1.5MB. Don't hammer the node with frequent, heavy requests.
//
// cadvisor records cumulative cpu usage in nanoseconds, so we need to have two
// stats points to compute the cpu usage over the interval. Assuming cadvisor
// polls every second, we'd need to get N stats points for N-second interval.
// Note that this is an approximation and may not be accurate, hence we also
// write the actual interval used for calculation (based on the timestamps of
// the stats points in ContainerResourceUsage.CPUInterval.
//
// containerNames is a function returning a collection of container names in which
// user is interested in.
func getOneTimeResourceUsageOnNode(
	c clientset.Interface,
	nodeName string,
	cpuInterval time.Duration,
	containerNames func() []string,
) (ResourceUsagePerContainer, error) {
	const (
		// cadvisor records stats about every second.
		cadvisorStatsPollingIntervalInSeconds float64 = 1.0
		// cadvisor caches up to 2 minutes of stats (configured by kubelet).
		maxNumStatsToRequest int = 120
	)

	numStats := int(float64(cpuInterval.Seconds()) / cadvisorStatsPollingIntervalInSeconds)
	if numStats < 2 || numStats > maxNumStatsToRequest {
		return nil, fmt.Errorf("numStats needs to be > 1 and < %d", maxNumStatsToRequest)
	}
	// Get information of all containers on the node.
	summary, err := getStatsSummary(c, nodeName)
	if err != nil {
		return nil, err
	}

	f := func(name string, newStats *stats.ContainerStats) *ContainerResourceUsage {
		if newStats == nil || newStats.CPU == nil || newStats.Memory == nil {
			return nil
		}
		return &ContainerResourceUsage{
			Name:                    name,
			Timestamp:               newStats.StartTime.Time,
			CPUUsageInCores:         float64(removeUint64Ptr(newStats.CPU.UsageNanoCores)) / 1000000000,
			MemoryUsageInBytes:      removeUint64Ptr(newStats.Memory.UsageBytes),
			MemoryWorkingSetInBytes: removeUint64Ptr(newStats.Memory.WorkingSetBytes),
			MemoryRSSInBytes:        removeUint64Ptr(newStats.Memory.RSSBytes),
			CPUInterval:             0,
		}
	}
	// Process container infos that are relevant to us.
	containers := containerNames()
	usageMap := make(ResourceUsagePerContainer, len(containers))
	observedContainers := []string{}
	for _, pod := range summary.Pods {
		for _, container := range pod.Containers {
			isInteresting := false
			for _, interestingContainerName := range containers {
				if container.Name == interestingContainerName {
					isInteresting = true
					observedContainers = append(observedContainers, container.Name)
					break
				}
			}
			if !isInteresting {
				continue
			}
			if usage := f(pod.PodRef.Name+"/"+container.Name, &container); usage != nil {
				usageMap[pod.PodRef.Name+"/"+container.Name] = usage
			}
		}
	}
	return usageMap, nil
}

func getNodeStatsSummary(c clientset.Interface, nodeName string) (*stats.Summary, error) {
	subResourceProxyAvailable, err := ServerVersionGTE(SubResourceServiceAndNodeProxyVersion, c.Discovery())
	if err != nil {
		return nil, err
	}

	var data []byte
	if subResourceProxyAvailable {
		data, err = c.Core().RESTClient().Get().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", nodeName, ports.KubeletPort)).
			Suffix("stats/summary").
			SetHeader("Content-Type", "application/json").
			Do().Raw()

	} else {
		data, err = c.Core().RESTClient().Get().
			Prefix("proxy").
			Resource("nodes").
			Name(fmt.Sprintf("%v:%v", nodeName, ports.KubeletPort)).
			Suffix("stats/summary").
			SetHeader("Content-Type", "application/json").
			Do().Raw()
	}
	if err != nil {
		return nil, err
	}

	var summary *stats.Summary
	err = json.Unmarshal(data, &summary)
	if err != nil {
		return nil, err
	}
	return summary, nil
}

func getSystemContainerStats(summary *stats.Summary) map[string]*stats.ContainerStats {
	statsList := summary.Node.SystemContainers
	statsMap := make(map[string]*stats.ContainerStats)
	for i := range statsList {
		statsMap[statsList[i].Name] = &statsList[i]
	}

	// Create a root container stats using information available in
	// stats.NodeStats. This is necessary since it is a different type.
	statsMap[rootContainerName] = &stats.ContainerStats{
		CPU:    summary.Node.CPU,
		Memory: summary.Node.Memory,
	}
	return statsMap
}

const (
	rootContainerName = "/"
)

// A list of containers for which we want to collect resource usage.
func TargetContainers() []string {
	return []string{
		rootContainerName,
		stats.SystemContainerRuntime,
		stats.SystemContainerKubelet,
	}
}

type ContainerResourceUsage struct {
	Name                    string
	Timestamp               time.Time
	CPUUsageInCores         float64
	MemoryUsageInBytes      uint64
	MemoryWorkingSetInBytes uint64
	MemoryRSSInBytes        uint64
	// The interval used to calculate CPUUsageInCores.
	CPUInterval time.Duration
}

func (r *ContainerResourceUsage) isStrictlyGreaterThan(rhs *ContainerResourceUsage) bool {
	return r.CPUUsageInCores > rhs.CPUUsageInCores && r.MemoryWorkingSetInBytes > rhs.MemoryWorkingSetInBytes
}

type ResourceUsagePerContainer map[string]*ContainerResourceUsage
type ResourceUsagePerNode map[string]ResourceUsagePerContainer

func formatResourceUsageStats(nodeName string, containerStats ResourceUsagePerContainer) string {
	// Example output:
	//
	// Resource usage for node "e2e-test-foo-node-abcde":
	// container        cpu(cores)  memory(MB)
	// "/"              0.363       2942.09
	// "/docker-daemon" 0.088       521.80
	// "/kubelet"       0.086       424.37
	// "/system"        0.007       119.88
	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	fmt.Fprintf(w, "container\tcpu(cores)\tmemory_working_set(MB)\tmemory_rss(MB)\n")
	for name, s := range containerStats {
		fmt.Fprintf(w, "%q\t%.3f\t%.2f\t%.2f\n", name, s.CPUUsageInCores, float64(s.MemoryWorkingSetInBytes)/(1024*1024), float64(s.MemoryRSSInBytes)/(1024*1024))
	}
	w.Flush()
	return fmt.Sprintf("Resource usage on node %q:\n%s", nodeName, buf.String())
}

type uint64arr []uint64

func (a uint64arr) Len() int           { return len(a) }
func (a uint64arr) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a uint64arr) Less(i, j int) bool { return a[i] < a[j] }

type usageDataPerContainer struct {
	cpuData        []float64
	memUseData     []uint64
	memWorkSetData []uint64
}

func GetKubeletHeapStats(c clientset.Interface, nodeName string) (string, error) {
	client, err := NodeProxyRequest(c, nodeName, "debug/pprof/heap")
	if err != nil {
		return "", err
	}
	raw, errRaw := client.Raw()
	if errRaw != nil {
		return "", err
	}
	stats := string(raw)
	// Only dumping the runtime.MemStats numbers to avoid polluting the log.
	numLines := 23
	lines := strings.Split(stats, "\n")
	return strings.Join(lines[len(lines)-numLines:], "\n"), nil
}

func PrintAllKubeletPods(c clientset.Interface, nodeName string) {
	podList, err := GetKubeletPods(c, nodeName)
	if err != nil {
		Logf("Unable to retrieve kubelet pods for node %v: %v", nodeName, err)
		return
	}
	for _, p := range podList.Items {
		Logf("%v from %v started at %v (%d container statuses recorded)", p.Name, p.Namespace, p.Status.StartTime, len(p.Status.ContainerStatuses))
		for _, c := range p.Status.ContainerStatuses {
			Logf("\tContainer %v ready: %v, restart count %v",
				c.Name, c.Ready, c.RestartCount)
		}
	}
}

func computeContainerResourceUsage(name string, oldStats, newStats *stats.ContainerStats) *ContainerResourceUsage {
	return &ContainerResourceUsage{
		Name:                    name,
		Timestamp:               newStats.CPU.Time.Time,
		CPUUsageInCores:         float64(*newStats.CPU.UsageCoreNanoSeconds-*oldStats.CPU.UsageCoreNanoSeconds) / float64(newStats.CPU.Time.Time.Sub(oldStats.CPU.Time.Time).Nanoseconds()),
		MemoryUsageInBytes:      *newStats.Memory.UsageBytes,
		MemoryWorkingSetInBytes: *newStats.Memory.WorkingSetBytes,
		MemoryRSSInBytes:        *newStats.Memory.RSSBytes,
		CPUInterval:             newStats.CPU.Time.Time.Sub(oldStats.CPU.Time.Time),
	}
}

// resourceCollector periodically polls the node, collect stats for a given
// list of containers, computes and cache resource usage up to
// maxEntriesPerContainer for each container.
type resourceCollector struct {
	lock            sync.RWMutex
	node            string
	containers      []string
	client          clientset.Interface
	buffers         map[string][]*ContainerResourceUsage
	pollingInterval time.Duration
	stopCh          chan struct{}
}

func newResourceCollector(c clientset.Interface, nodeName string, containerNames []string, pollingInterval time.Duration) *resourceCollector {
	buffers := make(map[string][]*ContainerResourceUsage)
	return &resourceCollector{
		node:            nodeName,
		containers:      containerNames,
		client:          c,
		buffers:         buffers,
		pollingInterval: pollingInterval,
	}
}

// Start starts a goroutine to Poll the node every pollingInterval.
func (r *resourceCollector) Start() {
	r.stopCh = make(chan struct{}, 1)
	// Keep the last observed stats for comparison.
	oldStats := make(map[string]*stats.ContainerStats)
	go wait.Until(func() { r.collectStats(oldStats) }, r.pollingInterval, r.stopCh)
}

// Stop sends a signal to terminate the stats collecting goroutine.
func (r *resourceCollector) Stop() {
	close(r.stopCh)
}

// collectStats gets the latest stats from kubelet stats summary API, computes
// the resource usage, and pushes it to the buffer.
func (r *resourceCollector) collectStats(oldStatsMap map[string]*stats.ContainerStats) {
	summary, err := getNodeStatsSummary(r.client, r.node)
	if err != nil {
		Logf("Error getting node stats summary on %q, err: %v", r.node, err)
		return
	}
	cStatsMap := getSystemContainerStats(summary)
	r.lock.Lock()
	defer r.lock.Unlock()
	for _, name := range r.containers {
		cStats, ok := cStatsMap[name]
		if !ok {
			Logf("Missing info/stats for container %q on node %q", name, r.node)
			return
		}

		if oldStats, ok := oldStatsMap[name]; ok {
			if oldStats.CPU.Time.Equal(cStats.CPU.Time) {
				// No change -> skip this stat.
				continue
			}
			r.buffers[name] = append(r.buffers[name], computeContainerResourceUsage(name, oldStats, cStats))
		}
		// Update the old stats.
		oldStatsMap[name] = cStats
	}
}

func (r *resourceCollector) GetLatest() (ResourceUsagePerContainer, error) {
	r.lock.RLock()
	defer r.lock.RUnlock()
	stats := make(ResourceUsagePerContainer)
	for _, name := range r.containers {
		contStats, ok := r.buffers[name]
		if !ok || len(contStats) == 0 {
			return nil, fmt.Errorf("Resource usage on node %q is not ready yet", r.node)
		}
		stats[name] = contStats[len(contStats)-1]
	}
	return stats, nil
}

// Reset frees the stats and start over.
func (r *resourceCollector) Reset() {
	r.lock.Lock()
	defer r.lock.Unlock()
	for _, name := range r.containers {
		r.buffers[name] = []*ContainerResourceUsage{}
	}
}

type resourceUsageByCPU []*ContainerResourceUsage

func (r resourceUsageByCPU) Len() int           { return len(r) }
func (r resourceUsageByCPU) Swap(i, j int)      { r[i], r[j] = r[j], r[i] }
func (r resourceUsageByCPU) Less(i, j int) bool { return r[i].CPUUsageInCores < r[j].CPUUsageInCores }

// The percentiles to report.
var percentiles = [...]float64{0.05, 0.20, 0.50, 0.70, 0.90, 0.95, 0.99}

// GetBasicCPUStats returns the percentiles the cpu usage in cores for
// containerName. This method examines all data currently in the buffer.
func (r *resourceCollector) GetBasicCPUStats(containerName string) map[float64]float64 {
	r.lock.RLock()
	defer r.lock.RUnlock()
	result := make(map[float64]float64, len(percentiles))
	usages := r.buffers[containerName]
	sort.Sort(resourceUsageByCPU(usages))
	for _, q := range percentiles {
		index := int(float64(len(usages))*q) - 1
		if index < 0 {
			// We don't have enough data.
			result[q] = 0
			continue
		}
		result[q] = usages[index].CPUUsageInCores
	}
	return result
}

// ResourceMonitor manages a resourceCollector per node.
type ResourceMonitor struct {
	client          clientset.Interface
	containers      []string
	pollingInterval time.Duration
	collectors      map[string]*resourceCollector
}

func NewResourceMonitor(c clientset.Interface, containerNames []string, pollingInterval time.Duration) *ResourceMonitor {
	return &ResourceMonitor{
		containers:      containerNames,
		client:          c,
		pollingInterval: pollingInterval,
	}
}

func (r *ResourceMonitor) Start() {
	// It should be OK to monitor unschedulable Nodes
	nodes, err := r.client.Core().Nodes().List(metav1.ListOptions{})
	if err != nil {
		Failf("ResourceMonitor: unable to get list of nodes: %v", err)
	}
	r.collectors = make(map[string]*resourceCollector, 0)
	for _, node := range nodes.Items {
		collector := newResourceCollector(r.client, node.Name, r.containers, r.pollingInterval)
		r.collectors[node.Name] = collector
		collector.Start()
	}
}

func (r *ResourceMonitor) Stop() {
	for _, collector := range r.collectors {
		collector.Stop()
	}
}

func (r *ResourceMonitor) Reset() {
	for _, collector := range r.collectors {
		collector.Reset()
	}
}

func (r *ResourceMonitor) LogLatest() {
	summary, err := r.GetLatest()
	if err != nil {
		Logf("%v", err)
	}
	Logf("%s", r.FormatResourceUsage(summary))
}

func (r *ResourceMonitor) FormatResourceUsage(s ResourceUsagePerNode) string {
	summary := []string{}
	for node, usage := range s {
		summary = append(summary, formatResourceUsageStats(node, usage))
	}
	return strings.Join(summary, "\n")
}

func (r *ResourceMonitor) GetLatest() (ResourceUsagePerNode, error) {
	result := make(ResourceUsagePerNode)
	errs := []error{}
	for key, collector := range r.collectors {
		s, err := collector.GetLatest()
		if err != nil {
			errs = append(errs, err)
			continue
		}
		result[key] = s
	}
	return result, utilerrors.NewAggregate(errs)
}

func (r *ResourceMonitor) GetMasterNodeLatest(usagePerNode ResourceUsagePerNode) ResourceUsagePerNode {
	result := make(ResourceUsagePerNode)
	var masterUsage ResourceUsagePerContainer
	var nodesUsage []ResourceUsagePerContainer
	for node, usage := range usagePerNode {
		if strings.HasSuffix(node, "master") {
			masterUsage = usage
		} else {
			nodesUsage = append(nodesUsage, usage)
		}
	}
	nodeAvgUsage := make(ResourceUsagePerContainer)
	for _, nodeUsage := range nodesUsage {
		for c, usage := range nodeUsage {
			if _, found := nodeAvgUsage[c]; !found {
				nodeAvgUsage[c] = &ContainerResourceUsage{Name: usage.Name}
			}
			nodeAvgUsage[c].CPUUsageInCores += usage.CPUUsageInCores
			nodeAvgUsage[c].MemoryUsageInBytes += usage.MemoryUsageInBytes
			nodeAvgUsage[c].MemoryWorkingSetInBytes += usage.MemoryWorkingSetInBytes
			nodeAvgUsage[c].MemoryRSSInBytes += usage.MemoryRSSInBytes
		}
	}
	for c := range nodeAvgUsage {
		nodeAvgUsage[c].CPUUsageInCores /= float64(len(nodesUsage))
		nodeAvgUsage[c].MemoryUsageInBytes /= uint64(len(nodesUsage))
		nodeAvgUsage[c].MemoryWorkingSetInBytes /= uint64(len(nodesUsage))
		nodeAvgUsage[c].MemoryRSSInBytes /= uint64(len(nodesUsage))
	}
	result["master"] = masterUsage
	result["node"] = nodeAvgUsage
	return result
}

// ContainersCPUSummary is indexed by the container name with each entry a
// (percentile, value) map.
type ContainersCPUSummary map[string]map[float64]float64

// NodesCPUSummary is indexed by the node name with each entry a
// ContainersCPUSummary map.
type NodesCPUSummary map[string]ContainersCPUSummary

func (r *ResourceMonitor) FormatCPUSummary(summary NodesCPUSummary) string {
	// Example output for a node (the percentiles may differ):
	// CPU usage of containers on node "e2e-test-foo-node-0vj7":
	// container        5th%  50th% 90th% 95th%
	// "/"              0.051 0.159 0.387 0.455
	// "/runtime        0.000 0.000 0.146 0.166
	// "/kubelet"       0.036 0.053 0.091 0.154
	// "/misc"          0.001 0.001 0.001 0.002
	var summaryStrings []string
	var header []string
	header = append(header, "container")
	for _, p := range percentiles {
		header = append(header, fmt.Sprintf("%.0fth%%", p*100))
	}
	for nodeName, containers := range summary {
		buf := &bytes.Buffer{}
		w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
		fmt.Fprintf(w, "%s\n", strings.Join(header, "\t"))
		for _, containerName := range TargetContainers() {
			var s []string
			s = append(s, fmt.Sprintf("%q", containerName))
			data, ok := containers[containerName]
			for _, p := range percentiles {
				value := "N/A"
				if ok {
					value = fmt.Sprintf("%.3f", data[p])
				}
				s = append(s, value)
			}
			fmt.Fprintf(w, "%s\n", strings.Join(s, "\t"))
		}
		w.Flush()
		summaryStrings = append(summaryStrings, fmt.Sprintf("CPU usage of containers on node %q\n:%s", nodeName, buf.String()))
	}
	return strings.Join(summaryStrings, "\n")
}

func (r *ResourceMonitor) LogCPUSummary() {
	summary := r.GetCPUSummary()
	Logf("%s", r.FormatCPUSummary(summary))
}

func (r *ResourceMonitor) GetCPUSummary() NodesCPUSummary {
	result := make(NodesCPUSummary)
	for nodeName, collector := range r.collectors {
		result[nodeName] = make(ContainersCPUSummary)
		for _, containerName := range TargetContainers() {
			data := collector.GetBasicCPUStats(containerName)
			result[nodeName][containerName] = data
		}
	}
	return result
}

func (r *ResourceMonitor) GetMasterNodeCPUSummary(summaryPerNode NodesCPUSummary) NodesCPUSummary {
	result := make(NodesCPUSummary)
	var masterSummary ContainersCPUSummary
	var nodesSummaries []ContainersCPUSummary
	for node, summary := range summaryPerNode {
		if strings.HasSuffix(node, "master") {
			masterSummary = summary
		} else {
			nodesSummaries = append(nodesSummaries, summary)
		}
	}

	nodeAvgSummary := make(ContainersCPUSummary)
	for _, nodeSummary := range nodesSummaries {
		for c, summary := range nodeSummary {
			if _, found := nodeAvgSummary[c]; !found {
				nodeAvgSummary[c] = map[float64]float64{}
			}
			for perc, value := range summary {
				nodeAvgSummary[c][perc] += value
			}
		}
	}
	for c := range nodeAvgSummary {
		for perc := range nodeAvgSummary[c] {
			nodeAvgSummary[c][perc] /= float64(len(nodesSummaries))
		}
	}
	result["master"] = masterSummary
	result["node"] = nodeAvgSummary
	return result
}
