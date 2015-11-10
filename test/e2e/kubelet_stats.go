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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	cadvisor "github.com/google/cadvisor/info/v1"
	"github.com/prometheus/common/model"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/master/ports"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

// KubeletMetric stores metrics scraped from the kubelet server's /metric endpoint.
// TODO: Get some more structure around the metrics and this type
type KubeletMetric struct {
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
type KubeletMetricByLatency []KubeletMetric

func (a KubeletMetricByLatency) Len() int           { return len(a) }
func (a KubeletMetricByLatency) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a KubeletMetricByLatency) Less(i, j int) bool { return a[i].Latency > a[j].Latency }

// ParseKubeletMetrics reads metrics from the kubelet server running on the given node
func ParseKubeletMetrics(metricsBlob string) ([]KubeletMetric, error) {
	samples, err := extractMetricSamples(metricsBlob)
	if err != nil {
		return nil, err
	}

	acceptedMethods := sets.NewString(
		metrics.PodWorkerLatencyKey,
		metrics.PodWorkerStartLatencyKey,
		metrics.SyncPodsLatencyKey,
		metrics.PodStartLatencyKey,
		metrics.PodStatusLatencyKey,
		metrics.ContainerManagerOperationsKey,
		metrics.DockerOperationsKey,
		metrics.DockerErrorsKey,
	)

	var kms []KubeletMetric
	for _, sample := range samples {
		const prefix = metrics.KubeletSubsystem + "_"
		metricName := string(sample.Metric[model.MetricNameLabel])
		if !strings.HasPrefix(metricName, prefix) {
			// Not a kubelet metric.
			continue
		}

		method := strings.TrimPrefix(metricName, prefix)
		if !acceptedMethods.Has(method) {
			continue
		}

		if method == metrics.DockerErrorsKey {
			Logf("ERROR %v", sample)
		}

		latency := sample.Value
		operation := string(sample.Metric["operation_type"])
		var quantile float64
		if val, ok := sample.Metric[model.QuantileLabel]; ok {
			var err error
			if quantile, err = strconv.ParseFloat(string(val), 64); err != nil {
				continue
			}
		}

		kms = append(kms, KubeletMetric{
			operation,
			method,
			quantile,
			time.Duration(int64(latency)) * time.Microsecond,
		})
	}
	return kms, nil
}

// HighLatencyKubeletOperations logs and counts the high latency metrics exported by the kubelet server via /metrics.
func HighLatencyKubeletOperations(c *client.Client, threshold time.Duration, nodeName string) ([]KubeletMetric, error) {
	var metricsBlob string
	var err error
	// If we haven't been given a client try scraping the nodename directly for a /metrics endpoint.
	if c == nil {
		metricsBlob, err = getKubeletMetricsThroughNode(nodeName)
	} else {
		metricsBlob, err = getKubeletMetricsThroughProxy(c, nodeName)
	}
	if err != nil {
		return []KubeletMetric{}, err
	}
	metric, err := ParseKubeletMetrics(metricsBlob)
	if err != nil {
		return []KubeletMetric{}, err
	}
	sort.Sort(KubeletMetricByLatency(metric))
	var badMetrics []KubeletMetric
	Logf("\nLatency metrics for node %v", nodeName)
	for _, m := range metric {
		if m.Latency > threshold {
			badMetrics = append(badMetrics, m)
			Logf("%+v", m)
		}
	}
	return badMetrics, nil
}

// getContainerInfo contacts kubelet for the container informaton. The "Stats"
// in the returned ContainerInfo is subject to the requirements in statsRequest.
func getContainerInfo(c *client.Client, nodeName string, req *kubelet.StatsRequest) (map[string]cadvisor.ContainerInfo, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	data, err := c.Post().
		Prefix("proxy").
		Resource("nodes").
		Name(fmt.Sprintf("%v:%v", nodeName, ports.KubeletPort)).
		Suffix("stats/container").
		SetHeader("Content-Type", "application/json").
		Body(reqBody).
		Do().Raw()

	var containers map[string]cadvisor.ContainerInfo
	err = json.Unmarshal(data, &containers)
	if err != nil {
		return nil, err
	}
	return containers, nil
}

const (
	// cadvisor records stats about every second.
	cadvisorStatsPollingIntervalInSeconds float64 = 1.0
	// cadvisor caches up to 2 minutes of stats (configured by kubelet).
	maxNumStatsToRequest int = 120
)

// A list of containers for which we want to collect resource usage.
func targetContainers() []string {
	if providerIs("gce", "gke") {
		return []string{
			"/",
			"/docker-daemon",
			"/kubelet",
			"/kube-proxy",
			"/system",
		}
	} else {
		return []string{
			"/",
		}
	}
}

type containerResourceUsage struct {
	Name                    string
	Timestamp               time.Time
	CPUUsageInCores         float64
	MemoryUsageInBytes      int64
	MemoryWorkingSetInBytes int64
	// The interval used to calculate CPUUsageInCores.
	CPUInterval time.Duration
}

func (r *containerResourceUsage) isStrictlyGreaterThan(rhs *containerResourceUsage) bool {
	return r.CPUUsageInCores > rhs.CPUUsageInCores && r.MemoryUsageInBytes > rhs.MemoryUsageInBytes && r.MemoryWorkingSetInBytes > rhs.MemoryWorkingSetInBytes
}

// getOneTimeResourceUsageOnNode queries the node's /stats/container endpoint
// and returns the resource usage of targetContainers for the past
// cpuInterval.
// The acceptable range of the interval is 2s~120s. Be warned that as the
// interval (and #containers) increases, the size of kubelet's response
// could be sigificant. E.g., the 60s interval stats for ~20 containers is
// ~1.5MB. Don't hammer the node with frequent, heavy requests.
//
// cadvisor records cumulative cpu usage in nanoseconds, so we need to have two
// stats points to compute the cpu usage over the interval. Assuming cadvisor
// polls every second, we'd need to get N stats points for N-second interval.
// Note that this is an approximation and may not be accurate, hence we also
// write the actual interval used for calcuation (based on the timestampes of
// the stats points in containerResourceUsage.CPUInterval.
func getOneTimeResourceUsageOnNode(c *client.Client, nodeName string, cpuInterval time.Duration) (map[string]*containerResourceUsage, error) {
	numStats := int(float64(cpuInterval.Seconds()) / cadvisorStatsPollingIntervalInSeconds)
	if numStats < 2 || numStats > maxNumStatsToRequest {
		return nil, fmt.Errorf("numStats needs to be > 1 and < %d", maxNumStatsToRequest)
	}
	// Get information of all containers on the node.
	containerInfos, err := getContainerInfo(c, nodeName, &kubelet.StatsRequest{
		ContainerName: "/",
		NumStats:      numStats,
		Subcontainers: true,
	})
	if err != nil {
		return nil, err
	}
	// Process container infos that are relevant to us.
	containers := targetContainers()
	usageMap := make(map[string]*containerResourceUsage, len(containers))
	for _, name := range containers {
		info, ok := containerInfos[name]
		if !ok {
			return nil, fmt.Errorf("missing info for container %q on node %q", name, nodeName)
		}
		first := info.Stats[0]
		last := info.Stats[len(info.Stats)-1]
		usageMap[name] = computeContainerResourceUsage(name, first, last)
	}
	return usageMap, nil
}

// logOneTimeResourceUsageSummary collects container resource for the list of
// nodes, formats and logs the stats.
func logOneTimeResourceUsageSummary(c *client.Client, nodeNames []string, cpuInterval time.Duration) {
	var summary []string
	for _, nodeName := range nodeNames {
		stats, err := getOneTimeResourceUsageOnNode(c, nodeName, cpuInterval)
		if err != nil {
			summary = append(summary, fmt.Sprintf("Error getting resource usage from node %q, err: %v", nodeName, err))
		} else {
			summary = append(summary, formatResourceUsageStats(nodeName, stats))
		}
	}
	Logf("\n%s", strings.Join(summary, "\n"))
}

func formatResourceUsageStats(nodeName string, containerStats map[string]*containerResourceUsage) string {
	// Example output:
	//
	// Resource usage for node "e2e-test-foo-minion-abcde":
	// container        cpu(cores)  memory(MB)
	// "/"              0.363       2942.09
	// "/docker-daemon" 0.088       521.80
	// "/kubelet"       0.086       424.37
	// "/kube-proxy"    0.011       4.66
	// "/system"        0.007       119.88
	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	fmt.Fprintf(w, "container\tcpu(cores)\tmemory(MB)\n")
	for name, s := range containerStats {
		fmt.Fprintf(w, "%q\t%.3f\t%.2f\n", name, s.CPUUsageInCores, float64(s.MemoryUsageInBytes)/1000000)
	}
	w.Flush()
	return fmt.Sprintf("Resource usage on node %q:\n%s", nodeName, buf.String())
}

// Performs a get on a node proxy endpoint given the nodename and rest client.
func nodeProxyRequest(c *client.Client, node, endpoint string) client.Result {
	return c.Get().
		Prefix("proxy").
		Resource("nodes").
		Name(fmt.Sprintf("%v:%v", node, ports.KubeletPort)).
		Suffix(endpoint).
		Do()
}

// Retrieve metrics from the kubelet server of the given node.
func getKubeletMetricsThroughProxy(c *client.Client, node string) (string, error) {
	metric, err := nodeProxyRequest(c, node, "metrics").Raw()
	if err != nil {
		return "", err
	}
	return string(metric), nil
}

// Retrieve metrics from the kubelet on the given node using a simple GET over http.
// Currently only used in integration tests.
func getKubeletMetricsThroughNode(nodeName string) (string, error) {
	resp, err := http.Get(fmt.Sprintf("http://%v/metrics", nodeName))
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return string(body), nil
}

// GetKubeletPods retrieves the list of running pods on the kubelet. The pods
// includes necessary information (e.g., UID, name, namespace for
// pods/containers), but do not contain the full spec.
func GetKubeletPods(c *client.Client, node string) (*api.PodList, error) {
	result := &api.PodList{}
	if err := nodeProxyRequest(c, node, "runningpods").Into(result); err != nil {
		return &api.PodList{}, err
	}
	return result, nil
}

func computeContainerResourceUsage(name string, oldStats, newStats *cadvisor.ContainerStats) *containerResourceUsage {
	return &containerResourceUsage{
		Name:                    name,
		Timestamp:               newStats.Timestamp,
		CPUUsageInCores:         float64(newStats.Cpu.Usage.Total-oldStats.Cpu.Usage.Total) / float64(newStats.Timestamp.Sub(oldStats.Timestamp).Nanoseconds()),
		MemoryUsageInBytes:      int64(newStats.Memory.Usage),
		MemoryWorkingSetInBytes: int64(newStats.Memory.WorkingSet),
		CPUInterval:             newStats.Timestamp.Sub(oldStats.Timestamp),
	}
}

// resourceCollector periodically polls the node, collect stats for a given
// list of containers, computes and cache resource usage up to
// maxEntriesPerContainer for each container.
type resourceCollector struct {
	lock            sync.RWMutex
	node            string
	containers      []string
	client          *client.Client
	buffers         map[string][]*containerResourceUsage
	pollingInterval time.Duration
	stopCh          chan struct{}
}

func newResourceCollector(c *client.Client, nodeName string, containerNames []string, pollingInterval time.Duration) *resourceCollector {
	buffers := make(map[string][]*containerResourceUsage)
	return &resourceCollector{
		node:            nodeName,
		containers:      containerNames,
		client:          c,
		buffers:         buffers,
		pollingInterval: pollingInterval,
	}
}

// Start starts a goroutine to poll the node every pollingInerval.
func (r *resourceCollector) Start() {
	r.stopCh = make(chan struct{}, 1)
	// Keep the last observed stats for comparison.
	oldStats := make(map[string]*cadvisor.ContainerStats)
	go util.Until(func() { r.collectStats(oldStats) }, r.pollingInterval, r.stopCh)
}

// Stop sends a signal to terminate the stats collecting goroutine.
func (r *resourceCollector) Stop() {
	close(r.stopCh)
}

// collectStats gets the latest stats from kubelet's /stats/container, computes
// the resource usage, and pushes it to the buffer.
func (r *resourceCollector) collectStats(oldStats map[string]*cadvisor.ContainerStats) {
	infos, err := getContainerInfo(r.client, r.node, &kubelet.StatsRequest{
		ContainerName: "/",
		NumStats:      1,
		Subcontainers: true,
	})
	if err != nil {
		Logf("Error getting container info on %q, err: %v", r.node, err)
		return
	}
	r.lock.Lock()
	defer r.lock.Unlock()
	for _, name := range r.containers {
		info, ok := infos[name]
		if !ok || len(info.Stats) < 1 {
			Logf("Missing info/stats for container %q on node %q", name, r.node)
			return
		}
		if _, ok := oldStats[name]; ok {
			r.buffers[name] = append(r.buffers[name], computeContainerResourceUsage(name, oldStats[name], info.Stats[0]))
		}
		oldStats[name] = info.Stats[0]
	}
}

// LogLatest logs the latest resource usage of each container.
func (r *resourceCollector) LogLatest() {
	r.lock.RLock()
	defer r.lock.RUnlock()
	stats := make(map[string]*containerResourceUsage)
	for _, name := range r.containers {
		contStats, ok := r.buffers[name]
		if !ok || len(contStats) == 0 {
			Logf("Resource usage on node %q is not ready yet", r.node)
			return
		}
		stats[name] = contStats[len(contStats)-1]
	}
	Logf("\n%s", formatResourceUsageStats(r.node, stats))
}

// Reset frees the stats and start over.
func (r *resourceCollector) Reset() {
	r.lock.Lock()
	defer r.lock.Unlock()
	for _, name := range r.containers {
		r.buffers[name] = []*containerResourceUsage{}
	}
}

type resourceUsageByCPU []*containerResourceUsage

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

// resourceMonitor manages a resourceCollector per node.
type resourceMonitor struct {
	client          *client.Client
	containers      []string
	pollingInterval time.Duration
	collectors      map[string]*resourceCollector
}

func newResourceMonitor(c *client.Client, containerNames []string, pollingInterval time.Duration) *resourceMonitor {
	return &resourceMonitor{
		containers:      containerNames,
		client:          c,
		pollingInterval: pollingInterval,
	}
}

func (r *resourceMonitor) Start() {
	nodes, err := r.client.Nodes().List(labels.Everything(), fields.Everything())
	if err != nil {
		Failf("resourceMonitor: unable to get list of nodes: %v", err)
	}
	r.collectors = make(map[string]*resourceCollector, 0)
	for _, node := range nodes.Items {
		collector := newResourceCollector(r.client, node.Name, r.containers, pollInterval)
		r.collectors[node.Name] = collector
		collector.Start()
	}
}

func (r *resourceMonitor) Stop() {
	for _, collector := range r.collectors {
		collector.Stop()
	}
}

func (r *resourceMonitor) Reset() {
	for _, collector := range r.collectors {
		collector.Reset()
	}
}

func (r *resourceMonitor) LogLatest() {
	for _, collector := range r.collectors {
		collector.LogLatest()
	}
}

func (r *resourceMonitor) LogCPUSummary() {
	// Example output for a node (the percentiles may differ):
	// CPU usage of containers on node "e2e-test-yjhong-minion-0vj7":
	// container        5th%  50th% 90th% 95th%
	// "/"              0.051 0.159 0.387 0.455
	// "/docker-daemon" 0.000 0.000 0.146 0.166
	// "/kubelet"       0.036 0.053 0.091 0.154
	// "/kube-proxy"    0.017 0.000 0.000 0.000
	// "/system"        0.001 0.001 0.001 0.002
	var header []string
	header = append(header, "container")
	for _, p := range percentiles {
		header = append(header, fmt.Sprintf("%.0fth%%", p*100))
	}
	for nodeName, collector := range r.collectors {
		buf := &bytes.Buffer{}
		w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
		fmt.Fprintf(w, "%s\n", strings.Join(header, "\t"))
		for _, containerName := range targetContainers() {
			data := collector.GetBasicCPUStats(containerName)
			var s []string
			s = append(s, fmt.Sprintf("%q", containerName))
			for _, p := range percentiles {
				s = append(s, fmt.Sprintf("%.3f", data[p]))
			}
			fmt.Fprintf(w, "%s\n", strings.Join(s, "\t"))
		}
		w.Flush()
		Logf("\nCPU usage of containers on node %q:\n%s", nodeName, buf.String())
	}
}
