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

	cadvisorapi "github.com/google/cadvisor/info/v1"
	"github.com/prometheus/common/model"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	kubeletstats "k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/master/ports"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	// timeout for proxy requests.
	proxyTimeout = 2 * time.Minute
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

// getContainerInfo contacts kubelet for the container information. The "Stats"
// in the returned ContainerInfo is subject to the requirements in statsRequest.
// TODO: This function uses the deprecated kubelet stats API; it should be
// removed.
func getContainerInfo(c *client.Client, nodeName string, req *kubeletstats.StatsRequest) (map[string]cadvisorapi.ContainerInfo, error) {
	reqBody, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	subResourceProxyAvailable, err := serverVersionGTE(subResourceServiceAndNodeProxyVersion, c)
	if err != nil {
		return nil, err
	}

	var data []byte
	if subResourceProxyAvailable {
		data, err = c.Post().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", nodeName, ports.KubeletPort)).
			Suffix("stats/container").
			SetHeader("Content-Type", "application/json").
			Body(reqBody).
			Do().Raw()

	} else {
		data, err = c.Post().
			Prefix("proxy").
			Resource("nodes").
			Name(fmt.Sprintf("%v:%v", nodeName, ports.KubeletPort)).
			Suffix("stats/container").
			SetHeader("Content-Type", "application/json").
			Body(reqBody).
			Do().Raw()
	}
	if err != nil {
		return nil, err
	}

	var containers map[string]cadvisorapi.ContainerInfo
	err = json.Unmarshal(data, &containers)
	if err != nil {
		return nil, err
	}
	return containers, nil
}

// getOneTimeResourceUsageOnNode queries the node's /stats/container endpoint
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
// the stats points in containerResourceUsage.CPUInterval.
//
// containerNames is a function returning a collection of container names in which
// user is interested in. ExpectMissingContainers is a flag which says if the test
// should fail if one of containers listed by containerNames is missing on any node
// (useful e.g. when looking for system containers or daemons). If set to true function
// is more forgiving and ignores missing containers.
// TODO: This function relies on the deprecated kubelet stats API and should be
// removed and/or rewritten.
func getOneTimeResourceUsageOnNode(
	c *client.Client,
	nodeName string,
	cpuInterval time.Duration,
	containerNames func() []string,
	expectMissingContainers bool,
) (resourceUsagePerContainer, error) {
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
	containerInfos, err := getContainerInfo(c, nodeName, &kubeletstats.StatsRequest{
		ContainerName: "/",
		NumStats:      numStats,
		Subcontainers: true,
	})
	if err != nil {
		return nil, err
	}

	f := func(name string, oldStats, newStats *cadvisorapi.ContainerStats) *containerResourceUsage {
		return &containerResourceUsage{
			Name:                    name,
			Timestamp:               newStats.Timestamp,
			CPUUsageInCores:         float64(newStats.Cpu.Usage.Total-oldStats.Cpu.Usage.Total) / float64(newStats.Timestamp.Sub(oldStats.Timestamp).Nanoseconds()),
			MemoryUsageInBytes:      newStats.Memory.Usage,
			MemoryWorkingSetInBytes: newStats.Memory.WorkingSet,
			MemoryRSSInBytes:        newStats.Memory.RSS,
			CPUInterval:             newStats.Timestamp.Sub(oldStats.Timestamp),
		}
	}
	// Process container infos that are relevant to us.
	containers := containerNames()
	usageMap := make(resourceUsagePerContainer, len(containers))
	for _, name := range containers {
		info, ok := containerInfos[name]
		if !ok {
			if !expectMissingContainers {
				return nil, fmt.Errorf("missing info for container %q on node %q", name, nodeName)
			}
			continue
		}
		first := info.Stats[0]
		last := info.Stats[len(info.Stats)-1]
		usageMap[name] = f(name, first, last)
	}
	return usageMap, nil
}

func getNodeStatsSummary(c *client.Client, nodeName string) (*stats.Summary, error) {
	subResourceProxyAvailable, err := serverVersionGTE(subResourceServiceAndNodeProxyVersion, c)
	if err != nil {
		return nil, err
	}

	var data []byte
	if subResourceProxyAvailable {
		data, err = c.Get().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", nodeName, ports.KubeletPort)).
			Suffix("stats/summary").
			SetHeader("Content-Type", "application/json").
			Do().Raw()

	} else {
		data, err = c.Get().
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
func targetContainers() []string {
	return []string{
		rootContainerName,
		stats.SystemContainerRuntime,
		stats.SystemContainerKubelet,
		stats.SystemContainerMisc,
	}
}

type containerResourceUsage struct {
	Name                    string
	Timestamp               time.Time
	CPUUsageInCores         float64
	MemoryUsageInBytes      uint64
	MemoryWorkingSetInBytes uint64
	MemoryRSSInBytes        uint64
	// The interval used to calculate CPUUsageInCores.
	CPUInterval time.Duration
}

func (r *containerResourceUsage) isStrictlyGreaterThan(rhs *containerResourceUsage) bool {
	return r.CPUUsageInCores > rhs.CPUUsageInCores && r.MemoryWorkingSetInBytes > rhs.MemoryWorkingSetInBytes
}

type resourceUsagePerContainer map[string]*containerResourceUsage
type resourceUsagePerNode map[string]resourceUsagePerContainer

func formatResourceUsageStats(nodeName string, containerStats resourceUsagePerContainer) string {
	// Example output:
	//
	// Resource usage for node "e2e-test-foo-minion-abcde":
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

// Performs a get on a node proxy endpoint given the nodename and rest client.
func nodeProxyRequest(c *client.Client, node, endpoint string) (restclient.Result, error) {
	// proxy tends to hang in some cases when Node is not ready. Add an artificial timeout for this call.
	// This will leak a goroutine if proxy hangs. #22165
	subResourceProxyAvailable, err := serverVersionGTE(subResourceServiceAndNodeProxyVersion, c)
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

// Retrieve metrics from the kubelet server of the given node.
func getKubeletMetricsThroughProxy(c *client.Client, node string) (string, error) {
	client, err := nodeProxyRequest(c, node, "metrics")
	if err != nil {
		return "", err
	}
	metric, errRaw := client.Raw()
	if errRaw != nil {
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

func getKubeletHeapStats(c *client.Client, nodeName string) (string, error) {
	client, err := nodeProxyRequest(c, nodeName, "debug/pprof/heap")
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

// GetKubeletPods retrieves the list of running pods on the kubelet. The pods
// includes necessary information (e.g., UID, name, namespace for
// pods/containers), but do not contain the full spec.
func GetKubeletPods(c *client.Client, node string) (*api.PodList, error) {
	result := &api.PodList{}
	client, err := nodeProxyRequest(c, node, "runningpods")
	if err != nil {
		return &api.PodList{}, err
	}
	if err = client.Into(result); err != nil {
		return &api.PodList{}, err
	}
	return result, nil
}

func PrintAllKubeletPods(c *client.Client, nodeName string) {
	result, err := nodeProxyRequest(c, nodeName, "pods")
	if err != nil {
		Logf("Unable to retrieve kubelet pods for node %v", nodeName)
		return
	}
	podList := &api.PodList{}
	err = result.Into(podList)
	if err != nil {
		Logf("Unable to cast result to pods for node %v", nodeName)
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

func computeContainerResourceUsage(name string, oldStats, newStats *stats.ContainerStats) *containerResourceUsage {
	return &containerResourceUsage{
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

// Start starts a goroutine to poll the node every pollingInterval.
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
	cStatsMap := getSystemContainerStats(summary)
	if err != nil {
		Logf("Error getting node stats summary on %q, err: %v", r.node, err)
		return
	}
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

func (r *resourceCollector) GetLatest() (resourceUsagePerContainer, error) {
	r.lock.RLock()
	defer r.lock.RUnlock()
	stats := make(resourceUsagePerContainer)
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
	// It should be OK to monitor unschedulable Nodes
	nodes, err := r.client.Nodes().List(api.ListOptions{})
	if err != nil {
		Failf("resourceMonitor: unable to get list of nodes: %v", err)
	}
	r.collectors = make(map[string]*resourceCollector, 0)
	for _, node := range nodes.Items {
		collector := newResourceCollector(r.client, node.Name, r.containers, r.pollingInterval)
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
	summary, err := r.GetLatest()
	if err != nil {
		Logf("%v", err)
	}
	Logf(r.FormatResourceUsage(summary))
}

func (r *resourceMonitor) FormatResourceUsage(s resourceUsagePerNode) string {
	summary := []string{}
	for node, usage := range s {
		summary = append(summary, formatResourceUsageStats(node, usage))
	}
	return strings.Join(summary, "\n")
}

func (r *resourceMonitor) GetLatest() (resourceUsagePerNode, error) {
	result := make(resourceUsagePerNode)
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

// containersCPUSummary is indexed by the container name with each entry a
// (percentile, value) map.
type containersCPUSummary map[string]map[float64]float64

// nodesCPUSummary is indexed by the node name with each entry a
// containersCPUSummary map.
type nodesCPUSummary map[string]containersCPUSummary

func (r *resourceMonitor) FormatCPUSummary(summary nodesCPUSummary) string {
	// Example output for a node (the percentiles may differ):
	// CPU usage of containers on node "e2e-test-foo-minion-0vj7":
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
		for _, containerName := range targetContainers() {
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

func (r *resourceMonitor) LogCPUSummary() {
	summary := r.GetCPUSummary()
	Logf(r.FormatCPUSummary(summary))
}

func (r *resourceMonitor) GetCPUSummary() nodesCPUSummary {
	result := make(nodesCPUSummary)
	for nodeName, collector := range r.collectors {
		result[nodeName] = make(containersCPUSummary)
		for _, containerName := range targetContainers() {
			data := collector.GetBasicCPUStats(containerName)
			result[nodeName][containerName] = data
		}
	}
	return result
}
