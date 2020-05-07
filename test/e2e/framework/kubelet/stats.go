/*
Copyright 2019 The Kubernetes Authors.

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

package kubelet

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"

	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
)

const (
	// timeout for proxy requests.
	proxyTimeout = 2 * time.Minute

	// dockerOperationsKey is the key for docker operation metrics.
	// copied from k8s.io/kubernetes/pkg/kubelet/dockershim/metrics
	dockerOperationsKey = "docker_operations_total"

	// dockerOperationsErrorsKey is the key for the operation error metrics.
	// copied from k8s.io/kubernetes/pkg/kubelet/dockershim/metrics
	dockerOperationsErrorsKey = "docker_operations_errors_total"

	// dockerOperationsTimeoutKey is the key for the operation timeout metrics.
	// copied from k8s.io/kubernetes/pkg/kubelet/dockershim/metrics
	dockerOperationsTimeoutKey = "docker_operations_timeout_total"

	// SystemContainerRuntime is the container name for the system container tracking the runtime (e.g. docker) usage.
        // NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
        SystemContainerRuntime = "runtime"
        // SystemContainerKubelet is the container name for the system container tracking Kubelet usage.
        // NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
        SystemContainerKubelet = "kubelet"
)

// ContainerResourceUsage is a structure for gathering container resource usage.
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

// ResourceUsagePerContainer is map of ContainerResourceUsage
type ResourceUsagePerContainer map[string]*ContainerResourceUsage

// ResourceUsagePerNode is map of ResourceUsagePerContainer.
type ResourceUsagePerNode map[string]ResourceUsagePerContainer

// ContainersCPUSummary is indexed by the container name with each entry a
// (percentile, value) map.
type ContainersCPUSummary map[string]map[float64]float64

// NodesCPUSummary is indexed by the node name with each entry a
// ContainersCPUSummary map.
type NodesCPUSummary map[string]ContainersCPUSummary

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

// Summary is a top-level container for holding NodeStats and PodStats.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type Summary struct {
        // Overall node stats.
        Node NodeStats `json:"node"`
        // Per-pod stats.
        Pods []PodStats `json:"pods"`
}

// NodeStats holds node-level unprocessed sample stats.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type NodeStats struct {
        // Reference to the measured Node.
        NodeName string `json:"nodeName"`
        // Stats of system daemons tracked as raw containers.
        // The system containers are named according to the SystemContainer* constants.
        // +optional
        // +patchMergeKey=name
        // +patchStrategy=merge
        SystemContainers []ContainerStats `json:"systemContainers,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
        // The time at which data collection for the node-scoped (i.e. aggregate) stats was (re)started.
        StartTime metav1.Time `json:"startTime"`
        // Stats pertaining to CPU resources.
        // +optional
        CPU *CPUStats `json:"cpu,omitempty"`
        // Stats pertaining to memory (RAM) resources.
        // +optional
        Memory *MemoryStats `json:"memory,omitempty"`
        // Stats pertaining to network resources.
        // +optional
        Network *NetworkStats `json:"network,omitempty"`
        // Stats pertaining to total usage of filesystem resources on the rootfs used by node k8s components.
        // NodeFs.Used is the total bytes used on the filesystem.
        // +optional
        Fs *FsStats `json:"fs,omitempty"`
        // Stats about the underlying container runtime.
        // +optional
        Runtime *RuntimeStats `json:"runtime,omitempty"`
        // Stats about the rlimit of system.
        // +optional
        Rlimit *RlimitStats `json:"rlimit,omitempty"`
}

// RlimitStats are stats rlimit of OS.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type RlimitStats struct {
        Time metav1.Time `json:"time"`

        // The max PID of OS.
        MaxPID *int64 `json:"maxpid,omitempty"`
        // The number of running process in the OS.
        NumOfRunningProcesses *int64 `json:"curproc,omitempty"`
}

// RuntimeStats are stats pertaining to the underlying container runtime.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type RuntimeStats struct {
        // Stats about the underlying filesystem where container images are stored.
        // This filesystem could be the same as the primary (root) filesystem.
        // Usage here refers to the total number of bytes occupied by images on the filesystem.
        // +optional
        ImageFs *FsStats `json:"imageFs,omitempty"`
}

// ProcessStats are stats pertaining to processes.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type ProcessStats struct {
        // Number of processes
        // +optional
        ProcessCount *uint64 `json:"process_count,omitempty"`
}

// PodStats holds pod-level unprocessed sample stats.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type PodStats struct {
        // Reference to the measured Pod.
        PodRef PodReference `json:"podRef"`
        // The time at which data collection for the pod-scoped (e.g. network) stats was (re)started.
        StartTime metav1.Time `json:"startTime"`
        // Stats of containers in the measured pod.
        // +patchMergeKey=name
        // +patchStrategy=merge
        Containers []ContainerStats `json:"containers" patchStrategy:"merge" patchMergeKey:"name"`
        // Stats pertaining to CPU resources consumed by pod cgroup (which includes all containers' resource usage and pod overhead).
        // +optional
        CPU *CPUStats `json:"cpu,omitempty"`
        // Stats pertaining to memory (RAM) resources consumed by pod cgroup (which includes all containers' resource usage and pod overhead).
        // +optional
        Memory *MemoryStats `json:"memory,omitempty"`
        // Stats pertaining to network resources.
        // +optional
        Network *NetworkStats `json:"network,omitempty"`
        // Stats pertaining to volume usage of filesystem resources.
        // VolumeStats.UsedBytes is the number of bytes used by the Volume
        // +optional
        // +patchMergeKey=name
        // +patchStrategy=merge
        VolumeStats []VolumeStats `json:"volume,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
        // EphemeralStorage reports the total filesystem usage for the containers and emptyDir-backed volumes in the measured Pod.
        // +optional
        EphemeralStorage *FsStats `json:"ephemeral-storage,omitempty"`
        // ProcessStats pertaining to processes.
        // +optional
        ProcessStats *ProcessStats `json:"process_stats,omitempty"`
}

// ContainerStats holds container-level unprocessed sample stats.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type ContainerStats struct {
        // Reference to the measured container.
        Name string `json:"name"`
        // The time at which data collection for this container was (re)started.
        StartTime metav1.Time `json:"startTime"`
        // Stats pertaining to CPU resources.
        // +optional
        CPU *CPUStats `json:"cpu,omitempty"`
        // Stats pertaining to memory (RAM) resources.
        // +optional
        Memory *MemoryStats `json:"memory,omitempty"`
        // Metrics for Accelerators. Each Accelerator corresponds to one element in the array.
        Accelerators []AcceleratorStats `json:"accelerators,omitempty"`
        // Stats pertaining to container rootfs usage of filesystem resources.
        // Rootfs.UsedBytes is the number of bytes used for the container write layer.
        // +optional
        Rootfs *FsStats `json:"rootfs,omitempty"`
        // Stats pertaining to container logs usage of filesystem resources.
        // Logs.UsedBytes is the number of bytes used for the container logs.
        // +optional
        Logs *FsStats `json:"logs,omitempty"`
        // User defined metrics that are exposed by containers in the pod. Typically, we expect only one container in the pod to be exposing user defined metrics. In the event of multiple containers exposing metrics, they will be combined here.
        // +patchMergeKey=name
        // +patchStrategy=merge
        UserDefinedMetrics []UserDefinedMetric `json:"userDefinedMetrics,omitempty" patchStrategy:"merge" patchMergeKey:"name"`
}

// PodReference contains enough information to locate the referenced pod.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type PodReference struct {
        Name      string `json:"name"`
        Namespace string `json:"namespace"`
        UID       string `json:"uid"`
}

// InterfaceStats contains resource value data about interface.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type InterfaceStats struct {
        // The name of the interface
        Name string `json:"name"`
        // Cumulative count of bytes received.
        // +optional
        RxBytes *uint64 `json:"rxBytes,omitempty"`
        // Cumulative count of receive errors encountered.
        // +optional
        RxErrors *uint64 `json:"rxErrors,omitempty"`
        // Cumulative count of bytes transmitted.
        // +optional
        TxBytes *uint64 `json:"txBytes,omitempty"`
        // Cumulative count of transmit errors encountered.
        // +optional
        TxErrors *uint64 `json:"txErrors,omitempty"`
}

// NetworkStats contains data about network resources.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type NetworkStats struct {
        // The time at which these stats were updated.
        Time metav1.Time `json:"time"`

        // Stats for the default interface, if found
        InterfaceStats `json:",inline"`

        Interfaces []InterfaceStats `json:"interfaces,omitempty"`
}

// CPUStats contains data about CPU usage.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type CPUStats struct {
        // The time at which these stats were updated.
        Time metav1.Time `json:"time"`
        // Total CPU usage (sum of all cores) averaged over the sample window.
        // The "core" unit can be interpreted as CPU core-nanoseconds per second.
        // +optional
        UsageNanoCores *uint64 `json:"usageNanoCores,omitempty"`
        // Cumulative CPU usage (sum of all cores) since object creation.
        // +optional
        UsageCoreNanoSeconds *uint64 `json:"usageCoreNanoSeconds,omitempty"`
}

// MemoryStats contains data about memory usage.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type MemoryStats struct {
        // The time at which these stats were updated.
        Time metav1.Time `json:"time"`
        // Available memory for use.  This is defined as the memory limit - workingSetBytes.
        // If memory limit is undefined, the available bytes is omitted.
        // +optional
        AvailableBytes *uint64 `json:"availableBytes,omitempty"`
        // Total memory in use. This includes all memory regardless of when it was accessed.
        // +optional
        UsageBytes *uint64 `json:"usageBytes,omitempty"`
        // The amount of working set memory. This includes recently accessed memory,
        // dirty memory, and kernel memory. WorkingSetBytes is <= UsageBytes
        // +optional
        WorkingSetBytes *uint64 `json:"workingSetBytes,omitempty"`
        // The amount of anonymous and swap cache memory (includes transparent
        // hugepages).
        // +optional
        RSSBytes *uint64 `json:"rssBytes,omitempty"`
        // Cumulative number of minor page faults.
        // +optional
        PageFaults *uint64 `json:"pageFaults,omitempty"`
        // Cumulative number of major page faults.
        // +optional
        MajorPageFaults *uint64 `json:"majorPageFaults,omitempty"`
}

// AcceleratorStats contains stats for accelerators attached to the container.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type AcceleratorStats struct {
        // Make of the accelerator (nvidia, amd, google etc.)
        Make string `json:"make"`

        // Model of the accelerator (tesla-p100, tesla-k80 etc.)
        Model string `json:"model"`

        // ID of the accelerator.
        ID string `json:"id"`

        // Total accelerator memory.
        // unit: bytes
        MemoryTotal uint64 `json:"memoryTotal"`

        // Total accelerator memory allocated.
        // unit: bytes
        MemoryUsed uint64 `json:"memoryUsed"`

        // Percent of time over the past sample period (10s) during which
        // the accelerator was actively processing.
        DutyCycle uint64 `json:"dutyCycle"`
}

// VolumeStats contains data about Volume filesystem usage.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type VolumeStats struct {
        // Embedded FsStats
        FsStats
        // Name is the name given to the Volume
        // +optional
        Name string `json:"name,omitempty"`
        // Reference to the PVC, if one exists
        // +optional
        PVCRef *PVCReference `json:"pvcRef,omitempty"`
}

// PVCReference contains enough information to describe the referenced PVC.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type PVCReference struct {
        Name      string `json:"name"`
        Namespace string `json:"namespace"`
}

// FsStats contains data about filesystem usage.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type FsStats struct {
        // The time at which these stats were updated.
        Time metav1.Time `json:"time"`
        // AvailableBytes represents the storage space available (bytes) for the filesystem.
        // +optional
        AvailableBytes *uint64 `json:"availableBytes,omitempty"`
        // CapacityBytes represents the total capacity (bytes) of the filesystems underlying storage.
        // +optional
        CapacityBytes *uint64 `json:"capacityBytes,omitempty"`
        // UsedBytes represents the bytes used for a specific task on the filesystem.
        // This may differ from the total bytes used on the filesystem and may not equal CapacityBytes - AvailableBytes.
        // e.g. For ContainerStats.Rootfs this is the bytes used by the container rootfs on the filesystem.
        // +optional
        UsedBytes *uint64 `json:"usedBytes,omitempty"`
        // InodesFree represents the free inodes in the filesystem.
        // +optional
        InodesFree *uint64 `json:"inodesFree,omitempty"`
        // Inodes represents the total inodes in the filesystem.
        // +optional
        Inodes *uint64 `json:"inodes,omitempty"`
        // InodesUsed represents the inodes used by the filesystem
        // This may not equal Inodes - InodesFree because this filesystem may share inodes with other "filesystems"
        // e.g. For ContainerStats.Rootfs, this is the inodes used only by that container, and does not count inodes used by other containers.
        InodesUsed *uint64 `json:"inodesUsed,omitempty"`
}

// UserDefinedMetricType defines how the metric should be interpreted by the user.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type UserDefinedMetricType string

// UserDefinedMetricDescriptor contains metadata that describes a user defined metric.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type UserDefinedMetricDescriptor struct {
        // The name of the metric.
        Name string `json:"name"`

        // Type of the metric.
        Type UserDefinedMetricType `json:"type"`

        // Display Units for the stats.
        Units string `json:"units"`

        // Metadata labels associated with this metric.
        // +optional
        Labels map[string]string `json:"labels,omitempty"`
}

// UserDefinedMetric represents a metric defined and generated by users.
// NOTE: Copied from pkg/kubelet/apis/stats/v1alpha1 to avoid pulling extra dependencies
type UserDefinedMetric struct {
        UserDefinedMetricDescriptor `json:",inline"`
        // The time at which these stats were updated.
        Time metav1.Time `json:"time"`
        // Value of the metric. Float64s have 53 bit precision.
        // We do not foresee any metrics exceeding that value.
        Value float64 `json:"value"`
}

// ProxyRequest performs a get on a node proxy endpoint given the nodename and rest client.
func ProxyRequest(c clientset.Interface, node, endpoint string, port int) (restclient.Result, error) {
	// proxy tends to hang in some cases when Node is not ready. Add an artificial timeout for this call. #22165
	var result restclient.Result
	finished := make(chan struct{}, 1)
	go func() {
		result = c.CoreV1().RESTClient().Get().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", node, port)).
			Suffix(endpoint).
			Do(context.TODO())

		finished <- struct{}{}
	}()
	select {
	case <-finished:
		return result, nil
	case <-time.After(proxyTimeout):
		return restclient.Result{}, nil
	}
}

// NewRuntimeOperationMonitor returns a new RuntimeOperationMonitor.
func NewRuntimeOperationMonitor(c clientset.Interface) *RuntimeOperationMonitor {
	m := &RuntimeOperationMonitor{
		client:          c,
		nodesRuntimeOps: make(map[string]NodeRuntimeOperationErrorRate),
	}
	nodes, err := m.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		framework.Failf("RuntimeOperationMonitor: unable to get list of nodes: %v", err)
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
			framework.Logf("GetRuntimeOperationErrorRate: unable to get kubelet metrics from node %q: %v", node, err)
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
			framework.Logf("GetLatestRuntimeOperationErrorRate: unable to get kubelet metrics from node %q: %v", node, err)
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
	ms, err := e2emetrics.GetKubeletMetrics(c, node)
	if err != nil {
		return result, err
	}
	// If no corresponding metrics are found, the returned samples will be empty. Then the following
	// loop will be skipped automatically.
	allOps := ms[dockerOperationsKey]
	errOps := ms[dockerOperationsErrorsKey]
	timeoutOps := ms[dockerOperationsTimeoutKey]
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

// GetStatsSummary contacts kubelet for the container information.
func GetStatsSummary(c clientset.Interface, nodeName string) (*Summary, error) {
	ctx, cancel := context.WithTimeout(context.Background(), framework.SingleCallTimeout)
	defer cancel()

	data, err := c.CoreV1().RESTClient().Get().
		Resource("nodes").
		SubResource("proxy").
		Name(fmt.Sprintf("%v:%v", nodeName, framework.KubeletPort)).
		Suffix("stats/summary").
		Do(ctx).Raw()

	if err != nil {
		return nil, err
	}

	summary := Summary{}
	err = json.Unmarshal(data, &summary)
	if err != nil {
		return nil, err
	}
	return &summary, nil
}

func getNodeStatsSummary(c clientset.Interface, nodeName string) (*Summary, error) {
	data, err := c.CoreV1().RESTClient().Get().
		Resource("nodes").
		SubResource("proxy").
		Name(fmt.Sprintf("%v:%v", nodeName, framework.KubeletPort)).
		Suffix("stats/summary").
		SetHeader("Content-Type", "application/json").
		Do(context.TODO()).Raw()

	if err != nil {
		return nil, err
	}

	var summary *Summary
	err = json.Unmarshal(data, &summary)
	if err != nil {
		return nil, err
	}
	return summary, nil
}

func getSystemContainerStats(summary *Summary) map[string]*ContainerStats {
	statsList := summary.Node.SystemContainers
	statsMap := make(map[string]*ContainerStats)
	for i := range statsList {
		statsMap[statsList[i].Name] = &statsList[i]
	}

	// Create a root container stats using information available in
	// stats.NodeStats. This is necessary since it is a different type.
	statsMap[rootContainerName] = &ContainerStats{
		CPU:    summary.Node.CPU,
		Memory: summary.Node.Memory,
	}
	return statsMap
}

const (
	rootContainerName = "/"
)

// TargetContainers returns a list of containers for which we want to collect resource usage.
func TargetContainers() []string {
	return []string{
		rootContainerName,
		SystemContainerRuntime,
		SystemContainerKubelet,
	}
}

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

// GetKubeletHeapStats returns stats of kubelet heap.
func GetKubeletHeapStats(c clientset.Interface, nodeName string) (string, error) {
	client, err := ProxyRequest(c, nodeName, "debug/pprof/heap", framework.KubeletPort)
	if err != nil {
		return "", err
	}
	raw, errRaw := client.Raw()
	if errRaw != nil {
		return "", err
	}
	kubeletstatsv1alpha1 := string(raw)
	// Only dumping the runtime.MemStats numbers to avoid polluting the log.
	numLines := 23
	lines := strings.Split(kubeletstatsv1alpha1, "\n")
	return strings.Join(lines[len(lines)-numLines:], "\n"), nil
}

func computeContainerResourceUsage(name string, oldStats, newStats *ContainerStats) *ContainerResourceUsage {
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
	oldStats := make(map[string]*ContainerStats)
	go wait.Until(func() { r.collectStats(oldStats) }, r.pollingInterval, r.stopCh)
}

// Stop sends a signal to terminate the stats collecting goroutine.
func (r *resourceCollector) Stop() {
	close(r.stopCh)
}

// collectStats gets the latest stats from kubelet stats summary API, computes
// the resource usage, and pushes it to the buffer.
func (r *resourceCollector) collectStats(oldStatsMap map[string]*ContainerStats) {
	summary, err := getNodeStatsSummary(r.client, r.node)
	if err != nil {
		framework.Logf("Error getting node stats summary on %q, err: %v", r.node, err)
		return
	}
	cStatsMap := getSystemContainerStats(summary)
	r.lock.Lock()
	defer r.lock.Unlock()
	for _, name := range r.containers {
		cStats, ok := cStatsMap[name]
		if !ok {
			framework.Logf("Missing info/stats for container %q on node %q", name, r.node)
			return
		}

		if oldStats, ok := oldStatsMap[name]; ok {
			if oldStats.CPU == nil || cStats.CPU == nil || oldStats.Memory == nil || cStats.Memory == nil {
				continue
			}
			if oldStats.CPU.Time.Equal(&cStats.CPU.Time) {
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
	kubeletstatsv1alpha1 := make(ResourceUsagePerContainer)
	for _, name := range r.containers {
		contStats, ok := r.buffers[name]
		if !ok || len(contStats) == 0 {
			return nil, fmt.Errorf("Resource usage on node %q is not ready yet", r.node)
		}
		kubeletstatsv1alpha1[name] = contStats[len(contStats)-1]
	}
	return kubeletstatsv1alpha1, nil
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

// NewResourceMonitor returns a new ResourceMonitor.
func NewResourceMonitor(c clientset.Interface, containerNames []string, pollingInterval time.Duration) *ResourceMonitor {
	return &ResourceMonitor{
		containers:      containerNames,
		client:          c,
		pollingInterval: pollingInterval,
	}
}

// Start starts collectors.
func (r *ResourceMonitor) Start() {
	// It should be OK to monitor unschedulable Nodes
	nodes, err := r.client.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		framework.Failf("ResourceMonitor: unable to get list of nodes: %v", err)
	}
	r.collectors = make(map[string]*resourceCollector, 0)
	for _, node := range nodes.Items {
		collector := newResourceCollector(r.client, node.Name, r.containers, r.pollingInterval)
		r.collectors[node.Name] = collector
		collector.Start()
	}
}

// Stop stops collectors.
func (r *ResourceMonitor) Stop() {
	for _, collector := range r.collectors {
		collector.Stop()
	}
}

// Reset resets collectors.
func (r *ResourceMonitor) Reset() {
	for _, collector := range r.collectors {
		collector.Reset()
	}
}

// LogLatest outputs the latest resource usage into log.
func (r *ResourceMonitor) LogLatest() {
	summary, err := r.GetLatest()
	if err != nil {
		framework.Logf("%v", err)
	}
	framework.Logf("%s", r.FormatResourceUsage(summary))
}

// FormatResourceUsage returns the formatted string for LogLatest().
// TODO(oomichi): This can be made to local function after making test/e2e/node/kubelet_perf.go use LogLatest directly instead.
func (r *ResourceMonitor) FormatResourceUsage(s ResourceUsagePerNode) string {
	summary := []string{}
	for node, usage := range s {
		summary = append(summary, formatResourceUsageStats(node, usage))
	}
	return strings.Join(summary, "\n")
}

// GetLatest returns the latest resource usage.
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

// GetMasterNodeLatest returns the latest resource usage of master and node.
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

// FormatCPUSummary returns the string of human-readable CPU summary from the specified summary data.
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

// LogCPUSummary outputs summary of CPU into log.
func (r *ResourceMonitor) LogCPUSummary() {
	summary := r.GetCPUSummary()
	framework.Logf("%s", r.FormatCPUSummary(summary))
}

// GetCPUSummary returns summary of CPU.
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

// GetMasterNodeCPUSummary returns summary of master node CPUs.
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
