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

package eviction

import (
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	v1qos "k8s.io/kubernetes/pkg/api/v1/helper/qos"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/quota/evaluator/core"
)

const (
	unsupportedEvictionSignal = "unsupported eviction signal %v"
	// the reason reported back in status.
	reason = "Evicted"
	// the message associated with the reason.
	message = "The node was low on resource: %v."
	// disk, in bytes.  internal to this module, used to account for local disk usage.
	resourceDisk v1.ResourceName = "disk"
	// inodes, number. internal to this module, used to account for local disk inode consumption.
	resourceInodes v1.ResourceName = "inodes"
	// imagefs, in bytes.  internal to this module, used to account for local image filesystem usage.
	resourceImageFs v1.ResourceName = "imagefs"
	// imagefs inodes, number.  internal to this module, used to account for local image filesystem inodes.
	resourceImageFsInodes v1.ResourceName = "imagefsInodes"
	// nodefs, in bytes.  internal to this module, used to account for local node root filesystem usage.
	resourceNodeFs v1.ResourceName = "nodefs"
	// nodefs inodes, number.  internal to this module, used to account for local node root filesystem inodes.
	resourceNodeFsInodes v1.ResourceName = "nodefsInodes"
	// container overlay storage, in bytes.  internal to this module, used to account for local disk usage for container overlay.
	resourceOverlay v1.ResourceName = "overlay"
)

var (
	// signalToNodeCondition maps a signal to the node condition to report if threshold is met.
	signalToNodeCondition map[evictionapi.Signal]v1.NodeConditionType
	// signalToResource maps a Signal to its associated Resource.
	signalToResource map[evictionapi.Signal]v1.ResourceName
	// resourceToSignal maps a Resource to its associated Signal
	resourceToSignal map[v1.ResourceName]evictionapi.Signal
)

func init() {
	// map eviction signals to node conditions
	signalToNodeCondition = map[evictionapi.Signal]v1.NodeConditionType{}
	signalToNodeCondition[evictionapi.SignalMemoryAvailable] = v1.NodeMemoryPressure
	signalToNodeCondition[evictionapi.SignalAllocatableMemoryAvailable] = v1.NodeMemoryPressure
	signalToNodeCondition[evictionapi.SignalImageFsAvailable] = v1.NodeDiskPressure
	signalToNodeCondition[evictionapi.SignalNodeFsAvailable] = v1.NodeDiskPressure
	signalToNodeCondition[evictionapi.SignalImageFsInodesFree] = v1.NodeDiskPressure
	signalToNodeCondition[evictionapi.SignalNodeFsInodesFree] = v1.NodeDiskPressure
	signalToNodeCondition[evictionapi.SignalAllocatableNodeFsAvailable] = v1.NodeDiskPressure

	// map signals to resources (and vice-versa)
	signalToResource = map[evictionapi.Signal]v1.ResourceName{}
	signalToResource[evictionapi.SignalMemoryAvailable] = v1.ResourceMemory
	signalToResource[evictionapi.SignalAllocatableMemoryAvailable] = v1.ResourceMemory
	signalToResource[evictionapi.SignalAllocatableNodeFsAvailable] = resourceNodeFs
	signalToResource[evictionapi.SignalImageFsAvailable] = resourceImageFs
	signalToResource[evictionapi.SignalImageFsInodesFree] = resourceImageFsInodes
	signalToResource[evictionapi.SignalNodeFsAvailable] = resourceNodeFs
	signalToResource[evictionapi.SignalNodeFsInodesFree] = resourceNodeFsInodes

	resourceToSignal = map[v1.ResourceName]evictionapi.Signal{}
	for key, value := range signalToResource {
		resourceToSignal[value] = key
	}
	// Hard-code here to make sure resourceNodeFs maps to evictionapi.SignalNodeFsAvailable
	// (TODO) resourceToSignal is a map from resource name to a list of signals
	resourceToSignal[resourceNodeFs] = evictionapi.SignalNodeFsAvailable
}

// validSignal returns true if the signal is supported.
func validSignal(signal evictionapi.Signal) bool {
	_, found := signalToResource[signal]
	return found
}

// ParseThresholdConfig parses the flags for thresholds.
func ParseThresholdConfig(allocatableConfig []string, evictionHard, evictionSoft, evictionSoftGracePeriod, evictionMinimumReclaim string) ([]evictionapi.Threshold, error) {
	results := []evictionapi.Threshold{}
	allocatableThresholds := getAllocatableThreshold(allocatableConfig)
	results = append(results, allocatableThresholds...)

	hardThresholds, err := parseThresholdStatements(evictionHard)
	if err != nil {
		return nil, err
	}
	results = append(results, hardThresholds...)

	softThresholds, err := parseThresholdStatements(evictionSoft)
	if err != nil {
		return nil, err
	}
	gracePeriods, err := parseGracePeriods(evictionSoftGracePeriod)
	if err != nil {
		return nil, err
	}
	minReclaims, err := parseMinimumReclaims(evictionMinimumReclaim)
	if err != nil {
		return nil, err
	}
	for i := range softThresholds {
		signal := softThresholds[i].Signal
		period, found := gracePeriods[signal]
		if !found {
			return nil, fmt.Errorf("grace period must be specified for the soft eviction threshold %v", signal)
		}
		softThresholds[i].GracePeriod = period
	}
	results = append(results, softThresholds...)
	for i := range results {
		for signal, minReclaim := range minReclaims {
			if results[i].Signal == signal {
				results[i].MinReclaim = &minReclaim
				break
			}
		}
	}
	return results, nil
}

// parseThresholdStatements parses the input statements into a list of Threshold objects.
func parseThresholdStatements(expr string) ([]evictionapi.Threshold, error) {
	if len(expr) == 0 {
		return nil, nil
	}
	results := []evictionapi.Threshold{}
	statements := strings.Split(expr, ",")
	signalsFound := sets.NewString()
	for _, statement := range statements {
		result, err := parseThresholdStatement(statement)
		if err != nil {
			return nil, err
		}
		if signalsFound.Has(string(result.Signal)) {
			return nil, fmt.Errorf("found duplicate eviction threshold for signal %v", result.Signal)
		}
		signalsFound.Insert(string(result.Signal))
		results = append(results, result)
	}
	return results, nil
}

// parseThresholdStatement parses a threshold statement.
func parseThresholdStatement(statement string) (evictionapi.Threshold, error) {
	tokens2Operator := map[string]evictionapi.ThresholdOperator{
		"<": evictionapi.OpLessThan,
	}
	var (
		operator evictionapi.ThresholdOperator
		parts    []string
	)
	for token := range tokens2Operator {
		parts = strings.Split(statement, token)
		// if we got a token, we know this was the operator...
		if len(parts) > 1 {
			operator = tokens2Operator[token]
			break
		}
	}
	if len(operator) == 0 || len(parts) != 2 {
		return evictionapi.Threshold{}, fmt.Errorf("invalid eviction threshold syntax %v, expected <signal><operator><value>", statement)
	}
	signal := evictionapi.Signal(parts[0])
	if !validSignal(signal) {
		return evictionapi.Threshold{}, fmt.Errorf(unsupportedEvictionSignal, signal)
	}

	quantityValue := parts[1]
	if strings.HasSuffix(quantityValue, "%") {
		percentage, err := parsePercentage(quantityValue)
		if err != nil {
			return evictionapi.Threshold{}, err
		}
		if percentage <= 0 {
			return evictionapi.Threshold{}, fmt.Errorf("eviction percentage threshold %v must be positive: %s", signal, quantityValue)
		}
		return evictionapi.Threshold{
			Signal:   signal,
			Operator: operator,
			Value: evictionapi.ThresholdValue{
				Percentage: percentage,
			},
		}, nil
	}
	quantity, err := resource.ParseQuantity(quantityValue)
	if err != nil {
		return evictionapi.Threshold{}, err
	}
	if quantity.Sign() < 0 || quantity.IsZero() {
		return evictionapi.Threshold{}, fmt.Errorf("eviction threshold %v must be positive: %s", signal, &quantity)
	}
	return evictionapi.Threshold{
		Signal:   signal,
		Operator: operator,
		Value: evictionapi.ThresholdValue{
			Quantity: &quantity,
		},
	}, nil
}

// getAllocatableThreshold returns the thresholds applicable for the allocatable configuration
func getAllocatableThreshold(allocatableConfig []string) []evictionapi.Threshold {
	for _, key := range allocatableConfig {
		if key == cm.NodeAllocatableEnforcementKey {
			return []evictionapi.Threshold{
				{
					Signal:   evictionapi.SignalAllocatableMemoryAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: resource.NewQuantity(int64(0), resource.BinarySI),
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: resource.NewQuantity(int64(0), resource.BinarySI),
					},
				},
				{
					Signal:   evictionapi.SignalAllocatableNodeFsAvailable,
					Operator: evictionapi.OpLessThan,
					Value: evictionapi.ThresholdValue{
						Quantity: resource.NewQuantity(int64(0), resource.BinarySI),
					},
					MinReclaim: &evictionapi.ThresholdValue{
						Quantity: resource.NewQuantity(int64(0), resource.BinarySI),
					},
				},
			}
		}
	}
	return []evictionapi.Threshold{}
}

// parsePercentage parses a string representing a percentage value
func parsePercentage(input string) (float32, error) {
	value, err := strconv.ParseFloat(strings.TrimRight(input, "%"), 32)
	if err != nil {
		return 0, err
	}
	return float32(value) / 100, nil
}

// parseGracePeriods parses the grace period statements
func parseGracePeriods(expr string) (map[evictionapi.Signal]time.Duration, error) {
	if len(expr) == 0 {
		return nil, nil
	}
	results := map[evictionapi.Signal]time.Duration{}
	statements := strings.Split(expr, ",")
	for _, statement := range statements {
		parts := strings.Split(statement, "=")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid eviction grace period syntax %v, expected <signal>=<duration>", statement)
		}
		signal := evictionapi.Signal(parts[0])
		if !validSignal(signal) {
			return nil, fmt.Errorf(unsupportedEvictionSignal, signal)
		}

		gracePeriod, err := time.ParseDuration(parts[1])
		if err != nil {
			return nil, err
		}
		if gracePeriod < 0 {
			return nil, fmt.Errorf("invalid eviction grace period specified: %v, must be a positive value", parts[1])
		}

		// check against duplicate statements
		if _, found := results[signal]; found {
			return nil, fmt.Errorf("duplicate eviction grace period specified for %v", signal)
		}
		results[signal] = gracePeriod
	}
	return results, nil
}

// parseMinimumReclaims parses the minimum reclaim statements
func parseMinimumReclaims(expr string) (map[evictionapi.Signal]evictionapi.ThresholdValue, error) {
	if len(expr) == 0 {
		return nil, nil
	}
	results := map[evictionapi.Signal]evictionapi.ThresholdValue{}
	statements := strings.Split(expr, ",")
	for _, statement := range statements {
		parts := strings.Split(statement, "=")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid eviction minimum reclaim syntax: %v, expected <signal>=<value>", statement)
		}
		signal := evictionapi.Signal(parts[0])
		if !validSignal(signal) {
			return nil, fmt.Errorf(unsupportedEvictionSignal, signal)
		}

		quantityValue := parts[1]
		if strings.HasSuffix(quantityValue, "%") {
			percentage, err := parsePercentage(quantityValue)
			if err != nil {
				return nil, err
			}
			if percentage <= 0 {
				return nil, fmt.Errorf("eviction percentage minimum reclaim %v must be positive: %s", signal, quantityValue)
			}
			// check against duplicate statements
			if _, found := results[signal]; found {
				return nil, fmt.Errorf("duplicate eviction minimum reclaim specified for %v", signal)
			}
			results[signal] = evictionapi.ThresholdValue{
				Percentage: percentage,
			}
			continue
		}
		// check against duplicate statements
		if _, found := results[signal]; found {
			return nil, fmt.Errorf("duplicate eviction minimum reclaim specified for %v", signal)
		}
		quantity, err := resource.ParseQuantity(parts[1])
		if quantity.Sign() < 0 {
			return nil, fmt.Errorf("negative eviction minimum reclaim specified for %v", signal)
		}
		if err != nil {
			return nil, err
		}
		results[signal] = evictionapi.ThresholdValue{
			Quantity: &quantity,
		}
	}
	return results, nil
}

// diskUsage converts used bytes into a resource quantity.
func diskUsage(fsStats *statsapi.FsStats) *resource.Quantity {
	if fsStats == nil || fsStats.UsedBytes == nil {
		return &resource.Quantity{Format: resource.BinarySI}
	}
	usage := int64(*fsStats.UsedBytes)
	return resource.NewQuantity(usage, resource.BinarySI)
}

// inodeUsage converts inodes consumed into a resource quantity.
func inodeUsage(fsStats *statsapi.FsStats) *resource.Quantity {
	if fsStats == nil || fsStats.InodesUsed == nil {
		return &resource.Quantity{Format: resource.BinarySI}
	}
	usage := int64(*fsStats.InodesUsed)
	return resource.NewQuantity(usage, resource.BinarySI)
}

// memoryUsage converts working set into a resource quantity.
func memoryUsage(memStats *statsapi.MemoryStats) *resource.Quantity {
	if memStats == nil || memStats.WorkingSetBytes == nil {
		return &resource.Quantity{Format: resource.BinarySI}
	}
	usage := int64(*memStats.WorkingSetBytes)
	return resource.NewQuantity(usage, resource.BinarySI)
}

// localVolumeNames returns the set of volumes for the pod that are local
// TODO: sumamry API should report what volumes consume local storage rather than hard-code here.
func localVolumeNames(pod *v1.Pod) []string {
	result := []string{}
	for _, volume := range pod.Spec.Volumes {
		if volume.HostPath != nil ||
			(volume.EmptyDir != nil && volume.EmptyDir.Medium != v1.StorageMediumMemory) ||
			volume.ConfigMap != nil ||
			volume.GitRepo != nil {
			result = append(result, volume.Name)
		}
	}
	return result
}

// podDiskUsage aggregates pod disk usage and inode consumption for the specified stats to measure.
func podDiskUsage(podStats statsapi.PodStats, pod *v1.Pod, statsToMeasure []fsStatsType) (v1.ResourceList, error) {
	disk := resource.Quantity{Format: resource.BinarySI}
	inodes := resource.Quantity{Format: resource.BinarySI}
	overlay := resource.Quantity{Format: resource.BinarySI}
	for _, container := range podStats.Containers {
		if hasFsStatsType(statsToMeasure, fsStatsRoot) {
			disk.Add(*diskUsage(container.Rootfs))
			inodes.Add(*inodeUsage(container.Rootfs))
			overlay.Add(*diskUsage(container.Rootfs))
		}
		if hasFsStatsType(statsToMeasure, fsStatsLogs) {
			disk.Add(*diskUsage(container.Logs))
			inodes.Add(*inodeUsage(container.Logs))
		}
	}
	if hasFsStatsType(statsToMeasure, fsStatsLocalVolumeSource) {
		volumeNames := localVolumeNames(pod)
		for _, volumeName := range volumeNames {
			for _, volumeStats := range podStats.VolumeStats {
				if volumeStats.Name == volumeName {
					disk.Add(*diskUsage(&volumeStats.FsStats))
					inodes.Add(*inodeUsage(&volumeStats.FsStats))
					break
				}
			}
		}
	}
	return v1.ResourceList{
		resourceDisk:    disk,
		resourceInodes:  inodes,
		resourceOverlay: overlay,
	}, nil
}

// podMemoryUsage aggregates pod memory usage.
func podMemoryUsage(podStats statsapi.PodStats) (v1.ResourceList, error) {
	disk := resource.Quantity{Format: resource.BinarySI}
	memory := resource.Quantity{Format: resource.BinarySI}
	for _, container := range podStats.Containers {
		// disk usage (if known)
		for _, fsStats := range []*statsapi.FsStats{container.Rootfs, container.Logs} {
			disk.Add(*diskUsage(fsStats))
		}
		// memory usage (if known)
		memory.Add(*memoryUsage(container.Memory))
	}
	return v1.ResourceList{
		v1.ResourceMemory: memory,
		resourceDisk:      disk,
	}, nil
}

// formatThreshold formats a threshold for logging.
func formatThreshold(threshold evictionapi.Threshold) string {
	return fmt.Sprintf("threshold(signal=%v, operator=%v, value=%v, gracePeriod=%v)", threshold.Signal, threshold.Operator, evictionapi.ThresholdValue(threshold.Value), threshold.GracePeriod)
}

// formatevictionapi.ThresholdValue formats a thresholdValue for logging.
func formatThresholdValue(value evictionapi.ThresholdValue) string {
	if value.Quantity != nil {
		return value.Quantity.String()
	}
	return fmt.Sprintf("%f%%", value.Percentage*float32(100))
}

// cachedStatsFunc returns a statsFunc based on the provided pod stats.
func cachedStatsFunc(podStats []statsapi.PodStats) statsFunc {
	uid2PodStats := map[string]statsapi.PodStats{}
	for i := range podStats {
		uid2PodStats[podStats[i].PodRef.UID] = podStats[i]
	}
	return func(pod *v1.Pod) (statsapi.PodStats, bool) {
		stats, found := uid2PodStats[string(pod.UID)]
		return stats, found
	}
}

// Cmp compares p1 and p2 and returns:
//
//   -1 if p1 <  p2
//    0 if p1 == p2
//   +1 if p1 >  p2
//
type cmpFunc func(p1, p2 *v1.Pod) int

// multiSorter implements the Sort interface, sorting changes within.
type multiSorter struct {
	pods []*v1.Pod
	cmp  []cmpFunc
}

// Sort sorts the argument slice according to the less functions passed to OrderedBy.
func (ms *multiSorter) Sort(pods []*v1.Pod) {
	ms.pods = pods
	sort.Sort(ms)
}

// OrderedBy returns a Sorter that sorts using the cmp functions, in order.
// Call its Sort method to sort the data.
func orderedBy(cmp ...cmpFunc) *multiSorter {
	return &multiSorter{
		cmp: cmp,
	}
}

// Len is part of sort.Interface.
func (ms *multiSorter) Len() int {
	return len(ms.pods)
}

// Swap is part of sort.Interface.
func (ms *multiSorter) Swap(i, j int) {
	ms.pods[i], ms.pods[j] = ms.pods[j], ms.pods[i]
}

// Less is part of sort.Interface.
func (ms *multiSorter) Less(i, j int) bool {
	p1, p2 := ms.pods[i], ms.pods[j]
	var k int
	for k = 0; k < len(ms.cmp)-1; k++ {
		cmpResult := ms.cmp[k](p1, p2)
		// p1 is less than p2
		if cmpResult < 0 {
			return true
		}
		// p1 is greater than p2
		if cmpResult > 0 {
			return false
		}
		// we don't know yet
	}
	// the last cmp func is the final decider
	return ms.cmp[k](p1, p2) < 0
}

// qosComparator compares pods by QoS (BestEffort < Burstable < Guaranteed)
func qosComparator(p1, p2 *v1.Pod) int {
	qosP1 := v1qos.GetPodQOS(p1)
	qosP2 := v1qos.GetPodQOS(p2)
	// its a tie
	if qosP1 == qosP2 {
		return 0
	}
	// if p1 is best effort, we know p2 is burstable or guaranteed
	if qosP1 == v1.PodQOSBestEffort {
		return -1
	}
	// we know p1 and p2 are not besteffort, so if p1 is burstable, p2 must be guaranteed
	if qosP1 == v1.PodQOSBurstable {
		if qosP2 == v1.PodQOSGuaranteed {
			return -1
		}
		return 1
	}
	// ok, p1 must be guaranteed.
	return 1
}

// memory compares pods by largest consumer of memory relative to request.
func memory(stats statsFunc) cmpFunc {
	return func(p1, p2 *v1.Pod) int {
		p1Stats, found := stats(p1)
		// if we have no usage stats for p1, we want p2 first
		if !found {
			return -1
		}
		// if we have no usage stats for p2, but p1 has usage, we want p1 first.
		p2Stats, found := stats(p2)
		if !found {
			return 1
		}
		// if we cant get usage for p1 measured, we want p2 first
		p1Usage, err := podMemoryUsage(p1Stats)
		if err != nil {
			return -1
		}
		// if we cant get usage for p2 measured, we want p1 first
		p2Usage, err := podMemoryUsage(p2Stats)
		if err != nil {
			return 1
		}

		// adjust p1, p2 usage relative to the request (if any)
		p1Memory := p1Usage[v1.ResourceMemory]
		p1Spec, err := core.PodUsageFunc(p1)
		if err != nil {
			return -1
		}
		p1Request := p1Spec[api.ResourceRequestsMemory]
		p1Memory.Sub(p1Request)

		p2Memory := p2Usage[v1.ResourceMemory]
		p2Spec, err := core.PodUsageFunc(p2)
		if err != nil {
			return 1
		}
		p2Request := p2Spec[api.ResourceRequestsMemory]
		p2Memory.Sub(p2Request)

		// if p2 is using more than p1, we want p2 first
		return p2Memory.Cmp(p1Memory)
	}
}

// disk compares pods by largest consumer of disk relative to request for the specified disk resource.
func disk(stats statsFunc, fsStatsToMeasure []fsStatsType, diskResource v1.ResourceName) cmpFunc {
	return func(p1, p2 *v1.Pod) int {
		p1Stats, found := stats(p1)
		// if we have no usage stats for p1, we want p2 first
		if !found {
			return -1
		}
		// if we have no usage stats for p2, but p1 has usage, we want p1 first.
		p2Stats, found := stats(p2)
		if !found {
			return 1
		}
		// if we cant get usage for p1 measured, we want p2 first
		p1Usage, err := podDiskUsage(p1Stats, p1, fsStatsToMeasure)
		if err != nil {
			return -1
		}
		// if we cant get usage for p2 measured, we want p1 first
		p2Usage, err := podDiskUsage(p2Stats, p2, fsStatsToMeasure)
		if err != nil {
			return 1
		}

		// disk is best effort, so we don't measure relative to a request.
		// TODO: add disk as a guaranteed resource
		p1Disk := p1Usage[diskResource]
		p2Disk := p2Usage[diskResource]
		// if p2 is using more than p1, we want p2 first
		return p2Disk.Cmp(p1Disk)
	}
}

// rankMemoryPressure orders the input pods for eviction in response to memory pressure.
func rankMemoryPressure(pods []*v1.Pod, stats statsFunc) {
	orderedBy(qosComparator, memory(stats)).Sort(pods)
}

// rankDiskPressureFunc returns a rankFunc that measures the specified fs stats.
func rankDiskPressureFunc(fsStatsToMeasure []fsStatsType, diskResource v1.ResourceName) rankFunc {
	return func(pods []*v1.Pod, stats statsFunc) {
		orderedBy(qosComparator, disk(stats, fsStatsToMeasure, diskResource)).Sort(pods)
	}
}

// byEvictionPriority implements sort.Interface for []v1.ResourceName.
type byEvictionPriority []v1.ResourceName

func (a byEvictionPriority) Len() int      { return len(a) }
func (a byEvictionPriority) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

// Less ranks memory before all other resources.
func (a byEvictionPriority) Less(i, j int) bool {
	return a[i] == v1.ResourceMemory
}

// makeSignalObservations derives observations using the specified summary provider.
func makeSignalObservations(summaryProvider stats.SummaryProvider, capacityProvider CapacityProvider, pods []*v1.Pod, withImageFs bool) (signalObservations, statsFunc, error) {
	summary, err := summaryProvider.Get()
	if err != nil {
		return nil, nil, err
	}
	// build the function to work against for pod stats
	statsFunc := cachedStatsFunc(summary.Pods)
	// build an evaluation context for current eviction signals
	result := signalObservations{}

	if memory := summary.Node.Memory; memory != nil && memory.AvailableBytes != nil && memory.WorkingSetBytes != nil {
		result[evictionapi.SignalMemoryAvailable] = signalObservation{
			available: resource.NewQuantity(int64(*memory.AvailableBytes), resource.BinarySI),
			capacity:  resource.NewQuantity(int64(*memory.AvailableBytes+*memory.WorkingSetBytes), resource.BinarySI),
			time:      memory.Time,
		}
	}
	if nodeFs := summary.Node.Fs; nodeFs != nil {
		if nodeFs.AvailableBytes != nil && nodeFs.CapacityBytes != nil {
			result[evictionapi.SignalNodeFsAvailable] = signalObservation{
				available: resource.NewQuantity(int64(*nodeFs.AvailableBytes), resource.BinarySI),
				capacity:  resource.NewQuantity(int64(*nodeFs.CapacityBytes), resource.BinarySI),
				time:      nodeFs.Time,
			}
		}
		if nodeFs.InodesFree != nil && nodeFs.Inodes != nil {
			result[evictionapi.SignalNodeFsInodesFree] = signalObservation{
				available: resource.NewQuantity(int64(*nodeFs.InodesFree), resource.BinarySI),
				capacity:  resource.NewQuantity(int64(*nodeFs.Inodes), resource.BinarySI),
				time:      nodeFs.Time,
			}
		}
	}
	if summary.Node.Runtime != nil {
		if imageFs := summary.Node.Runtime.ImageFs; imageFs != nil {
			if imageFs.AvailableBytes != nil && imageFs.CapacityBytes != nil {
				result[evictionapi.SignalImageFsAvailable] = signalObservation{
					available: resource.NewQuantity(int64(*imageFs.AvailableBytes), resource.BinarySI),
					capacity:  resource.NewQuantity(int64(*imageFs.CapacityBytes), resource.BinarySI),
					time:      imageFs.Time,
				}
				if imageFs.InodesFree != nil && imageFs.Inodes != nil {
					result[evictionapi.SignalImageFsInodesFree] = signalObservation{
						available: resource.NewQuantity(int64(*imageFs.InodesFree), resource.BinarySI),
						capacity:  resource.NewQuantity(int64(*imageFs.Inodes), resource.BinarySI),
						time:      imageFs.Time,
					}
				}
			}
		}
	}

	nodeCapacity := capacityProvider.GetCapacity()
	allocatableReservation := capacityProvider.GetNodeAllocatableReservation()

	memoryAllocatableCapacity, memoryAllocatableAvailable, exist := getResourceAllocatable(nodeCapacity, allocatableReservation, v1.ResourceMemory)
	if exist {
		for _, pod := range summary.Pods {
			mu, err := podMemoryUsage(pod)
			if err == nil {
				memoryAllocatableAvailable.Sub(mu[v1.ResourceMemory])
			}
		}
		result[evictionapi.SignalAllocatableMemoryAvailable] = signalObservation{
			available: memoryAllocatableAvailable,
			capacity:  memoryAllocatableCapacity,
		}
	}

	storageScratchCapacity, storageScratchAllocatable, exist := getResourceAllocatable(nodeCapacity, allocatableReservation, v1.ResourceStorageScratch)
	if exist {
		for _, pod := range pods {
			podStat, ok := statsFunc(pod)
			if !ok {
				continue
			}

			usage, err := podDiskUsage(podStat, pod, []fsStatsType{fsStatsLogs, fsStatsLocalVolumeSource, fsStatsRoot})
			if err != nil {
				glog.Warningf("eviction manager: error getting pod disk usage %v", err)
				continue
			}
			// If there is a seperate imagefs set up for container runtimes, the scratch disk usage from nodefs should exclude the overlay usage
			if withImageFs {
				diskUsage := usage[resourceDisk]
				diskUsageP := &diskUsage
				diskUsagep := diskUsageP.Copy()
				diskUsagep.Sub(usage[resourceOverlay])
				storageScratchAllocatable.Sub(*diskUsagep)
			} else {
				storageScratchAllocatable.Sub(usage[resourceDisk])
			}
		}
		result[evictionapi.SignalAllocatableNodeFsAvailable] = signalObservation{
			available: storageScratchAllocatable,
			capacity:  storageScratchCapacity,
		}
	}

	return result, statsFunc, nil
}

func getResourceAllocatable(capacity v1.ResourceList, reservation v1.ResourceList, resourceName v1.ResourceName) (*resource.Quantity, *resource.Quantity, bool) {
	if capacity, ok := capacity[resourceName]; ok {
		allocate := capacity.Copy()
		if reserved, exists := reservation[resourceName]; exists {
			allocate.Sub(reserved)
		}
		return capacity.Copy(), allocate, true
	}
	glog.Errorf("Could not find capacity information for resource %v", resourceName)
	return nil, nil, false
}

// thresholdsMet returns the set of thresholds that were met independent of grace period
func thresholdsMet(thresholds []evictionapi.Threshold, observations signalObservations, enforceMinReclaim bool) []evictionapi.Threshold {
	results := []evictionapi.Threshold{}
	for i := range thresholds {
		threshold := thresholds[i]
		observed, found := observations[threshold.Signal]
		if !found {
			glog.Warningf("eviction manager: no observation found for eviction signal %v", threshold.Signal)
			continue
		}
		// determine if we have met the specified threshold
		thresholdMet := false
		quantity := evictionapi.GetThresholdQuantity(threshold.Value, observed.capacity)
		// if enforceMinReclaim is specified, we compare relative to value - minreclaim
		if enforceMinReclaim && threshold.MinReclaim != nil {
			quantity.Add(*evictionapi.GetThresholdQuantity(*threshold.MinReclaim, observed.capacity))
		}
		thresholdResult := quantity.Cmp(*observed.available)
		switch threshold.Operator {
		case evictionapi.OpLessThan:
			thresholdMet = thresholdResult > 0
		}
		if thresholdMet {
			results = append(results, threshold)
		}
	}
	return results
}

func debugLogObservations(logPrefix string, observations signalObservations) {
	if !glog.V(3) {
		return
	}
	for k, v := range observations {
		if !v.time.IsZero() {
			glog.Infof("eviction manager: %v: signal=%v, available: %v, capacity: %v, time: %v", logPrefix, k, v.available, v.capacity, v.time)
		} else {
			glog.Infof("eviction manager: %v: signal=%v, available: %v, capacity: %v", logPrefix, k, v.available, v.capacity)
		}
	}
}

func debugLogThresholdsWithObservation(logPrefix string, thresholds []evictionapi.Threshold, observations signalObservations) {
	if !glog.V(3) {
		return
	}
	for i := range thresholds {
		threshold := thresholds[i]
		observed, found := observations[threshold.Signal]
		if found {
			quantity := evictionapi.GetThresholdQuantity(threshold.Value, observed.capacity)
			glog.Infof("eviction manager: %v: threshold [signal=%v, quantity=%v] observed %v", logPrefix, threshold.Signal, quantity, observed.available)
		} else {
			glog.Infof("eviction manager: %v: threshold [signal=%v] had no observation", logPrefix, threshold.Signal)
		}
	}
}

func thresholdsUpdatedStats(thresholds []evictionapi.Threshold, observations, lastObservations signalObservations) []evictionapi.Threshold {
	results := []evictionapi.Threshold{}
	for i := range thresholds {
		threshold := thresholds[i]
		observed, found := observations[threshold.Signal]
		if !found {
			glog.Warningf("eviction manager: no observation found for eviction signal %v", threshold.Signal)
			continue
		}
		last, found := lastObservations[threshold.Signal]
		if !found || observed.time.IsZero() || observed.time.After(last.time.Time) {
			results = append(results, threshold)
		}
	}
	return results
}

// thresholdsFirstObservedAt merges the input set of thresholds with the previous observation to determine when active set of thresholds were initially met.
func thresholdsFirstObservedAt(thresholds []evictionapi.Threshold, lastObservedAt thresholdsObservedAt, now time.Time) thresholdsObservedAt {
	results := thresholdsObservedAt{}
	for i := range thresholds {
		observedAt, found := lastObservedAt[thresholds[i]]
		if !found {
			observedAt = now
		}
		results[thresholds[i]] = observedAt
	}
	return results
}

// thresholdsMetGracePeriod returns the set of thresholds that have satisfied associated grace period
func thresholdsMetGracePeriod(observedAt thresholdsObservedAt, now time.Time) []evictionapi.Threshold {
	results := []evictionapi.Threshold{}
	for threshold, at := range observedAt {
		duration := now.Sub(at)
		if duration < threshold.GracePeriod {
			glog.V(2).Infof("eviction manager: eviction criteria not yet met for %v, duration: %v", formatThreshold(threshold), duration)
			continue
		}
		results = append(results, threshold)
	}
	return results
}

// nodeConditions returns the set of node conditions associated with a threshold
func nodeConditions(thresholds []evictionapi.Threshold) []v1.NodeConditionType {
	results := []v1.NodeConditionType{}
	for _, threshold := range thresholds {
		if nodeCondition, found := signalToNodeCondition[threshold.Signal]; found {
			if !hasNodeCondition(results, nodeCondition) {
				results = append(results, nodeCondition)
			}
		}
	}
	return results
}

// nodeConditionsLastObservedAt merges the input with the previous observation to determine when a condition was most recently met.
func nodeConditionsLastObservedAt(nodeConditions []v1.NodeConditionType, lastObservedAt nodeConditionsObservedAt, now time.Time) nodeConditionsObservedAt {
	results := nodeConditionsObservedAt{}
	// the input conditions were observed "now"
	for i := range nodeConditions {
		results[nodeConditions[i]] = now
	}
	// the conditions that were not observed now are merged in with their old time
	for key, value := range lastObservedAt {
		_, found := results[key]
		if !found {
			results[key] = value
		}
	}
	return results
}

// nodeConditionsObservedSince returns the set of conditions that have been observed within the specified period
func nodeConditionsObservedSince(observedAt nodeConditionsObservedAt, period time.Duration, now time.Time) []v1.NodeConditionType {
	results := []v1.NodeConditionType{}
	for nodeCondition, at := range observedAt {
		duration := now.Sub(at)
		if duration < period {
			results = append(results, nodeCondition)
		}
	}
	return results
}

// hasFsStatsType returns true if the fsStat is in the input list
func hasFsStatsType(inputs []fsStatsType, item fsStatsType) bool {
	for _, input := range inputs {
		if input == item {
			return true
		}
	}
	return false
}

// hasNodeCondition returns true if the node condition is in the input list
func hasNodeCondition(inputs []v1.NodeConditionType, item v1.NodeConditionType) bool {
	for _, input := range inputs {
		if input == item {
			return true
		}
	}
	return false
}

// mergeThresholds will merge both threshold lists eliminating duplicates.
func mergeThresholds(inputsA []evictionapi.Threshold, inputsB []evictionapi.Threshold) []evictionapi.Threshold {
	results := inputsA
	for _, threshold := range inputsB {
		if !hasThreshold(results, threshold) {
			results = append(results, threshold)
		}
	}
	return results
}

// hasThreshold returns true if the threshold is in the input list
func hasThreshold(inputs []evictionapi.Threshold, item evictionapi.Threshold) bool {
	for _, input := range inputs {
		if input.GracePeriod == item.GracePeriod && input.Operator == item.Operator && input.Signal == item.Signal && compareThresholdValue(input.Value, item.Value) {
			return true
		}
	}
	return false
}

// compareThresholdValue returns true if the two thresholdValue objects are logically the same
func compareThresholdValue(a evictionapi.ThresholdValue, b evictionapi.ThresholdValue) bool {
	if a.Quantity != nil {
		if b.Quantity == nil {
			return false
		}
		return a.Quantity.Cmp(*b.Quantity) == 0
	}
	if b.Quantity != nil {
		return false
	}
	return a.Percentage == b.Percentage
}

// getStarvedResources returns the set of resources that are starved based on thresholds met.
func getStarvedResources(thresholds []evictionapi.Threshold) []v1.ResourceName {
	results := []v1.ResourceName{}
	for _, threshold := range thresholds {
		if starvedResource, found := signalToResource[threshold.Signal]; found {
			results = append(results, starvedResource)
		}
	}
	return results
}

// isSoftEviction returns true if the thresholds met for the starved resource are only soft thresholds
func isSoftEvictionThresholds(thresholds []evictionapi.Threshold, starvedResource v1.ResourceName) bool {
	for _, threshold := range thresholds {
		if resourceToCheck := signalToResource[threshold.Signal]; resourceToCheck != starvedResource {
			continue
		}
		if isHardEvictionThreshold(threshold) {
			return false
		}
	}
	return true
}

// isHardEvictionThreshold returns true if eviction should immediately occur
func isHardEvictionThreshold(threshold evictionapi.Threshold) bool {
	return threshold.GracePeriod == time.Duration(0)
}

// buildResourceToRankFunc returns ranking functions associated with resources
func buildResourceToRankFunc(withImageFs bool) map[v1.ResourceName]rankFunc {
	resourceToRankFunc := map[v1.ResourceName]rankFunc{
		v1.ResourceMemory: rankMemoryPressure,
	}
	// usage of an imagefs is optional
	if withImageFs {
		// with an imagefs, nodefs pod rank func for eviction only includes logs and local volumes
		resourceToRankFunc[resourceNodeFs] = rankDiskPressureFunc([]fsStatsType{fsStatsLogs, fsStatsLocalVolumeSource}, resourceDisk)
		resourceToRankFunc[resourceNodeFsInodes] = rankDiskPressureFunc([]fsStatsType{fsStatsLogs, fsStatsLocalVolumeSource}, resourceInodes)
		// with an imagefs, imagefs pod rank func for eviction only includes rootfs
		resourceToRankFunc[resourceImageFs] = rankDiskPressureFunc([]fsStatsType{fsStatsRoot}, resourceDisk)
		resourceToRankFunc[resourceImageFsInodes] = rankDiskPressureFunc([]fsStatsType{fsStatsRoot}, resourceInodes)
	} else {
		// without an imagefs, nodefs pod rank func for eviction looks at all fs stats.
		// since imagefs and nodefs share a common device, they share common ranking functions.
		resourceToRankFunc[resourceNodeFs] = rankDiskPressureFunc([]fsStatsType{fsStatsRoot, fsStatsLogs, fsStatsLocalVolumeSource}, resourceDisk)
		resourceToRankFunc[resourceNodeFsInodes] = rankDiskPressureFunc([]fsStatsType{fsStatsRoot, fsStatsLogs, fsStatsLocalVolumeSource}, resourceInodes)
		resourceToRankFunc[resourceImageFs] = rankDiskPressureFunc([]fsStatsType{fsStatsRoot, fsStatsLogs, fsStatsLocalVolumeSource}, resourceDisk)
		resourceToRankFunc[resourceImageFsInodes] = rankDiskPressureFunc([]fsStatsType{fsStatsRoot, fsStatsLogs, fsStatsLocalVolumeSource}, resourceInodes)
	}
	return resourceToRankFunc
}

// PodIsEvicted returns true if the reported pod status is due to an eviction.
func PodIsEvicted(podStatus v1.PodStatus) bool {
	return podStatus.Phase == v1.PodFailed && podStatus.Reason == reason
}

// buildResourceToNodeReclaimFuncs returns reclaim functions associated with resources.
func buildResourceToNodeReclaimFuncs(imageGC ImageGC, containerGC ContainerGC, withImageFs bool) map[v1.ResourceName]nodeReclaimFuncs {
	resourceToReclaimFunc := map[v1.ResourceName]nodeReclaimFuncs{}
	// usage of an imagefs is optional
	if withImageFs {
		// with an imagefs, nodefs pressure should just delete logs
		resourceToReclaimFunc[resourceNodeFs] = nodeReclaimFuncs{}
		resourceToReclaimFunc[resourceNodeFsInodes] = nodeReclaimFuncs{}
		// with an imagefs, imagefs pressure should delete unused images
		resourceToReclaimFunc[resourceImageFs] = nodeReclaimFuncs{deleteTerminatedContainers(containerGC), deleteImages(imageGC, true)}
		resourceToReclaimFunc[resourceImageFsInodes] = nodeReclaimFuncs{deleteTerminatedContainers(containerGC), deleteImages(imageGC, false)}
	} else {
		// without an imagefs, nodefs pressure should delete logs, and unused images
		// since imagefs and nodefs share a common device, they share common reclaim functions
		resourceToReclaimFunc[resourceNodeFs] = nodeReclaimFuncs{deleteTerminatedContainers(containerGC), deleteImages(imageGC, true)}
		resourceToReclaimFunc[resourceNodeFsInodes] = nodeReclaimFuncs{deleteTerminatedContainers(containerGC), deleteImages(imageGC, false)}
		resourceToReclaimFunc[resourceImageFs] = nodeReclaimFuncs{deleteTerminatedContainers(containerGC), deleteImages(imageGC, true)}
		resourceToReclaimFunc[resourceImageFsInodes] = nodeReclaimFuncs{deleteTerminatedContainers(containerGC), deleteImages(imageGC, false)}
	}
	return resourceToReclaimFunc
}

// deleteTerminatedContainers will delete terminated containers to free up disk pressure.
func deleteTerminatedContainers(containerGC ContainerGC) nodeReclaimFunc {
	return func() (*resource.Quantity, error) {
		glog.Infof("eviction manager: attempting to delete unused containers")
		err := containerGC.DeleteAllUnusedContainers()
		// Calculating bytes freed is not yet supported.
		return resource.NewQuantity(int64(0), resource.BinarySI), err
	}
}

// deleteImages will delete unused images to free up disk pressure.
func deleteImages(imageGC ImageGC, reportBytesFreed bool) nodeReclaimFunc {
	return func() (*resource.Quantity, error) {
		glog.Infof("eviction manager: attempting to delete unused images")
		bytesFreed, err := imageGC.DeleteUnusedImages()
		reclaimed := int64(0)
		if reportBytesFreed {
			reclaimed = bytesFreed
		}
		return resource.NewQuantity(reclaimed, resource.BinarySI), err
	}
}
