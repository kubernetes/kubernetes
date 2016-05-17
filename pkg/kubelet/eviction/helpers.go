/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	statsapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	qosutil "k8s.io/kubernetes/pkg/kubelet/qos/util"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/quota/evaluator/core"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	unsupportedEvictionSignal = "unsupported eviction signal %v"
	// the reason reported back in status.
	reason = "Evicted"
	// the message associated with the reason.
	message = "The node was low on compute resources."
	// disk, in bytes.  internal to this module, used to account for local disk usage.
	resourceDisk api.ResourceName = "disk"
)

// resourceToRankFunc maps a resource to ranking function for that resource.
var resourceToRankFunc = map[api.ResourceName]rankFunc{
	api.ResourceMemory: rankMemoryPressure,
}

// signalToNodeCondition maps a signal to the node condition to report if threshold is met.
var signalToNodeCondition = map[Signal]api.NodeConditionType{
	SignalMemoryAvailable: api.NodeMemoryPressure,
}

// signalToResource maps a Signal to its associated Resource.
var signalToResource = map[Signal]api.ResourceName{
	SignalMemoryAvailable: api.ResourceMemory,
}

// validSignal returns true if the signal is supported.
func validSignal(signal Signal) bool {
	_, found := signalToResource[signal]
	return found
}

// ParseThresholdConfig parses the flags for thresholds.
func ParseThresholdConfig(evictionHard, evictionSoft, evictionSoftGracePeriod string) ([]Threshold, error) {
	results := []Threshold{}

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
	for i := range softThresholds {
		signal := softThresholds[i].Signal
		period, found := gracePeriods[signal]
		if !found {
			return nil, fmt.Errorf("grace period must be specified for the soft eviction threshold %v", signal)
		}
		softThresholds[i].GracePeriod = period
	}
	results = append(results, softThresholds...)
	return results, nil
}

// parseThresholdStatements parses the input statements into a list of Threshold objects.
func parseThresholdStatements(expr string) ([]Threshold, error) {
	if len(expr) == 0 {
		return nil, nil
	}
	results := []Threshold{}
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
func parseThresholdStatement(statement string) (Threshold, error) {
	tokens2Operator := map[string]ThresholdOperator{
		"<": OpLessThan,
	}
	var (
		operator ThresholdOperator
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
		return Threshold{}, fmt.Errorf("invalid eviction threshold syntax %v, expected <signal><operator><value>", statement)
	}
	signal := Signal(parts[0])
	if !validSignal(signal) {
		return Threshold{}, fmt.Errorf(unsupportedEvictionSignal, signal)
	}

	quantity, err := resource.ParseQuantity(parts[1])
	if err != nil {
		return Threshold{}, err
	}
	return Threshold{
		Signal:   signal,
		Operator: operator,
		Value:    &quantity,
	}, nil
}

// parseGracePeriods parses the grace period statements
func parseGracePeriods(expr string) (map[Signal]time.Duration, error) {
	if len(expr) == 0 {
		return nil, nil
	}
	results := map[Signal]time.Duration{}
	statements := strings.Split(expr, ",")
	for _, statement := range statements {
		parts := strings.Split(statement, "=")
		if len(parts) != 2 {
			return nil, fmt.Errorf("invalid eviction grace period syntax %v, expected <signal>=<duration>", statement)
		}
		signal := Signal(parts[0])
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

// diskUsage converts used bytes into a resource quantity.
func diskUsage(fsStats *statsapi.FsStats) *resource.Quantity {
	if fsStats == nil || fsStats.UsedBytes == nil {
		return &resource.Quantity{Format: resource.BinarySI}
	}
	usage := int64(*fsStats.UsedBytes)
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

// podUsage aggregates usage of compute resources.
// it supports the following memory and disk.
func podUsage(podStats statsapi.PodStats) (api.ResourceList, error) {
	disk := resource.Quantity{Format: resource.BinarySI}
	memory := resource.Quantity{Format: resource.BinarySI}
	for _, container := range podStats.Containers {
		// disk usage (if known)
		// TODO: need to handle volumes
		for _, fsStats := range []*statsapi.FsStats{container.Rootfs, container.Logs} {
			disk.Add(*diskUsage(fsStats))
		}
		// memory usage (if known)
		memory.Add(*memoryUsage(container.Memory))
	}
	return api.ResourceList{
		api.ResourceMemory: memory,
		resourceDisk:       disk,
	}, nil
}

// formatThreshold formats a threshold for logging.
func formatThreshold(threshold Threshold) string {
	return fmt.Sprintf("threshold(signal=%v, operator=%v, value=%v, gracePeriod=%v)", threshold.Signal, threshold.Value.String(), threshold.Operator, threshold.GracePeriod)
}

// cachedStatsFunc returns a statsFunc based on the provided pod stats.
func cachedStatsFunc(podStats []statsapi.PodStats) statsFunc {
	uid2PodStats := map[string]statsapi.PodStats{}
	for i := range podStats {
		uid2PodStats[podStats[i].PodRef.UID] = podStats[i]
	}
	return func(pod *api.Pod) (statsapi.PodStats, bool) {
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
type cmpFunc func(p1, p2 *api.Pod) int

// multiSorter implements the Sort interface, sorting changes within.
type multiSorter struct {
	pods []*api.Pod
	cmp  []cmpFunc
}

// Sort sorts the argument slice according to the less functions passed to OrderedBy.
func (ms *multiSorter) Sort(pods []*api.Pod) {
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

// qos compares pods by QoS (BestEffort < Burstable < Guaranteed)
func qos(p1, p2 *api.Pod) int {
	qosP1 := qosutil.GetPodQos(p1)
	qosP2 := qosutil.GetPodQos(p2)
	// its a tie
	if qosP1 == qosP2 {
		return 0
	}
	// if p1 is best effort, we know p2 is burstable or guaranteed
	if qosP1 == qosutil.BestEffort {
		return -1
	}
	// we know p1 and p2 are not besteffort, so if p1 is burstable, p2 must be guaranteed
	if qosP1 == qosutil.Burstable {
		if qosP2 == qosutil.Guaranteed {
			return -1
		}
		return 1
	}
	// ok, p1 must be guaranteed.
	return 1
}

// memory compares pods by largest consumer of memory relative to request.
func memory(stats statsFunc) cmpFunc {
	return func(p1, p2 *api.Pod) int {
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
		p1Usage, err := podUsage(p1Stats)
		if err != nil {
			return -1
		}
		// if we cant get usage for p2 measured, we want p1 first
		p2Usage, err := podUsage(p2Stats)
		if err != nil {
			return 1
		}

		// adjust p1, p2 usage relative to the request (if any)
		p1Memory := p1Usage[api.ResourceMemory]
		p1Spec := core.PodUsageFunc(p1)
		p1Request := p1Spec[api.ResourceRequestsMemory]
		p1Memory.Sub(p1Request)

		p2Memory := p2Usage[api.ResourceMemory]
		p2Spec := core.PodUsageFunc(p2)
		p2Request := p2Spec[api.ResourceRequestsMemory]
		p2Memory.Sub(p2Request)

		// if p2 is using more than p1, we want p2 first
		return p2Memory.Cmp(p1Memory)
	}
}

// disk compares pods by largest consumer of disk relative to request.
func disk(stats statsFunc) cmpFunc {
	return func(p1, p2 *api.Pod) int {
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
		p1Usage, err := podUsage(p1Stats)
		if err != nil {
			return -1
		}
		// if we cant get usage for p2 measured, we want p1 first
		p2Usage, err := podUsage(p2Stats)
		if err != nil {
			return 1
		}

		// disk is best effort, so we don't measure relative to a request.
		// TODO: add disk as a guaranteed resource
		p1Disk := p1Usage[api.ResourceStorage]
		p2Disk := p2Usage[api.ResourceStorage]
		// if p2 is using more than p1, we want p2 first
		return p2Disk.Cmp(p1Disk)
	}
}

// rankMemoryPressure orders the input pods for eviction in response to memory pressure.
func rankMemoryPressure(pods []*api.Pod, stats statsFunc) {
	orderedBy(qos, memory(stats)).Sort(pods)
}

// rankDiskPressure orders the input pods for eviction in response to disk pressure.
func rankDiskPressure(pods []*api.Pod, stats statsFunc) {
	orderedBy(qos, disk(stats)).Sort(pods)
}

// byEvictionPriority implements sort.Interface for []api.ResourceName.
type byEvictionPriority []api.ResourceName

func (a byEvictionPriority) Len() int      { return len(a) }
func (a byEvictionPriority) Swap(i, j int) { a[i], a[j] = a[j], a[i] }

// Less ranks memory before all other resources.
func (a byEvictionPriority) Less(i, j int) bool {
	return a[i] == api.ResourceMemory
}

// makeSignalObservations derives observations using the specified summary provider.
func makeSignalObservations(summaryProvider stats.SummaryProvider) (signalObservations, statsFunc, error) {
	summary, err := summaryProvider.Get()
	if err != nil {
		return nil, nil, err
	}
	// build the function to work against for pod stats
	statsFunc := cachedStatsFunc(summary.Pods)
	// build an evaluation context for current eviction signals
	result := signalObservations{}
	result[SignalMemoryAvailable] = resource.NewQuantity(int64(*summary.Node.Memory.AvailableBytes), resource.BinarySI)
	return result, statsFunc, nil
}

// thresholdsMet returns the set of thresholds that were met independent of grace period
func thresholdsMet(thresholds []Threshold, observations signalObservations) []Threshold {
	results := []Threshold{}
	for i := range thresholds {
		threshold := thresholds[i]
		observed, found := observations[threshold.Signal]
		if !found {
			glog.Warningf("eviction manager: no observation found for eviction signal %v", threshold.Signal)
			continue
		}
		// determine if we have met the specified threshold
		thresholdMet := false
		thresholdResult := threshold.Value.Cmp(*observed)
		switch threshold.Operator {
		case OpLessThan:
			thresholdMet = thresholdResult > 0
		}
		if thresholdMet {
			results = append(results, threshold)
		}
	}
	return results
}

// thresholdsFirstObservedAt merges the input set of thresholds with the previous observation to determine when active set of thresholds were initially met.
func thresholdsFirstObservedAt(thresholds []Threshold, lastObservedAt thresholdsObservedAt, now time.Time) thresholdsObservedAt {
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
func thresholdsMetGracePeriod(observedAt thresholdsObservedAt, now time.Time) []Threshold {
	results := []Threshold{}
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
func nodeConditions(thresholds []Threshold) []api.NodeConditionType {
	results := []api.NodeConditionType{}
	for _, threshold := range thresholds {
		if nodeCondition, found := signalToNodeCondition[threshold.Signal]; found {
			results = append(results, nodeCondition)
		}
	}
	return results
}

// nodeConditionsLastObservedAt merges the input with the previous observation to determine when a condition was most recently met.
func nodeConditionsLastObservedAt(nodeConditions []api.NodeConditionType, lastObservedAt nodeConditionsObservedAt, now time.Time) nodeConditionsObservedAt {
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
func nodeConditionsObservedSince(observedAt nodeConditionsObservedAt, period time.Duration, now time.Time) []api.NodeConditionType {
	results := []api.NodeConditionType{}
	for nodeCondition, at := range observedAt {
		duration := now.Sub(at)
		if duration < period {
			results = append(results, nodeCondition)
		}
	}
	return results
}

// hasNodeCondition returns true if the node condition is in the input list
func hasNodeCondition(inputs []api.NodeConditionType, item api.NodeConditionType) bool {
	for _, input := range inputs {
		if input == item {
			return true
		}
	}
	return false
}

// hasThreshold returns true if the node condition is in the input list
func hasThreshold(inputs []Threshold, item Threshold) bool {
	for _, input := range inputs {
		if input.GracePeriod == item.GracePeriod && input.Operator == item.Operator && input.Signal == item.Signal && input.Value.Cmp(*item.Value) == 0 {
			return true
		}
	}
	return false
}

// reclaimResources returns the set of resources that are starved based on thresholds met.
func reclaimResources(thresholds []Threshold) []api.ResourceName {
	results := []api.ResourceName{}
	for _, threshold := range thresholds {
		if starvedResource, found := signalToResource[threshold.Signal]; found {
			results = append(results, starvedResource)
		}
	}
	return results
}

// isSoftEviction returns true if the thresholds met for the starved resource are only soft thresholds
func isSoftEviction(thresholds []Threshold, starvedResource api.ResourceName) bool {
	for _, threshold := range thresholds {
		if resourceToCheck := signalToResource[threshold.Signal]; resourceToCheck != starvedResource {
			continue
		}
		if threshold.GracePeriod == time.Duration(0) {
			return false
		}
	}
	return true
}
