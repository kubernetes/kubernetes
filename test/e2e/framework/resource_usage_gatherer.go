/*
Copyright 2015 The Kubernetes Authors.

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
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientset "k8s.io/client-go/kubernetes"
	kubeletstatsv1alpha1 "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/test/e2e/system"

	// TODO: Remove the following imports (ref: https://github.com/kubernetes/kubernetes/issues/81245)
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
)

// ResourceConstraint is a struct to hold constraints.
type ResourceConstraint struct {
	CPUConstraint    float64
	MemoryConstraint uint64
}

// SingleContainerSummary is a struct to hold single container summary.
type SingleContainerSummary struct {
	Name string
	CPU  float64
	Mem  uint64
}

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

// ResourceUsageSummary is a struct to hold resource usage summary.
// we can't have int here, as JSON does not accept integer keys.
type ResourceUsageSummary map[string][]SingleContainerSummary

// PrintHumanReadable prints resource usage summary in human readable.
func (s *ResourceUsageSummary) PrintHumanReadable() string {
	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	for perc, summaries := range *s {
		buf.WriteString(fmt.Sprintf("%v percentile:\n", perc))
		fmt.Fprintf(w, "container\tcpu(cores)\tmemory(MB)\n")
		for _, summary := range summaries {
			fmt.Fprintf(w, "%q\t%.3f\t%.2f\n", summary.Name, summary.CPU, float64(summary.Mem)/(1024*1024))
		}
		w.Flush()
	}
	return buf.String()
}

// PrintJSON prints resource usage summary in JSON.
func (s *ResourceUsageSummary) PrintJSON() string {
	return PrettyPrintJSON(*s)
}

// SummaryKind returns string of ResourceUsageSummary
func (s *ResourceUsageSummary) SummaryKind() string {
	return "ResourceUsageSummary"
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

func computePercentiles(timeSeries []ResourceUsagePerContainer, percentilesToCompute []int) map[int]ResourceUsagePerContainer {
	if len(timeSeries) == 0 {
		return make(map[int]ResourceUsagePerContainer)
	}
	dataMap := make(map[string]*usageDataPerContainer)
	for i := range timeSeries {
		for name, data := range timeSeries[i] {
			if dataMap[name] == nil {
				dataMap[name] = &usageDataPerContainer{
					cpuData:        make([]float64, 0, len(timeSeries)),
					memUseData:     make([]uint64, 0, len(timeSeries)),
					memWorkSetData: make([]uint64, 0, len(timeSeries)),
				}
			}
			dataMap[name].cpuData = append(dataMap[name].cpuData, data.CPUUsageInCores)
			dataMap[name].memUseData = append(dataMap[name].memUseData, data.MemoryUsageInBytes)
			dataMap[name].memWorkSetData = append(dataMap[name].memWorkSetData, data.MemoryWorkingSetInBytes)
		}
	}
	for _, v := range dataMap {
		sort.Float64s(v.cpuData)
		sort.Sort(uint64arr(v.memUseData))
		sort.Sort(uint64arr(v.memWorkSetData))
	}

	result := make(map[int]ResourceUsagePerContainer)
	for _, perc := range percentilesToCompute {
		data := make(ResourceUsagePerContainer)
		for k, v := range dataMap {
			percentileIndex := int(math.Ceil(float64(len(v.cpuData)*perc)/100)) - 1
			data[k] = &ContainerResourceUsage{
				Name:                    k,
				CPUUsageInCores:         v.cpuData[percentileIndex],
				MemoryUsageInBytes:      v.memUseData[percentileIndex],
				MemoryWorkingSetInBytes: v.memWorkSetData[percentileIndex],
			}
		}
		result[perc] = data
	}
	return result
}

func leftMergeData(left, right map[int]ResourceUsagePerContainer) map[int]ResourceUsagePerContainer {
	result := make(map[int]ResourceUsagePerContainer)
	for percentile, data := range left {
		result[percentile] = data
		if _, ok := right[percentile]; !ok {
			continue
		}
		for k, v := range right[percentile] {
			result[percentile][k] = v
		}
	}
	return result
}

type resourceGatherWorker struct {
	c                           clientset.Interface
	nodeName                    string
	wg                          *sync.WaitGroup
	containerIDs                []string
	stopCh                      chan struct{}
	dataSeries                  []ResourceUsagePerContainer
	finished                    bool
	inKubemark                  bool
	resourceDataGatheringPeriod time.Duration
	probeDuration               time.Duration
	printVerboseLogs            bool
}

func (w *resourceGatherWorker) singleProbe() {
	data := make(ResourceUsagePerContainer)
	if w.inKubemark {
		kubemarkData := getKubemarkMasterComponentsResourceUsage()
		if kubemarkData == nil {
			return
		}
		for k, v := range kubemarkData {
			data[k] = &ContainerResourceUsage{
				Name:                    v.Name,
				MemoryWorkingSetInBytes: v.MemoryWorkingSetInBytes,
				CPUUsageInCores:         v.CPUUsageInCores,
			}
		}
	} else {
		nodeUsage, err := getOneTimeResourceUsageOnNode(w.c, w.nodeName, w.probeDuration, func() []string { return w.containerIDs })
		if err != nil {
			Logf("Error while reading data from %v: %v", w.nodeName, err)
			return
		}
		for k, v := range nodeUsage {
			data[k] = v
			if w.printVerboseLogs {
				Logf("Get container %v usage on node %v. CPUUsageInCores: %v, MemoryUsageInBytes: %v, MemoryWorkingSetInBytes: %v", k, w.nodeName, v.CPUUsageInCores, v.MemoryUsageInBytes, v.MemoryWorkingSetInBytes)
			}
		}
	}
	w.dataSeries = append(w.dataSeries, data)
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

	f := func(name string, newStats *kubeletstatsv1alpha1.ContainerStats) *ContainerResourceUsage {
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
	for _, pod := range summary.Pods {
		for _, container := range pod.Containers {
			isInteresting := false
			for _, interestingContainerName := range containers {
				if container.Name == interestingContainerName {
					isInteresting = true
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

// getStatsSummary contacts kubelet for the container information.
func getStatsSummary(c clientset.Interface, nodeName string) (*kubeletstatsv1alpha1.Summary, error) {
	ctx, cancel := context.WithTimeout(context.Background(), SingleCallTimeout)
	defer cancel()

	data, err := c.CoreV1().RESTClient().Get().
		Resource("nodes").
		SubResource("proxy").
		Name(fmt.Sprintf("%v:%v", nodeName, KubeletPort)).
		Suffix("stats/summary").
		Do(ctx).Raw()

	if err != nil {
		return nil, err
	}

	summary := kubeletstatsv1alpha1.Summary{}
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

func (w *resourceGatherWorker) gather(initialSleep time.Duration) {
	defer utilruntime.HandleCrash()
	defer w.wg.Done()
	defer Logf("Closing worker for %v", w.nodeName)
	defer func() { w.finished = true }()
	select {
	case <-time.After(initialSleep):
		w.singleProbe()
		for {
			select {
			case <-time.After(w.resourceDataGatheringPeriod):
				w.singleProbe()
			case <-w.stopCh:
				return
			}
		}
	case <-w.stopCh:
		return
	}
}

// ContainerResourceGatherer is a struct for gathering container resource.
type ContainerResourceGatherer struct {
	client       clientset.Interface
	stopCh       chan struct{}
	workers      []resourceGatherWorker
	workerWg     sync.WaitGroup
	containerIDs []string
	options      ResourceGathererOptions
}

// ResourceGathererOptions is a struct to hold options for resource.
type ResourceGathererOptions struct {
	InKubemark                  bool
	Nodes                       NodesSet
	ResourceDataGatheringPeriod time.Duration
	ProbeDuration               time.Duration
	PrintVerboseLogs            bool
}

// NodesSet is a value of nodes set.
type NodesSet int

const (
	// AllNodes means all containers on all nodes.
	AllNodes NodesSet = 0
	// MasterNodes means all containers on Master nodes only.
	MasterNodes NodesSet = 1
	// MasterAndDNSNodes means all containers on Master nodes and DNS containers on other nodes.
	MasterAndDNSNodes NodesSet = 2
)

// NewResourceUsageGatherer returns a new ContainerResourceGatherer.
func NewResourceUsageGatherer(c clientset.Interface, options ResourceGathererOptions, pods *v1.PodList) (*ContainerResourceGatherer, error) {
	g := ContainerResourceGatherer{
		client:       c,
		stopCh:       make(chan struct{}),
		containerIDs: make([]string, 0),
		options:      options,
	}

	if options.InKubemark {
		g.workerWg.Add(1)
		g.workers = append(g.workers, resourceGatherWorker{
			inKubemark:                  true,
			stopCh:                      g.stopCh,
			wg:                          &g.workerWg,
			finished:                    false,
			resourceDataGatheringPeriod: options.ResourceDataGatheringPeriod,
			probeDuration:               options.ProbeDuration,
			printVerboseLogs:            options.PrintVerboseLogs,
		})
		return &g, nil
	}

	// Tracks kube-system pods if no valid PodList is passed in.
	var err error
	if pods == nil {
		pods, err = c.CoreV1().Pods("kube-system").List(context.TODO(), metav1.ListOptions{})
		if err != nil {
			Logf("Error while listing Pods: %v", err)
			return nil, err
		}
	}
	dnsNodes := make(map[string]bool)
	for _, pod := range pods.Items {
		if (options.Nodes == MasterNodes) && !system.DeprecatedMightBeMasterNode(pod.Spec.NodeName) {
			continue
		}
		if (options.Nodes == MasterAndDNSNodes) && !system.DeprecatedMightBeMasterNode(pod.Spec.NodeName) && pod.Labels["k8s-app"] != "kube-dns" {
			continue
		}
		for _, container := range pod.Status.InitContainerStatuses {
			g.containerIDs = append(g.containerIDs, container.Name)
		}
		for _, container := range pod.Status.ContainerStatuses {
			g.containerIDs = append(g.containerIDs, container.Name)
		}
		if options.Nodes == MasterAndDNSNodes {
			dnsNodes[pod.Spec.NodeName] = true
		}
	}
	nodeList, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		Logf("Error while listing Nodes: %v", err)
		return nil, err
	}

	for _, node := range nodeList.Items {
		if options.Nodes == AllNodes || system.DeprecatedMightBeMasterNode(node.Name) || dnsNodes[node.Name] {
			g.workerWg.Add(1)
			g.workers = append(g.workers, resourceGatherWorker{
				c:                           c,
				nodeName:                    node.Name,
				wg:                          &g.workerWg,
				containerIDs:                g.containerIDs,
				stopCh:                      g.stopCh,
				finished:                    false,
				inKubemark:                  false,
				resourceDataGatheringPeriod: options.ResourceDataGatheringPeriod,
				probeDuration:               options.ProbeDuration,
				printVerboseLogs:            options.PrintVerboseLogs,
			})
			if options.Nodes == MasterNodes {
				break
			}
		}
	}
	return &g, nil
}

// StartGatheringData starts a stat gathering worker blocks for each node to track,
// and blocks until StopAndSummarize is called.
func (g *ContainerResourceGatherer) StartGatheringData() {
	if len(g.workers) == 0 {
		return
	}
	delayPeriod := g.options.ResourceDataGatheringPeriod / time.Duration(len(g.workers))
	delay := time.Duration(0)
	for i := range g.workers {
		go g.workers[i].gather(delay)
		delay += delayPeriod
	}
	g.workerWg.Wait()
}

// StopAndSummarize stops stat gathering workers, processes the collected stats,
// generates resource summary for the passed-in percentiles, and returns the summary.
// It returns an error if the resource usage at any percentile is beyond the
// specified resource constraints.
func (g *ContainerResourceGatherer) StopAndSummarize(percentiles []int, constraints map[string]ResourceConstraint) (*ResourceUsageSummary, error) {
	close(g.stopCh)
	Logf("Closed stop channel. Waiting for %v workers", len(g.workers))
	finished := make(chan struct{}, 1)
	go func() {
		g.workerWg.Wait()
		finished <- struct{}{}
	}()
	select {
	case <-finished:
		Logf("Waitgroup finished.")
	case <-time.After(2 * time.Minute):
		unfinished := make([]string, 0)
		for i := range g.workers {
			if !g.workers[i].finished {
				unfinished = append(unfinished, g.workers[i].nodeName)
			}
		}
		Logf("Timed out while waiting for waitgroup, some workers failed to finish: %v", unfinished)
	}

	if len(percentiles) == 0 {
		Logf("Warning! Empty percentile list for stopAndPrintData.")
		return &ResourceUsageSummary{}, fmt.Errorf("Failed to get any resource usage data")
	}
	data := make(map[int]ResourceUsagePerContainer)
	for i := range g.workers {
		if g.workers[i].finished {
			stats := computePercentiles(g.workers[i].dataSeries, percentiles)
			data = leftMergeData(stats, data)
		}
	}

	// Workers has been stopped. We need to gather data stored in them.
	sortedKeys := []string{}
	for name := range data[percentiles[0]] {
		sortedKeys = append(sortedKeys, name)
	}
	sort.Strings(sortedKeys)
	violatedConstraints := make([]string, 0)
	summary := make(ResourceUsageSummary)
	for _, perc := range percentiles {
		for _, name := range sortedKeys {
			usage := data[perc][name]
			summary[strconv.Itoa(perc)] = append(summary[strconv.Itoa(perc)], SingleContainerSummary{
				Name: name,
				CPU:  usage.CPUUsageInCores,
				Mem:  usage.MemoryWorkingSetInBytes,
			})

			// Verifying 99th percentile of resource usage
			if perc != 99 {
				continue
			}
			// Name has a form: <pod_name>/<container_name>
			containerName := strings.Split(name, "/")[1]
			constraint, ok := constraints[containerName]
			if !ok {
				continue
			}
			if usage.CPUUsageInCores > constraint.CPUConstraint {
				violatedConstraints = append(
					violatedConstraints,
					fmt.Sprintf("Container %v is using %v/%v CPU",
						name,
						usage.CPUUsageInCores,
						constraint.CPUConstraint,
					),
				)
			}
			if usage.MemoryWorkingSetInBytes > constraint.MemoryConstraint {
				violatedConstraints = append(
					violatedConstraints,
					fmt.Sprintf("Container %v is using %v/%v MB of memory",
						name,
						float64(usage.MemoryWorkingSetInBytes)/(1024*1024),
						float64(constraint.MemoryConstraint)/(1024*1024),
					),
				)
			}
		}
	}
	if len(violatedConstraints) > 0 {
		return &summary, fmt.Errorf(strings.Join(violatedConstraints, "\n"))
	}
	return &summary, nil
}

// kubemarkResourceUsage is a struct for tracking the resource usage of kubemark.
type kubemarkResourceUsage struct {
	Name                    string
	MemoryWorkingSetInBytes uint64
	CPUUsageInCores         float64
}

func getMasterUsageByPrefix(prefix string) (string, error) {
	sshResult, err := e2essh.SSH(fmt.Sprintf("ps ax -o %%cpu,rss,command | tail -n +2 | grep %v | sed 's/\\s+/ /g'", prefix), GetMasterHost()+":22", TestContext.Provider)
	if err != nil {
		return "", err
	}
	return sshResult.Stdout, nil
}

// getKubemarkMasterComponentsResourceUsage returns the resource usage of kubemark which contains multiple combinations of cpu and memory usage for each pod name.
func getKubemarkMasterComponentsResourceUsage() map[string]*kubemarkResourceUsage {
	result := make(map[string]*kubemarkResourceUsage)
	// Get kubernetes component resource usage
	sshResult, err := getMasterUsageByPrefix("kube")
	if err != nil {
		Logf("Error when trying to SSH to master machine. Skipping probe. %v", err)
		return nil
	}
	scanner := bufio.NewScanner(strings.NewReader(sshResult))
	for scanner.Scan() {
		var cpu float64
		var mem uint64
		var name string
		fmt.Sscanf(strings.TrimSpace(scanner.Text()), "%f %d /usr/local/bin/kube-%s", &cpu, &mem, &name)
		if name != "" {
			// Gatherer expects pod_name/container_name format
			fullName := name + "/" + name
			result[fullName] = &kubemarkResourceUsage{Name: fullName, MemoryWorkingSetInBytes: mem * 1024, CPUUsageInCores: cpu / 100}
		}
	}
	// Get etcd resource usage
	sshResult, err = getMasterUsageByPrefix("bin/etcd")
	if err != nil {
		Logf("Error when trying to SSH to master machine. Skipping probe")
		return nil
	}
	scanner = bufio.NewScanner(strings.NewReader(sshResult))
	for scanner.Scan() {
		var cpu float64
		var mem uint64
		var etcdKind string
		fmt.Sscanf(strings.TrimSpace(scanner.Text()), "%f %d /bin/sh -c /usr/local/bin/etcd", &cpu, &mem)
		dataDirStart := strings.Index(scanner.Text(), "--data-dir")
		if dataDirStart < 0 {
			continue
		}
		fmt.Sscanf(scanner.Text()[dataDirStart:], "--data-dir=/var/%s", &etcdKind)
		if etcdKind != "" {
			// Gatherer expects pod_name/container_name format
			fullName := "etcd/" + etcdKind
			result[fullName] = &kubemarkResourceUsage{Name: fullName, MemoryWorkingSetInBytes: mem * 1024, CPUUsageInCores: cpu / 100}
		}
	}
	return result
}
