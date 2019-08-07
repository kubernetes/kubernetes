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
	"bytes"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/system"
	e2ekubelet "k8s.io/kubernetes/test/e2e/framework/kubelet"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
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

// ResourceUsageSummary is a struct to hold resource usage summary.
// we can't have int here, as JSON does not accept integer keys.
type ResourceUsageSummary map[string][]SingleContainerSummary

// NoCPUConstraint is the number of constraint for CPU.
const NoCPUConstraint = math.MaxFloat64

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
	return e2emetrics.PrettyPrintJSON(*s)
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

func computePercentiles(timeSeries []e2ekubelet.ResourceUsagePerContainer, percentilesToCompute []int) map[int]e2ekubelet.ResourceUsagePerContainer {
	if len(timeSeries) == 0 {
		return make(map[int]e2ekubelet.ResourceUsagePerContainer)
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

	result := make(map[int]e2ekubelet.ResourceUsagePerContainer)
	for _, perc := range percentilesToCompute {
		data := make(e2ekubelet.ResourceUsagePerContainer)
		for k, v := range dataMap {
			percentileIndex := int(math.Ceil(float64(len(v.cpuData)*perc)/100)) - 1
			data[k] = &e2ekubelet.ContainerResourceUsage{
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

func leftMergeData(left, right map[int]e2ekubelet.ResourceUsagePerContainer) map[int]e2ekubelet.ResourceUsagePerContainer {
	result := make(map[int]e2ekubelet.ResourceUsagePerContainer)
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
	dataSeries                  []e2ekubelet.ResourceUsagePerContainer
	finished                    bool
	inKubemark                  bool
	resourceDataGatheringPeriod time.Duration
	probeDuration               time.Duration
	printVerboseLogs            bool
}

func (w *resourceGatherWorker) singleProbe() {
	data := make(e2ekubelet.ResourceUsagePerContainer)
	if w.inKubemark {
		kubemarkData := GetKubemarkMasterComponentsResourceUsage()
		if data == nil {
			return
		}
		for k, v := range kubemarkData {
			data[k] = &e2ekubelet.ContainerResourceUsage{
				Name:                    v.Name,
				MemoryWorkingSetInBytes: v.MemoryWorkingSetInBytes,
				CPUUsageInCores:         v.CPUUsageInCores,
			}
		}
	} else {
		nodeUsage, err := e2ekubelet.GetOneTimeResourceUsageOnNode(w.c, w.nodeName, w.probeDuration, func() []string { return w.containerIDs })
		if err != nil {
			e2elog.Logf("Error while reading data from %v: %v", w.nodeName, err)
			return
		}
		for k, v := range nodeUsage {
			data[k] = v
			if w.printVerboseLogs {
				e2elog.Logf("Get container %v usage on node %v. CPUUsageInCores: %v, MemoryUsageInBytes: %v, MemoryWorkingSetInBytes: %v", k, w.nodeName, v.CPUUsageInCores, v.MemoryUsageInBytes, v.MemoryWorkingSetInBytes)
			}
		}
	}
	w.dataSeries = append(w.dataSeries, data)
}

func (w *resourceGatherWorker) gather(initialSleep time.Duration) {
	defer utilruntime.HandleCrash()
	defer w.wg.Done()
	defer e2elog.Logf("Closing worker for %v", w.nodeName)
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
		pods, err = c.CoreV1().Pods("kube-system").List(metav1.ListOptions{})
		if err != nil {
			e2elog.Logf("Error while listing Pods: %v", err)
			return nil, err
		}
	}
	dnsNodes := make(map[string]bool)
	for _, pod := range pods.Items {
		if (options.Nodes == MasterNodes) && !system.IsMasterNode(pod.Spec.NodeName) {
			continue
		}
		if (options.Nodes == MasterAndDNSNodes) && !system.IsMasterNode(pod.Spec.NodeName) && pod.Labels["k8s-app"] != "kube-dns" {
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
	nodeList, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	if err != nil {
		e2elog.Logf("Error while listing Nodes: %v", err)
		return nil, err
	}

	for _, node := range nodeList.Items {
		if options.Nodes == AllNodes || system.IsMasterNode(node.Name) || dnsNodes[node.Name] {
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
	e2elog.Logf("Closed stop channel. Waiting for %v workers", len(g.workers))
	finished := make(chan struct{})
	go func() {
		g.workerWg.Wait()
		finished <- struct{}{}
	}()
	select {
	case <-finished:
		e2elog.Logf("Waitgroup finished.")
	case <-time.After(2 * time.Minute):
		unfinished := make([]string, 0)
		for i := range g.workers {
			if !g.workers[i].finished {
				unfinished = append(unfinished, g.workers[i].nodeName)
			}
		}
		e2elog.Logf("Timed out while waiting for waitgroup, some workers failed to finish: %v", unfinished)
	}

	if len(percentiles) == 0 {
		e2elog.Logf("Warning! Empty percentile list for stopAndPrintData.")
		return &ResourceUsageSummary{}, fmt.Errorf("Failed to get any resource usage data")
	}
	data := make(map[int]e2ekubelet.ResourceUsagePerContainer)
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
