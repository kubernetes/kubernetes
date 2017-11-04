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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/util/system"
)

const (
	resourceDataGatheringPeriod = 60 * time.Second
	probeDuration               = 15 * time.Second
)

type ResourceConstraint struct {
	CPUConstraint    float64
	MemoryConstraint uint64
}

type SingleContainerSummary struct {
	Name string
	Cpu  float64
	Mem  uint64
}

// we can't have int here, as JSON does not accept integer keys.
type ResourceUsageSummary map[string][]SingleContainerSummary

func (s *ResourceUsageSummary) PrintHumanReadable() string {
	buf := &bytes.Buffer{}
	w := tabwriter.NewWriter(buf, 1, 0, 1, ' ', 0)
	for perc, summaries := range *s {
		buf.WriteString(fmt.Sprintf("%v percentile:\n", perc))
		fmt.Fprintf(w, "container\tcpu(cores)\tmemory(MB)\n")
		for _, summary := range summaries {
			fmt.Fprintf(w, "%q\t%.3f\t%.2f\n", summary.Name, summary.Cpu, float64(summary.Mem)/(1024*1024))
		}
		w.Flush()
	}
	return buf.String()
}

func (s *ResourceUsageSummary) PrintJSON() string {
	return PrettyPrintJSON(*s)
}

func (s *ResourceUsageSummary) SummaryKind() string {
	return "ResourceUsageSummary"
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
	c            clientset.Interface
	nodeName     string
	wg           *sync.WaitGroup
	containerIDs []string
	stopCh       chan struct{}
	dataSeries   []ResourceUsagePerContainer
	finished     bool
	inKubemark   bool
}

func (w *resourceGatherWorker) singleProbe() {
	data := make(ResourceUsagePerContainer)
	if w.inKubemark {
		kubemarkData := GetKubemarkMasterComponentsResourceUsage()
		if data == nil {
			return
		}
		for k, v := range kubemarkData {
			data[k] = &ContainerResourceUsage{
				Name: v.Name,
				MemoryWorkingSetInBytes: v.MemoryWorkingSetInBytes,
				CPUUsageInCores:         v.CPUUsageInCores,
			}
		}
	} else {
		nodeUsage, err := getOneTimeResourceUsageOnNode(w.c, w.nodeName, probeDuration, func() []string { return w.containerIDs })
		if err != nil {
			Logf("Error while reading data from %v: %v", w.nodeName, err)
			return
		}
		for k, v := range nodeUsage {
			data[k] = v
		}
	}
	w.dataSeries = append(w.dataSeries, data)
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
			case <-time.After(resourceDataGatheringPeriod):
				w.singleProbe()
			case <-w.stopCh:
				return
			}
		}
	case <-w.stopCh:
		return
	}
}

func (g *containerResourceGatherer) getKubeSystemContainersResourceUsage(c clientset.Interface) {
	if len(g.workers) == 0 {
		return
	}
	delayPeriod := resourceDataGatheringPeriod / time.Duration(len(g.workers))
	delay := time.Duration(0)
	for i := range g.workers {
		go g.workers[i].gather(delay)
		delay += delayPeriod
	}
	g.workerWg.Wait()
}

type containerResourceGatherer struct {
	client       clientset.Interface
	stopCh       chan struct{}
	workers      []resourceGatherWorker
	workerWg     sync.WaitGroup
	containerIDs []string
	options      ResourceGathererOptions
}

type ResourceGathererOptions struct {
	inKubemark bool
	masterOnly bool
}

func NewResourceUsageGatherer(c clientset.Interface, options ResourceGathererOptions) (*containerResourceGatherer, error) {
	g := containerResourceGatherer{
		client:       c,
		stopCh:       make(chan struct{}),
		containerIDs: make([]string, 0),
		options:      options,
	}

	if options.inKubemark {
		g.workerWg.Add(1)
		g.workers = append(g.workers, resourceGatherWorker{
			inKubemark: true,
			stopCh:     g.stopCh,
			wg:         &g.workerWg,
			finished:   false,
		})
	} else {
		pods, err := c.CoreV1().Pods("kube-system").List(metav1.ListOptions{})
		if err != nil {
			Logf("Error while listing Pods: %v", err)
			return nil, err
		}
		for _, pod := range pods.Items {
			for _, container := range pod.Status.ContainerStatuses {
				g.containerIDs = append(g.containerIDs, container.Name)
			}
		}
		nodeList, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
		if err != nil {
			Logf("Error while listing Nodes: %v", err)
			return nil, err
		}

		for _, node := range nodeList.Items {
			if !options.masterOnly || system.IsMasterNode(node.Name) {
				g.workerWg.Add(1)
				g.workers = append(g.workers, resourceGatherWorker{
					c:            c,
					nodeName:     node.Name,
					wg:           &g.workerWg,
					containerIDs: g.containerIDs,
					stopCh:       g.stopCh,
					finished:     false,
					inKubemark:   false,
				})
				if options.masterOnly {
					break
				}
			}
		}
	}
	return &g, nil
}

// startGatheringData blocks until stopAndSummarize is called.
func (g *containerResourceGatherer) startGatheringData() {
	g.getKubeSystemContainersResourceUsage(g.client)
}

func (g *containerResourceGatherer) stopAndSummarize(percentiles []int, constraints map[string]ResourceConstraint) (*ResourceUsageSummary, error) {
	close(g.stopCh)
	Logf("Closed stop channel. Waiting for %v workers", len(g.workers))
	finished := make(chan struct{})
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
				Cpu:  usage.CPUUsageInCores,
				Mem:  usage.MemoryWorkingSetInBytes,
			})
			// Verifying 99th percentile of resource usage
			if perc == 99 {
				// Name has a form: <pod_name>/<container_name>
				containerName := strings.Split(name, "/")[1]
				if constraint, ok := constraints[containerName]; ok {
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
		}
	}
	if len(violatedConstraints) > 0 {
		return &summary, fmt.Errorf(strings.Join(violatedConstraints, "\n"))
	}
	return &summary, nil
}
