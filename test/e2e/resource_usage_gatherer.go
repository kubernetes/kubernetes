/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"text/tabwriter"
	"time"

	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/unversioned"
)

const (
	resourceDataGatheringPeriodSeconds = 60
)

type resourceConstraint struct {
	cpuConstraint    float64
	memoryConstraint uint64
}

type containerResourceGatherer struct {
	usageTimeseries map[time.Time]resourceUsagePerContainer
	stopCh          chan struct{}
	timer           *time.Ticker
	wg              sync.WaitGroup
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
	return prettyPrintJSON(*s)
}

func (g *containerResourceGatherer) startGatheringData(c *client.Client, period time.Duration) {
	g.usageTimeseries = make(map[time.Time]resourceUsagePerContainer)
	g.wg.Add(1)
	g.stopCh = make(chan struct{})
	g.timer = time.NewTicker(period)
	go func() error {
		for {
			select {
			case <-g.timer.C:
				now := time.Now()
				data, err := g.getKubeSystemContainersResourceUsage(c)
				if err != nil {
					Logf("Error while getting resource usage: %v", err)
					continue
				}
				g.usageTimeseries[now] = data
			case <-g.stopCh:
				Logf("Stop channel is closed. Stopping gatherer.")
				g.wg.Done()
				return nil
			}
		}
	}()
}

func (g *containerResourceGatherer) stopAndSummarize(percentiles []int, constraints map[string]resourceConstraint) *ResourceUsageSummary {
	close(g.stopCh)
	Logf("Closed stop channel.")
	g.wg.Wait()
	Logf("Waitgroup finished.")
	if len(percentiles) == 0 {
		Logf("Warning! Empty percentile list for stopAndPrintData.")
		return &ResourceUsageSummary{}
	}
	stats := g.computePercentiles(g.usageTimeseries, percentiles)
	sortedKeys := []string{}
	for name := range stats[percentiles[0]] {
		sortedKeys = append(sortedKeys, name)
	}
	sort.Strings(sortedKeys)
	violatedConstraints := make([]string, 0)
	summary := make(ResourceUsageSummary)
	for _, perc := range percentiles {
		for _, name := range sortedKeys {
			usage := stats[perc][name]
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
					if usage.CPUUsageInCores > constraint.cpuConstraint {
						violatedConstraints = append(
							violatedConstraints,
							fmt.Sprintf("Container %v is using %v/%v CPU",
								name,
								usage.CPUUsageInCores,
								constraint.cpuConstraint,
							),
						)
					}
					if usage.MemoryWorkingSetInBytes > constraint.memoryConstraint {
						violatedConstraints = append(
							violatedConstraints,
							fmt.Sprintf("Container %v is using %v/%v MB of memory",
								name,
								float64(usage.MemoryWorkingSetInBytes)/(1024*1024),
								float64(constraint.memoryConstraint)/(1024*1024),
							),
						)
					}
				}
			}
		}
	}
	Expect(violatedConstraints).To(BeEmpty())
	return &summary
}

func (g *containerResourceGatherer) computePercentiles(timeSeries map[time.Time]resourceUsagePerContainer, percentilesToCompute []int) map[int]resourceUsagePerContainer {
	if len(timeSeries) == 0 {
		return make(map[int]resourceUsagePerContainer)
	}
	dataMap := make(map[string]*usageDataPerContainer)
	for _, singleStatistic := range timeSeries {
		for name, data := range singleStatistic {
			if dataMap[name] == nil {
				dataMap[name] = &usageDataPerContainer{
					cpuData:        make([]float64, len(timeSeries)),
					memUseData:     make([]uint64, len(timeSeries)),
					memWorkSetData: make([]uint64, len(timeSeries)),
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

	result := make(map[int]resourceUsagePerContainer)
	for _, perc := range percentilesToCompute {
		data := make(resourceUsagePerContainer)
		for k, v := range dataMap {
			percentileIndex := int(math.Ceil(float64(len(v.cpuData)*perc)/100)) - 1
			data[k] = &containerResourceUsage{
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

func (g *containerResourceGatherer) getKubeSystemContainersResourceUsage(c *client.Client) (resourceUsagePerContainer, error) {
	pods, err := c.Pods("kube-system").List(api.ListOptions{})
	if err != nil {
		return resourceUsagePerContainer{}, err
	}
	nodes, err := c.Nodes().List(api.ListOptions{})
	if err != nil {
		return resourceUsagePerContainer{}, err
	}
	containerIDToNameMap := make(map[string]string)
	containerIDs := make([]string, 0)
	for _, pod := range pods.Items {
		for _, container := range pod.Status.ContainerStatuses {
			containerID := strings.TrimPrefix(container.ContainerID, "docker:/")
			containerIDToNameMap[containerID] = pod.Name + "/" + container.Name
			containerIDs = append(containerIDs, containerID)
		}
	}

	mutex := sync.Mutex{}
	wg := sync.WaitGroup{}
	wg.Add(len(nodes.Items))
	errors := make([]error, 0)
	nameToUsageMap := make(resourceUsagePerContainer, len(containerIDToNameMap))
	for _, node := range nodes.Items {
		go func(nodeName string) {
			defer wg.Done()
			nodeUsage, err := getOneTimeResourceUsageOnNode(c, nodeName, 15*time.Second, func() []string { return containerIDs }, true)
			mutex.Lock()
			defer mutex.Unlock()
			if err != nil {
				errors = append(errors, err)
				return
			}
			for k, v := range nodeUsage {
				nameToUsageMap[containerIDToNameMap[k]] = v
			}
		}(node.Name)
	}
	wg.Wait()
	if len(errors) != 0 {
		return resourceUsagePerContainer{}, fmt.Errorf("Errors while gathering usage data: %v", errors)
	}
	return nameToUsageMap, nil
}
