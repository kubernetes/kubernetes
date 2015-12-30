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

/* A group of functions for automatic comparison for metrics embedded in test outputs */

package resultcompare

import (
	"math"
	"sort"
	"strings"

	"k8s.io/kubernetes/test/e2e"

	"github.com/golang/glog"
)

const (
	ResourceUsageVarianceAllowedPercent = 50
	// To avoid false negatives we assume that minimal CPU usage is 5% and memory 50MB
	minCpu = 0.05
	minMem = int64(50 * 1024 * 1024)
)

type percentileUsageData struct {
	perc    int
	cpuData []float64
	memData []int64
}

type ViolatingDataPair struct {
	perc         int
	leftCpuData  []float64
	rightCpuData []float64
	leftMemData  []int64
	rightMemData []int64
}

type int64arr []int64

func (a int64arr) Len() int           { return len(a) }
func (a int64arr) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a int64arr) Less(i, j int) bool { return a[i] < a[j] }

func max(left, right int64) int64 {
	if left > right {
		return left
	}
	return right
}

func min(left, right int64) int64 {
	if left < right {
		return left
	}
	return right
}

func getContainerKind(containerName string) string {
	return containerName[strings.LastIndex(containerName, "/")+1:]
}

/* A simple comparison checking if minimum and maximums in both datasets are within allowedVariance */
func isUsageSimilarEnough(left, right percentileUsageData, allowedVariance float64) bool {
	if len(left.cpuData) == 0 || len(left.memData) == 0 || len(right.cpuData) == 0 || len(right.memData) == 0 {
		glog.Warningf("Length of at least one data vector is zero. Returning false for the lack of data.")
		return false
	}

	sort.Float64s(left.cpuData)
	sort.Float64s(right.cpuData)
	sort.Sort(int64arr(left.memData))
	sort.Sort(int64arr(right.memData))

	leftCpuMin := math.Max(left.cpuData[0], minCpu)
	leftCpuMax := math.Max(left.cpuData[len(left.cpuData)-1], minCpu)
	leftMemMin := max(left.memData[0], minMem)
	leftMemMax := max(left.memData[len(left.memData)-1], minMem)
	rightCpuMin := math.Max(right.cpuData[0], minCpu)
	rightCpuMax := math.Max(right.cpuData[len(right.cpuData)-1], minCpu)
	rightMemMin := max(right.memData[0], minMem)
	rightMemMax := max(right.memData[len(right.memData)-1], minMem)

	return leftCpuMin < allowedVariance*rightCpuMin &&
		rightCpuMin < allowedVariance*leftCpuMin &&
		leftCpuMax < allowedVariance*rightCpuMax &&
		rightCpuMax < allowedVariance*leftCpuMax &&
		float64(leftMemMin) < allowedVariance*float64(rightMemMin) &&
		float64(rightMemMin) < allowedVariance*float64(leftMemMin) &&
		float64(leftMemMax) < allowedVariance*float64(rightMemMax) &&
		float64(rightMemMax) < allowedVariance*float64(leftMemMax)
}

/* Pivoting the data from percentile -> container to container_kind -> percentine */
func computeAggregates(data e2e.ResourceUsageSummary) map[string][]percentileUsageData {
	aggregates := make(map[string][]percentileUsageData)
	sortedPercentiles := make([]int, 0)
	for perc := range data {
		sortedPercentiles = append(sortedPercentiles, perc)
	}
	sort.Ints(sortedPercentiles)
	for _, perc := range sortedPercentiles {
		for i := range data[perc] {
			name := getContainerKind(data[perc][i].Name)
			aggregate, ok := aggregates[name]
			if !ok || aggregate[len(aggregate)-1].perc != perc {
				aggregates[name] = append(aggregates[name],
					percentileUsageData{perc: perc})
			}
			aggregates[name][len(aggregates[name])-1].cpuData = append(aggregates[name][len(aggregates[name])-1].cpuData, data[perc][i].Cpu)
			aggregates[name][len(aggregates[name])-1].memData = append(aggregates[name][len(aggregates[name])-1].memData, data[perc][i].Mem)
		}
	}
	return aggregates
}

func compareResourceUsages(left e2e.ResourceUsageSummary, right e2e.ResourceUsageSummary) map[string]ViolatingDataPair {
	leftAggregates := computeAggregates(left)
	rightAggregates := computeAggregates(right)

	allowedVariance := float64(100+ResourceUsageVarianceAllowedPercent) / float64(100)
	violatingContainers := make(map[string]ViolatingDataPair)
	for container := range leftAggregates {
		if _, ok := rightAggregates[container]; !ok {
			glog.Warningf("Missing results for container %v on right-hand side.", container)
			continue
		}
		j := 0
		for i := range leftAggregates[container] {
			for j < len(rightAggregates[container]) && rightAggregates[container][j].perc < leftAggregates[container][i].perc {
				j++
			}
			if j >= len(rightAggregates[container]) || rightAggregates[container][j].perc != leftAggregates[container][i].perc {
				glog.Warningf("Right-hand data for %v missing percentile: %v, skipping", container, leftAggregates[container][i].perc)
				continue
			}
			if !isUsageSimilarEnough(leftAggregates[container][i], rightAggregates[container][j], allowedVariance) {
				violatingContainers[container] = ViolatingDataPair{
					perc:         leftAggregates[container][i].perc,
					leftCpuData:  leftAggregates[container][i].cpuData,
					rightCpuData: rightAggregates[container][i].cpuData,
					leftMemData:  leftAggregates[container][i].memData,
					rightMemData: rightAggregates[container][i].memData,
				}
			}
		}
	}
	if len(violatingContainers) == 0 {
		return nil
	}
	return violatingContainers
}
