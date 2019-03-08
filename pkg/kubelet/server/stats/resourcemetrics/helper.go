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

package resourcemetrics

import (
	"k8s.io/klog"
	"regexp"
	"time"

	infov1 "github.com/google/cadvisor/info/v1"
	info "github.com/google/cadvisor/info/v2"
	libcontainercgroups "github.com/opencontainers/runc/libcontainer/cgroups"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	summary "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

func containerNameToDockerId(cn cm.CgroupName) string {
	var dockerCgroupRegexp = regexp.MustCompile(`([a-z0-9]{64})`)
	if matches := dockerCgroupRegexp.FindStringSubmatch(cn.ToCgroupfs()); matches != nil {
		return matches[1]
	}
	return cn.ToCgroupfs()
}

func isPodManagedContainer(labels map[string]string) bool {
	podName := kubetypes.GetPodName(labels)
	podNamespace := kubetypes.GetPodNamespace(labels)
	managed := podName != "" && podNamespace != ""
	if !managed && podName != podNamespace {
		klog.Warningf(
			"Expect container to have either both podName (%s) and podNamespace (%s) labels, or neither.",
			podName, podNamespace)
	}
	return managed
}
func setCpuStats(s *libcontainercgroups.Stats, ret *infov1.CpuStats) {
	ret.Usage.User = s.CpuStats.CpuUsage.UsageInUsermode
	ret.Usage.System = s.CpuStats.CpuUsage.UsageInKernelmode
	ret.Usage.Total = s.CpuStats.CpuUsage.TotalUsage
	ret.CFS.Periods = s.CpuStats.ThrottlingData.Periods
	ret.CFS.ThrottledPeriods = s.CpuStats.ThrottlingData.ThrottledPeriods
	ret.CFS.ThrottledTime = s.CpuStats.ThrottlingData.ThrottledTime
}

func setMemoryStats(s *libcontainercgroups.Stats, ret *infov1.MemoryStats) {
	ret.Usage = s.MemoryStats.Usage.Usage
	ret.MaxUsage = s.MemoryStats.Usage.MaxUsage
	ret.Failcnt = s.MemoryStats.Usage.Failcnt

	if s.MemoryStats.UseHierarchy {
		ret.Cache = s.MemoryStats.Stats["total_cache"]
		ret.RSS = s.MemoryStats.Stats["total_rss"]
		ret.Swap = s.MemoryStats.Stats["total_swap"]
		ret.MappedFile = s.MemoryStats.Stats["total_mapped_file"]
	} else {
		ret.Cache = s.MemoryStats.Stats["cache"]
		ret.RSS = s.MemoryStats.Stats["rss"]
		ret.Swap = s.MemoryStats.Stats["swap"]
		ret.MappedFile = s.MemoryStats.Stats["mapped_file"]
	}
	if v, ok := s.MemoryStats.Stats["pgfault"]; ok {
		ret.ContainerData.Pgfault = v
		ret.HierarchicalData.Pgfault = v
	}
	if v, ok := s.MemoryStats.Stats["pgmajfault"]; ok {
		ret.ContainerData.Pgmajfault = v
		ret.HierarchicalData.Pgmajfault = v
	}

	workingSet := ret.Usage
	if v, ok := s.MemoryStats.Stats["total_inactive_file"]; ok {
		if workingSet < v {
			workingSet = 0
		} else {
			workingSet -= v
		}
	}
	ret.WorkingSet = workingSet
}

func cadvisorInfoToCPUandMemoryStats(stats *info.ContainerStats) (*summary.CPUStats, *summary.MemoryStats) {
	var cpuStats *summary.CPUStats
	var memoryStats *summary.MemoryStats
	cpuStats = &summary.CPUStats{
		Time: metav1.NewTime(stats.Timestamp),
	}
	if stats.CpuInst != nil {
		cpuStats.UsageNanoCores = &stats.CpuInst.Usage.Total
	}
	if stats.Cpu != nil {
		cpuStats.UsageCoreNanoSeconds = &stats.Cpu.Usage.Total
	}

	pageFaults := stats.Memory.ContainerData.Pgfault
	majorPageFaults := stats.Memory.ContainerData.Pgmajfault
	memoryStats = &summary.MemoryStats{
		Time:            metav1.NewTime(stats.Timestamp),
		UsageBytes:      &stats.Memory.Usage,
		WorkingSetBytes: &stats.Memory.WorkingSet,
		RSSBytes:        &stats.Memory.RSS,
		PageFaults:      &pageFaults,
		MajorPageFaults: &majorPageFaults,
	}

	return cpuStats, memoryStats
}

func buildPodRef(containerLabels map[string]string) summary.PodReference {
	podName := kubetypes.GetPodName(containerLabels)
	podNamespace := kubetypes.GetPodNamespace(containerLabels)
	podUID := kubetypes.GetPodUID(containerLabels)
	return summary.PodReference{Name: podName, Namespace: podNamespace, UID: podUID}
}

func extractContainerInfo(cg cm.CgroupName, cm cm.ContainerManager) (*summary.CPUStats, *summary.MemoryStats, time.Time, error) {
	s, time, err := cm.GetResourceMetrics(cg)
	if err != nil {
		return nil, nil, time, err
	}

	ret := &info.ContainerStats{
		Timestamp: time,
	}

	ret.Cpu = &infov1.CpuStats{}
	ret.Memory = &infov1.MemoryStats{}
	setCpuStats(s, ret.Cpu)
	setMemoryStats(s, ret.Memory)
	cpu, memory := cadvisorInfoToCPUandMemoryStats(ret)
	return cpu, memory, time, nil
}

func uint64Ptr(u uint64) *uint64 {
	return &u
}
