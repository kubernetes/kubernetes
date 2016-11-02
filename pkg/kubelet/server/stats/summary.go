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

package stats

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/leaky"
	"k8s.io/kubernetes/pkg/kubelet/network"
	"k8s.io/kubernetes/pkg/kubelet/types"
	kubetypes "k8s.io/kubernetes/pkg/types"

	"github.com/golang/glog"

	cadvisorapiv1 "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
)

type SummaryProvider interface {
	// Get provides a new Summary using the latest results from cadvisor
	Get() (*stats.Summary, error)
}

type summaryProviderImpl struct {
	provider           StatsProvider
	fsResourceAnalyzer fsResourceAnalyzerInterface
	runtime            container.Runtime
}

var _ SummaryProvider = &summaryProviderImpl{}

// NewSummaryProvider returns a new SummaryProvider
func NewSummaryProvider(statsProvider StatsProvider, fsResourceAnalyzer fsResourceAnalyzerInterface, cruntime container.Runtime) SummaryProvider {
	return &summaryProviderImpl{statsProvider, fsResourceAnalyzer, cruntime}
}

// Get implements the SummaryProvider interface
// Query cadvisor for the latest resource metrics and build into a summary
func (sp *summaryProviderImpl) Get() (*stats.Summary, error) {
	options := cadvisorapiv2.RequestOptions{
		IdType:    cadvisorapiv2.TypeName,
		Count:     2, // 2 samples are needed to compute "instantaneous" CPU
		Recursive: true,
	}
	infos, err := sp.provider.GetContainerInfoV2("/", options)
	if err != nil {
		if _, ok := infos["/"]; ok {
			// If the failure is partial, log it and return a best-effort response.
			glog.Errorf("Partial failure issuing GetContainerInfoV2: %v", err)
		} else {
			return nil, fmt.Errorf("failed GetContainerInfoV2: %v", err)
		}
	}

	// TODO(timstclair): Consider returning a best-effort response if any of the following errors
	// occur.
	node, err := sp.provider.GetNode()
	if err != nil {
		return nil, fmt.Errorf("failed GetNode: %v", err)
	}

	nodeConfig := sp.provider.GetNodeConfig()
	rootFsInfo, err := sp.provider.RootFsInfo()
	if err != nil {
		return nil, fmt.Errorf("failed RootFsInfo: %v", err)
	}
	imageFsInfo, err := sp.provider.ImagesFsInfo()
	if err != nil {
		return nil, fmt.Errorf("failed DockerImagesFsInfo: %v", err)
	}
	imageStats, err := sp.runtime.ImageStats()
	if err != nil || imageStats == nil {
		return nil, fmt.Errorf("failed ImageStats: %v", err)
	}
	sb := &summaryBuilder{sp.fsResourceAnalyzer, node, nodeConfig, rootFsInfo, imageFsInfo, *imageStats, infos}
	return sb.build()
}

// summaryBuilder aggregates the datastructures provided by cadvisor into a Summary result
type summaryBuilder struct {
	fsResourceAnalyzer fsResourceAnalyzerInterface
	node               *api.Node
	nodeConfig         cm.NodeConfig
	rootFsInfo         cadvisorapiv2.FsInfo
	imageFsInfo        cadvisorapiv2.FsInfo
	imageStats         container.ImageStats
	infos              map[string]cadvisorapiv2.ContainerInfo
}

// build returns a Summary from aggregating the input data
func (sb *summaryBuilder) build() (*stats.Summary, error) {
	rootInfo, found := sb.infos["/"]
	if !found {
		return nil, fmt.Errorf("Missing stats for root container")
	}

	var nodeFsInodesUsed *uint64
	if sb.rootFsInfo.Inodes != nil && sb.rootFsInfo.InodesFree != nil {
		nodeFsIU := *sb.rootFsInfo.Inodes - *sb.rootFsInfo.InodesFree
		nodeFsInodesUsed = &nodeFsIU
	}

	var imageFsInodesUsed *uint64
	if sb.imageFsInfo.Inodes != nil && sb.imageFsInfo.InodesFree != nil {
		imageFsIU := *sb.imageFsInfo.Inodes - *sb.imageFsInfo.InodesFree
		imageFsInodesUsed = &imageFsIU
	}

	rootStats := sb.containerInfoV2ToStats("", &rootInfo)
	nodeStats := stats.NodeStats{
		NodeName: sb.node.Name,
		CPU:      rootStats.CPU,
		Memory:   rootStats.Memory,
		Network:  sb.containerInfoV2ToNetworkStats("node:"+sb.node.Name, &rootInfo),
		Fs: &stats.FsStats{
			AvailableBytes: &sb.rootFsInfo.Available,
			CapacityBytes:  &sb.rootFsInfo.Capacity,
			UsedBytes:      &sb.rootFsInfo.Usage,
			InodesFree:     sb.rootFsInfo.InodesFree,
			Inodes:         sb.rootFsInfo.Inodes,
			InodesUsed:     nodeFsInodesUsed,
		},
		StartTime: rootStats.StartTime,
		Runtime: &stats.RuntimeStats{
			ImageFs: &stats.FsStats{
				AvailableBytes: &sb.imageFsInfo.Available,
				CapacityBytes:  &sb.imageFsInfo.Capacity,
				UsedBytes:      &sb.imageStats.TotalStorageBytes,
				InodesFree:     sb.imageFsInfo.InodesFree,
				Inodes:         sb.imageFsInfo.Inodes,
				InodesUsed:     imageFsInodesUsed,
			},
		},
	}

	systemContainers := map[string]string{
		stats.SystemContainerKubelet: sb.nodeConfig.KubeletCgroupsName,
		stats.SystemContainerRuntime: sb.nodeConfig.RuntimeCgroupsName,
		stats.SystemContainerMisc:    sb.nodeConfig.SystemCgroupsName,
	}
	for sys, name := range systemContainers {
		if info, ok := sb.infos[name]; ok {
			sysCont := sb.containerInfoV2ToStats(sys, &info)
			// System containers don't have a filesystem associated with them.
			sysCont.Rootfs = nil
			sysCont.Logs = nil
			nodeStats.SystemContainers = append(nodeStats.SystemContainers, sysCont)
		}
	}

	summary := stats.Summary{
		Node: nodeStats,
		Pods: sb.buildSummaryPods(),
	}
	return &summary, nil
}

// containerInfoV2FsStats populates the container fs stats
func (sb *summaryBuilder) containerInfoV2FsStats(
	info *cadvisorapiv2.ContainerInfo,
	cs *stats.ContainerStats) {

	// The container logs live on the node rootfs device
	cs.Logs = &stats.FsStats{
		AvailableBytes: &sb.rootFsInfo.Available,
		CapacityBytes:  &sb.rootFsInfo.Capacity,
		InodesFree:     sb.rootFsInfo.InodesFree,
		Inodes:         sb.rootFsInfo.Inodes,
	}

	if sb.rootFsInfo.Inodes != nil && sb.rootFsInfo.InodesFree != nil {
		logsInodesUsed := *sb.rootFsInfo.Inodes - *sb.rootFsInfo.InodesFree
		cs.Logs.InodesUsed = &logsInodesUsed
	}

	// The container rootFs lives on the imageFs devices (which may not be the node root fs)
	cs.Rootfs = &stats.FsStats{
		AvailableBytes: &sb.imageFsInfo.Available,
		CapacityBytes:  &sb.imageFsInfo.Capacity,
		InodesFree:     sb.imageFsInfo.InodesFree,
		Inodes:         sb.imageFsInfo.Inodes,
	}
	lcs, found := sb.latestContainerStats(info)
	if !found {
		return
	}
	cfs := lcs.Filesystem

	if cfs != nil {
		if cfs.BaseUsageBytes != nil {
			rootfsUsage := *cfs.BaseUsageBytes
			cs.Rootfs.UsedBytes = &rootfsUsage
			if cfs.TotalUsageBytes != nil {
				logsUsage := *cfs.TotalUsageBytes - *cfs.BaseUsageBytes
				cs.Logs.UsedBytes = &logsUsage
			}
		}
		if cfs.InodeUsage != nil {
			rootInodes := *cfs.InodeUsage
			cs.Rootfs.InodesUsed = &rootInodes
		}
	}
}

// latestContainerStats returns the latest container stats from cadvisor, or nil if none exist
func (sb *summaryBuilder) latestContainerStats(info *cadvisorapiv2.ContainerInfo) (*cadvisorapiv2.ContainerStats, bool) {
	stats := info.Stats
	if len(stats) < 1 {
		return nil, false
	}
	latest := stats[len(stats)-1]
	if latest == nil {
		return nil, false
	}
	return latest, true
}

// buildSummaryPods aggregates and returns the container stats in cinfos by the Pod managing the container.
// Containers not managed by a Pod are omitted.
func (sb *summaryBuilder) buildSummaryPods() []stats.PodStats {
	// Map each container to a pod and update the PodStats with container data
	podToStats := map[stats.PodReference]*stats.PodStats{}
	for key, cinfo := range sb.infos {
		// on systemd using devicemapper each mount into the container has an associated cgroup.
		// we ignore them to ensure we do not get duplicate entries in our summary.
		// for details on .mount units: http://man7.org/linux/man-pages/man5/systemd.mount.5.html
		if strings.HasSuffix(key, ".mount") {
			continue
		}
		// Build the Pod key if this container is managed by a Pod
		if !sb.isPodManagedContainer(&cinfo) {
			continue
		}
		ref := sb.buildPodRef(&cinfo)

		// Lookup the PodStats for the pod using the PodRef.  If none exists, initialize a new entry.
		podStats, found := podToStats[ref]
		if !found {
			podStats = &stats.PodStats{PodRef: ref}
			podToStats[ref] = podStats
		}

		// Update the PodStats entry with the stats from the container by adding it to stats.Containers
		containerName := types.GetContainerName(cinfo.Spec.Labels)
		if containerName == leaky.PodInfraContainerName {
			// Special case for infrastructure container which is hidden from the user and has network stats
			podStats.Network = sb.containerInfoV2ToNetworkStats("pod:"+ref.Namespace+"_"+ref.Name, &cinfo)
			podStats.StartTime = unversioned.NewTime(cinfo.Spec.CreationTime)
		} else {
			podStats.Containers = append(podStats.Containers, sb.containerInfoV2ToStats(containerName, &cinfo))
		}
	}

	// Add each PodStats to the result
	result := make([]stats.PodStats, 0, len(podToStats))
	for _, podStats := range podToStats {
		// Lookup the volume stats for each pod
		podUID := kubetypes.UID(podStats.PodRef.UID)
		if vstats, found := sb.fsResourceAnalyzer.GetPodVolumeStats(podUID); found {
			podStats.VolumeStats = vstats.Volumes
		}
		result = append(result, *podStats)
	}
	return result
}

// buildPodRef returns a PodReference that identifies the Pod managing cinfo
func (sb *summaryBuilder) buildPodRef(cinfo *cadvisorapiv2.ContainerInfo) stats.PodReference {
	podName := types.GetPodName(cinfo.Spec.Labels)
	podNamespace := types.GetPodNamespace(cinfo.Spec.Labels)
	podUID := types.GetPodUID(cinfo.Spec.Labels)
	return stats.PodReference{Name: podName, Namespace: podNamespace, UID: podUID}
}

// isPodManagedContainer returns true if the cinfo container is managed by a Pod
func (sb *summaryBuilder) isPodManagedContainer(cinfo *cadvisorapiv2.ContainerInfo) bool {
	podName := types.GetPodName(cinfo.Spec.Labels)
	podNamespace := types.GetPodNamespace(cinfo.Spec.Labels)
	managed := podName != "" && podNamespace != ""
	if !managed && podName != podNamespace {
		glog.Warningf(
			"Expect container to have either both podName (%s) and podNamespace (%s) labels, or neither.",
			podName, podNamespace)
	}
	return managed
}

func (sb *summaryBuilder) containerInfoV2ToStats(
	name string,
	info *cadvisorapiv2.ContainerInfo) stats.ContainerStats {
	cStats := stats.ContainerStats{
		StartTime: unversioned.NewTime(info.Spec.CreationTime),
		Name:      name,
	}
	cstat, found := sb.latestContainerStats(info)
	if !found {
		return cStats
	}
	if info.Spec.HasCpu {
		cpuStats := stats.CPUStats{
			Time: unversioned.NewTime(cstat.Timestamp),
		}
		if cstat.CpuInst != nil {
			cpuStats.UsageNanoCores = &cstat.CpuInst.Usage.Total
		}
		if cstat.Cpu != nil {
			cpuStats.UsageCoreNanoSeconds = &cstat.Cpu.Usage.Total
		}
		cStats.CPU = &cpuStats
	}
	if info.Spec.HasMemory {
		pageFaults := cstat.Memory.ContainerData.Pgfault
		majorPageFaults := cstat.Memory.ContainerData.Pgmajfault
		cStats.Memory = &stats.MemoryStats{
			Time:            unversioned.NewTime(cstat.Timestamp),
			UsageBytes:      &cstat.Memory.Usage,
			WorkingSetBytes: &cstat.Memory.WorkingSet,
			RSSBytes:        &cstat.Memory.RSS,
			PageFaults:      &pageFaults,
			MajorPageFaults: &majorPageFaults,
		}
		// availableBytes = memory  limit (if known) - workingset
		if !isMemoryUnlimited(info.Spec.Memory.Limit) {
			availableBytes := info.Spec.Memory.Limit - cstat.Memory.WorkingSet
			cStats.Memory.AvailableBytes = &availableBytes
		}
	}

	sb.containerInfoV2FsStats(info, &cStats)
	cStats.UserDefinedMetrics = sb.containerInfoV2ToUserDefinedMetrics(info)
	return cStats
}

// Size after which we consider memory to be "unlimited". This is not
// MaxInt64 due to rounding by the kernel.
// TODO: cadvisor should export this https://github.com/google/cadvisor/blob/master/metrics/prometheus.go#L596
const maxMemorySize = uint64(1 << 62)

func isMemoryUnlimited(v uint64) bool {
	return v > maxMemorySize
}

func (sb *summaryBuilder) containerInfoV2ToNetworkStats(name string, info *cadvisorapiv2.ContainerInfo) *stats.NetworkStats {
	if !info.Spec.HasNetwork {
		return nil
	}
	cstat, found := sb.latestContainerStats(info)
	if !found {
		return nil
	}
	for _, inter := range cstat.Network.Interfaces {
		if inter.Name == network.DefaultInterfaceName {
			return &stats.NetworkStats{
				Time:     unversioned.NewTime(cstat.Timestamp),
				RxBytes:  &inter.RxBytes,
				RxErrors: &inter.RxErrors,
				TxBytes:  &inter.TxBytes,
				TxErrors: &inter.TxErrors,
			}
		}
	}
	glog.V(4).Infof("Missing default interface %q for %s", network.DefaultInterfaceName, name)
	return nil
}

func (sb *summaryBuilder) containerInfoV2ToUserDefinedMetrics(info *cadvisorapiv2.ContainerInfo) []stats.UserDefinedMetric {
	type specVal struct {
		ref     stats.UserDefinedMetricDescriptor
		valType cadvisorapiv1.DataType
		time    time.Time
		value   float64
	}
	udmMap := map[string]*specVal{}
	for _, spec := range info.Spec.CustomMetrics {
		udmMap[spec.Name] = &specVal{
			ref: stats.UserDefinedMetricDescriptor{
				Name:  spec.Name,
				Type:  stats.UserDefinedMetricType(spec.Type),
				Units: spec.Units,
			},
			valType: spec.Format,
		}
	}
	for _, stat := range info.Stats {
		for name, values := range stat.CustomMetrics {
			specVal, ok := udmMap[name]
			if !ok {
				glog.Warningf("spec for custom metric %q is missing from cAdvisor output. Spec: %+v, Metrics: %+v", name, info.Spec, stat.CustomMetrics)
				continue
			}
			for _, value := range values {
				// Pick the most recent value
				if value.Timestamp.Before(specVal.time) {
					continue
				}
				specVal.time = value.Timestamp
				specVal.value = value.FloatValue
				if specVal.valType == cadvisorapiv1.IntType {
					specVal.value = float64(value.IntValue)
				}
			}
		}
	}
	var udm []stats.UserDefinedMetric
	for _, specVal := range udmMap {
		udm = append(udm, stats.UserDefinedMetric{
			UserDefinedMetricDescriptor: specVal.ref,
			Time:  unversioned.NewTime(specVal.time),
			Value: specVal.value,
		})
	}
	return udm
}
