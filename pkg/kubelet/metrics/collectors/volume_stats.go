/*
Copyright 2018 The Kubernetes Authors.

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

package collectors

import (
	"context"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-base/metrics"
	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	serverstats "k8s.io/kubernetes/pkg/kubelet/server/stats"
)

var (
	volumeStatsCapacityBytesDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsCapacityBytesKey),
		"Capacity in bytes of the volume",
		[]string{"namespace", "persistentvolumeclaim"}, nil,
		metrics.ALPHA, "",
	)
	volumeStatsAvailableBytesDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsAvailableBytesKey),
		"Number of available bytes in the volume",
		[]string{"namespace", "persistentvolumeclaim"}, nil,
		metrics.ALPHA, "",
	)
	volumeStatsUsedBytesDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsUsedBytesKey),
		"Number of used bytes in the volume",
		[]string{"namespace", "persistentvolumeclaim"}, nil,
		metrics.ALPHA, "",
	)
	volumeStatsInodesDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsInodesKey),
		"Maximum number of inodes in the volume",
		[]string{"namespace", "persistentvolumeclaim"}, nil,
		metrics.ALPHA, "",
	)
	volumeStatsInodesFreeDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsInodesFreeKey),
		"Number of free inodes in the volume",
		[]string{"namespace", "persistentvolumeclaim"}, nil,
		metrics.ALPHA, "",
	)
	volumeStatsInodesUsedDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsInodesUsedKey),
		"Number of used inodes in the volume",
		[]string{"namespace", "persistentvolumeclaim"}, nil,
		metrics.ALPHA, "",
	)

	// Pod-scoped volume metrics cover volumes that aren't backed by a PVC (e.g. emptyDir,
	// secret, configMap, downwardAPI, projected), which the metrics above never report since
	// they're keyed on PVCReference. Labeled by the pod-spec volume name instead of a claim
	// name, since there is no claim to reference.
	volumeStatsPodCapacityBytesDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsPodCapacityBytesKey),
		"Capacity in bytes of a pod-scoped (non-PVC) volume",
		[]string{"namespace", "pod", "volume_name"}, nil,
		metrics.ALPHA, "",
	)
	volumeStatsPodAvailableBytesDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsPodAvailableBytesKey),
		"Number of available bytes in a pod-scoped (non-PVC) volume",
		[]string{"namespace", "pod", "volume_name"}, nil,
		metrics.ALPHA, "",
	)
	volumeStatsPodUsedBytesDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsPodUsedBytesKey),
		"Number of used bytes in a pod-scoped (non-PVC) volume",
		[]string{"namespace", "pod", "volume_name"}, nil,
		metrics.ALPHA, "",
	)
	volumeStatsPodInodesDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsPodInodesKey),
		"Maximum number of inodes in a pod-scoped (non-PVC) volume",
		[]string{"namespace", "pod", "volume_name"}, nil,
		metrics.ALPHA, "",
	)
	volumeStatsPodInodesFreeDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsPodInodesFreeKey),
		"Number of free inodes in a pod-scoped (non-PVC) volume",
		[]string{"namespace", "pod", "volume_name"}, nil,
		metrics.ALPHA, "",
	)
	volumeStatsPodInodesUsedDesc = metrics.NewDesc(
		metrics.BuildFQName("", kubeletmetrics.KubeletSubsystem, kubeletmetrics.VolumeStatsPodInodesUsedKey),
		"Number of used inodes in a pod-scoped (non-PVC) volume",
		[]string{"namespace", "pod", "volume_name"}, nil,
		metrics.ALPHA, "",
	)
)

type volumeStatsCollector struct {
	metrics.BaseStableCollector

	statsProvider serverstats.Provider
}

// Check if volumeStatsCollector implements necessary interface
var _ metrics.StableCollector = &volumeStatsCollector{}

// NewVolumeStatsCollector creates a volume stats metrics.StableCollector.
func NewVolumeStatsCollector(statsProvider serverstats.Provider) metrics.StableCollector {
	return &volumeStatsCollector{statsProvider: statsProvider}
}

// DescribeWithStability implements the metrics.StableCollector interface.
func (collector *volumeStatsCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- volumeStatsCapacityBytesDesc
	ch <- volumeStatsAvailableBytesDesc
	ch <- volumeStatsUsedBytesDesc
	ch <- volumeStatsInodesDesc
	ch <- volumeStatsInodesFreeDesc
	ch <- volumeStatsInodesUsedDesc
	ch <- volumeStatsPodCapacityBytesDesc
	ch <- volumeStatsPodAvailableBytesDesc
	ch <- volumeStatsPodUsedBytesDesc
	ch <- volumeStatsPodInodesDesc
	ch <- volumeStatsPodInodesFreeDesc
	ch <- volumeStatsPodInodesUsedDesc
}

// CollectWithStability implements the metrics.StableCollector interface.
func (collector *volumeStatsCollector) CollectWithStability(ch chan<- metrics.Metric) {
	// Use context.TODO() because we currently do not have a proper context to pass in.
	// Replace this with an appropriate context when refactoring this function to accept a context parameter.
	ctx := context.TODO()
	podStats, err := collector.statsProvider.ListPodStats(ctx)
	if err != nil {
		return
	}
	addGauge := func(desc *metrics.Desc, pvcRef *stats.PVCReference, v float64, lv ...string) {
		lv = append([]string{pvcRef.Namespace, pvcRef.Name}, lv...)
		ch <- metrics.NewLazyConstMetric(desc, metrics.GaugeValue, v, lv...)
	}
	addPodGauge := func(desc *metrics.Desc, podRef stats.PodReference, volumeName string, v float64, lv ...string) {
		lv = append([]string{podRef.Namespace, podRef.Name, volumeName}, lv...)
		ch <- metrics.NewLazyConstMetric(desc, metrics.GaugeValue, v, lv...)
	}
	allPVCs := sets.Set[stats.PVCReference]{}
	for _, podStat := range podStats {
		if podStat.VolumeStats == nil {
			continue
		}
		for _, volumeStat := range podStat.VolumeStats {
			pvcRef := volumeStat.PVCRef
			if pvcRef == nil {
				// No PVC reference: a pod-scoped volume (emptyDir, secret, configMap,
				// downwardAPI, projected, ...). Report it labeled by pod + volume name
				// instead of skipping it entirely.
				addPodGauge(volumeStatsPodCapacityBytesDesc, podStat.PodRef, volumeStat.Name, float64(*volumeStat.CapacityBytes))
				addPodGauge(volumeStatsPodAvailableBytesDesc, podStat.PodRef, volumeStat.Name, float64(*volumeStat.AvailableBytes))
				addPodGauge(volumeStatsPodUsedBytesDesc, podStat.PodRef, volumeStat.Name, float64(*volumeStat.UsedBytes))
				addPodGauge(volumeStatsPodInodesDesc, podStat.PodRef, volumeStat.Name, float64(*volumeStat.Inodes))
				addPodGauge(volumeStatsPodInodesFreeDesc, podStat.PodRef, volumeStat.Name, float64(*volumeStat.InodesFree))
				addPodGauge(volumeStatsPodInodesUsedDesc, podStat.PodRef, volumeStat.Name, float64(*volumeStat.InodesUsed))
				continue
			}
			if allPVCs.Has(*pvcRef) {
				// ignore if already collected
				continue
			}
			addGauge(volumeStatsCapacityBytesDesc, pvcRef, float64(*volumeStat.CapacityBytes))
			addGauge(volumeStatsAvailableBytesDesc, pvcRef, float64(*volumeStat.AvailableBytes))
			addGauge(volumeStatsUsedBytesDesc, pvcRef, float64(*volumeStat.UsedBytes))
			addGauge(volumeStatsInodesDesc, pvcRef, float64(*volumeStat.Inodes))
			addGauge(volumeStatsInodesFreeDesc, pvcRef, float64(*volumeStat.InodesFree))
			addGauge(volumeStatsInodesUsedDesc, pvcRef, float64(*volumeStat.InodesUsed))
			allPVCs.Insert(*pvcRef)
		}
	}
}
