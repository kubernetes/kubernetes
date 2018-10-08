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
	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/api/core/v1"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	serverstats "k8s.io/kubernetes/pkg/kubelet/server/stats"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/volume"
)

var (
	emptyDirVolumeUsedDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", metrics.KubeletSubsystem, metrics.EmptyDirVolumeUsedKey),
		"Number of used bytes in the EmptyDir volume",
		[]string{"pod_uid", "volume_name"}, nil,
	)
	emptyDirVolumeCapacityDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", metrics.KubeletSubsystem, metrics.EmptyDirVolumeCapacityKey),
		"Capacity of the EmptyDir volume",
		[]string{"pod_uid", "volume_name"}, nil,
	)
)

type emptyDirCollector struct {
	statsProvider serverstats.StatsProvider
}

// NewEmptyDirStatsCollector creates a emptydir volume stats prometheus collector.
func NewEmptyDirStatsCollector(statsProvider serverstats.StatsProvider) prometheus.Collector {
	return &emptyDirCollector{statsProvider: statsProvider}
}

// Describe implements the prometheus.Collector interface.
func (collector *emptyDirCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- emptyDirVolumeUsedDesc
	ch <- emptyDirVolumeCapacityDesc
}

// Collect implements the prometheus.Collector interface.
func (collector *emptyDirCollector) Collect(ch chan<- prometheus.Metric) {
	pods := collector.statsProvider.GetPods()
	addGauge := func(desc *prometheus.Desc, podUID, volumeName string, v float64) {
		metric, err := prometheus.NewConstMetric(desc, prometheus.GaugeValue, v, []string{podUID, volumeName}...)
		if err != nil {
			klog.Warningf("Failed to generate metric: %v", err)
			return
		}
		ch <- metric
	}

	for _, pod := range pods {
		volumes, found := collector.statsProvider.ListVolumesForPod(pod.UID)
		if !found {
			return
		}

		// Get volume sources for the pod - key'd by volume name
		volumeSources := make(map[string]v1.Volume)
		for _, v := range pod.Spec.Volumes {
			volumeSources[v.Name] = v
		}

		for name, v := range volumes {
			metric, err := v.GetMetrics()
			if err != nil {
				// Expected for Volumes that don't support Metrics
				if !volume.IsNotSupported(err) {
					klog.V(4).Infof("Failed to calculate volume metrics for pod %s volume %s: %+v", format.Pod(pod), name, err)
				}
				continue
			}

			volSource, ok := volumeSources[name]
			if !ok {
				klog.Errorf("Volume %s is not the volume of Pod %s", name, pod.UID)
				continue
			} else {
				if volSource.EmptyDir != nil && volSource.EmptyDir.Medium == v1.StorageMediumDefault {
					addGauge(emptyDirVolumeUsedDesc, string(pod.UID), name, float64(metric.Used.Value()))
					if volSource.EmptyDir.SizeLimit == nil {
						addGauge(emptyDirVolumeCapacityDesc, string(pod.UID), name, float64(metric.Capacity.Value()))
					} else {
						addGauge(emptyDirVolumeCapacityDesc, string(pod.UID), name, float64(volSource.EmptyDir.SizeLimit.Value()))
					}
				} else {
					klog.V(6).Infof("volume %s is not emptydir or medium is not default", name)
				}
			}

		}

	}

}
