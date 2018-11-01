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

package metrics

import (
	"sync"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/apimachinery/pkg/labels"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/util"
	"k8s.io/kubernetes/pkg/volume"
)

const pluginNameNotAvailable = "N/A"

var (
	inUseVolumeMetricDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", "storage_count", "attachable_volumes_in_use"),
		"Measure number of volumes in use",
		[]string{"node", "volume_plugin"}, nil)

	totalVolumesMetricDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", "attachdetach_controller", "total_volumes"),
		"Number of volumes in A/D Controller",
		[]string{"plugin_name", "state"}, nil)

	forcedDetachMetricCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Name: "attachdetach_controller_forced_detaches",
			Help: "Number of times the A/D Controller performed a forced detach"})
)
var registerMetrics sync.Once

// Register registers metrics in A/D Controller.
func Register(pvcLister corelisters.PersistentVolumeClaimLister,
	pvLister corelisters.PersistentVolumeLister,
	podLister corelisters.PodLister,
	asw cache.ActualStateOfWorld,
	dsw cache.DesiredStateOfWorld,
	pluginMgr *volume.VolumePluginMgr) {
	registerMetrics.Do(func() {
		prometheus.MustRegister(newAttachDetachStateCollector(pvcLister,
			podLister,
			pvLister,
			asw,
			dsw,
			pluginMgr))
		prometheus.MustRegister(forcedDetachMetricCounter)
	})
}

type attachDetachStateCollector struct {
	pvcLister       corelisters.PersistentVolumeClaimLister
	podLister       corelisters.PodLister
	pvLister        corelisters.PersistentVolumeLister
	asw             cache.ActualStateOfWorld
	dsw             cache.DesiredStateOfWorld
	volumePluginMgr *volume.VolumePluginMgr
}

// volumeCount is a map of maps used as a counter, e.g.:
//     node 172.168.1.100.ec2.internal has 10 EBS and 3 glusterfs PVC in use:
//     {"172.168.1.100.ec2.internal": {"aws-ebs": 10, "glusterfs": 3}}
//     state actual_state_of_world contains a total of 10 EBS volumes:
//     {"actual_state_of_world": {"aws-ebs": 10}}
type volumeCount map[string]map[string]int64

func (v volumeCount) add(typeKey, counterKey string) {
	count, ok := v[typeKey]
	if !ok {
		count = map[string]int64{}
	}
	count[counterKey]++
	v[typeKey] = count
}

func newAttachDetachStateCollector(
	pvcLister corelisters.PersistentVolumeClaimLister,
	podLister corelisters.PodLister,
	pvLister corelisters.PersistentVolumeLister,
	asw cache.ActualStateOfWorld,
	dsw cache.DesiredStateOfWorld,
	pluginMgr *volume.VolumePluginMgr) *attachDetachStateCollector {
	return &attachDetachStateCollector{pvcLister, podLister, pvLister, asw, dsw, pluginMgr}
}

// Check if our collector implements necessary collector interface
var _ prometheus.Collector = &attachDetachStateCollector{}

func (collector *attachDetachStateCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- inUseVolumeMetricDesc
	ch <- totalVolumesMetricDesc
}

func (collector *attachDetachStateCollector) Collect(ch chan<- prometheus.Metric) {
	nodeVolumeMap := collector.getVolumeInUseCount()
	for nodeName, pluginCount := range nodeVolumeMap {
		for pluginName, count := range pluginCount {
			metric, err := prometheus.NewConstMetric(inUseVolumeMetricDesc,
				prometheus.GaugeValue,
				float64(count),
				string(nodeName),
				pluginName)
			if err != nil {
				glog.Warningf("Failed to create metric : %v", err)
			}
			ch <- metric
		}
	}

	stateVolumeMap := collector.getTotalVolumesCount()
	for stateName, pluginCount := range stateVolumeMap {
		for pluginName, count := range pluginCount {
			metric, err := prometheus.NewConstMetric(totalVolumesMetricDesc,
				prometheus.GaugeValue,
				float64(count),
				pluginName,
				string(stateName))
			if err != nil {
				glog.Warningf("Failed to create metric : %v", err)
			}
			ch <- metric
		}
	}
}

func (collector *attachDetachStateCollector) getVolumeInUseCount() volumeCount {
	pods, err := collector.podLister.List(labels.Everything())
	if err != nil {
		glog.Errorf("Error getting pod list")
		return nil
	}

	nodeVolumeMap := make(volumeCount)
	for _, pod := range pods {
		if len(pod.Spec.Volumes) <= 0 {
			continue
		}

		if pod.Spec.NodeName == "" {
			continue
		}
		for _, podVolume := range pod.Spec.Volumes {
			volumeSpec, err := util.CreateVolumeSpec(podVolume, pod.Namespace, collector.pvcLister, collector.pvLister)
			if err != nil {
				continue
			}
			volumePlugin, err := collector.volumePluginMgr.FindPluginBySpec(volumeSpec)
			if err != nil {
				continue
			}
			nodeVolumeMap.add(pod.Spec.NodeName, volumePlugin.GetPluginName())
		}
	}
	return nodeVolumeMap
}

func (collector *attachDetachStateCollector) getTotalVolumesCount() volumeCount {
	stateVolumeMap := make(volumeCount)
	for _, v := range collector.dsw.GetVolumesToAttach() {
		if plugin, err := collector.volumePluginMgr.FindPluginBySpec(v.VolumeSpec); err == nil {
			pluginName := pluginNameNotAvailable
			if plugin != nil {
				pluginName = plugin.GetPluginName()
			}
			stateVolumeMap.add("desired_state_of_world", pluginName)
		}
	}
	for _, v := range collector.asw.GetAttachedVolumes() {
		if plugin, err := collector.volumePluginMgr.FindPluginBySpec(v.VolumeSpec); err == nil {
			pluginName := pluginNameNotAvailable
			if plugin != nil {
				pluginName = plugin.GetPluginName()
			}
			stateVolumeMap.add("actual_state_of_world", pluginName)
		}
	}
	return stateVolumeMap
}

// RecordForcedDetachMetric register a forced detach metric.
func RecordForcedDetachMetric() {
	forcedDetachMetricCounter.Inc()
}
