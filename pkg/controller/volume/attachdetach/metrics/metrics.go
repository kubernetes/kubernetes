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
	"time"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/cache"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/util"
	"k8s.io/kubernetes/pkg/volume"
)

var registerMetrics sync.Once

// ADCMetricsRecorder is an interface used to record metrics in A/D Controller.
type ADCMetricsRecorder interface {
	Run(stopCh <-chan struct{})
	RecordForcedDetach()
}

type adcMetrics struct {
	volumesInUse *volumeInUseCollector
	totalVolumes *totalVolumes
}

func (m *adcMetrics) register() {
	registerMetrics.Do(func() {
		prometheus.MustRegister(m.volumesInUse)
		prometheus.MustRegister(m.totalVolumes.volumesStateMetric)
		prometheus.MustRegister(m.totalVolumes.forcedDetachMetric)
	})
}

func (m *adcMetrics) Run(stopCh <-chan struct{}) {
	m.register()
	m.totalVolumes.recordVolumesStates(stopCh)
}

func (m *adcMetrics) RecordForcedDetach() {
	m.totalVolumes.forcedDetachMetric.Inc()
}

// NewADCMetricsRecorder returns an implementation the ADCMetricsRecorder interface.
func NewADCMetricsRecorder(loopSleepDuration time.Duration,
	pluginMgr *volume.VolumePluginMgr,
	dsw cache.DesiredStateOfWorld,
	asw cache.ActualStateOfWorld,
	pvcLister corelisters.PersistentVolumeClaimLister,
	podLister corelisters.PodLister,
	pvLister corelisters.PersistentVolumeLister) ADCMetricsRecorder {

	inUse := &volumeInUseCollector{
		pvcLister:       pvcLister,
		podLister:       podLister,
		pvLister:        pvLister,
		volumePluginMgr: pluginMgr,
		inUseVolumeMetricDesc: prometheus.NewDesc(
			prometheus.BuildFQName("", "storage_count", "attachable_volumes_in_use"),
			"Measure number of volumes in use",
			[]string{"node", "volume_plugin"},
			nil,
		),
	}

	total := &totalVolumes{
		loopSleepDuration: loopSleepDuration,
		pluginMgr:         pluginMgr,
		dsw:               dsw,
		asw:               asw,
		volumesStateMetric: prometheus.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "attachdetach_controller_total_volumes",
				Help: "Number of volumes in A/D Controller",
			},
			[]string{"plugin_name", "state"},
		),
		forcedDetachMetric: prometheus.NewCounter(
			prometheus.CounterOpts{
				Name: "attachdetach_controller_total_forced_detaches",
				Help: "Number of times the A/D Controller performed a forced detach",
			},
		),
	}

	return &adcMetrics{inUse, total}
}

type volumeInUseCollector struct {
	pvcLister       corelisters.PersistentVolumeClaimLister
	podLister       corelisters.PodLister
	pvLister        corelisters.PersistentVolumeLister
	volumePluginMgr *volume.VolumePluginMgr

	inUseVolumeMetricDesc *prometheus.Desc
}

// nodeVolumeCount contains map of {"nodeName": {"pluginName": volume_count }}
// For example :
//     node 172.168.1.100.ec2.internal has 10 EBS and 3 glusterfs PVC in use
//     {"172.168.1.100.ec2.internal": {"aws-ebs": 10, "glusterfs": 3}}
type nodeVolumeCount map[types.NodeName]map[string]int

func (volumeInUse nodeVolumeCount) add(nodeName types.NodeName, pluginName string) {
	nodeCount, ok := volumeInUse[nodeName]
	if !ok {
		nodeCount = map[string]int{}
	}
	nodeCount[pluginName]++
	volumeInUse[nodeName] = nodeCount
}

// Check if our collector implements necessary collector interface
var _ prometheus.Collector = &volumeInUseCollector{}

func (collector *volumeInUseCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- collector.inUseVolumeMetricDesc
}

func (collector *volumeInUseCollector) Collect(ch chan<- prometheus.Metric) {
	nodeVolumeMap := collector.getVolumeInUseCount()
	for nodeName, pluginCount := range nodeVolumeMap {
		for pluginName, count := range pluginCount {
			metric, err := prometheus.NewConstMetric(collector.inUseVolumeMetricDesc,
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
}

func (collector *volumeInUseCollector) getVolumeInUseCount() nodeVolumeCount {
	pods, err := collector.podLister.List(labels.Everything())
	if err != nil {
		glog.Errorf("Error getting pod list")
		return nil
	}

	nodeVolumeMap := make(nodeVolumeCount)
	for _, pod := range pods {
		if len(pod.Spec.Volumes) <= 0 {
			continue
		}

		nodeName := types.NodeName(pod.Spec.NodeName)
		if nodeName == "" {
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
			nodeVolumeMap.add(nodeName, volumePlugin.GetPluginName())
		}
	}
	return nodeVolumeMap
}

type totalVolumes struct {
	loopSleepDuration time.Duration
	pluginMgr         *volume.VolumePluginMgr
	dsw               cache.DesiredStateOfWorld
	asw               cache.ActualStateOfWorld

	forcedDetachMetric prometheus.Counter
	volumesStateMetric *prometheus.GaugeVec
}

func (m *totalVolumes) recordVolumesStates(stopCh <-chan struct{}) {
	metricsLoop := func() {
		m.volumesStateMetric.Reset()
		for _, v := range m.dsw.GetVolumesToAttach() {
			if plugin, err := m.pluginMgr.FindPluginBySpec(v.VolumeSpec); err == nil {
				m.recordVolumeState(plugin.GetPluginName(), "desired_state_of_world")
			}
		}
		for _, v := range m.asw.GetAttachedVolumes() {
			if plugin, err := m.pluginMgr.FindPluginBySpec(v.VolumeSpec); err == nil {
				m.recordVolumeState(plugin.GetPluginName(), "actual_state_of_world")
			}
		}
	}
	wait.Until(metricsLoop, m.loopSleepDuration, stopCh)
}

func (m *totalVolumes) recordVolumeState(pluginName, state string) {
	if pluginName == "" {
		pluginName = "N/A"
	}
	m.volumesStateMetric.WithLabelValues(pluginName, state).Inc()
}
