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
	"k8s.io/apimachinery/pkg/types"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller/volume/attachdetach/util"
	"k8s.io/kubernetes/pkg/volume"
)

var (
	inUseVolumeMetricDesc = prometheus.NewDesc(
		prometheus.BuildFQName("", "storage_count", "attachable_volumes_in_use"),
		"Measure number of volumes in use",
		[]string{"node", "volume_plugin"}, nil)
)
var registerMetrics sync.Once

type volumeInUseCollector struct {
	pvcLister       corelisters.PersistentVolumeClaimLister
	podLister       corelisters.PodLister
	pvLister        corelisters.PersistentVolumeLister
	volumePluginMgr *volume.VolumePluginMgr
}

// nodeVolumeCount contains map of {"nodeName": {"pluginName": volume_count }}
// For example :
//     node 172.168.1.100.ec2.internal has 10 EBS and 3 glusterfs PVC in use
//     {"172.168.1.100.ec2.internal": {"aws-ebs": 10, "glusterfs": 3}}
type nodeVolumeCount map[types.NodeName]map[string]int

// Register registers pvc's in-use metrics
func Register(pvcLister corelisters.PersistentVolumeClaimLister,
	pvLister corelisters.PersistentVolumeLister,
	podLister corelisters.PodLister,
	pluginMgr *volume.VolumePluginMgr) {
	registerMetrics.Do(func() {
		prometheus.MustRegister(newVolumeInUseCollector(pvcLister, podLister, pvLister, pluginMgr))
	})

}

func (volumeInUse nodeVolumeCount) add(nodeName types.NodeName, pluginName string) {
	nodeCount, ok := volumeInUse[nodeName]
	if !ok {
		nodeCount = map[string]int{}
	}
	nodeCount[pluginName]++
	volumeInUse[nodeName] = nodeCount
}

func newVolumeInUseCollector(
	pvcLister corelisters.PersistentVolumeClaimLister,
	podLister corelisters.PodLister,
	pvLister corelisters.PersistentVolumeLister,
	pluginMgr *volume.VolumePluginMgr) *volumeInUseCollector {
	return &volumeInUseCollector{pvcLister, podLister, pvLister, pluginMgr}
}

// Check if our collector implements necessary collector interface
var _ prometheus.Collector = &volumeInUseCollector{}

func (collector *volumeInUseCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- inUseVolumeMetricDesc
}

func (collector *volumeInUseCollector) Collect(ch chan<- prometheus.Metric) {
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
