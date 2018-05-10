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
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	pluginNameNotAvailable = "N/A"

	// Metric keys for Volume Manager.
	volumeManagerTotalVolumes = "volume_manager_total_volumes"
)

var (
	registerMetrics sync.Once

	totalVolumesDesc = prometheus.NewDesc(
		volumeManagerTotalVolumes,
		"Number of volumes in Volume Manager",
		[]string{"plugin_name", "state"},
		nil,
	)
)

// volumeCount is a map of maps used as a counter.
type volumeCount map[string]map[string]int64

func (v volumeCount) add(state, plugin string) {
	count, ok := v[state]
	if !ok {
		count = map[string]int64{}
	}
	count[plugin]++
	v[state] = count
}

// Register registers Volume Manager metrics.
func Register(asw cache.ActualStateOfWorld, dsw cache.DesiredStateOfWorld, pluginMgr *volume.VolumePluginMgr) {
	registerMetrics.Do(func() {
		prometheus.MustRegister(&totalVolumesCollector{asw, dsw, pluginMgr})
	})
}

type totalVolumesCollector struct {
	asw       cache.ActualStateOfWorld
	dsw       cache.DesiredStateOfWorld
	pluginMgr *volume.VolumePluginMgr
}

var _ prometheus.Collector = &totalVolumesCollector{}

// Describe implements the prometheus.Collector interface.
func (c *totalVolumesCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- totalVolumesDesc
}

// Collect implements the prometheus.Collector interface.
func (c *totalVolumesCollector) Collect(ch chan<- prometheus.Metric) {
	for stateName, pluginCount := range c.getVolumeCount() {
		for pluginName, count := range pluginCount {
			metric, err := prometheus.NewConstMetric(totalVolumesDesc,
				prometheus.GaugeValue,
				float64(count),
				pluginName,
				stateName)
			if err != nil {
				glog.Warningf("Failed to create metric : %v", err)
			}
			ch <- metric
		}
	}
}

func (c *totalVolumesCollector) getVolumeCount() volumeCount {
	counter := make(volumeCount)
	for _, mountedVolume := range c.asw.GetMountedVolumes() {
		pluginName := mountedVolume.PluginName
		if pluginName == "" {
			pluginName = pluginNameNotAvailable
		}
		counter.add("actual_state_of_world", pluginName)
	}

	for _, volumeToMount := range c.dsw.GetVolumesToMount() {
		pluginName := pluginNameNotAvailable
		if plugin, err := c.pluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec); err == nil {
			pluginName = plugin.GetPluginName()
		}
		counter.add("desired_state_of_world", pluginName)
	}
	return counter
}
