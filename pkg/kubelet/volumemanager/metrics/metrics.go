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

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	pluginNameNotAvailable = "N/A"

	// Metric keys for Volume Manager.
	volumeManagerTotalVolumes                     = "volume_manager_total_volumes"
	reconstructVolumeOperationsTotal              = "reconstruct_volume_operations_total"
	reconstructVolumeOperationsErrorsTotal        = "reconstruct_volume_operations_errors_total"
	forceCleanedFailedVolumeOperationsTotal       = "force_cleaned_failed_volume_operations_total"
	forceCleanedFailedVolumeOperationsErrorsTotal = "force_cleaned_failed_volume_operation_errors_total"
)

var (
	registerMetrics sync.Once

	totalVolumesDesc = metrics.NewDesc(
		volumeManagerTotalVolumes,
		"Number of volumes in Volume Manager",
		[]string{"plugin_name", "state"},
		nil,
		metrics.ALPHA, "",
	)

	ReconstructVolumeOperationsTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           reconstructVolumeOperationsTotal,
			Help:           "The number of volumes that were attempted to be reconstructed from the operating system during kubelet startup. This includes both successful and failed reconstruction.",
			StabilityLevel: metrics.ALPHA,
		},
	)
	ReconstructVolumeOperationsErrorsTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           reconstructVolumeOperationsErrorsTotal,
			Help:           "The number of volumes that failed reconstruction from the operating system during kubelet startup.",
			StabilityLevel: metrics.ALPHA,
		},
	)

	ForceCleanedFailedVolumeOperationsTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           forceCleanedFailedVolumeOperationsTotal,
			Help:           "The number of volumes that were force cleaned after their reconstruction failed during kubelet startup. This includes both successful and failed cleanups.",
			StabilityLevel: metrics.ALPHA,
		},
	)
	ForceCleanedFailedVolumeOperationsErrorsTotal = metrics.NewCounter(
		&metrics.CounterOpts{
			Name:           forceCleanedFailedVolumeOperationsErrorsTotal,
			Help:           "The number of volumes that failed force cleanup after their reconstruction failed during kubelet startup.",
			StabilityLevel: metrics.ALPHA,
		},
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
		legacyregistry.CustomMustRegister(&totalVolumesCollector{asw: asw, dsw: dsw, pluginMgr: pluginMgr})
		legacyregistry.MustRegister(ReconstructVolumeOperationsTotal)
		legacyregistry.MustRegister(ReconstructVolumeOperationsErrorsTotal)
		legacyregistry.MustRegister(ForceCleanedFailedVolumeOperationsTotal)
		legacyregistry.MustRegister(ForceCleanedFailedVolumeOperationsErrorsTotal)
	})
}

type totalVolumesCollector struct {
	metrics.BaseStableCollector

	asw       cache.ActualStateOfWorld
	dsw       cache.DesiredStateOfWorld
	pluginMgr *volume.VolumePluginMgr
}

var _ metrics.StableCollector = &totalVolumesCollector{}

// DescribeWithStability implements the metrics.StableCollector interface.
func (c *totalVolumesCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- totalVolumesDesc
}

// CollectWithStability implements the metrics.StableCollector interface.
func (c *totalVolumesCollector) CollectWithStability(ch chan<- metrics.Metric) {
	for stateName, pluginCount := range c.getVolumeCount() {
		for pluginName, count := range pluginCount {
			ch <- metrics.NewLazyConstMetric(totalVolumesDesc,
				metrics.GaugeValue,
				float64(count),
				pluginName,
				stateName)
		}
	}
}

func (c *totalVolumesCollector) getVolumeCount() volumeCount {
	counter := make(volumeCount)
	for _, mountedVolume := range c.asw.GetMountedVolumes() {
		pluginName := volumeutil.GetFullQualifiedPluginNameForVolume(mountedVolume.PluginName, mountedVolume.VolumeSpec)
		if pluginName == "" {
			pluginName = pluginNameNotAvailable
		}
		counter.add("actual_state_of_world", pluginName)
	}

	for _, volumeToMount := range c.dsw.GetVolumesToMount() {
		pluginName := pluginNameNotAvailable
		if plugin, err := c.pluginMgr.FindPluginBySpec(volumeToMount.VolumeSpec); err == nil {
			pluginName = volumeutil.GetFullQualifiedPluginNameForVolume(plugin.GetPluginName(), volumeToMount.VolumeSpec)
		}
		counter.add("desired_state_of_world", pluginName)
	}
	return counter
}
