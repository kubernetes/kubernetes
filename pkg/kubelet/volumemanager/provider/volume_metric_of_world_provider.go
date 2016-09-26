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

// Package provider implements interfaces that provide the metrics of volumes
// which exist in the actual state of the world.
package provider

import (
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/kubelet/volumemanager/cache"
)

const (
	longDu             = 1 * time.Second
	longCacheDuration  = 24 * time.Hour
	maxDuBackoffFactor = 20
	// The maximum number of routine that is running `du` tasks at once.
	maxConsecutiveRoutinesRunningDu = 20
)

// A pool for restricting the number of consecutive routines running `du` tasks.
var routinePoolForRunningDu = make(chan struct{}, maxConsecutiveRoutinesRunningDu)

func claimRoutineTokenForRunningDu() {
	<-routinePoolForRunningDu
}

func releaseRoutineTokenForRunningDu() {
	routinePoolForRunningDu <- struct{}{}
}

func initRoutineToken() {
	for i := 0; i < maxConsecutiveRoutinesRunningDu; i++ {
		routinePoolForRunningDu <- struct{}{}
	}
}

// VolumesMetricsOfWorldProvider periodically loops through the map containing
// metrics providers which generated from actual world cache and update each
// volume using provider on the path to which the volume should be mounted for
// a pod.
type VolumesMetricsOfWorldProvider interface {
	// Starts running the provider loop which executes periodically,
	// checks if volume metrics that should be updated will be measured
	// using provider.
	Run(stopCh <-chan struct{})
}

// NewVolumesMetricsOfWorldProvider returns a new instance of
// NewVolumesMetricsOfWorldProvider.
//
// metricsCacheDuration - the amount of time the metrics data is effective
// 	in the cache
// loopSleepDuration - the amount of time the provider loop sleeps between
//     successive executions
// actualStateOfWorld - the cache of actual world
// volumesMetricsOfWorld - the cache to keep volume metrics data
func NewVolumesMetricsOfWorldProvider(
	metricsCacheDuration time.Duration,
	loopSleepDuration time.Duration,
	housekeepingDuration time.Duration,
	actualStateOfWorld cache.ActualStateOfWorld,
	volumesMetricsOfWorld cache.VolumesMetricsOfWorld) VolumesMetricsOfWorldProvider {
	return &volumesMetricsOfWorldProvider{
		metricsCacheDuration:  metricsCacheDuration,
		loopSleepDuration:     loopSleepDuration,
		housekeepingDuration:  housekeepingDuration,
		actualStateOfWorld:    actualStateOfWorld,
		volumesMetricsOfWorld: volumesMetricsOfWorld,
	}
}

type volumesMetricsOfWorldProvider struct {
	metricsCacheDuration  time.Duration
	loopSleepDuration     time.Duration
	housekeepingDuration  time.Duration
	actualStateOfWorld    cache.ActualStateOfWorld
	volumesMetricsOfWorld cache.VolumesMetricsOfWorld
}

// provideVolumeMetrics use the proper provider for the various volumes.
func (vmwp *volumesMetricsOfWorldProvider) provideVolumeMetrics(providerWrapper cache.MetricsProviderWrapper, cacheDuration time.Duration) {
	// TODO: Use Constant as case values.
	switch providerWrapper.PluginName {
	case "kubernetes.io/nfs", "kubernetes.io/empty-dir":
		// It should mark the volume being measured before starting a routine.
		// It would attempt to avoid running 'du' on paths simultaneously to which the same volume is mounted.
		// Ignore side effect of 'HousekeepVolumeMetricsWorld' that remove the metrics data while 'du' is running.
		vmwp.volumesMetricsOfWorld.MarkVolumeIsMeasuring(providerWrapper.VolumeName)
		go vmwp.provideVolumeMetricsByDu(providerWrapper, cacheDuration)

	case "kubernetes.io/aws-ebs", "kubernetes.io/azure-file", "kubernetes.io/gce-pd":
		vmwp.provideVolumeMetricsByStatFS(providerWrapper, cacheDuration)

	case "kubernetes.io/secret":
		// Excute the next updating after 'longCacheDuration' hours.
		vmwp.provideVolumeMetricsByDu(providerWrapper, longCacheDuration)
	// TODO:Provide other volume plugins.
	default:
	}
}

// provideVolumeMetricsByDu get metrics by 'StatFS' and update volume
// metrics in cahce.
func (vmwp *volumesMetricsOfWorldProvider) provideVolumeMetricsByStatFS(providerWrapper cache.MetricsProviderWrapper, cacheDuration time.Duration) {
	metrics, err := providerWrapper.Provider.GetMetrics()
	if err != nil {
		glog.Errorf("Failed to get FsInfo of volume with name %s due to error %v.", providerWrapper.VolumeName, err)
		return
	}
	vmwp.volumesMetricsOfWorld.UpdateVolumeMetricsData(providerWrapper.VolumeName,
		metrics,
		providerWrapper.OuterVolumeSpecName,
		vmwp.metricsCacheDuration,
		time.Now())
}

// provideVolumeMetricsByDu get metrics by 'Du' and update volume
// metrics in cahce.
func (vmwp *volumesMetricsOfWorldProvider) provideVolumeMetricsByDu(providerWrapper cache.MetricsProviderWrapper, cacheDuration time.Duration) {
	// startMeasureVolume Call will block here until the number of 'du' task
	// running at once is less than maxConsecutiveRoutinesRunningDu.
	claimRoutineTokenForRunningDu()
	defer releaseRoutineTokenForRunningDu()

	start := time.Now()
	// GetMetrics Call maybe block here for some time.
	metrics, err := providerWrapper.Provider.GetMetrics()
	// If fail to get the volume metrics, double the cache time.
	// The cache time of volume metrics determine how frequently to provide the metrics.
	// The cache time is not greater than maxDuBackoffFactor*vmwp.metricsCacheDuration.
	// Just as cadvisor does.
	if err != nil {
		glog.Errorf("failed to excute 'du' on volume with name %s - %v", providerWrapper.VolumeName, err)
		cacheDuration = cacheDuration * 2
		if cacheDuration > maxDuBackoffFactor*vmwp.metricsCacheDuration {
			cacheDuration = maxDuBackoffFactor * vmwp.metricsCacheDuration
		}

		vmwp.volumesMetricsOfWorld.UpdateVolumeMetricsStatus(providerWrapper.VolumeName,
			cacheDuration,
			time.Now())

		return
	}

	duration := time.Since(start)
	if duration > longDu {
		glog.V(2).Infof("`du` on volume with name %s took %v", providerWrapper.VolumeName, duration)
	}

	vmwp.volumesMetricsOfWorld.UpdateVolumeMetricsData(providerWrapper.VolumeName,
		metrics,
		providerWrapper.OuterVolumeSpecName,
		vmwp.metricsCacheDuration,
		time.Now())
}

func (vmwp *volumesMetricsOfWorldProvider) Run(stopCh <-chan struct{}) {
	initRoutineToken()
	for {
		select {
		case <-stopCh:
			return

		case <-time.After(vmwp.housekeepingDuration):
			// Clear the dirty volume metrics.
			volumeNamesInActualWorld := vmwp.actualStateOfWorld.GetMountedVolumeNames()
			vmwp.volumesMetricsOfWorld.HousekeepVolumeMetricsWorld(volumeNamesInActualWorld)

		case <-time.After(vmwp.loopSleepDuration):
			metricsProviderWrappers := vmwp.actualStateOfWorld.GetMountedMetricsProviderWrappers()
			for path := range metricsProviderWrappers {
				metricsProviderWrapper := metricsProviderWrappers[path]
				// A volume mabye mounted to multiple paths.
				// Running 'du' on only one of the paths.
				measureRequired, cacheDuration := vmwp.volumesMetricsOfWorld.GetVolumeMetricsStatus(metricsProviderWrapper.VolumeName)
				if measureRequired {
					// Set cache duration of new volume metrics as
					// the default value 'vmwp.metricsCacheDuration'.
					if cacheDuration == nil {
						cacheDuration = &vmwp.metricsCacheDuration
					}
					vmwp.provideVolumeMetrics(metricsProviderWrapper, *cacheDuration)
				}
			}
		}
	}
}
