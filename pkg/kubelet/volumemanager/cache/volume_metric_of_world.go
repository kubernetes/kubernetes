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

/*
Package cache implements data structures used by the kubelet volume manager to
keep track of attached volumes and the pods that mounted them.
*/
package cache

import (
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/volume"
)

// VolumesMetricsOfWorld defines a set of thread-safe operations for the kubelet
// volume manager's volumes metrics of the world cache.
// This cache contains volumes->metrics data.
type VolumesMetricsOfWorld interface {
	// MarkVolumeIsMeasuring marks the volume as being measured or not.
	// The volume may be mounted on multiple paths for multiple pods.
	// It is neccery to avoid excute 'du' on all paths that mountes the same volume.
	// When mark it as true indicates the volume is being measuring.
	// if the volume does not exist in the cache , an error is returned.
	MarkVolumeIsMeasuring(volumeName v1.UniqueVolumeName)

	// SetVolumeMetricsData sets the volume metrics data value for the given volume.
	// volumeMetrics is the metrics of the volume.
	// cacheDuration is the amount of time the metrics is available in the cache.
	// dataTimestamp is the time of last measure of the metrics.
	UpdateVolumeMetricsData(volumeName v1.UniqueVolumeName, volumeMetrics *volume.Metrics, outerVolumeSpecName string, cacheDuration time.Duration, dataTimestamp time.Time)

	// UpdateVolumeMetricsStatus update cache time and timestap
	// when fail to provide volume metrics.
	UpdateVolumeMetricsStatus(volumeName v1.UniqueVolumeName, cacheDuration time.Duration, dataTimestamp time.Time)

	// HousekeepVolumeMetricsWorld removes the volume metrics from the
	// cache when the volume has not been in the actual world.
	HousekeepVolumeMetricsWorld(volumeNamesInActualWorld []v1.UniqueVolumeName)

	// GetVolumeMetricsCacheDuration return whether volume requires measure or not
	// and volume metrics cache duration.
	// If return 'measureRequired' as false, return 'cacheDuration' as nil.
	GetVolumeMetricsStatus(volumeName v1.UniqueVolumeName) (measureRequired bool, cacheDuration *time.Duration)

	// GetVolumeMetrics return the volume metrics and volume outerVolumeSpecName.
	GetVolumeMetrics(volumeName v1.UniqueVolumeName) (*volume.Metrics, string)
}

// NewVolumeMetricsOfWorld returns a new instance of VolumesMetricsOfWorld.
func NewVolumeMetricsOfWorld() VolumesMetricsOfWorld {
	return &volumeMetricsOfWorld{
		mountedVolumesMetricsCache: make(map[v1.UniqueVolumeName]volumeMetricsData),
	}
}

type volumeMetricsData struct {
	metrics             *volume.Metrics
	outerVolumeSpecName string
	isMeasuring         bool
	cacheDuration       *time.Duration
	metricsTimestamp    *time.Time
}

type volumeMetricsOfWorld struct {
	mountedVolumesMetricsCache map[v1.UniqueVolumeName]volumeMetricsData
	sync.RWMutex
}

func (vmw *volumeMetricsOfWorld) MarkVolumeIsMeasuring(volumeName v1.UniqueVolumeName) {
	vmw.RLock()
	defer vmw.RUnlock()
	metricsData, exist := vmw.mountedVolumesMetricsCache[volumeName]
	if exist {
		metricsData.isMeasuring = true
	} else {
		metricsData = volumeMetricsData{isMeasuring: true}
	}

	vmw.mountedVolumesMetricsCache[volumeName] = metricsData
}

func (vmw *volumeMetricsOfWorld) UpdateVolumeMetricsData(volumeName v1.UniqueVolumeName,
	volumeMetrics *volume.Metrics,
	outerVolumeSpecName string,
	cacheDuration time.Duration,
	dataTimestamp time.Time) {
	vmw.RLock()
	defer vmw.RUnlock()
	metricsData := volumeMetricsData{
		metrics:             volumeMetrics,
		outerVolumeSpecName: outerVolumeSpecName,
		isMeasuring:         false,
		cacheDuration:       &cacheDuration,
		metricsTimestamp:    &dataTimestamp}
	vmw.mountedVolumesMetricsCache[volumeName] = metricsData
}

func (vmw *volumeMetricsOfWorld) UpdateVolumeMetricsStatus(volumeName v1.UniqueVolumeName,
	cacheDuration time.Duration,
	dataTimestamp time.Time) {
	vmw.RLock()
	defer vmw.RUnlock()
	metricsData, exist := vmw.mountedVolumesMetricsCache[volumeName]
	if exist {
		metricsData.isMeasuring = false
		metricsData.cacheDuration = &cacheDuration
		metricsData.metricsTimestamp = &dataTimestamp
	} else {
		metricsData = volumeMetricsData{
			isMeasuring:      false,
			cacheDuration:    &cacheDuration,
			metricsTimestamp: &dataTimestamp}
	}

	vmw.mountedVolumesMetricsCache[volumeName] = metricsData
}

func (vmw *volumeMetricsOfWorld) HousekeepVolumeMetricsWorld(volumeNamesInActualWorld []v1.UniqueVolumeName) {
	vmw.RLock()
	defer vmw.RUnlock()
	for volumeName := range vmw.mountedVolumesMetricsCache {
		dataRequireDeleted := true
		for index := range volumeNamesInActualWorld {
			if volumeName == volumeNamesInActualWorld[index] {
				dataRequireDeleted = false
				break
			}
		}
		if dataRequireDeleted {
			delete(vmw.mountedVolumesMetricsCache, volumeName)
		}
	}
}

func (vmw *volumeMetricsOfWorld) GetVolumeMetricsStatus(volumeName v1.UniqueVolumeName) (bool, *time.Duration) {
	vmw.RLock()
	defer vmw.RUnlock()

	metricsData, exist := vmw.mountedVolumesMetricsCache[volumeName]
	// The volume appear first time in actual world and requires measure.
	if !exist {
		return true, nil
	}

	// The volume is being measured by 'du' .
	if !metricsData.isMeasuring {
		return false, nil
	}

	if metricsData.metricsTimestamp != nil && metricsData.cacheDuration != nil {
		isExpired := metricsData.metricsTimestamp.Add(*metricsData.cacheDuration).Before(time.Now())
		// The volume metrics is expired and requires measure.
		if isExpired {
			return true, metricsData.cacheDuration
		}
	}

	return false, nil
}

func (vmw *volumeMetricsOfWorld) GetVolumeMetrics(volumeName v1.UniqueVolumeName) (*volume.Metrics, string) {
	vmw.RLock()
	defer vmw.RUnlock()

	metricsData, exist := vmw.mountedVolumesMetricsCache[volumeName]
	if !exist || metricsData.metrics == nil {
		return nil, ""
	}

	return metricsData.metrics, metricsData.outerVolumeSpecName
}
