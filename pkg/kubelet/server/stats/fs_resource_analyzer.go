/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"sync/atomic"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
)

// Map to PodVolumeStats pointers since the addresses for map values are not constant and can cause pain
// if we need ever to get a pointer to one of the values (e.g. you can't)
type Cache map[types.UID]*PodVolumeStats

// PodVolumeStats encapsulates all VolumeStats for a pod
type PodVolumeStats struct {
	Volumes []stats.VolumeStats
}

// fsResourceAnalyzerInterface is for embedding fs functions into ResourceAnalyzer
type fsResourceAnalyzerInterface interface {
	GetPodVolumeStats(uid types.UID) (PodVolumeStats, bool)
}

// diskResourceAnalyzer provider stats about fs resource usage
type fsResourceAnalyzer struct {
	statsProvider     StatsProvider
	calcVolumePeriod  time.Duration
	cachedVolumeStats atomic.Value
}

var _ fsResourceAnalyzerInterface = &fsResourceAnalyzer{}

// newFsResourceAnalyzer returns a new fsResourceAnalyzer implementation
func newFsResourceAnalyzer(statsProvider StatsProvider, calcVolumePeriod time.Duration) *fsResourceAnalyzer {
	return &fsResourceAnalyzer{
		statsProvider:    statsProvider,
		calcVolumePeriod: calcVolumePeriod,
	}
}

// Start eager background caching of volume stats.
func (s *fsResourceAnalyzer) Start() {
	if s.calcVolumePeriod <= 0 {
		glog.Info("Volume stats collection disabled.")
		return
	}
	glog.Info("Starting FS ResourceAnalyzer")
	go wait.Forever(func() {
		startTime := time.Now()
		s.updateCachedPodVolumeStats()
		glog.V(3).Infof("Finished calculating volume stats in %v.", time.Now().Sub(startTime))
		metrics.MetricsVolumeCalcLatency.Observe(metrics.SinceInMicroseconds(startTime))
	}, s.calcVolumePeriod)
}

// updateCachedPodVolumeStats calculates and caches the PodVolumeStats for every Pod known to the kubelet.
func (s *fsResourceAnalyzer) updateCachedPodVolumeStats() {
	// Calculate the new volume stats map
	pods := s.statsProvider.GetPods()
	newCache := make(Cache)
	// TODO: Prevent 1 pod metrics hanging from blocking other pods.  Schedule pods independently and spaced
	// evenly across the period to prevent cpu spikes.  Ideally resource collection consumes the resources
	// allocated to the pod itself to isolate bad actors.
	// See issue #20675
	for _, pod := range pods {
		podUid := pod.GetUID()
		stats, found := s.getPodVolumeStats(pod)
		if !found {
			glog.Warningf("Could not locate volumes for pod %s", format.Pod(pod))
			continue
		}
		newCache[podUid] = &stats
	}
	// Update the cache reference
	s.cachedVolumeStats.Store(newCache)
}

// getPodVolumeStats calculates PodVolumeStats for a given pod and returns the result.
func (s *fsResourceAnalyzer) getPodVolumeStats(pod *api.Pod) (PodVolumeStats, bool) {
	// Find all Volumes for the Pod
	volumes, found := s.statsProvider.ListVolumesForPod(pod.UID)
	if !found {
		return PodVolumeStats{}, found
	}

	// Call GetMetrics on each Volume and copy the result to a new VolumeStats.FsStats
	stats := make([]stats.VolumeStats, 0, len(volumes))
	for name, v := range volumes {
		metric, err := v.GetMetrics()
		if err != nil {
			// Expected for Volumes that don't support Metrics
			// TODO: Disambiguate unsupported from errors
			// See issue #20676
			glog.V(4).Infof("Failed to calculate volume metrics for pod %s volume %s: %+v",
				format.Pod(pod), name, err)
			continue
		}
		stats = append(stats, s.parsePodVolumeStats(name, metric))
	}
	return PodVolumeStats{Volumes: stats}, true
}

func (s *fsResourceAnalyzer) parsePodVolumeStats(podName string, metric *volume.Metrics) stats.VolumeStats {
	available := uint64(metric.Available.Value())
	capacity := uint64(metric.Capacity.Value())
	used := uint64((metric.Used.Value()))
	return stats.VolumeStats{
		Name: podName,
		FsStats: stats.FsStats{
			AvailableBytes: &available,
			CapacityBytes:  &capacity,
			UsedBytes:      &used}}
}

// GetPodVolumeStats returns the PodVolumeStats for a given pod.  Results are looked up from a cache that
// is eagerly populated in the background, and never calculated on the fly.
func (s *fsResourceAnalyzer) GetPodVolumeStats(uid types.UID) (PodVolumeStats, bool) {
	// Cache hasn't been initialized yet
	if s.cachedVolumeStats.Load() == nil {
		return PodVolumeStats{}, false
	}
	cache := s.cachedVolumeStats.Load().(Cache)
	stats, f := cache[uid]
	if !f {
		// TODO: Differentiate between stats being empty
		// See issue #20679
		return PodVolumeStats{}, false
	}
	return *stats, true
}
