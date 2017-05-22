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

package stats

import (
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/api/v1"
	stats "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/golang/glog"
)

// volumeStatCalculator calculates volume metrics for a given pod periodically in the background and caches the result
type volumeStatCalculator struct {
	statsProvider StatsProvider
	jitterPeriod  time.Duration
	pod           *v1.Pod
	stopChannel   chan struct{}
	startO        sync.Once
	stopO         sync.Once
	latest        atomic.Value
}

// PodVolumeStats encapsulates all VolumeStats for a pod
type PodVolumeStats struct {
	Volumes []stats.VolumeStats
}

// newVolumeStatCalculator creates a new VolumeStatCalculator
func newVolumeStatCalculator(statsProvider StatsProvider, jitterPeriod time.Duration, pod *v1.Pod) *volumeStatCalculator {
	return &volumeStatCalculator{
		statsProvider: statsProvider,
		jitterPeriod:  jitterPeriod,
		pod:           pod,
		stopChannel:   make(chan struct{}),
	}
}

// StartOnce starts pod volume calc that will occur periodically in the background until s.StopOnce is called
func (s *volumeStatCalculator) StartOnce() *volumeStatCalculator {
	s.startO.Do(func() {
		go wait.JitterUntil(func() {
			s.calcAndStoreStats()
		}, s.jitterPeriod, 1.0, true, s.stopChannel)
	})
	return s
}

// StopOnce stops background pod volume calculation.  Will not stop a currently executing calculations until
// they complete their current iteration.
func (s *volumeStatCalculator) StopOnce() *volumeStatCalculator {
	s.stopO.Do(func() {
		close(s.stopChannel)
	})
	return s
}

// getLatest returns the most recent PodVolumeStats from the cache
func (s *volumeStatCalculator) GetLatest() (PodVolumeStats, bool) {
	if result := s.latest.Load(); result == nil {
		return PodVolumeStats{}, false
	} else {
		return result.(PodVolumeStats), true
	}
}

// calcAndStoreStats calculates PodVolumeStats for a given pod and writes the result to the s.latest cache.
func (s *volumeStatCalculator) calcAndStoreStats() {
	// Find all Volumes for the Pod
	volumes, found := s.statsProvider.ListVolumesForPod(s.pod.UID)
	if !found {
		return
	}

	// Call GetMetrics on each Volume and copy the result to a new VolumeStats.FsStats
	stats := make([]stats.VolumeStats, 0, len(volumes))
	for name, v := range volumes {
		metric, err := v.GetMetrics()
		if err != nil {
			// Expected for Volumes that don't support Metrics
			if !volume.IsNotSupported(err) {
				glog.V(4).Infof("Failed to calculate volume metrics for pod %s volume %s: %+v", format.Pod(s.pod), name, err)
			}
			continue
		}
		stats = append(stats, s.parsePodVolumeStats(name, metric))
	}

	// Store the new stats
	s.latest.Store(PodVolumeStats{Volumes: stats})
}

// parsePodVolumeStats converts (internal) volume.Metrics to (external) stats.VolumeStats structures
func (s *volumeStatCalculator) parsePodVolumeStats(podName string, metric *volume.Metrics) stats.VolumeStats {
	available := uint64(metric.Available.Value())
	capacity := uint64(metric.Capacity.Value())
	used := uint64(metric.Used.Value())
	inodes := uint64(metric.Inodes.Value())
	inodesFree := uint64(metric.InodesFree.Value())
	inodesUsed := uint64(metric.InodesUsed.Value())
	return stats.VolumeStats{
		Name: podName,
		FsStats: stats.FsStats{Time: metric.Time, AvailableBytes: &available, CapacityBytes: &capacity,
			UsedBytes: &used, Inodes: &inodes, InodesFree: &inodesFree, InodesUsed: &inodesUsed},
	}
}
