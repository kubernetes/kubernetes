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
	"fmt"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/volume"

	"github.com/stretchr/testify/assert"
)

// TestGetPodVolumeStats tests that GetPodVolumeStats reads from the cache and returns the value
func TestGetPodVolumeStats(t *testing.T) {
	instance := newFsResourceAnalyzer(&MockStatsProvider{}, time.Minute*5)
	stats, found := instance.GetPodVolumeStats("testpod1")
	assert.False(t, found)
	assert.Equal(t, PodVolumeStats{}, stats)

	instance.cachedVolumeStats.Store(make(Cache))
	stats, found = instance.GetPodVolumeStats("testpod1")
	assert.False(t, found)
	assert.Equal(t, PodVolumeStats{}, stats)

	available := uint64(100)
	used := uint64(200)
	capacity := uint64(400)
	vs1 := VolumeStats{
		Name: "vol1",
		FsStats: FsStats{
			AvailableBytes: &available,
			UsedBytes:      &used,
			CapacityBytes:  &capacity,
		},
	}
	pvs := &PodVolumeStats{
		Volumes: []VolumeStats{vs1},
	}

	instance.cachedVolumeStats.Load().(Cache)["testpod1"] = pvs
	stats, found = instance.GetPodVolumeStats("testpod1")
	assert.True(t, found)
	assert.Equal(t, *pvs, stats)
}

// TestUpdateCachedPodVolumeStats tests that the cache is updated from the stats provider
func TestUpdateCachedPodVolumeStats(t *testing.T) {
	statsPr := &MockStatsProvider{}
	instance := newFsResourceAnalyzer(statsPr, time.Minute*5)

	// Mock retrieving pods
	pods := []*api.Pod{
		{ObjectMeta: api.ObjectMeta{UID: "testpod1"}},
		{ObjectMeta: api.ObjectMeta{UID: "testpod2"}},
	}
	statsPr.On("GetPods").Return(pods)

	// Mock volumes for pod1
	m1 := &volume.Metrics{
		Available: resource.NewQuantity(100, resource.DecimalSI),
		Used:      resource.NewQuantity(200, resource.DecimalSI),
		Capacity:  resource.NewQuantity(400, resource.DecimalSI),
	}
	v1 := &volume.MockVolume{}
	v1.On("GetMetrics").Return(m1, nil)

	m2 := &volume.Metrics{
		Available: resource.NewQuantity(600, resource.DecimalSI),
		Used:      resource.NewQuantity(700, resource.DecimalSI),
		Capacity:  resource.NewQuantity(1400, resource.DecimalSI),
	}
	v2 := &volume.MockVolume{}
	v2.On("GetMetrics").Return(m2, nil)
	tp1Volumes := map[string]volume.Volume{
		"v1": v1,
		"v2": v2,
	}
	statsPr.On("ListVolumesForPod", types.UID("testpod1")).Return(tp1Volumes, true)

	// Mock volumes for pod2
	m3 := &volume.Metrics{
		Available: resource.NewQuantity(800, resource.DecimalSI),
		Used:      resource.NewQuantity(900, resource.DecimalSI),
		Capacity:  resource.NewQuantity(1800, resource.DecimalSI),
	}
	v3 := &volume.MockVolume{}
	v3.On("GetMetrics").Return(m3, nil)
	v4 := &volume.MockVolume{}
	v4.On("GetMetrics").Return(nil, fmt.Errorf("Error calculating stats"))
	tp2Volumes := map[string]volume.Volume{
		"v3": v3,
		"v4": v4,
	}
	statsPr.On("ListVolumesForPod", types.UID("testpod2")).Return(tp2Volumes, true)

	instance.updateCachedPodVolumeStats()

	actual1, found := instance.GetPodVolumeStats("testpod1")
	assert.True(t, found)
	assert.Len(t, actual1.Volumes, 2)
	v1available := uint64(100)
	v1used := uint64(200)
	v1capacity := uint64(400)
	assert.Contains(t, actual1.Volumes, VolumeStats{
		Name: "v1",
		FsStats: FsStats{
			AvailableBytes: &v1available,
			UsedBytes:      &v1used,
			CapacityBytes:  &v1capacity,
		},
	})

	v2available := uint64(600)
	v2used := uint64(700)
	v2capacity := uint64(1400)
	assert.Contains(t, actual1.Volumes, VolumeStats{
		Name: "v2",
		FsStats: FsStats{
			AvailableBytes: &v2available,
			UsedBytes:      &v2used,
			CapacityBytes:  &v2capacity,
		},
	})

	v3available := uint64(800)
	v3used := uint64(900)
	v3capacity := uint64(1800)
	actual2, found := instance.GetPodVolumeStats("testpod2")
	assert.True(t, found)
	assert.Len(t, actual2.Volumes, 1)
	assert.Contains(t, actual2.Volumes, VolumeStats{
		Name: "v3",
		FsStats: FsStats{
			AvailableBytes: &v3available,
			UsedBytes:      &v3used,
			CapacityBytes:  &v3capacity,
		},
	})

	// Make sure the cache gets updated.  The mocking libraries have trouble
	pods = []*api.Pod{
		{ObjectMeta: api.ObjectMeta{UID: "testpod3"}},
	}
	statsPr.On("GetPods").Return(pods)

	// pod3 volumes
	m1 = &volume.Metrics{
		Available: resource.NewQuantity(150, resource.DecimalSI),
		Used:      resource.NewQuantity(200, resource.DecimalSI),
		Capacity:  resource.NewQuantity(600, resource.DecimalSI),
	}
	v1 = &volume.MockVolume{}
	v1.On("GetMetrics").Return(m1, nil)

	tp1Volumes = map[string]volume.Volume{
		"v1": v1,
	}
	statsPr.On("ListVolumesForPod", types.UID("testpod3")).Return(tp1Volumes, true)
}
