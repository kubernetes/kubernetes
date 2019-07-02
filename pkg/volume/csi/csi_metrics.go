/*
Copyright 2019 The Kubernetes Authors.

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

package csi

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/volume"
)

var _ volume.MetricsProvider = &metricsCsi{}

// metricsCsi represents a MetricsProvider that calculates the used,free and
// capacity information for volume using volume path.

type metricsCsi struct {
	// the directory path the volume is mounted to.
	targetPath string

	// Volume handle or id
	volumeID string

	//csiClient with cache
	csiClientGetter
}

// NewMetricsCsi creates a new metricsCsi with the Volume ID and path.
func NewMetricsCsi(volumeID string, targetPath string, driverName csiDriverName) volume.MetricsProvider {
	mc := &metricsCsi{volumeID: volumeID, targetPath: targetPath}
	mc.csiClientGetter.driverName = driverName
	return mc
}

func (mc *metricsCsi) GetMetrics() (*volume.Metrics, error) {
	currentTime := metav1.Now()
	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()
	// Get CSI client
	csiClient, err := mc.csiClientGetter.Get()
	if err != nil {
		return nil, err
	}
	// Check whether "GET_VOLUME_STATS" is set
	volumeStatsSet, err := csiClient.NodeSupportsVolumeStats(ctx)
	if err != nil {
		return nil, err
	}
	// if plugin doesnot support volume status, return.
	if !volumeStatsSet {
		return nil, nil
	}
	// Get Volumestatus
	metrics, err := csiClient.NodeGetVolumeStats(ctx, mc.volumeID, mc.targetPath)
	if err != nil {
		return nil, err
	}
	if metrics == nil {
		return nil, fmt.Errorf("csi.NodeGetVolumeStats returned nil metrics for volume %s", mc.volumeID)
	}
	//set recorded time
	metrics.Time = currentTime
	return metrics, nil
}
