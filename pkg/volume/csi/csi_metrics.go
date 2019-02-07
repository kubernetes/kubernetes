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
	// targetPath is the directory path the volume is mounted to
	targetPath string

	// volumeID is the volume's handle or ID
	volumeID string

	// driverName is the CSI driver's name
	// Currently onl used to enrich log messages
	driverName string

	// csiClientGetter is the nested struct to construct, retrieve and cache the
	// CSI client
	csiClientGetter
}

// NewMetricsCsi creates a new metricsCsi with the Volume ID and path.
func NewMetricsCsi(driverName, volumeID, targetPath string, clientCreator clientCreatorFunc) volume.MetricsProvider {
	mc := &metricsCsi{volumeID: volumeID, targetPath: targetPath, driverName: driverName}
	mc.csiClientGetter.clientCreator = clientCreator
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
		return nil, volume.NewNotSupportedErrorWithDriverName(mc.driverName)
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
