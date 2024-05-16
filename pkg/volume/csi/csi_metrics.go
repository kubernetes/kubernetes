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
	"time"

	"google.golang.org/grpc"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	servermetrics "k8s.io/kubernetes/pkg/kubelet/server/metrics"
	"k8s.io/kubernetes/pkg/volume"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
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
	startTime := time.Now()
	defer servermetrics.CollectVolumeStatCalDuration(string(mc.csiClientGetter.driverName), startTime)
	currentTime := metav1.Now()
	ctx, cancel := context.WithTimeout(context.Background(), csiTimeout)
	defer cancel()
	// Get CSI client
	csiClient, err := mc.csiClientGetter.Get()
	if err != nil {
		// Treat the absence of the CSI driver as a transient error
		// See https://github.com/kubernetes/kubernetes/issues/120268
		return nil, volumetypes.NewTransientOperationFailure(err.Error())
	}
	// Check whether "GET_VOLUME_STATS" is set
	volumeStatsSet, err := csiClient.NodeSupportsVolumeStats(ctx)
	if err != nil {
		return nil, err
	}

	// if plugin doesnot support volume status, return.
	if !volumeStatsSet {
		return nil, volume.NewNotSupportedErrorWithDriverName(
			string(mc.csiClientGetter.driverName))
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

// MetricsManager defines the metrics manager for CSI operation
type MetricsManager struct {
	driverName string
}

// NewCSIMetricsManager creates a CSIMetricsManager object
func NewCSIMetricsManager(driverName string) *MetricsManager {
	cmm := MetricsManager{
		driverName: driverName,
	}
	return &cmm
}

type additionalInfo struct {
	Migrated string
}
type additionalInfoKeyType struct{}

var additionalInfoKey additionalInfoKeyType

// RecordMetricsInterceptor is a grpc interceptor that is used to
// record CSI operation
func (cmm *MetricsManager) RecordMetricsInterceptor(
	ctx context.Context,
	method string,
	req, reply interface{},
	cc *grpc.ClientConn,
	invoker grpc.UnaryInvoker,
	opts ...grpc.CallOption) error {
	start := time.Now()
	err := invoker(ctx, method, req, reply, cc, opts...)
	duration := time.Since(start)
	// Check if this is migrated operation
	additionalInfoVal := ctx.Value(additionalInfoKey)
	migrated := "false"
	if additionalInfoVal != nil {
		additionalInfoVal, ok := additionalInfoVal.(additionalInfo)
		if !ok {
			return err
		}
		migrated = additionalInfoVal.Migrated
	}
	// Record the metric latency
	volumeutil.RecordCSIOperationLatencyMetrics(cmm.driverName, method, err, duration, migrated)

	return err
}
