/*
Copyright 2017 The Kubernetes Authors.

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

package vsphere

import (
	"github.com/prometheus/client_golang/prometheus"
	"time"
)

const (
	api_createvolume = "CreateVolume"
	api_deletevolume = "DeleteVolume"
	api_attachvolume = "AttachVolume"
	api_detachvolume = "DetachVolume"
)

const (
	operation_deletevolume                      = "DeleteVolumeOperation"
	operation_attachvolume                      = "AttachVolumeOperation"
	operation_detachvolume                      = "DetachVolumeOperation"
	operation_diskIsAttached                    = "DiskIsAttachedOperation"
	operation_disksAreAttached                  = "DisksAreAttachedOperation"
	operation_createvolume                      = "CreateVolumeOperation"
	operation_createvolume_with_policy          = "CreateVolumeWithPolicyOperation"
	operation_createvolume_with_raw_vsan_policy = "CreateVolumeWithRawVSANPolicyOperation"
)

// vsphereApiMetric is for recording latency of Single API Call.
var vsphereApiMetric = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Name: "cloudprovider_vsphere_api_request_duration_seconds",
		Help: "Latency of vsphere api call",
	},
	[]string{"request"},
)

var vsphereApiErrorMetric = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "cloudprovider_vsphere_api_request_errors",
		Help: "vsphere Api errors",
	},
	[]string{"request"},
)

// vsphereOperationMetric is for recording latency of vSphere Operation which invokes multiple APIs to get the task done.
var vsphereOperationMetric = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Name: "cloudprovider_vsphere_operation_duration_seconds",
		Help: "Latency of vsphere operation call",
	},
	[]string{"operation"},
)

var vsphereOperationErrorMetric = prometheus.NewCounterVec(
	prometheus.CounterOpts{
		Name: "cloudprovider_vsphere_operation_errors",
		Help: "vsphere operation errors",
	},
	[]string{"operation"},
)

func registerMetrics() {
	prometheus.MustRegister(vsphereApiMetric)
	prometheus.MustRegister(vsphereApiErrorMetric)
	prometheus.MustRegister(vsphereOperationMetric)
	prometheus.MustRegister(vsphereOperationErrorMetric)
}

func recordvSphereMetric(actionName string, requestTime time.Time, err error) {
	switch actionName {
	case api_createvolume, api_deletevolume, api_attachvolume, api_detachvolume:
		recordvSphereAPIMetric(actionName, requestTime, err)
	default:
		recordvSphereOperationMetric(actionName, requestTime, err)
	}
}

func recordvSphereAPIMetric(actionName string, requestTime time.Time, err error) {
	if err != nil {
		vsphereApiErrorMetric.With(prometheus.Labels{"request": actionName}).Inc()
	} else {
		vsphereApiMetric.With(prometheus.Labels{"request": actionName}).Observe(calculateTimeTaken(requestTime))
	}
}

func recordvSphereOperationMetric(actionName string, requestTime time.Time, err error) {
	if err != nil {
		vsphereOperationErrorMetric.With(prometheus.Labels{"operation": actionName}).Inc()
	} else {
		vsphereOperationMetric.With(prometheus.Labels{"operation": actionName}).Observe(calculateTimeTaken(requestTime))
	}
}

func recordCreateVolumeMetric(volumeOptions *VolumeOptions, requestTime time.Time, err error) {
	var actionName string
	if volumeOptions.StoragePolicyName != "" {
		actionName = operation_createvolume_with_policy
	} else if volumeOptions.VSANStorageProfileData != "" {
		actionName = operation_createvolume_with_raw_vsan_policy
	} else {
		actionName = operation_createvolume
	}
	recordvSphereMetric(actionName, requestTime, err)
}

func calculateTimeTaken(requestBeginTime time.Time) (timeTaken float64) {
	if !requestBeginTime.IsZero() {
		timeTaken = time.Since(requestBeginTime).Seconds()
	} else {
		timeTaken = 0
	}
	return timeTaken
}
