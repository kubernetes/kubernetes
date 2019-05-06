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

package vclib

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// Cloud Provider API constants
const (
	APICreateVolume = "CreateVolume"
	APIDeleteVolume = "DeleteVolume"
	APIAttachVolume = "AttachVolume"
	APIDetachVolume = "DetachVolume"
)

// Cloud Provider Operation constants
const (
	OperationDeleteVolume                  = "DeleteVolumeOperation"
	OperationAttachVolume                  = "AttachVolumeOperation"
	OperationDetachVolume                  = "DetachVolumeOperation"
	OperationDiskIsAttached                = "DiskIsAttachedOperation"
	OperationDisksAreAttached              = "DisksAreAttachedOperation"
	OperationCreateVolume                  = "CreateVolumeOperation"
	OperationCreateVolumeWithPolicy        = "CreateVolumeWithPolicyOperation"
	OperationCreateVolumeWithRawVSANPolicy = "CreateVolumeWithRawVSANPolicyOperation"
)

// vsphereAPIMetric is for recording latency of Single API Call.
var vsphereAPIMetric = prometheus.NewHistogramVec(
	prometheus.HistogramOpts{
		Name: "cloudprovider_vsphere_api_request_duration_seconds",
		Help: "Latency of vsphere api call",
	},
	[]string{"request"},
)

var vsphereAPIErrorMetric = prometheus.NewCounterVec(
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

// RegisterMetrics registers all the API and Operation metrics
func RegisterMetrics() {
	prometheus.MustRegister(vsphereAPIMetric)
	prometheus.MustRegister(vsphereAPIErrorMetric)
	prometheus.MustRegister(vsphereOperationMetric)
	prometheus.MustRegister(vsphereOperationErrorMetric)
}

// RecordvSphereMetric records the vSphere API and Operation metrics
func RecordvSphereMetric(actionName string, requestTime time.Time, err error) {
	switch actionName {
	case APICreateVolume, APIDeleteVolume, APIAttachVolume, APIDetachVolume:
		recordvSphereAPIMetric(actionName, requestTime, err)
	default:
		recordvSphereOperationMetric(actionName, requestTime, err)
	}
}

func recordvSphereAPIMetric(actionName string, requestTime time.Time, err error) {
	if err != nil {
		vsphereAPIErrorMetric.With(prometheus.Labels{"request": actionName}).Inc()
	} else {
		vsphereAPIMetric.With(prometheus.Labels{"request": actionName}).Observe(calculateTimeTaken(requestTime))
	}
}

func recordvSphereOperationMetric(actionName string, requestTime time.Time, err error) {
	if err != nil {
		vsphereOperationErrorMetric.With(prometheus.Labels{"operation": actionName}).Inc()
	} else {
		vsphereOperationMetric.With(prometheus.Labels{"operation": actionName}).Observe(calculateTimeTaken(requestTime))
	}
}

// RecordCreateVolumeMetric records the Create Volume metric
func RecordCreateVolumeMetric(volumeOptions *VolumeOptions, requestTime time.Time, err error) {
	var actionName string
	if volumeOptions.StoragePolicyName != "" {
		actionName = OperationCreateVolumeWithPolicy
	} else if volumeOptions.VSANStorageProfileData != "" {
		actionName = OperationCreateVolumeWithRawVSANPolicy
	} else {
		actionName = OperationCreateVolume
	}
	RecordvSphereMetric(actionName, requestTime, err)
}

func calculateTimeTaken(requestBeginTime time.Time) (timeTaken float64) {
	if !requestBeginTime.IsZero() {
		timeTaken = time.Since(requestBeginTime).Seconds()
	} else {
		timeTaken = 0
	}
	return timeTaken
}
