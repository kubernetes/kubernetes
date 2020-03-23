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
	"os"
	"strconv"

	"github.com/onsi/gomega"
	"k8s.io/kubernetes/test/e2e/framework"
)

// environment variables related to datastore parameters
const (
	SPBMPolicyName             = "VSPHERE_SPBM_POLICY_NAME"
	StorageClassDatastoreName  = "VSPHERE_DATASTORE"
	SecondSharedDatastore      = "VSPHERE_SECOND_SHARED_DATASTORE"
	KubernetesClusterName      = "VSPHERE_KUBERNETES_CLUSTER"
	SPBMTagPolicy              = "VSPHERE_SPBM_TAG_POLICY"
	VCPClusterDatastore        = "CLUSTER_DATASTORE"
	SPBMPolicyDataStoreCluster = "VSPHERE_SPBM_POLICY_DS_CLUSTER"
)

// environment variables used for scaling tests
const (
	VCPScaleVolumeCount   = "VCP_SCALE_VOLUME_COUNT"
	VCPScaleVolumesPerPod = "VCP_SCALE_VOLUME_PER_POD"
	VCPScaleInstances     = "VCP_SCALE_INSTANCES"
)

// environment variables used for stress tests
const (
	VCPStressInstances  = "VCP_STRESS_INSTANCES"
	VCPStressIterations = "VCP_STRESS_ITERATIONS"
)

// environment variables used for performance tests
const (
	VCPPerfVolumeCount   = "VCP_PERF_VOLUME_COUNT"
	VCPPerfVolumesPerPod = "VCP_PERF_VOLUME_PER_POD"
	VCPPerfIterations    = "VCP_PERF_ITERATIONS"
)

// environment variables used for zone tests
const (
	VCPZoneVsanDatastore1      = "VCP_ZONE_VSANDATASTORE1"
	VCPZoneVsanDatastore2      = "VCP_ZONE_VSANDATASTORE2"
	VCPZoneLocalDatastore      = "VCP_ZONE_LOCALDATASTORE"
	VCPZoneCompatPolicyName    = "VCP_ZONE_COMPATPOLICY_NAME"
	VCPZoneNonCompatPolicyName = "VCP_ZONE_NONCOMPATPOLICY_NAME"
	VCPZoneA                   = "VCP_ZONE_A"
	VCPZoneB                   = "VCP_ZONE_B"
	VCPZoneC                   = "VCP_ZONE_C"
	VCPZoneD                   = "VCP_ZONE_D"
	VCPInvalidZone             = "VCP_INVALID_ZONE"
)

// storage class parameters
const (
	Datastore                    = "datastore"
	PolicyDiskStripes            = "diskStripes"
	PolicyHostFailuresToTolerate = "hostFailuresToTolerate"
	PolicyCacheReservation       = "cacheReservation"
	PolicyObjectSpaceReservation = "objectSpaceReservation"
	PolicyIopsLimit              = "iopsLimit"
	DiskFormat                   = "diskformat"
	SpbmStoragePolicy            = "storagepolicyname"
)

// test values for storage class parameters
const (
	ThinDisk                                   = "thin"
	BronzeStoragePolicy                        = "bronze"
	HostFailuresToTolerateCapabilityVal        = "0"
	CacheReservationCapabilityVal              = "20"
	DiskStripesCapabilityVal                   = "1"
	ObjectSpaceReservationCapabilityVal        = "30"
	IopsLimitCapabilityVal                     = "100"
	StripeWidthCapabilityVal                   = "2"
	DiskStripesCapabilityInvalidVal            = "14"
	HostFailuresToTolerateCapabilityInvalidVal = "4"
)

// GetAndExpectStringEnvVar returns the string value of an environment variable or fails if
// the variable is not set
func GetAndExpectStringEnvVar(varName string) string {
	varValue := os.Getenv(varName)
	gomega.Expect(varValue).NotTo(gomega.BeEmpty(), "ENV "+varName+" is not set")
	return varValue
}

// GetAndExpectIntEnvVar returns the integer value of an environment variable or fails if
// the variable is not set
func GetAndExpectIntEnvVar(varName string) int {
	varValue := GetAndExpectStringEnvVar(varName)
	varIntValue, err := strconv.Atoi(varValue)
	framework.ExpectNoError(err, "Error Parsing "+varName)
	return varIntValue
}
