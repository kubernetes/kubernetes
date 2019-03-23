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
	. "github.com/onsi/gomega"
	"os"
	"strconv"
)

const (
	SPBMPolicyName            = "VSPHERE_SPBM_POLICY_NAME"
	StorageClassDatastoreName = "VSPHERE_DATASTORE"
	SecondSharedDatastore     = "VSPHERE_SECOND_SHARED_DATASTORE"
	KubernetesClusterName     = "VSPHERE_KUBERNETES_CLUSTER"
	SPBMTagPolicy             = "VSPHERE_SPBM_TAG_POLICY"
)

const (
	VCPClusterDatastore        = "CLUSTER_DATASTORE"
	SPBMPolicyDataStoreCluster = "VSPHERE_SPBM_POLICY_DS_CLUSTER"
)

const (
	VCPScaleVolumeCount   = "VCP_SCALE_VOLUME_COUNT"
	VCPScaleVolumesPerPod = "VCP_SCALE_VOLUME_PER_POD"
	VCPScaleInstances     = "VCP_SCALE_INSTANCES"
)

const (
	VCPStressInstances  = "VCP_STRESS_INSTANCES"
	VCPStressIterations = "VCP_STRESS_ITERATIONS"
)

const (
	VCPPerfVolumeCount   = "VCP_PERF_VOLUME_COUNT"
	VCPPerfVolumesPerPod = "VCP_PERF_VOLUME_PER_POD"
	VCPPerfIterations    = "VCP_PERF_ITERATIONS"
)

const (
	VCPZoneVsanDatastore1      = "VCP_ZONE_VSANDATASTORE1"
	VCPZoneVsanDatastore2      = "VCP_ZONE_VSANDATASTORE2"
	VCPZoneCompatPolicyName    = "VCP_ZONE_COMPATPOLICY_NAME"
	VCPZoneNonCompatPolicyName = "VCP_ZONE_NONCOMPATPOLICY_NAME"
	VCPZoneA                   = "VCP_ZONE_A"
	VCPZoneB                   = "VCP_ZONE_B"
	VCPZoneC                   = "VCP_ZONE_C"
	VCPZoneD                   = "VCP_ZONE_D"
)

func GetAndExpectStringEnvVar(varName string) string {
	varValue := os.Getenv(varName)
	Expect(varValue).NotTo(BeEmpty(), "ENV "+varName+" is not set")
	return varValue
}

func GetAndExpectIntEnvVar(varName string) int {
	varValue := GetAndExpectStringEnvVar(varName)
	varIntValue, err := strconv.Atoi(varValue)
	Expect(err).NotTo(HaveOccurred(), "Error Parsing "+varName)
	return varIntValue
}
