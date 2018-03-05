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

package openstack

import (
	. "github.com/onsi/gomega"
	"os"
	"strconv"
)

const (
	SPBMPolicyName            = "OPENSTACK_SPBM_POLICY_NAME"
	StorageClassDatastoreName = "OPENSTACK_DATASTORE"
	SecondSharedDatastore     = "OPENSTACK_SECOND_SHARED_DATASTORE"
	KubernetesClusterName     = "OPENSTACK_KUBERNETES_CLUSTER"
	SPBMTagPolicy             = "OPENSTACK_SPBM_TAG_POLICY"
)

const (
	OSPClusterDatastore        = "CLUSTER_DATASTORE"
	SPBMPolicyDataStoreCluster = "OPENSTACK_SPBM_POLICY_DS_CLUSTER"
)

const (
	OSPScaleVolumeCount   = "OSP_SCALE_VOLUME_COUNT"
	OSPScaleVolumesPerPod = "OSP_SCALE_VOLUME_PER_POD"
	OSPScaleInstances     = "OSP_SCALE_INSTANCES"
)

const (
	OSPStressInstances  = "OSP_STRESS_INSTANCES"
	OSPStressIterations = "OSP_STRESS_ITERATIONS"
)

const (
	OSPPerfVolumeCount   = "OSP_PERF_VOLUME_COUNT"
	OSPPerfVolumesPerPod = "OSP_PERF_VOLUME_PER_POD"
	OSPPerfIterations    = "OSP_PERF_ITERATIONS"
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
