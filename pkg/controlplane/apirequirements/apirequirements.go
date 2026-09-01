/*
Copyright The Kubernetes Authors.

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

// Package apirequirements declares which APIs must be served for a feature gate to function.
//
// It is deliberately kept free of control plane dependencies: cmd/genfeaturegates renders these
// requirements into the generated feature gate reference, and runs on every pull request, so it
// must not have to compile the apiserver to do so.
package apirequirements

import (
	"maps"

	certificatesapiv1 "k8s.io/api/certificates/v1"
	lifecyclev1alpha1 "k8s.io/api/lifecycle/v1alpha1"
	schedulingapiv1 "k8s.io/api/scheduling/v1"
	svmv1 "k8s.io/api/storagemigration/v1"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/kubernetes/pkg/features"
)

// DefaultForGenericControlPlane returns the APIs that must be served for a feature gate to
// function in a generic control plane.
func DefaultForGenericControlPlane() serverstorage.FeatureGateAPIRequirements {
	return serverstorage.FeatureGateAPIRequirements{
		features.ClusterTrustBundle:     {{Group: certificatesapiv1.GroupName, Resource: "clustertrustbundles"}},
		features.StorageVersionMigrator: {{Group: svmv1.GroupName, Resource: "storageversionmigrations"}},
	}
}

// DefaultForKubeAPIServer returns the APIs that must be served for a feature gate to function in
// kube-apiserver.
func DefaultForKubeAPIServer() serverstorage.FeatureGateAPIRequirements {
	// start with the requirements shared with a generic control plane
	ret := DefaultForGenericControlPlane()
	// PodCertificateRequest is here rather than in the generic set because the certificates
	// provider skips podcertificaterequests when it serves a generic control plane.
	maps.Copy(ret, serverstorage.FeatureGateAPIRequirements{
		features.CompositePodGroup: {{Group: schedulingapiv1.GroupName, Resource: "compositepodgroups"}},
		features.EvictionRequestAPI: {
			{Group: lifecyclev1alpha1.GroupName, Resource: "evictions"},
			{Group: lifecyclev1alpha1.GroupName, Resource: "evictionrequests"},
		},
		features.GenericWorkload: {
			{Group: schedulingapiv1.GroupName, Resource: "workloads"},
			{Group: schedulingapiv1.GroupName, Resource: "podgroups"},
		},
		features.PodCertificateRequest: {{Group: certificatesapiv1.GroupName, Resource: "podcertificaterequests"}},
	})
	return ret
}
