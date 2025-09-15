/*
Copyright 2024 The Kubernetes Authors.

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

// +groupName=output.kubeadm.k8s.io
// +k8s:deepcopy-gen=package
// +k8s:conversion-gen=k8s.io/kubernetes/cmd/kubeadm/app/apis/output

// Package v1alpha3 defines the v1alpha3 version of the kubeadm data structures
// related to structured output
// The purpose of the kubeadm structured output is to have a well
// defined versioned output format that other software that uses
// kubeadm for cluster deployments can use and rely on.
//
// Changes since v1alpha2:
//   - Added support for outputting certificate expiration information for "kubeadm certs check-expiration"
//     with the CertificateExpirationInfo structure.
//   - Introduce a (breaking) change to the UpgradePlan structure used by "kubeadm upgrade plan".
//     UpgradePlan now contains a list of AvailableUpgrade structures.
package v1alpha3
