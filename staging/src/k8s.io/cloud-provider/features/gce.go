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

package features

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// TODO: this file should ideally live in k8s.io/cloud-provider-gcp, but it is
// temporarily placed here to remove dependencies to k8s.io/kubernetes in the
// in-tree GCE cloud provider. Move this to k8s.io/cloud-provider-gcp as soon
// as it's ready to be used
const (
	// owner: @verult
	// GA: v1.13
	//
	// Enables the regional PD feature on GCE.
	GCERegionalPersistentDisk utilfeature.Feature = "GCERegionalPersistentDisk"
)
