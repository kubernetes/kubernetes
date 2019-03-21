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

package features

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // alpha: v1.4
	// MyFeature() bool

	// owner: @sttts, @nikhita
	// alpha: v1.8
	// beta: v1.9
	//
	// CustomResourceValidation is a list of validation methods for CustomResources
	CustomResourceValidation utilfeature.Feature = "CustomResourceValidation"

	// owner: @roycaihw, @sttts
	// alpha: v1.14
	//
	// CustomResourcePublishOpenAPI enables publishing of CRD OpenAPI specs.
	CustomResourcePublishOpenAPI utilfeature.Feature = "CustomResourcePublishOpenAPI"

	// owner: @sttts, @nikhita
	// alpha: v1.10
	// beta: v1.11
	//
	// CustomResourceSubresources defines the subresources for CustomResources
	CustomResourceSubresources utilfeature.Feature = "CustomResourceSubresources"

	// owner: @mbohlool, @roycaihw
	// alpha: v1.13
	//
	// CustomResourceWebhookConversion defines the webhook conversion for Custom Resources.
	CustomResourceWebhookConversion utilfeature.Feature = "CustomResourceWebhookConversion"
)

func init() {
	utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates)
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[utilfeature.Feature]utilfeature.FeatureSpec{
	CustomResourceValidation:        {Default: true, PreRelease: utilfeature.Beta},
	CustomResourceSubresources:      {Default: true, PreRelease: utilfeature.Beta},
	CustomResourceWebhookConversion: {Default: false, PreRelease: utilfeature.Alpha},
	CustomResourcePublishOpenAPI:    {Default: false, PreRelease: utilfeature.Alpha},
}
