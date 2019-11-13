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
	"k8s.io/component-base/featuregate"
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
	// GA: v1.16
	//
	// CustomResourceValidation is a list of validation methods for CustomResources
	CustomResourceValidation featuregate.Feature = "CustomResourceValidation"

	// owner: @roycaihw, @sttts
	// alpha: v1.14
	// beta: v1.15
	// GA: v1.16
	//
	// CustomResourcePublishOpenAPI enables publishing of CRD OpenAPI specs.
	CustomResourcePublishOpenAPI featuregate.Feature = "CustomResourcePublishOpenAPI"

	// owner: @sttts, @nikhita
	// alpha: v1.10
	// beta: v1.11
	// GA: v1.16
	//
	// CustomResourceSubresources defines the subresources for CustomResources
	CustomResourceSubresources featuregate.Feature = "CustomResourceSubresources"

	// owner: @mbohlool, @roycaihw
	// alpha: v1.13
	// beta: v1.15
	// GA: v1.16
	//
	// CustomResourceWebhookConversion defines the webhook conversion for Custom Resources.
	CustomResourceWebhookConversion featuregate.Feature = "CustomResourceWebhookConversion"

	// owner: @sttts
	// alpha: v1.15
	// beta: v1.16
	// GA: v1.17
	//
	// CustomResourceDefaulting enables OpenAPI defaulting in CustomResources.
	//
	// TODO: remove in 1.18
	CustomResourceDefaulting featuregate.Feature = "CustomResourceDefaulting"
)

func init() {
	utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates)
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	CustomResourceValidation:        {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	CustomResourceSubresources:      {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	CustomResourceWebhookConversion: {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	CustomResourcePublishOpenAPI:    {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	CustomResourceDefaulting:        {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
}
