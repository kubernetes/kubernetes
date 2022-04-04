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
	"k8s.io/apimachinery/pkg/util/runtime"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
)

const (
	// Every feature gate should add method here following this template:
	//
	// // owner: @username
	// // alpha: v1.4
	// MyFeature() bool

	// owner: @tallclair
	// alpha: v1.7
	// beta: v1.8
	// GA: v1.12
	//
	// AdvancedAuditing enables a much more general API auditing pipeline, which includes support for
	// pluggable output backends and an audit policy specifying how different requests should be
	// audited.
	AdvancedAuditing featuregate.Feature = "AdvancedAuditing"

	// owner: @ilackams
	// alpha: v1.7
	// beta: v1.16
	//
	// Enables compression of REST responses (GET and LIST only)
	APIResponseCompression featuregate.Feature = "APIResponseCompression"

	// owner: @smarterclayton
	// alpha: v1.8
	// beta: v1.9
	//
	// Allow API clients to retrieve resource lists in chunks rather than
	// all at once.
	APIListChunking featuregate.Feature = "APIListChunking"

	// owner: @apelisse
	// alpha: v1.12
	// beta: v1.13
	// stable: v1.18
	//
	// Allow requests to be processed but not stored, so that
	// validation, merging, mutation can be tested without
	// committing.
	DryRun featuregate.Feature = "DryRun"

	// owner: @caesarxuchao
	// alpha: v1.15
	// beta: v1.16
	//
	// Allow apiservers to show a count of remaining items in the response
	// to a chunking list request.
	RemainingItemCount featuregate.Feature = "RemainingItemCount"

	// owner: @apelisse, @lavalamp
	// alpha: v1.14
	// beta: v1.16
	// stable: v1.22
	//
	// Server-side apply. Merging happens on the server.
	ServerSideApply featuregate.Feature = "ServerSideApply"

	// owner: @caesarxuchao
	// alpha: v1.14
	// beta: v1.15
	//
	// Allow apiservers to expose the storage version hash in the discovery
	// document.
	StorageVersionHash featuregate.Feature = "StorageVersionHash"

	// owner: @caesarxuchao @roycaihw
	// alpha: v1.20
	//
	// Enable the storage version API.
	StorageVersionAPI featuregate.Feature = "StorageVersionAPI"

	// owner: @wojtek-t
	// alpha: v1.15
	// beta: v1.16
	// GA: v1.17
	//
	// Enables support for watch bookmark events.
	WatchBookmark featuregate.Feature = "WatchBookmark"

	// owner: @MikeSpreitzer @yue9944882
	// alpha: v1.18
	// beta: v1.20
	//
	// Enables managing request concurrency with prioritization and fairness at each server.
	// The FeatureGate was introduced in release 1.15 but the feature
	// was not really implemented before 1.18.
	APIPriorityAndFairness featuregate.Feature = "APIPriorityAndFairness"

	// owner: @wojtek-t
	// alpha: v1.16
	// beta: v1.20
	// GA: v1.24
	//
	// Deprecates and removes SelfLink from ObjectMeta and ListMeta.
	RemoveSelfLink featuregate.Feature = "RemoveSelfLink"

	// owner: @shaloulcy, @wojtek-t
	// alpha: v1.18
	// beta: v1.19
	// GA: v1.20
	//
	// Allows label and field based indexes in apiserver watch cache to accelerate list operations.
	SelectorIndex featuregate.Feature = "SelectorIndex"

	// owner: @wojtek-t
	// alpha: v1.20
	// beta: v1.21
	// GA: v1.24
	//
	// Allows for updating watchcache resource version with progress notify events.
	EfficientWatchResumption featuregate.Feature = "EfficientWatchResumption"

	// owner: @roycaihw
	// alpha: v1.20
	//
	// Assigns each kube-apiserver an ID in a cluster.
	APIServerIdentity featuregate.Feature = "APIServerIdentity"

	// owner: @dashpole
	// alpha: v1.22
	//
	// Add support for distributed tracing in the API Server
	APIServerTracing featuregate.Feature = "APIServerTracing"

	// owner: @jiahuif
	// kep: http://kep.k8s.io/2887
	// alpha: v1.23
	// beta: v1.24
	//
	// Enables populating "enum" field of OpenAPI schemas
	// in the spec returned from kube-apiserver.
	OpenAPIEnums featuregate.Feature = "OpenAPIEnums"

	// owner: @cici37
	// kep: http://kep.k8s.io/2876
	// alpha: v1.23
	//
	// Enables expression validation for Custom Resource
	CustomResourceValidationExpressions featuregate.Feature = "CustomResourceValidationExpressions"

	// owner: @jefftree
	// kep: http://kep.k8s.io/2896
	// alpha: v1.23
	// beta: v1.24
	//
	// Enables kubernetes to publish OpenAPI v3
	OpenAPIV3 featuregate.Feature = "OpenAPIV3"

	// owner: @kevindelgado
	// kep: http://kep.k8s.io/2885
	// alpha: v1.23
	//
	// Enables server-side field validation.
	ServerSideFieldValidation featuregate.Feature = "ServerSideFieldValidation"
)

func init() {
	runtime.Must(utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates))
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{
	AdvancedAuditing:                    {Default: true, PreRelease: featuregate.GA},
	APIResponseCompression:              {Default: true, PreRelease: featuregate.Beta},
	APIListChunking:                     {Default: true, PreRelease: featuregate.Beta},
	DryRun:                              {Default: true, PreRelease: featuregate.GA},
	RemainingItemCount:                  {Default: true, PreRelease: featuregate.Beta},
	ServerSideApply:                     {Default: true, PreRelease: featuregate.GA},
	StorageVersionHash:                  {Default: true, PreRelease: featuregate.Beta},
	StorageVersionAPI:                   {Default: false, PreRelease: featuregate.Alpha},
	WatchBookmark:                       {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	APIPriorityAndFairness:              {Default: true, PreRelease: featuregate.Beta},
	RemoveSelfLink:                      {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	SelectorIndex:                       {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	EfficientWatchResumption:            {Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	APIServerIdentity:                   {Default: false, PreRelease: featuregate.Alpha},
	APIServerTracing:                    {Default: false, PreRelease: featuregate.Alpha},
	OpenAPIEnums:                        {Default: true, PreRelease: featuregate.Beta},
	CustomResourceValidationExpressions: {Default: false, PreRelease: featuregate.Alpha},
	OpenAPIV3:                           {Default: true, PreRelease: featuregate.Beta},
	ServerSideFieldValidation:           {Default: false, PreRelease: featuregate.Alpha},
}
