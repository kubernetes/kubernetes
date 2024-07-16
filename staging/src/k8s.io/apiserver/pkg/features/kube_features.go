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
	// MyFeature featuregate.Feature = "MyFeature"
	//
	// Feature gates should be listed in alphabetical, case-sensitive
	// (upper before any lower case character) order. This reduces the risk
	// of code conflicts because changes are more likely to be scattered
	// across the file.

	// owner: @ivelichkovich, @tallclair
	// alpha: v1.27
	// beta: v1.28
	// stable: v1.30
	// kep: https://kep.k8s.io/3716
	//
	// Enables usage of MatchConditions fields to use CEL expressions for matching on admission webhooks
	AdmissionWebhookMatchConditions featuregate.Feature = "AdmissionWebhookMatchConditions"

	// owner: @jefftree @alexzielenski
	// alpha: v1.26
	// beta: v1.27
	// stable: v1.30
	//
	// Enables an single HTTP endpoint /discovery/<version> which supports native HTTP
	// caching with ETags containing all APIResources known to the apiserver.
	AggregatedDiscoveryEndpoint featuregate.Feature = "AggregatedDiscoveryEndpoint"

	// owner: @vinayakankugoyal
	// kep: https://kep.k8s.io/4633
	// alpha: v1.31
	//
	// Allows us to enable anonymous auth for only certain apiserver endpoints.
	AnonymousAuthConfigurableEndpoints featuregate.Feature = "AnonymousAuthConfigurableEndpoints"

	// owner: @smarterclayton
	// alpha: v1.8
	// beta: v1.9
	// stable: 1.29
	//
	// Allow API clients to retrieve resource lists in chunks rather than
	// all at once.
	APIListChunking featuregate.Feature = "APIListChunking"

	// owner: @ilackams
	// alpha: v1.7
	// beta: v1.16
	//
	// Enables compression of REST responses (GET and LIST only)
	APIResponseCompression featuregate.Feature = "APIResponseCompression"

	// owner: @roycaihw
	// alpha: v1.20
	//
	// Assigns each kube-apiserver an ID in a cluster.
	APIServerIdentity featuregate.Feature = "APIServerIdentity"

	// owner: @dashpole
	// alpha: v1.22
	// beta: v1.27
	//
	// Add support for distributed tracing in the API Server
	APIServerTracing featuregate.Feature = "APIServerTracing"

	// owner: @linxiulei
	// beta: v1.30
	//
	// Enables serving watch requests in separate goroutines.
	APIServingWithRoutine featuregate.Feature = "APIServingWithRoutine"

	// owner: @cici37 @jpbetz
	// kep: http://kep.k8s.io/3488
	// alpha: v1.26
	// beta: v1.28
	// stable: v1.30
	//
	// Note: the feature gate can be removed in 1.32
	// Enables expression validation in Admission Control
	ValidatingAdmissionPolicy featuregate.Feature = "ValidatingAdmissionPolicy"

	// alpha: v1.20
	// beta: v1.21
	// GA: v1.24
	//
	// Allows for updating watchcache resource version with progress notify events.
	EfficientWatchResumption featuregate.Feature = "EfficientWatchResumption"

	// owner: @aramase
	// kep: https://kep.k8s.io/3299
	// deprecated: v1.28
	//
	// Enables KMS v1 API for encryption at rest.
	KMSv1 featuregate.Feature = "KMSv1"

	// owner: @aramase
	// kep: https://kep.k8s.io/3299
	// alpha: v1.25
	// beta: v1.27
	// stable: v1.29
	//
	// Enables KMS v2 API for encryption at rest.
	KMSv2 featuregate.Feature = "KMSv2"

	// owner: @enj
	// kep: https://kep.k8s.io/3299
	// beta: v1.28
	// stable: v1.29
	//
	// Enables the use of derived encryption keys with KMS v2.
	KMSv2KDF featuregate.Feature = "KMSv2KDF"

	// owner: @alexzielenski, @cici37, @jiahuif
	// kep: https://kep.k8s.io/3962
	// alpha: v1.30
	//
	// Enables the MutatingAdmissionPolicy in Admission Chain
	MutatingAdmissionPolicy featuregate.Feature = "MutatingAdmissionPolicy"

	// owner: @jiahuif
	// kep: https://kep.k8s.io/2887
	// alpha: v1.23
	// beta: v1.24
	//
	// Enables populating "enum" field of OpenAPI schemas
	// in the spec returned from kube-apiserver.
	OpenAPIEnums featuregate.Feature = "OpenAPIEnums"

	// owner: @caesarxuchao
	// alpha: v1.15
	// beta: v1.16
	// stable: 1.29
	//
	// Allow apiservers to show a count of remaining items in the response
	// to a chunking list request.
	RemainingItemCount featuregate.Feature = "RemainingItemCount"

	// owner: @wojtek-t
	// beta: v1.31
	//
	// Enables resilient watchcache initialization to avoid controlplane
	// overload.
	ResilientWatchCacheInitialization featuregate.Feature = "ResilientWatchCacheInitialization"

	// owner: @serathius
	// beta: v1.30
	//
	// Allow watch cache to create a watch on a dedicated RPC.
	// This prevents watch cache from being starved by other watches.
	SeparateCacheWatchRPC featuregate.Feature = "SeparateCacheWatchRPC"

	// owner: @apelisse, @lavalamp
	// alpha: v1.14
	// beta: v1.16
	// stable: v1.22
	//
	// Server-side apply. Merging happens on the server.
	ServerSideApply featuregate.Feature = "ServerSideApply"

	// owner: @kevindelgado
	// kep: https://kep.k8s.io/2885
	// alpha: v1.23
	// beta: v1.24
	//
	// Enables server-side field validation.
	ServerSideFieldValidation featuregate.Feature = "ServerSideFieldValidation"

	// owner: @enj
	// beta: v1.29
	//
	// Enables http2 DOS mitigations for unauthenticated clients.
	//
	// Some known reasons to disable these mitigations:
	//
	// An API server that is fronted by an L7 load balancer that is set up
	// to mitigate http2 attacks may opt to disable this protection to prevent
	// unauthenticated clients from disabling connection reuse between the load
	// balancer and the API server (many incoming connections could share the
	// same backend connection).
	//
	// An API server that is on a private network may opt to disable this
	// protection to prevent performance regressions for unauthenticated
	// clients.
	UnauthenticatedHTTP2DOSMitigation featuregate.Feature = "UnauthenticatedHTTP2DOSMitigation"

	// owner: @jpbetz
	// alpha: v1.30
	// Resource create requests using generateName are retried automatically by the apiserver
	// if the generated name conflicts with an existing resource name, up to a maximum number of 7 retries.
	RetryGenerateName featuregate.Feature = "RetryGenerateName"

	// owner: @cici37
	// alpha: v1.30
	//
	// StrictCostEnforcementForVAP is used to apply strict CEL cost validation for ValidatingAdmissionPolicy.
	// It will be set to off by default for certain time of period to prevent the impact on the existing users.
	// It is strongly recommended to enable this feature gate as early as possible.
	// The strict cost is specific for the extended libraries whose cost defined under k8s/apiserver/pkg/cel/library.
	StrictCostEnforcementForVAP featuregate.Feature = "StrictCostEnforcementForVAP"

	// owner: @cici37
	// alpha: v1.30
	//
	// StrictCostEnforcementForWebhooks is used to apply strict CEL cost validation for matchConditions in Webhooks.
	// It will be set to off by default for certain time of period to prevent the impact on the existing users.
	// It is strongly recommended to enable this feature gate as early as possible.
	// The strict cost is specific for the extended libraries whose cost defined under k8s/apiserver/pkg/cel/library.
	StrictCostEnforcementForWebhooks featuregate.Feature = "StrictCostEnforcementForWebhooks"

	// owner: @caesarxuchao @roycaihw
	// alpha: v1.20
	//
	// Enable the storage version API.
	StorageVersionAPI featuregate.Feature = "StorageVersionAPI"

	// owner: @caesarxuchao
	// alpha: v1.14
	// beta: v1.15
	//
	// Allow apiservers to expose the storage version hash in the discovery
	// document.
	StorageVersionHash featuregate.Feature = "StorageVersionHash"

	// owner: @aramase, @enj, @nabokihms
	// kep: https://kep.k8s.io/3331
	// alpha: v1.29
	// beta: v1.30
	//
	// Enables Structured Authentication Configuration
	StructuredAuthenticationConfiguration featuregate.Feature = "StructuredAuthenticationConfiguration"

	// owner: @palnabarun
	// kep: https://kep.k8s.io/3221
	// alpha: v1.29
	// beta: v1.30
	//
	// Enables Structured Authorization Configuration
	StructuredAuthorizationConfiguration featuregate.Feature = "StructuredAuthorizationConfiguration"

	// owner: @wojtek-t
	// alpha: v1.15
	// beta: v1.16
	// GA: v1.17
	//
	// Enables support for watch bookmark events.
	WatchBookmark featuregate.Feature = "WatchBookmark"

	// owner: @wojtek-t
	// beta: v1.31
	//
	// Enables post-start-hook for storage readiness
	WatchCacheInitializationPostStartHook featuregate.Feature = "WatchCacheInitializationPostStartHook"

	// owner: @serathius
	// beta: 1.30
	// Enables watches without resourceVersion to be served from storage.
	// Used to prevent https://github.com/kubernetes/kubernetes/issues/123072 until etcd fixes the issue.
	WatchFromStorageWithoutResourceVersion featuregate.Feature = "WatchFromStorageWithoutResourceVersion"

	// owner: @vinaykul
	// kep: http://kep.k8s.io/1287
	// alpha: v1.27
	//
	// Enables In-Place Pod Vertical Scaling
	InPlacePodVerticalScaling featuregate.Feature = "InPlacePodVerticalScaling"

	// owner: @p0lyn0mial
	// alpha: v1.27
	// beta: v1.31
	//
	// Allow the API server to stream individual items instead of chunking
	WatchList featuregate.Feature = "WatchList"

	// owner: @serathius
	// kep: http://kep.k8s.io/2340
	// alpha: v1.28
	// beta: v1.31
	//
	// Allow the API server to serve consistent lists from cache
	ConsistentListFromCache featuregate.Feature = "ConsistentListFromCache"

	// owner: @tkashem
	// beta: v1.29
	// GA: v1.30
	//
	// Allow Priority & Fairness in the API server to use a zero value for
	// the 'nominalConcurrencyShares' field of the 'limited' section of a
	// priority level.
	ZeroLimitedNominalConcurrencyShares featuregate.Feature = "ZeroLimitedNominalConcurrencyShares"
)

func init() {
	runtime.Must(utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates))
	runtime.Must(utilfeature.DefaultMutableFeatureGate.AddVersioned(defaultVersionedKubernetesFeatureGates))
}

// defaultVersionedKubernetesFeatureGates consists of all known Kubernetes-specific feature keys with VersionedSpecs.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	// Example:
	// EmulationVersion: {
	// 	{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	// },
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[featuregate.Feature]featuregate.FeatureSpec{

	AnonymousAuthConfigurableEndpoints: {Default: false, PreRelease: featuregate.Alpha},

	AggregatedDiscoveryEndpoint: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.33

	AdmissionWebhookMatchConditions: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.33

	APIListChunking: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32

	APIResponseCompression: {Default: true, PreRelease: featuregate.Beta},

	APIServerIdentity: {Default: true, PreRelease: featuregate.Beta},

	APIServerTracing: {Default: true, PreRelease: featuregate.Beta},

	APIServingWithRoutine: {Default: true, PreRelease: featuregate.Beta},

	ValidatingAdmissionPolicy: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32

	EfficientWatchResumption: {Default: true, PreRelease: featuregate.GA, LockToDefault: true},

	KMSv1: {Default: false, PreRelease: featuregate.Deprecated},

	KMSv2: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	KMSv2KDF: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.31

	OpenAPIEnums: {Default: true, PreRelease: featuregate.Beta},

	RemainingItemCount: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32

	ResilientWatchCacheInitialization: {Default: true, PreRelease: featuregate.Beta},

	RetryGenerateName: {Default: true, PreRelease: featuregate.Beta},

	SeparateCacheWatchRPC: {Default: true, PreRelease: featuregate.Beta},

	ServerSideApply: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	ServerSideFieldValidation: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.29

	StorageVersionAPI: {Default: false, PreRelease: featuregate.Alpha},

	StorageVersionHash: {Default: true, PreRelease: featuregate.Beta},

	StrictCostEnforcementForVAP: {Default: false, PreRelease: featuregate.Beta},

	StrictCostEnforcementForWebhooks: {Default: false, PreRelease: featuregate.Beta},

	StructuredAuthenticationConfiguration: {Default: true, PreRelease: featuregate.Beta},

	StructuredAuthorizationConfiguration: {Default: true, PreRelease: featuregate.Beta},

	UnauthenticatedHTTP2DOSMitigation: {Default: true, PreRelease: featuregate.Beta},

	WatchBookmark: {Default: true, PreRelease: featuregate.GA, LockToDefault: true},

	WatchCacheInitializationPostStartHook: {Default: false, PreRelease: featuregate.Beta},

	WatchFromStorageWithoutResourceVersion: {Default: false, PreRelease: featuregate.Beta},

	InPlacePodVerticalScaling: {Default: false, PreRelease: featuregate.Alpha},

	WatchList: {Default: true, PreRelease: featuregate.Beta},

	ConsistentListFromCache: {Default: true, PreRelease: featuregate.Beta},

	ZeroLimitedNominalConcurrencyShares: {Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.32
}
