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
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	zpagesfeatures "k8s.io/component-base/zpages/features"
)

// Every feature gate should have an entry here following this template:
//
// // owner: @username
// MyFeature featuregate.Feature = "MyFeature"
//
// Feature gates should be listed in alphabetical, case-sensitive
// (upper before any lower case character) order. This reduces the risk
// of code conflicts because changes are more likely to be scattered
// across the file.
const (
	// owner: @ilackams
	//
	// Enables compression of REST responses (GET and LIST only)
	APIResponseCompression featuregate.Feature = "APIResponseCompression"

	// owner: @roycaihw
	//
	// Assigns each kube-apiserver an ID in a cluster.
	APIServerIdentity featuregate.Feature = "APIServerIdentity"

	// owner: @dashpole
	//
	// Add support for distributed tracing in the API Server
	APIServerTracing featuregate.Feature = "APIServerTracing"

	// owner: @linxiulei
	//
	// Enables serving watch requests in separate goroutines.
	APIServingWithRoutine featuregate.Feature = "APIServingWithRoutine"

	// owner: @jefftree
	//
	// Remove the v2beta1 apidiscovery.k8s.io/v2beta1 group version. Aggregated
	// discovery implements its own handlers and follows a different lifecycle than
	// traditional k8s resources.
	AggregatedDiscoveryRemoveBetaType featuregate.Feature = "AggregatedDiscoveryRemoveBetaType"

	// owner: @modulitos
	//
	// Allow user.DefaultInfo.UID to be set from x509 cert during cert auth.
	AllowParsingUserUIDFromCertAuth featuregate.Feature = "AllowParsingUserUIDFromCertAuth"

	// owner: @stlaz @tkashem @dgrisonnet
	// kep: https://kep.k8s.io/3926
	//
	// Enables the cluster admin to identify resources that fail to
	// decrypt or fail to be decoded into an object, and introduces
	// a new delete option to allow deletion of such corrupt
	// resources using the Kubernetes API only.
	AllowUnsafeMalformedObjectDeletion featuregate.Feature = "AllowUnsafeMalformedObjectDeletion"

	// owner: @vinayakankugoyal
	// kep: https://kep.k8s.io/4633
	//
	// Allows us to enable anonymous auth for only certain apiserver endpoints.
	AnonymousAuthConfigurableEndpoints featuregate.Feature = "AnonymousAuthConfigurableEndpoints"

	// owner: @deads2k
	// kep: https://kep.k8s.io/4601
	//
	// Allows authorization to use field and label selectors.
	AuthorizeWithSelectors featuregate.Feature = "AuthorizeWithSelectors"

	// owner: @serathius
	//
	// Replaces watch cache hashmap implementation with a btree based one, bringing performance improvements.
	BtreeWatchCache featuregate.Feature = "BtreeWatchCache"

	// owner: @benluddy
	// kep: https://kep.k8s.io/4222
	//
	// Enables CBOR as a supported encoding for requests and responses, and as the
	// preferred storage encoding for custom resources.
	CBORServingAndStorage featuregate.Feature = "CBORServingAndStorage"

	// owner: @serathius
	// Enables concurrent watch object decoding to avoid starving watch cache when conversion webhook is installed.
	ConcurrentWatchObjectDecode featuregate.Feature = "ConcurrentWatchObjectDecode"

	// owner: @serathius
	// kep: http://kep.k8s.io/2340
	//
	// Allow the API server to serve consistent lists from cache
	ConsistentListFromCache featuregate.Feature = "ConsistentListFromCache"

	// owner: @enj @qiujian16
	// kep: https://kep.k8s.io/5284
	//
	// Enables impersonation that is constrained to specific requests instead of being all or nothing.
	ConstrainedImpersonation featuregate.Feature = "ConstrainedImpersonation"

	// owner: @jefftree
	// kep: https://kep.k8s.io/4355
	//
	// Enables coordinated leader election in the API server
	CoordinatedLeaderElection featuregate.Feature = "CoordinatedLeaderElection"

	// owner: @jpbetz @aaron-prindle @yongruilin
	// kep: http://kep.k8s.io/5073
	// beta: v1.33
	//
	// Enables running declarative validation of APIs, where declared. When enabled, APIs with
	// declarative validation rules will validate objects using the generated
	// declarative validation code and compare the results to the regular imperative validation.
	// See DeclarativeValidationTakeover for more.
	DeclarativeValidation featuregate.Feature = "DeclarativeValidation"

	// owner: @jpbetz @aaron-prindle @yongruilin
	// kep: http://kep.k8s.io/5073
	// beta: v1.33
	//
	// When enabled, declarative validation errors are returned directly to the caller,
	// replacing hand-written validation errors for rules that have declarative implementations.
	// When disabled, hand-written validation errors are always returned, effectively putting
	// declarative validation in a "shadow mode" that monitors but does not affect API responses.
	// Note: Although declarative validation aims for functional equivalence with hand-written validation,
	// the exact number, format, and content of error messages may differ between the two approaches.
	DeclarativeValidationTakeover featuregate.Feature = "DeclarativeValidationTakeover"

	// owner: @serathius
	// kep: https://kep.k8s.io/4988
	//
	// Enabled cache inconsistency detection.
	DetectCacheInconsistency featuregate.Feature = "DetectCacheInconsistency"

	// owner: @aramase
	// kep: https://kep.k8s.io/3299
	// deprecated: v1.28
	//
	// Enables KMS v1 API for encryption at rest.
	KMSv1 featuregate.Feature = "KMSv1"

	// owner: @serathius
	// kep: https://kep.k8s.io/4988
	//
	// Enables generating snapshots of watch cache store and using them to serve LIST requests.
	ListFromCacheSnapshot featuregate.Feature = "ListFromCacheSnapshot"

	// owner: @alexzielenski, @cici37, @jiahuif, @jpbetz
	// kep: https://kep.k8s.io/3962
	//
	// Enables the MutatingAdmissionPolicy in Admission Chain
	MutatingAdmissionPolicy featuregate.Feature = "MutatingAdmissionPolicy"

	// owner: @jiahuif
	// kep: https://kep.k8s.io/2887
	//
	// Enables populating "enum" field of OpenAPI schemas
	// in the spec returned from kube-apiserver.
	OpenAPIEnums featuregate.Feature = "OpenAPIEnums"

	// owner: @stlaz
	//
	// Enable kube-apiserver to accept UIDs via request header authentication.
	// This will also make the kube-apiserver's API aggregator add UIDs via standard
	// headers when forwarding requests to the servers serving the aggregated API.
	RemoteRequestHeaderUID featuregate.Feature = "RemoteRequestHeaderUID"

	// owner: @wojtek-t
	//
	// Enables resilient watchcache initialization to avoid controlplane
	// overload.
	ResilientWatchCacheInitialization featuregate.Feature = "ResilientWatchCacheInitialization"

	// owner: @jpbetz
	// Resource create requests using generateName are retried automatically by the apiserver
	// if the generated name conflicts with an existing resource name, up to a maximum number of 7 retries.
	RetryGenerateName featuregate.Feature = "RetryGenerateName"

	// owner: @cici37
	//
	// Allow watch cache to create a watch on a dedicated RPC.
	// This prevents watch cache from being starved by other watches.
	SeparateCacheWatchRPC featuregate.Feature = "SeparateCacheWatchRPC"

	// owner: @serathius
	//
	// Enables APF to use size of objects for estimating request cost.
	SizeBasedListCostEstimate featuregate.Feature = "SizeBasedListCostEstimate"

	// owner: @caesarxuchao @roycaihw
	//
	// Enable the storage version API.
	StorageVersionAPI featuregate.Feature = "StorageVersionAPI"

	// owner: @caesarxuchao
	//
	// Allow apiservers to expose the storage version hash in the discovery
	// document.
	StorageVersionHash featuregate.Feature = "StorageVersionHash"

	// owner: @serathius
	// Allow API server JSON encoder to encode collections item by item, instead of all at once.
	StreamingCollectionEncodingToJSON featuregate.Feature = "StreamingCollectionEncodingToJSON"

	// owner: @serathius
	// Allow API server Protobuf encoder to encode collections item by item, instead of all at once.
	StreamingCollectionEncodingToProtobuf featuregate.Feature = "StreamingCollectionEncodingToProtobuf"

	// owner: @aramase, @enj, @nabokihms
	// kep: https://kep.k8s.io/3331
	//
	// Enables Structured Authentication Configuration
	StructuredAuthenticationConfiguration featuregate.Feature = "StructuredAuthenticationConfiguration"

	// owner: @aramase, @enj, @nabokihms
	// kep: https://kep.k8s.io/3331
	//
	// Enables Egress Selector in Structured Authentication Configuration
	StructuredAuthenticationConfigurationEgressSelector featuregate.Feature = "StructuredAuthenticationConfigurationEgressSelector"

	// owner: @aramase, @enj, @nabokihms
	// kep: https://kep.k8s.io/3331
	//
	// Enables JWKs metrics for Structured Authentication Configuration
	StructuredAuthenticationConfigurationJWKSMetrics featuregate.Feature = "StructuredAuthenticationConfigurationJWKSMetrics"

	// owner: @palnabarun
	// kep: https://kep.k8s.io/3221
	//
	// Enables Structured Authorization Configuration
	StructuredAuthorizationConfiguration featuregate.Feature = "StructuredAuthorizationConfiguration"

	// owner: @aramase
	//
	// Enables validation of service account UID in TokenRequest API.
	//
	// This feature gate is used to ensure that the UID provided in the TokenRequest
	// matches the UID of the service account for which the token is being requested.
	// It helps prevent misuse of the TokenRequest API by ensuring that tokens are only
	// issued for the correct service account.
	TokenRequestServiceAccountUIDValidation featuregate.Feature = "TokenRequestServiceAccountUIDValidation"

	// owner: @enj
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

	// owner: @richabanker
	//
	// Proxies client to an apiserver capable of serving the request in the event of version skew.
	UnknownVersionInteroperabilityProxy featuregate.Feature = "UnknownVersionInteroperabilityProxy"

	// owner: @wojtek-t
	//
	// Enables post-start-hook for storage readiness
	WatchCacheInitializationPostStartHook featuregate.Feature = "WatchCacheInitializationPostStartHook"

	// owner: @serathius
	// Enables watches without resourceVersion to be served from storage.
	// Used to prevent https://github.com/kubernetes/kubernetes/issues/123072 until etcd fixes the issue.
	WatchFromStorageWithoutResourceVersion featuregate.Feature = "WatchFromStorageWithoutResourceVersion"

	// owner: @p0lyn0mial
	//
	// Allow the API server to stream individual items instead of chunking
	WatchList featuregate.Feature = "WatchList"
)

func init() {
	runtime.Must(utilfeature.DefaultMutableFeatureGate.AddVersioned(defaultVersionedKubernetesFeatureGates))
	runtime.Must(zpagesfeatures.AddFeatureGates(utilfeature.DefaultMutableFeatureGate))
}

// defaultVersionedKubernetesFeatureGates consists of all known Kubernetes-specific feature keys with VersionedSpecs.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
//
// Entries are alphabetized and separated from each other with blank lines to avoid sweeping gofmt changes
// when adding or removing one entry.
var defaultVersionedKubernetesFeatureGates = map[featuregate.Feature]featuregate.VersionedSpecs{
	APIResponseCompression: {
		{Version: version.MustParse("1.8"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.16"), Default: true, PreRelease: featuregate.Beta},
	},

	APIServerIdentity: {
		{Version: version.MustParse("1.20"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.26"), Default: true, PreRelease: featuregate.Beta},
	},

	APIServerTracing: {
		{Version: version.MustParse("1.22"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.27"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	APIServingWithRoutine: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
	},

	AggregatedDiscoveryRemoveBetaType: {
		{Version: version.MustParse("1.0"), Default: false, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Deprecated},
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Deprecated, LockToDefault: true},
	},

	AllowParsingUserUIDFromCertAuth: {
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	AllowUnsafeMalformedObjectDeletion: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
	},

	AnonymousAuthConfigurableEndpoints: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	AuthorizeWithSelectors: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // remove in 1.37
	},

	BtreeWatchCache: {
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	CBORServingAndStorage: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
	},

	ConcurrentWatchObjectDecode: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
	},

	ConsistentListFromCache: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	ConstrainedImpersonation: {
		{Version: version.MustParse("1.35"), Default: false, PreRelease: featuregate.Alpha},
	},

	CoordinatedLeaderElection: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
	},

	DeclarativeValidation: {
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	DeclarativeValidationTakeover: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
	},

	DetectCacheInconsistency: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	KMSv1: {
		{Version: version.MustParse("1.0"), Default: true, PreRelease: featuregate.GA},
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Deprecated},
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Deprecated},
	},

	ListFromCacheSnapshot: {
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	MutatingAdmissionPolicy: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.34"), Default: false, PreRelease: featuregate.Beta},
	},

	OpenAPIEnums: {
		{Version: version.MustParse("1.23"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.24"), Default: true, PreRelease: featuregate.Beta},
	},

	RemoteRequestHeaderUID: {
		{Version: version.MustParse("1.32"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
	},

	ResilientWatchCacheInitialization: {
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	RetryGenerateName: {
		{Version: version.MustParse("1.30"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.31"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, LockToDefault: true, PreRelease: featuregate.GA},
	},

	SeparateCacheWatchRPC: {
		{Version: version.MustParse("1.28"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Deprecated},
	},

	SizeBasedListCostEstimate: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	StorageVersionAPI: {
		{Version: version.MustParse("1.20"), Default: false, PreRelease: featuregate.Alpha},
	},

	StorageVersionHash: {
		{Version: version.MustParse("1.14"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.15"), Default: true, PreRelease: featuregate.Beta},
	},

	StreamingCollectionEncodingToJSON: {
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	StreamingCollectionEncodingToProtobuf: {
		{Version: version.MustParse("1.33"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	StructuredAuthenticationConfiguration: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.GA, LockToDefault: true}, // GA and LockToDefault in 1.34, remove in 1.37
	},

	StructuredAuthenticationConfigurationEgressSelector: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	StructuredAuthenticationConfigurationJWKSMetrics: {
		{Version: version.MustParse("1.35"), Default: true, PreRelease: featuregate.Beta},
	},

	StructuredAuthorizationConfiguration: {
		{Version: version.MustParse("1.29"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.30"), Default: true, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.GA, LockToDefault: true},
	},

	TokenRequestServiceAccountUIDValidation: {
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},

	UnauthenticatedHTTP2DOSMitigation: {
		{Version: version.MustParse("1.25"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.29"), Default: true, PreRelease: featuregate.Beta},
	},

	UnknownVersionInteroperabilityProxy: {
		{Version: version.MustParse("1.28"), Default: false, PreRelease: featuregate.Alpha},
	},

	WatchCacheInitializationPostStartHook: {
		{Version: version.MustParse("1.31"), Default: false, PreRelease: featuregate.Beta},
	},

	WatchFromStorageWithoutResourceVersion: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Deprecated, LockToDefault: true},
	},

	WatchList: {
		{Version: version.MustParse("1.27"), Default: false, PreRelease: featuregate.Alpha},
		{Version: version.MustParse("1.32"), Default: true, PreRelease: featuregate.Beta},
		// switch this back to false because the json and proto streaming encoders appear to work better.
		{Version: version.MustParse("1.33"), Default: false, PreRelease: featuregate.Beta},
		{Version: version.MustParse("1.34"), Default: true, PreRelease: featuregate.Beta},
	},
}
