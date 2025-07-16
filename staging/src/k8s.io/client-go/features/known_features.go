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

package features

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

	// owner: @benluddy
	// kep: https://kep.k8s.io/4222
	// alpha: 1.32
	//
	// If disabled, clients configured to accept "application/cbor" will instead accept
	// "application/json" with the same relative preference, and clients configured to write
	// "application/cbor" or "application/apply-patch+cbor" will instead write
	// "application/json" or "application/apply-patch+yaml", respectively.
	ClientsAllowCBOR Feature = "ClientsAllowCBOR"

	// owner: @benluddy
	// kep: https://kep.k8s.io/4222
	// alpha: 1.32
	//
	// If enabled, and only if ClientsAllowCBOR is also enabled, the default request content
	// type (if not explicitly configured) and the dynamic client's request content type both
	// become "application/cbor" instead of "application/json". The default content type for
	// apply patch requests becomes "application/apply-patch+cbor" instead of
	// "application/apply-patch+yaml".
	ClientsPreferCBOR Feature = "ClientsPreferCBOR"

	// owner: @nilekhc
	// alpha: v1.30
	InformerResourceVersion Feature = "InformerResourceVersion"

	// owner: @deads2k
	// beta: v1.33
	//
	// Refactor informers to deliver watch stream events in order instead of out of order.
	InOrderInformers Feature = "InOrderInformers"

	// owner: @p0lyn0mial
	// beta: v1.30
	//
	// Allow the client to get a stream of individual items instead of chunking from the server.
	//
	// NOTE:
	//  The feature is disabled in Beta by default because
	//  it will only be turned on for selected control plane component(s).
	WatchListClient Feature = "WatchListClient"
)

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
//
// To add a new feature, define a key for it above and add it here.
// After registering with the binary, the features are, by default, controllable using environment variables.
// For more details, please see envVarFeatureGates implementation.
var defaultKubernetesFeatureGates = map[Feature]FeatureSpec{
	ClientsAllowCBOR:        {Default: false, PreRelease: Alpha},
	ClientsPreferCBOR:       {Default: false, PreRelease: Alpha},
	InformerResourceVersion: {Default: false, PreRelease: Alpha},
	InOrderInformers:        {Default: true, PreRelease: Beta},
	WatchListClient:         {Default: false, PreRelease: Beta},
}
