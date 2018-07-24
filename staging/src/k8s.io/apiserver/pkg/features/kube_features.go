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

	// owner: @tallclair
	// alpha: v1.5
	//
	// StreamingProxyRedirects controls whether the apiserver should intercept (and follow)
	// redirects from the backend (Kubelet) for streaming requests (exec/attach/port-forward).
	StreamingProxyRedirects utilfeature.Feature = "StreamingProxyRedirects"

	// owner: @tallclair
	// alpha: v1.7
	// beta: v1.8
	//
	// AdvancedAuditing enables a much more general API auditing pipeline, which includes support for
	// pluggable output backends and an audit policy specifying how different requests should be
	// audited.
	AdvancedAuditing utilfeature.Feature = "AdvancedAuditing"

	// owner: @ilackams
	// alpha: v1.7
	//
	// Enables compression of REST responses (GET and LIST only)
	APIResponseCompression utilfeature.Feature = "APIResponseCompression"

	// owner: @smarterclayton
	// alpha: v1.7
	//
	// Allow asynchronous coordination of object creation.
	// Auto-enabled by the Initializers admission plugin.
	Initializers utilfeature.Feature = "Initializers"

	// owner: @smarterclayton
	// alpha: v1.8
	// beta: v1.9
	//
	// Allow API clients to retrieve resource lists in chunks rather than
	// all at once.
	APIListChunking utilfeature.Feature = "APIListChunking"

	// owner: @apelisse
	// alpha: v1.12
	//
	// Allow requests to be processed but not stored, so that
	// validation, merging, mutation can be tested without
	// committing.
	DryRun utilfeature.Feature = "DryRun"
)

func init() {
	utilfeature.DefaultFeatureGate.Add(defaultKubernetesFeatureGates)
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[utilfeature.Feature]utilfeature.FeatureSpec{
	StreamingProxyRedirects: {Default: true, PreRelease: utilfeature.Beta},
	AdvancedAuditing:        {Default: true, PreRelease: utilfeature.Beta},
	APIResponseCompression:  {Default: false, PreRelease: utilfeature.Alpha},
	Initializers:            {Default: false, PreRelease: utilfeature.Alpha},
	APIListChunking:         {Default: true, PreRelease: utilfeature.Beta},
	DryRun:                  {Default: false, PreRelease: utilfeature.Alpha},
}
