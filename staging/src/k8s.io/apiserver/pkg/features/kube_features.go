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
	// beta: v1.6
	//
	// StreamingProxyRedirects controls whether the apiserver should intercept (and follow)
	// redirects from the backend (Kubelet) for streaming requests (exec/attach/port-forward).
	StreamingProxyRedirects utilfeature.Feature = "StreamingProxyRedirects"

	// owner: @tallclair
	// alpha: v1.10
	// beta: v1.14
	//
	// ValidateProxyRedirects controls whether the apiserver should validate that redirects are only
	// followed to the same host. Only used if StreamingProxyRedirects is enabled.
	ValidateProxyRedirects utilfeature.Feature = "ValidateProxyRedirects"

	// owner: @tallclair
	// alpha: v1.7
	// beta: v1.8
	// GA: v1.12
	//
	// AdvancedAuditing enables a much more general API auditing pipeline, which includes support for
	// pluggable output backends and an audit policy specifying how different requests should be
	// audited.
	AdvancedAuditing utilfeature.Feature = "AdvancedAuditing"

	// owner: @pbarker
	// alpha: v1.13
	//
	// DynamicAuditing enables configuration of audit policy and webhook backends through an
	// AuditSink API object.
	DynamicAuditing utilfeature.Feature = "DynamicAuditing"

	// owner: @ilackams
	// alpha: v1.7
	//
	// Enables compression of REST responses (GET and LIST only)
	APIResponseCompression utilfeature.Feature = "APIResponseCompression"

	// owner: @smarterclayton
	// alpha: v1.8
	// beta: v1.9
	//
	// Allow API clients to retrieve resource lists in chunks rather than
	// all at once.
	APIListChunking utilfeature.Feature = "APIListChunking"

	// owner: @apelisse
	// alpha: v1.12
	// beta: v1.13
	//
	// Allow requests to be processed but not stored, so that
	// validation, merging, mutation can be tested without
	// committing.
	DryRun utilfeature.Feature = "DryRun"

	// owner: @apelisse, @lavalamp
	// alpha: v1.14
	//
	// Server-side apply. Merging happens on the server.
	ServerSideApply utilfeature.Feature = "ServerSideApply"

	// owner: @ksubrmnn
	// alpha: v1.14
	//
	// Allows kube-proxy to run in Overlay mode for Windows
	WinOverlay utilfeature.Feature = "WinOverlay"

	// owner: @ksubrmnn
	// alpha: v1.14
	//
	// Allows kube-proxy to create DSR loadbalancers for Windows
	WinDSR utilfeature.Feature = "WinDSR"
)

func init() {
	utilfeature.DefaultMutableFeatureGate.Add(defaultKubernetesFeatureGates)
}

// defaultKubernetesFeatureGates consists of all known Kubernetes-specific feature keys.
// To add a new feature, define a key for it above and add it here. The features will be
// available throughout Kubernetes binaries.
var defaultKubernetesFeatureGates = map[utilfeature.Feature]utilfeature.FeatureSpec{
	StreamingProxyRedirects: {Default: true, PreRelease: utilfeature.Beta},
	ValidateProxyRedirects:  {Default: true, PreRelease: utilfeature.Beta},
	AdvancedAuditing:        {Default: true, PreRelease: utilfeature.GA},
	DynamicAuditing:         {Default: false, PreRelease: utilfeature.Alpha},
	APIResponseCompression:  {Default: false, PreRelease: utilfeature.Alpha},
	APIListChunking:         {Default: true, PreRelease: utilfeature.Beta},
	DryRun:                  {Default: true, PreRelease: utilfeature.Beta},
	ServerSideApply:         {Default: false, PreRelease: utilfeature.Alpha},
	WinOverlay:              {Default: false, PreRelease: utilfeature.Alpha},
	WinDSR:                  {Default: false, PreRelease: utilfeature.Alpha},
}
