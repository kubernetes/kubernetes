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

package endpointslice

import "k8s.io/endpointslice/topologycache"

type ReconcilerOption func(*Reconciler)

// WithTopologyCache tracks the distribution of Nodes and endpoints across zones
// to enable TopologyAwareHints.
func WithTopologyCache(topologyCache *topologycache.TopologyCache) ReconcilerOption {
	return func(r *Reconciler) {
		r.topologyCache = topologyCache
	}
}

// WithTrafficDistribution controls whether the Reconciler considers the
// `trafficDistribution` field while reconciling EndpointSlices.
func WithTrafficDistribution(enabled bool) ReconcilerOption {
	return func(r *Reconciler) {
		r.trafficDistributionEnabled = enabled
	}
}

// WithPlaceholder controls whether the Reconciler must set placeholder
// endpointslices or not.
func WithPlaceholder(enabled bool) ReconcilerOption {
	return func(r *Reconciler) {
		r.placeholderEnabled = enabled
	}
}

// WithOwnershipEnforced indicates, if set to true, that existing EndpointSlices
// passed as parameter of Reconcile that are not owned will be deleted as part of
// the reconciliation.
func WithOwnershipEnforced(enabled bool) ReconcilerOption {
	return func(r *Reconciler) {
		r.ownershipEnforced = enabled
	}
}
