/*
Copyright The Kubernetes Authors.

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

package storage

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-base/featuregate"
)

// FeatureGateAPIRequirements declares the GroupResources that must be served per feature
// gate. A gate requires its resources, and a resource requires every gate it is listed under.
type FeatureGateAPIRequirements map[featuregate.Feature][]schema.GroupResource

// explicitFeatureGate is satisfied by featuregate.MutableVersionedFeatureGate, which is what
// the ComponentGlobalsRegistry hands out and what utilfeature.DefaultFeatureGate is.
type explicitFeatureGate interface {
	ExplicitlySet(name featuregate.Feature) bool
}

// explicitlyEnabled reports whether feature was enabled through --feature-gates rather than by
// default. A gate that cannot report explicitness is treated as if every feature were enabled
// by default, so that the validation never rejects a default configuration.
func explicitlyEnabled(gate featuregate.FeatureGate, feature featuregate.Feature) bool {
	explicit, ok := gate.(explicitFeatureGate)
	return ok && explicit.ExplicitlySet(feature) && gate.Enabled(feature)
}

// Validate checks every enabled feature gate against the resources that are served.
func (r FeatureGateAPIRequirements) Validate(gate featuregate.FeatureGate, served sets.Set[schema.GroupResource]) (warnings []string, err error) {
	var errs []error

	// Sorted so a misconfiguration touching several gates reports them in a stable order.
	for _, feature := range slices.Sorted(maps.Keys(r)) {
		if !gate.Enabled(feature) {
			continue
		}

		var missing []string
		for _, gr := range r[feature] {
			if !served.Has(gr) {
				missing = append(missing, gr.String())
			}
		}
		if len(missing) == 0 {
			continue
		}

		if explicitlyEnabled(gate, feature) {
			errs = append(errs, fmt.Errorf("feature gate %s is enabled, but requires API resources that are not served: %s; enable them with --runtime-config or disable the feature gate",
				feature, strings.Join(missing, ", ")))
		} else {
			warnings = append(warnings, fmt.Sprintf("feature gate %s is enabled by default, but requires API resources that are not served: %s; the feature is inactive until they are enabled with --runtime-config",
				feature, strings.Join(missing, ", ")))
		}
	}

	return warnings, utilerrors.NewAggregate(errs)
}

// gatesByResource maps each resource to the gates that list it, the reverse of the
// declaration. A resource listed under several gates requires all of them, matching the
// conjunction the storage providers use to guard it.
func (r FeatureGateAPIRequirements) gatesByResource() map[schema.GroupResource][]featuregate.Feature {
	ret := map[schema.GroupResource][]featuregate.Feature{}
	for _, feature := range slices.Sorted(maps.Keys(r)) {
		for _, gr := range r[feature] {
			ret[gr] = append(ret[gr], feature)
		}
	}
	return ret
}

// ValidateExplicitlyEnabledAPIs reports every required API resource that was explicitly
// enabled by name with --runtime-config (group/version/resource=true) while a feature gate it
// requires is disabled. The storage providers skip such a resource, so the explicit request
// could never be honoured.
func (r FeatureGateAPIRequirements) ValidateExplicitlyEnabledAPIs(gate featuregate.FeatureGate,
	cfg APIResourceConfigSource, registered sets.Set[schema.GroupVersionResource]) error {
	byResource := r.gatesByResource()

	type request struct {
		keys     sets.Set[string]
		disabled sets.Set[featuregate.Feature]
	}
	requests := map[schema.GroupResource]*request{}

	for _, gvr := range slices.SortedFunc(maps.Keys(registered), func(a, b schema.GroupVersionResource) int {
		return strings.Compare(a.String(), b.String())
	}) {
		gates, ok := byResource[gvr.GroupResource()]
		if !ok {
			continue
		}
		// Only enablement is what the storage providers consult, so a resource that is
		// explicitly enabled but ends up disabled anyway is not something the server was
		// going to serve or complain about.
		if !cfg.ResourceEnabled(gvr) || !cfg.ResourceExplicitlyEnabled(gvr) {
			continue
		}
		key := gvr.GroupVersion().String() + "/" + gvr.Resource

		for _, feature := range gates {
			if gate.Enabled(feature) {
				continue
			}
			req := requests[gvr.GroupResource()]
			if req == nil {
				req = &request{keys: sets.New[string](), disabled: sets.New[featuregate.Feature]()}
				requests[gvr.GroupResource()] = req
			}
			req.keys.Insert(key)
			req.disabled.Insert(feature)
		}
	}

	var errs []error
	for _, gr := range slices.SortedFunc(maps.Keys(requests), func(a, b schema.GroupResource) int {
		return strings.Compare(a.String(), b.String())
	}) {
		req := requests[gr]
		disabled := make([]string, 0, req.disabled.Len())
		for _, feature := range sets.List(req.disabled) {
			disabled = append(disabled, string(feature))
		}
		errs = append(errs, fmt.Errorf("API resource %s was explicitly enabled with --runtime-config (%s), but requires feature gates that are disabled: %s; enable them with --feature-gates or do not enable the API resource",
			gr, strings.Join(sets.List(req.keys), ", "), strings.Join(disabled, ", ")))
	}

	return utilerrors.NewAggregate(errs)
}
