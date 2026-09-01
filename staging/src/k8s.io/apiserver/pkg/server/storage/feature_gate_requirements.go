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
// gate. A gate with no entry requires no API.
//
// Requirements are version-agnostic. A feature gate outlives the API versions its types are
// served from, so a requirement pinned to a version would break as the API graduates, and
// would be unsatisfiable under an --emulation-version predating that version.
type FeatureGateAPIRequirements map[featuregate.Feature][]schema.GroupResource

// Validate reports every enabled feature gate whose required resources are absent from served.
// Resolved against --runtime-config, --emulation-version, --emulation-forward-compatible and
// --runtime-config-emulation-forward-compatible, and filtered by the per-kind API lifecycle.
// A less resolved view, such as a merged ResourceConfig, reports resources as available that
// the server goes on to drop.
func (r FeatureGateAPIRequirements) Validate(gate featuregate.FeatureGate, served sets.Set[schema.GroupResource]) error {
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
		if len(missing) > 0 {
			errs = append(errs, fmt.Errorf("feature gate %s is enabled, but requires API resources that are not served: %s; enable them with --runtime-config or disable the feature gate",
				feature, strings.Join(missing, ", ")))
		}
	}

	return utilerrors.NewAggregate(errs)
}
