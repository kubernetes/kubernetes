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

package apiserver

import (
	"fmt"
	"slices"

	"k8s.io/apimachinery/pkg/runtime/schema"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/component-base/featuregate"
)

// validateFeatureGateAPIDependencies ensures that every enabled feature gate which
// requires an API to be served has a serving version of each required resource enabled
// in the effective APIResourceConfigSource. It is fail-fast when an enabled feature whose
// API is not served is a misconfiguration.
//
// A requirement is evaluated per GroupResource. The resource must be served in at least
// one of the versions declared for it.
//
// resources must be the effective, fully-resolved config so the check reflects what
// kube-apiserver will actually serve.
func validateFeatureGateAPIDependencies(gate featuregate.FeatureGate,
	resources serverstorage.APIResourceConfigSource,
	deps map[featuregate.Feature][]schema.GroupVersionResource) error {
	var errs []error

	features := make([]featuregate.Feature, 0, len(deps))
	for feature := range deps {
		features = append(features, feature)
	}
	slices.Sort(features)

	for _, feature := range features {
		if !gate.Enabled(feature) {
			continue
		}

		// Group the required GVRs by GroupResource, preserving first-seen order.
		versionsByResource := map[schema.GroupResource][]string{}
		order := []schema.GroupResource{}
		for _, gvr := range deps[feature] {
			gr := gvr.GroupResource()
			if _, seen := versionsByResource[gr]; !seen {
				order = append(order, gr)
			}
			versionsByResource[gr] = append(versionsByResource[gr], gvr.Version)
		}

		for _, gr := range order {
			served := false
			for _, version := range versionsByResource[gr] {
				if resources.ResourceEnabled(gr.WithVersion(version)) {
					served = true
					break
				}
			}
			if !served {
				errs = append(errs, fmt.Errorf("%s is enabled, but requires an API resource that is not served: %s (none of %v enabled)", feature, groupResourceString(gr), versionsByResource[gr]))
			}
		}
	}

	return utilerrors.NewAggregate(errs)
}

// groupResourceString renders a GroupResource as group/resource, matching the
// style used by the --runtime-config flag.
func groupResourceString(gr schema.GroupResource) string {
	if gr.Group == "" {
		return gr.Resource
	}
	return gr.Group + "/" + gr.Resource
}
