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

package storagecompression

import (
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

var anyGroupAnyResource = schema.GroupResource{Group: "*", Resource: "*"}

// selectorRules holds what differs between EncryptionConfiguration's selector
// checks and compression's: the messages, which name the feature, and the
// resources that wildcards such as '*.' and '*.*' never select (core secrets for
// compression, none for encryption). The rest of this file applies the same rules
// as validateResourceNames and validateResourceOverlap in validation_encryption.go,
// but reports each problem once, on the selector at fault.
type selectorRules struct {
	bareWildcardErr string
	nonRESTErr      string
	extensionsErr   string
	acrossGroupsErr string
	overlapErr      string
	// notSelectedByWildcards holds resources no wildcard selects, so they can sit
	// beside a wildcard without overlapping it.
	notSelectedByWildcards sets.Set[schema.GroupResource]
}

// validate runs ValidateSelectors' checks with r's messages and exceptions, so
// another configuration can reuse them with its own rules.
func (r selectorRules) validate(resources []string, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	earlier := make([]schema.GroupResource, 0, len(resources))
	for j, res := range resources {
		if err := r.validateName(res, fldPath.Index(j)); err != nil {
			allErrs = append(allErrs, err)
			continue
		}
		gr := schema.ParseGroupResource(res)
		switch {
		case slices.Contains(earlier, gr):
			allErrs = append(allErrs, field.Duplicate(fldPath.Index(j), res))
		case slices.ContainsFunc(earlier, func(e schema.GroupResource) bool { return r.overlap(e, gr) }):
			allErrs = append(allErrs, field.Invalid(fldPath.Index(j), res, r.overlapErr).WithOrigin("overlap"))
		}
		earlier = append(earlier, gr)
	}
	return allErrs
}

// validateName applies EncryptionConfiguration's resource name rules, in the
// same order, and returns the first one res breaks.
func (r selectorRules) validateName(res string, p *field.Path) *field.Error {
	gr := schema.ParseGroupResource(res)
	switch {
	case strings.ToLower(res) != res:
		return field.Invalid(p, res, "resource name should not contain capital letters").WithOrigin("lowercase")
	case res == "*":
		return field.Invalid(p, res, r.bareWildcardErr).WithOrigin("bareWildcard")
	case res == "apiserveripinfo" || res == "serviceipallocations" || res == "servicenodeportallocations":
		return field.Invalid(p, res, r.nonRESTErr).WithOrigin("nonRESTResource")
	case gr.Group == "events.k8s.io":
		return field.Invalid(p, res, "'*.events.k8s.io' objects are stored using the 'events' API group in etcd. Use 'events' instead in the config file").WithOrigin("eventsGroup")
	case gr.Group == "extensions":
		return field.Invalid(p, res, r.extensionsErr).WithOrigin("extensionsGroup")
	case gr.Group == "*" && gr.Resource != "*":
		return field.Invalid(p, res, r.acrossGroupsErr).WithOrigin("acrossGroups")
	}
	return nil
}

// overlap reports whether two different selectors can select the same resource.
func (r selectorRules) overlap(a, b schema.GroupResource) bool {
	// A resource that no wildcard selects can sit beside any wildcard.
	if r.notSelectedByWildcards.Has(a) || r.notSelectedByWildcards.Has(b) {
		return false
	}
	// '*.*' selects every resource in every group.
	if a == anyGroupAnyResource || b == anyGroupAnyResource {
		return true
	}
	// Otherwise only '*.<group>' overlaps, and only with selectors in that group.
	return a.Group == b.Group && (a.Resource == "*" || b.Resource == "*")
}
