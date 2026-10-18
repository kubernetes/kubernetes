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

package validation

import (
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

const (
	eventsGroupErr       = "'*.events.k8s.io' objects are stored using the 'events' API group in etcd. Use 'events' instead in the config file"
	resourceNameErr      = "resource name should not contain capital letters"
	duplicateResourceErr = "the same resource cannot be specified multiple times"
)

// resourceRules holds what differs between configurations that select resources
// the way EncryptionConfiguration does: the error messages that name the feature,
// and the resources that no wildcard selects.
type resourceRules struct {
	starResourceErr       string
	overlapErr            string
	nonRESTAPIResourceErr string
	extensionsGroupErr    string
	acrossGroupErr        string
	// notSelectedByWildcards holds resources no wildcard selects, so they can sit
	// beside a wildcard without overlapping it.
	notSelectedByWildcards sets.Set[schema.GroupResource]
}

var anyGroupAnyResource = schema.GroupResource{
	Group:    "*",
	Resource: "*",
}

func validateResourceOverlap(resources []string, fieldPath *field.Path, rules resourceRules) field.ErrorList {
	if len(resources) < 2 { // cannot have overlap with a single resource
		return nil
	}

	var allErrs field.ErrorList

	r := make([]schema.GroupResource, 0, len(resources))
	for _, resource := range resources {
		r = append(r, schema.ParseGroupResource(resource))
	}

	var hasOverlap, hasDuplicate bool

	for i, r1 := range r {
		for j, r2 := range r {
			if i == j {
				continue
			}

			if r1 == r2 && !hasDuplicate {
				hasDuplicate = true
				continue
			}

			if hasOverlap {
				continue
			}

			if rules.notSelectedByWildcards.Has(r1) || rules.notSelectedByWildcards.Has(r2) {
				continue
			}

			if r1 == anyGroupAnyResource {
				hasOverlap = true
				continue
			}

			if r1.Group != r2.Group {
				continue
			}

			if r1.Resource == "*" || r2.Resource == "*" {
				hasOverlap = true
				continue
			}
		}
	}

	if hasDuplicate {
		allErrs = append(
			allErrs,
			field.Invalid(
				fieldPath,
				resources,
				duplicateResourceErr,
			),
		)
	}

	if hasOverlap {
		allErrs = append(
			allErrs,
			field.Invalid(
				fieldPath,
				resources,
				rules.overlapErr,
			),
		)
	}

	return allErrs
}

func validateResourceNames(resources []string, fieldPath *field.Path, rules resourceRules) field.ErrorList {
	var allErrs field.ErrorList

	for j, res := range resources {
		jj := fieldPath.Index(j)

		// check if resource name has capital letters
		if hasCapital(res) {
			allErrs = append(
				allErrs,
				field.Invalid(
					jj,
					resources[j],
					resourceNameErr,
				),
			)
			continue
		}

		// check if resource is '*'
		if res == "*" {
			allErrs = append(
				allErrs,
				field.Invalid(
					jj,
					resources[j],
					rules.starResourceErr,
				),
			)
			continue
		}

		// check if resource is:
		// 'apiserveripinfo' OR
		// 'serviceipallocations' OR
		// 'servicenodeportallocations' OR
		if res == "apiserveripinfo" ||
			res == "serviceipallocations" ||
			res == "servicenodeportallocations" {
			allErrs = append(
				allErrs,
				field.Invalid(
					jj,
					resources[j],
					rules.nonRESTAPIResourceErr,
				),
			)
			continue
		}

		// check if group is 'events.k8s.io'
		gr := schema.ParseGroupResource(res)
		if gr.Group == "events.k8s.io" {
			allErrs = append(
				allErrs,
				field.Invalid(
					jj,
					resources[j],
					eventsGroupErr,
				),
			)
			continue
		}

		// check if group is 'extensions'
		if gr.Group == "extensions" {
			allErrs = append(
				allErrs,
				field.Invalid(
					jj,
					resources[j],
					rules.extensionsGroupErr,
				),
			)
			continue
		}

		// disallow resource.* as selecting the same resource across groups does not make sense
		if gr.Group == "*" && gr.Resource != "*" {
			allErrs = append(
				allErrs,
				field.Invalid(
					jj,
					resources[j],
					rules.acrossGroupErr,
				),
			)
			continue
		}
	}

	return allErrs
}

func hasCapital(input string) bool {
	return strings.ToLower(input) != input
}
