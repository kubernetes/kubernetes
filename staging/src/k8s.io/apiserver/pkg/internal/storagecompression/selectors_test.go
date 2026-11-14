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
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestValidateSelectors(t *testing.T) {
	selector := func(j int) *field.Path { return field.NewPath("resources").Index(j) }
	for _, tc := range []struct {
		desc      string
		resources []string
		want      field.ErrorList
	}{{
		desc:      "named resources and a group wildcard",
		resources: []string{"events", "widgets.example.com", "*.coordination.k8s.io"},
	}, {
		desc:      "the core wildcard",
		resources: []string{"*."},
	}, {
		desc:      "the everything wildcard",
		resources: []string{"*.*"},
	}, {
		desc: "selectors that can't match a stored resource",
		resources: []string{
			"ConfigMaps", "*", "apiserveripinfo", "serviceipallocations", "servicenodeportallocations",
			"events.events.k8s.io", "*.events.k8s.io", "deployments.extensions", "configmaps.*",
		},
		want: field.ErrorList{
			field.Invalid(selector(0), nil, "").WithOrigin("lowercase"),
			field.Invalid(selector(1), nil, "").WithOrigin("bareWildcard"),
			field.Invalid(selector(2), nil, "").WithOrigin("nonRESTResource"),
			field.Invalid(selector(3), nil, "").WithOrigin("nonRESTResource"),
			field.Invalid(selector(4), nil, "").WithOrigin("nonRESTResource"),
			field.Invalid(selector(5), nil, "").WithOrigin("eventsGroup"),
			field.Invalid(selector(6), nil, "").WithOrigin("eventsGroup"),
			field.Invalid(selector(7), nil, "").WithOrigin("extensionsGroup"),
			field.Invalid(selector(8), nil, "").WithOrigin("acrossGroups"),
		},
	}, {
		desc:      "a resource beside its group wildcard",
		resources: []string{"configmaps", "*."},
		want:      field.ErrorList{field.Invalid(selector(1), nil, "").WithOrigin("overlap")},
	}, {
		desc:      "a group wildcard beside the everything wildcard",
		resources: []string{"*.*", "*.apps"},
		want:      field.ErrorList{field.Invalid(selector(1), nil, "").WithOrigin("overlap")},
	}, {
		desc:      "each overlap reported once, on the later selector",
		resources: []string{"*.apps", "*.", "deployments.apps", "*.*"},
		want: field.ErrorList{
			field.Invalid(selector(2), nil, "").WithOrigin("overlap"),
			field.Invalid(selector(3), nil, "").WithOrigin("overlap"),
		},
	}, {
		desc:      "an overlap with a selector other than the first",
		resources: []string{"configmaps", "deployments.apps", "*.apps"},
		want:      field.ErrorList{field.Invalid(selector(2), nil, "").WithOrigin("overlap")},
	}, {
		desc:      "a selector reported for an overlap still overlaps later ones",
		resources: []string{"configmaps", "*.", "pods"},
		want: field.ErrorList{
			field.Invalid(selector(1), nil, "").WithOrigin("overlap"),
			field.Invalid(selector(2), nil, "").WithOrigin("overlap"),
		},
	}, {
		desc:      "a rejected selector takes no part in overlaps",
		resources: []string{"*", "*.", "ConfigMaps", "configmaps"},
		want: field.ErrorList{
			field.Invalid(selector(0), nil, "").WithOrigin("bareWildcard"),
			field.Invalid(selector(2), nil, "").WithOrigin("lowercase"),
			field.Invalid(selector(3), nil, "").WithOrigin("overlap"),
		},
	}, {
		desc:      "core secrets beside wildcards, which never select it",
		resources: []string{"secrets", "*.", "*.apps", "*.*"},
		want:      field.ErrorList{field.Invalid(selector(3), nil, "").WithOrigin("overlap")},
	}, {
		desc:      "core secrets after the wildcards",
		resources: []string{"*.", "*.apps", "secrets"},
	}, {
		desc:      "secrets in another group is an ordinary resource",
		resources: []string{"*.apps", "secrets.apps"},
		want:      field.ErrorList{field.Invalid(selector(1), nil, "").WithOrigin("overlap")},
	}, {
		desc:      "every repeat of a resource, however it's spelled",
		resources: []string{"secrets", "secrets.", "secrets"},
		want:      field.ErrorList{field.Duplicate(selector(1), nil), field.Duplicate(selector(2), nil)},
	}, {
		desc:      "a repeated wildcard is only a duplicate",
		resources: []string{"*.apps", "*.apps"},
		want:      field.ErrorList{field.Duplicate(selector(1), nil)},
	}} {
		t.Run(tc.desc, func(t *testing.T) {
			testErrors(t, tc.want, ValidateSelectors(tc.resources, field.NewPath("resources")))
		})
	}
}
