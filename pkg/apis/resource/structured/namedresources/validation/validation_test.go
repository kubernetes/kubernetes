/*
Copyright 2022 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/validation/field"
	resourceapi "k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/utils/ptr"
)

func testResources(instances []resourceapi.NamedResourcesInstance) *resourceapi.NamedResourcesResources {
	resources := &resourceapi.NamedResourcesResources{
		Instances: instances,
	}
	return resources
}

func TestValidateResources(t *testing.T) {
	goodName := "foo"
	badName := "!@#$%^"
	quantity := resource.MustParse("1")

	scenarios := map[string]struct {
		resources    *resourceapi.NamedResourcesResources
		wantFailures field.ErrorList
	}{
		"empty": {
			resources: testResources(nil),
		},
		"good": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName}}),
		},
		"bad-name": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("instances").Index(0).Child("name"), badName, "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')")},
			resources:    testResources([]resourceapi.NamedResourcesInstance{{Name: badName}}),
		},
		"duplicate-name": {
			wantFailures: field.ErrorList{field.Duplicate(field.NewPath("instances").Index(1).Child("name"), goodName)},
			resources:    testResources([]resourceapi.NamedResourcesInstance{{Name: goodName}, {Name: goodName}}),
		},
		"quantity": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{QuantityValue: &quantity}}}}}),
		},
		"bool": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{BoolValue: ptr.To(true)}}}}}),
		},
		"int": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{IntValue: ptr.To(int64(1))}}}}}),
		},
		"int-slice": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{IntSliceValue: &resourceapi.NamedResourcesIntSlice{Ints: []int64{1, 2, 3}}}}}}}),
		},
		"string": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{StringValue: ptr.To("hello")}}}}}),
		},
		"string-slice": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{StringSliceValue: &resourceapi.NamedResourcesStringSlice{Strings: []string{"hello"}}}}}}}),
		},
		"version-okay": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To("1.0.0")}}}}}),
		},
		"version-beta": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To("1.0.0-beta")}}}}}),
		},
		"version-beta-1": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To("1.0.0-beta.1")}}}}}),
		},
		"version-build": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To("1.0.0+build")}}}}}),
		},
		"version-build-1": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To("1.0.0+build.1")}}}}}),
		},
		"version-beta-1-build-1": {
			resources: testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To("1.0.0-beta.1+build.1")}}}}}),
		},
		"version-bad": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("instances").Index(0).Child("attributes").Index(0).Child("version"), "1.0", "must be a string compatible with semver.org spec 2.0.0")},
			resources:    testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To("1.0")}}}}}),
		},
		"version-bad-leading-zeros": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("instances").Index(0).Child("attributes").Index(0).Child("version"), "01.0.0", "must be a string compatible with semver.org spec 2.0.0")},
			resources:    testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To("01.0.0")}}}}}),
		},
		"version-bad-leading-zeros-middle": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("instances").Index(0).Child("attributes").Index(0).Child("version"), "1.00.0", "must be a string compatible with semver.org spec 2.0.0")},
			resources:    testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To("1.00.0")}}}}}),
		},
		"version-bad-leading-zeros-end": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("instances").Index(0).Child("attributes").Index(0).Child("version"), "1.0.00", "must be a string compatible with semver.org spec 2.0.0")},
			resources:    testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To("1.0.00")}}}}}),
		},
		"version-bad-spaces": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("instances").Index(0).Child("attributes").Index(0).Child("version"), " 1.0.0 ", "must be a string compatible with semver.org spec 2.0.0")},
			resources:    testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{VersionValue: ptr.To(" 1.0.0 ")}}}}}),
		},
		"empty-attribute": {
			wantFailures: field.ErrorList{field.Required(field.NewPath("instances").Index(0).Child("attributes").Index(0), "exactly one value must be set")},
			resources:    testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName}}}}),
		},
		"duplicate-value": {
			wantFailures: field.ErrorList{field.Invalid(field.NewPath("instances").Index(0).Child("attributes").Index(0), []string{"bool", "int"}, "exactly one field must be set, not several")},
			resources:    testResources([]resourceapi.NamedResourcesInstance{{Name: goodName, Attributes: []resourceapi.NamedResourcesAttribute{{Name: goodName, NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{BoolValue: ptr.To(true), IntValue: ptr.To(int64(1))}}}}}),
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			errs := ValidateResources(scenario.resources, nil)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}

func TestValidateSelector(t *testing.T) {
	scenarios := map[string]struct {
		selector     string
		wantFailures field.ErrorList
	}{
		"okay": {
			selector: "true",
		},
		"empty": {
			selector:     "",
			wantFailures: field.ErrorList{field.Required(nil, "")},
		},
		"undefined": {
			selector:     "nosuchvar",
			wantFailures: field.ErrorList{field.Invalid(nil, "nosuchvar", "compilation failed: ERROR: <input>:1:1: undeclared reference to 'nosuchvar' (in container '')\n | nosuchvar\n | ^")},
		},
		"wrong-type": {
			selector:     "1",
			wantFailures: field.ErrorList{field.Invalid(nil, "1", "must evaluate to bool")},
		},
		"quantity": {
			selector: `attributes.quantity["name"].isGreaterThan(quantity("0"))`,
		},
		"bool": {
			selector: `attributes.bool["name"]`,
		},
		"int": {
			selector: `attributes.int["name"] > 0`,
		},
		"intslice": {
			selector: `attributes.intslice["name"].isSorted()`,
		},
		"string": {
			selector: `attributes.string["name"] == "fish"`,
		},
		"stringslice": {
			selector: `attributes.stringslice["name"].isSorted()`,
		},
		"version": {
			selector: `attributes.version["name"].isGreaterThan(semver("1.0.0"))`,
		},
	}

	for name, scenario := range scenarios {
		t.Run(name, func(t *testing.T) {
			// At the moment, there's no difference between stored and new expressions.
			// This uses the stricter validation.
			opts := Options{
				StoredExpressions: false,
			}
			errs := validateSelector(opts, scenario.selector, nil)
			assert.Equal(t, scenario.wantFailures, errs)
		})
	}
}
