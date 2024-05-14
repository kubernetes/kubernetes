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

package cel

import (
	"strings"
	"testing"

	resourceapi "k8s.io/api/resource/v1alpha2"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apiserver/pkg/cel/environment"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func TestCompile(t *testing.T) {
	for name, scenario := range map[string]struct {
		expression         string
		attributes         []resourceapi.NamedResourcesAttribute
		expectCompileError string
		expectMatchError   string
		expectMatch        bool
	}{
		"true": {
			expression:  "true",
			expectMatch: true,
		},
		"false": {
			expression:  "false",
			expectMatch: false,
		},
		"syntax-error": {
			expression:         "?!",
			expectCompileError: "Syntax error",
		},
		"type-error": {
			expression:         `attributes.quantity["no-such-attr"]`,
			expectCompileError: "must evaluate to bool",
		},
		"runtime-error": {
			expression:       `attributes.quantity["no-such-attr"].isGreaterThan(quantity("0"))`,
			expectMatchError: "no such key: no-such-attr",
		},
		"quantity": {
			expression:  `attributes.quantity["name"].isGreaterThan(quantity("0"))`,
			attributes:  []resourceapi.NamedResourcesAttribute{{Name: "name", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{QuantityValue: ptr.To(resource.MustParse("1"))}}},
			expectMatch: true,
		},
		"bool": {
			expression:  `attributes.bool["name"]`,
			attributes:  []resourceapi.NamedResourcesAttribute{{Name: "name", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{BoolValue: ptr.To(true)}}},
			expectMatch: true,
		},
		"int": {
			expression:  `attributes.int["name"] > 0`,
			attributes:  []resourceapi.NamedResourcesAttribute{{Name: "name", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{IntValue: ptr.To(int64(1))}}},
			expectMatch: true,
		},
		"intslice": {
			expression:  `attributes.intslice["name"].isSorted() && attributes.intslice["name"].indexOf(3) == 2`,
			attributes:  []resourceapi.NamedResourcesAttribute{{Name: "name", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{IntSliceValue: &resourceapi.NamedResourcesIntSlice{Ints: []int64{1, 2, 3}}}}},
			expectMatch: true,
		},
		"empty-intslice": {
			expression:  `size(attributes.intslice["name"]) == 0`,
			attributes:  []resourceapi.NamedResourcesAttribute{{Name: "name", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{IntSliceValue: &resourceapi.NamedResourcesIntSlice{}}}},
			expectMatch: true,
		},
		"string": {
			expression:  `attributes.string["name"] == "fish"`,
			attributes:  []resourceapi.NamedResourcesAttribute{{Name: "name", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{StringValue: ptr.To("fish")}}},
			expectMatch: true,
		},
		"stringslice": {
			expression:  `attributes.stringslice["name"].isSorted() && attributes.stringslice["name"].indexOf("a") == 0`,
			attributes:  []resourceapi.NamedResourcesAttribute{{Name: "name", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{StringSliceValue: &resourceapi.NamedResourcesStringSlice{Strings: []string{"a", "b", "c"}}}}},
			expectMatch: true,
		},
		"empty-stringslice": {
			expression:  `size(attributes.stringslice["name"]) == 0`,
			attributes:  []resourceapi.NamedResourcesAttribute{{Name: "name", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{StringSliceValue: &resourceapi.NamedResourcesStringSlice{}}}},
			expectMatch: true,
		},
		"all": {
			expression: `attributes.quantity["quantity"].isGreaterThan(quantity("0")) &&
attributes.bool["bool"] &&
attributes.int["int"] > 0 &&
attributes.intslice["intslice"].isSorted() &&
attributes.string["string"] == "fish" &&
attributes.stringslice["stringslice"].isSorted()`,
			attributes: []resourceapi.NamedResourcesAttribute{
				{Name: "quantity", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{QuantityValue: ptr.To(resource.MustParse("1"))}},
				{Name: "bool", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{BoolValue: ptr.To(true)}},
				{Name: "int", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{IntValue: ptr.To(int64(1))}},
				{Name: "intslice", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{IntSliceValue: &resourceapi.NamedResourcesIntSlice{Ints: []int64{1, 2, 3}}}},
				{Name: "string", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{StringValue: ptr.To("fish")}},
				{Name: "stringslice", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{StringSliceValue: &resourceapi.NamedResourcesStringSlice{Strings: []string{"a", "b", "c"}}}},
			},
			expectMatch: true,
		},
		"many": {
			expression: `attributes.bool["a"] && attributes.bool["b"] && attributes.bool["c"]`,
			attributes: []resourceapi.NamedResourcesAttribute{
				{Name: "a", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{BoolValue: ptr.To(true)}},
				{Name: "b", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{BoolValue: ptr.To(true)}},
				{Name: "c", NamedResourcesAttributeValue: resourceapi.NamedResourcesAttributeValue{BoolValue: ptr.To(true)}},
			},
			expectMatch: true,
		},
	} {
		t.Run(name, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)
			result := Compiler.CompileCELExpression(scenario.expression, environment.StoredExpressions)
			if scenario.expectCompileError != "" && result.Error == nil {
				t.Fatalf("expected compile error %q, got none", scenario.expectCompileError)
			}
			if result.Error != nil {
				if scenario.expectCompileError == "" {
					t.Fatalf("unexpected compile error: %v", result.Error)
				}
				if !strings.Contains(result.Error.Error(), scenario.expectCompileError) {
					t.Fatalf("expected compile error to contain %q, but got instead: %v", scenario.expectCompileError, result.Error)
				}
				return
			}
			match, err := result.Evaluate(ctx, scenario.attributes)
			if err != nil {
				if scenario.expectMatchError == "" {
					t.Fatalf("unexpected evaluation error: %v", err)
				}
				if !strings.Contains(err.Error(), scenario.expectMatchError) {
					t.Fatalf("expected evaluation error to contain %q, but got instead: %v", scenario.expectMatchError, err)
				}
				return
			}
			if match != scenario.expectMatch {
				t.Fatalf("expected result %v, got %v", scenario.expectMatch, match)
			}
		})
	}
}
