/*
Copyright 2018 The Kubernetes Authors.

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

package generators

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/gengo/examples/set-gen/sets"
	"k8s.io/gengo/types"
)

func TestSingleTagExtension(t *testing.T) {

	// Comments only contain one tag extension and one value.
	var tests = []struct {
		comments        []string
		extensionTag    string
		extensionName   string
		extensionValues []string
	}{
		{
			comments:        []string{"+patchMergeKey=name"},
			extensionTag:    "patchMergeKey",
			extensionName:   "x-kubernetes-patch-merge-key",
			extensionValues: []string{"name"},
		},
		{
			comments:        []string{"+patchStrategy=merge"},
			extensionTag:    "patchStrategy",
			extensionName:   "x-kubernetes-patch-strategy",
			extensionValues: []string{"merge"},
		},
		{
			comments:        []string{"+listType=atomic"},
			extensionTag:    "listType",
			extensionName:   "x-kubernetes-list-type",
			extensionValues: []string{"atomic"},
		},
		{
			comments:        []string{"+listMapKey=port"},
			extensionTag:    "listMapKey",
			extensionName:   "x-kubernetes-list-map-keys",
			extensionValues: []string{"port"},
		},
		{
			comments:        []string{"+mapType=granular"},
			extensionTag:    "mapType",
			extensionName:   "x-kubernetes-map-type",
			extensionValues: []string{"granular"},
		},
		{
			comments:        []string{"+k8s:openapi-gen=x-kubernetes-member-tag:member_test"},
			extensionTag:    "k8s:openapi-gen",
			extensionName:   "x-kubernetes-member-tag",
			extensionValues: []string{"member_test"},
		},
		{
			comments:        []string{"+k8s:openapi-gen=x-kubernetes-member-tag:member_test:member_test2"},
			extensionTag:    "k8s:openapi-gen",
			extensionName:   "x-kubernetes-member-tag",
			extensionValues: []string{"member_test:member_test2"},
		},
		{
			// Test that poorly formatted extensions aren't added.
			comments: []string{
				"+k8s:openapi-gen=x-kubernetes-no-value",
				"+k8s:openapi-gen=x-kubernetes-member-success:success",
				"+k8s:openapi-gen=x-kubernetes-wrong-separator;error",
			},
			extensionTag:    "k8s:openapi-gen",
			extensionName:   "x-kubernetes-member-success",
			extensionValues: []string{"success"},
		},
	}
	for _, test := range tests {
		extensions, _ := parseExtensions(test.comments)
		actual := extensions[0]
		if actual.idlTag != test.extensionTag {
			t.Errorf("Extension Tag: expected (%s), actual (%s)\n", test.extensionTag, actual.idlTag)
		}
		if actual.xName != test.extensionName {
			t.Errorf("Extension Name: expected (%s), actual (%s)\n", test.extensionName, actual.xName)
		}
		if !reflect.DeepEqual(actual.values, test.extensionValues) {
			t.Errorf("Extension Values: expected (%s), actual (%s)\n", test.extensionValues, actual.values)
		}
		if actual.hasMultipleValues() {
			t.Errorf("%s: hasMultipleValues() should be false\n", actual.xName)
		}
	}

}

func TestMultipleTagExtensions(t *testing.T) {

	var tests = []struct {
		comments        []string
		extensionTag    string
		extensionName   string
		extensionValues []string
	}{
		{
			comments: []string{
				"+listMapKey=port",
				"+listMapKey=protocol",
			},
			extensionTag:    "listMapKey",
			extensionName:   "x-kubernetes-list-map-keys",
			extensionValues: []string{"port", "protocol"},
		},
	}
	for _, test := range tests {
		extensions, errors := parseExtensions(test.comments)
		if len(errors) > 0 {
			t.Errorf("Unexpected errors: %v\n", errors)
		}
		actual := extensions[0]
		if actual.idlTag != test.extensionTag {
			t.Errorf("Extension Tag: expected (%s), actual (%s)\n", test.extensionTag, actual.idlTag)
		}
		if actual.xName != test.extensionName {
			t.Errorf("Extension Name: expected (%s), actual (%s)\n", test.extensionName, actual.xName)
		}
		if !reflect.DeepEqual(actual.values, test.extensionValues) {
			t.Errorf("Extension Values: expected (%s), actual (%s)\n", test.extensionValues, actual.values)
		}
		if !actual.hasMultipleValues() {
			t.Errorf("%s: hasMultipleValues() should be true\n", actual.xName)
		}
	}

}

func TestExtensionParseErrors(t *testing.T) {

	var tests = []struct {
		comments     []string
		errorMessage string
	}{
		{
			// Missing extension value should be an error.
			comments: []string{
				"+k8s:openapi-gen=x-kubernetes-no-value",
			},
			errorMessage: "x-kubernetes-no-value",
		},
		{
			// Wrong separator should be an error.
			comments: []string{
				"+k8s:openapi-gen=x-kubernetes-wrong-separator;error",
			},
			errorMessage: "x-kubernetes-wrong-separator;error",
		},
	}

	for _, test := range tests {
		_, errors := parseExtensions(test.comments)
		if len(errors) == 0 {
			t.Errorf("Expected errors while parsing: %v\n", test.comments)
		}
		error := errors[0]
		if !strings.Contains(error.Error(), test.errorMessage) {
			t.Errorf("Error (%v) should contain substring (%s)\n", error, test.errorMessage)
		}
	}
}

func TestExtensionAllowedValues(t *testing.T) {

	var methodTests = []struct {
		e             extension
		allowedValues sets.String
	}{
		{
			e: extension{
				idlTag: "patchStrategy",
			},
			allowedValues: sets.NewString("merge", "retainKeys"),
		},
		{
			e: extension{
				idlTag: "patchMergeKey",
			},
			allowedValues: nil,
		},
		{
			e: extension{
				idlTag: "listType",
			},
			allowedValues: sets.NewString("atomic", "set", "map"),
		},
		{
			e: extension{
				idlTag: "listMapKey",
			},
			allowedValues: nil,
		},
		{
			e: extension{
				idlTag: "mapType",
			},
			allowedValues: sets.NewString("atomic", "granular"),
		},
		{
			e: extension{
				idlTag: "structType",
			},
			allowedValues: sets.NewString("atomic", "granular"),
		},
		{
			e: extension{
				idlTag: "k8s:openapi-gen",
			},
			allowedValues: nil,
		},
	}
	for _, test := range methodTests {
		if test.allowedValues != nil {
			if !test.e.hasAllowedValues() {
				t.Errorf("hasAllowedValues() expected (true), but received: false")
			}
			if !reflect.DeepEqual(test.allowedValues, test.e.allowedValues()) {
				t.Errorf("allowedValues() expected (%v), but received: %v",
					test.allowedValues, test.e.allowedValues())
			}
		}
		if test.allowedValues == nil && test.e.hasAllowedValues() {
			t.Errorf("hasAllowedValues() expected (false), but received: true")
		}
	}

	var successTests = []struct {
		e extension
	}{
		{
			e: extension{
				idlTag: "patchStrategy",
				xName:  "x-kubernetes-patch-strategy",
				values: []string{"merge"},
			},
		},
		{
			// Validate multiple values.
			e: extension{
				idlTag: "patchStrategy",
				xName:  "x-kubernetes-patch-strategy",
				values: []string{"merge", "retainKeys"},
			},
		},
		{
			e: extension{
				idlTag: "patchMergeKey",
				xName:  "x-kubernetes-patch-merge-key",
				values: []string{"key1"},
			},
		},
		{
			e: extension{
				idlTag: "listType",
				xName:  "x-kubernetes-list-type",
				values: []string{"atomic"},
			},
		},
		{
			e: extension{
				idlTag: "mapType",
				xName:  "x-kubernetes-map-type",
				values: []string{"atomic"},
			},
		},
		{
			e: extension{
				idlTag: "structType",
				xName:  "x-kubernetes-map-type",
				values: []string{"granular"},
			},
		},
	}
	for _, test := range successTests {
		actualErr := test.e.validateAllowedValues()
		if actualErr != nil {
			t.Errorf("Expected no error for (%v), but received: %v\n", test.e, actualErr)
		}
	}

	var failureTests = []struct {
		e extension
	}{
		{
			// Every value must be allowed.
			e: extension{
				idlTag: "patchStrategy",
				xName:  "x-kubernetes-patch-strategy",
				values: []string{"disallowed", "merge"},
			},
		},
		{
			e: extension{
				idlTag: "patchStrategy",
				xName:  "x-kubernetes-patch-strategy",
				values: []string{"foo"},
			},
		},
		{
			e: extension{
				idlTag: "listType",
				xName:  "x-kubernetes-list-type",
				values: []string{"not-allowed"},
			},
		},
		{
			e: extension{
				idlTag: "mapType",
				xName:  "x-kubernetes-map-type",
				values: []string{"something-pretty-wrong"},
			},
		},
		{
			e: extension{
				idlTag: "structType",
				xName:  "x-kubernetes-map-type",
				values: []string{"not-quite-right"},
			},
		},
	}
	for _, test := range failureTests {
		actualErr := test.e.validateAllowedValues()
		if actualErr == nil {
			t.Errorf("Expected error, but received none: %v\n", test.e)
		}
	}

}

func TestExtensionKind(t *testing.T) {

	var methodTests = []struct {
		e    extension
		kind types.Kind
	}{
		{
			e: extension{
				idlTag: "patchStrategy",
			},
			kind: types.Slice,
		},
		{
			e: extension{
				idlTag: "patchMergeKey",
			},
			kind: types.Slice,
		},
		{
			e: extension{
				idlTag: "listType",
			},
			kind: types.Slice,
		},
		{
			e: extension{
				idlTag: "mapType",
			},
			kind: types.Map,
		},
		{
			e: extension{
				idlTag: "structType",
			},
			kind: types.Struct,
		},
		{
			e: extension{
				idlTag: "listMapKey",
			},
			kind: types.Slice,
		},
		{
			e: extension{
				idlTag: "k8s:openapi-gen",
			},
			kind: "",
		},
	}
	for _, test := range methodTests {
		if len(test.kind) > 0 {
			if !test.e.hasKind() {
				t.Errorf("%v: hasKind() expected (true), but received: false", test.e)
			}
			if test.kind != test.e.kind() {
				t.Errorf("%v: kind() expected (%v), but received: %v", test.e, test.kind, test.e.kind())
			}
		} else {
			if test.e.hasKind() {
				t.Errorf("%v: hasKind() expected (false), but received: true", test.e)
			}
		}
	}
}

func TestValidateMemberExtensions(t *testing.T) {

	patchStrategyExtension := extension{
		idlTag: "patchStrategy",
		xName:  "x-kubernetes-patch-strategy",
		values: []string{"merge"},
	}
	patchMergeKeyExtension := extension{
		idlTag: "patchMergeKey",
		xName:  "x-kubernetes-patch-merge-key",
		values: []string{"key1", "key2"},
	}
	listTypeExtension := extension{
		idlTag: "listType",
		xName:  "x-kubernetes-list-type",
		values: []string{"atomic"},
	}
	listMapKeysExtension := extension{
		idlTag: "listMapKey",
		xName:  "x-kubernetes-map-keys",
		values: []string{"key1"},
	}
	genExtension := extension{
		idlTag: "k8s:openapi-gen",
		xName:  "x-kubernetes-member-type",
		values: []string{"value1"},
	}

	sliceField := types.Member{
		Name: "Containers",
		Type: &types.Type{
			Kind: types.Slice,
		},
	}
	mapField := types.Member{
		Name: "Containers",
		Type: &types.Type{
			Kind: types.Map,
		},
	}

	var successTests = []struct {
		extensions []extension
		member     types.Member
	}{
		// Test single member extension
		{
			extensions: []extension{patchStrategyExtension},
			member:     sliceField,
		},
		// Test multiple member extensions
		{
			extensions: []extension{
				patchMergeKeyExtension,
				listTypeExtension,
				listMapKeysExtension,
				genExtension, // Should not generate errors during type validation
			},
			member: sliceField,
		},
	}
	for _, test := range successTests {
		errors := validateMemberExtensions(test.extensions, &test.member)
		if len(errors) > 0 {
			t.Errorf("validateMemberExtensions: %v should have produced no errors. Errors: %v",
				test.extensions, errors)
		}
	}

	var failureTests = []struct {
		extensions []extension
		member     types.Member
	}{
		// Test single member extension
		{
			extensions: []extension{patchStrategyExtension},
			member:     mapField,
		},
		// Test multiple member extensions
		{
			extensions: []extension{
				patchMergeKeyExtension,
				listTypeExtension,
				listMapKeysExtension,
			},
			member: mapField,
		},
	}
	for _, test := range failureTests {
		errors := validateMemberExtensions(test.extensions, &test.member)
		if len(errors) != len(test.extensions) {
			t.Errorf("validateMemberExtensions: %v should have produced all errors. Errors: %v",
				test.extensions, errors)
		}
	}

}
