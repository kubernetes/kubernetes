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

package validate

import (
	"context"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestShortName(t *testing.T) {
	ctx := context.Background()
	fldPath := field.NewPath("test")

	testCases := []struct {
		name     string
		input    string
		wantErrs field.ErrorList
	}{{
		name:     "valid name",
		input:    "valid-name",
		wantErrs: nil,
	}, {
		name:     "valid single character name",
		input:    "a",
		wantErrs: nil,
	}, {
		name:     "valid name with numbers",
		input:    "123-abc",
		wantErrs: nil,
	}, {
		name:  "invalid: uppercase characters",
		input: "Invalid-Name",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:  "invalid: starts with dash",
		input: "-invalid-name",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:  "invalid: ends with dash",
		input: "invalid-name-",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:  "invalid: contains dots",
		input: "invalid.name",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:  "invalid: contains special characters",
		input: "invalid@name",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:  "invalid: too long",
		input: "a" + strings.Repeat("b", 62) + "c", // 64 characters
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-short-name"),
		},
	}, {
		name:     "valid: max length",
		input:    "a" + strings.Repeat("b", 61) + "c", // 63 characters
		wantErrs: nil,
	}, {
		name:  "invalid: empty string",
		input: "",
		wantErrs: field.ErrorList{
			field.Invalid(fldPath, nil, "").WithOrigin("format=k8s-short-name"),
		},
	}}

	matcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin()
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			value := tc.input
			gotErrs := ShortName(ctx, operation.Operation{}, fldPath, &value, nil)

			matcher.Test(t, tc.wantErrs, gotErrs)
		})
	}
}
