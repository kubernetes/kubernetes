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

package external

import (
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/code-generator/cmd/validation-gen/output_tests/multiple_packages/registered"
	"k8s.io/code-generator/cmd/validation-gen/output_tests/multiple_packages/types"
)

// TestSelfContained runs this non-registering package's validators via direct
// calls (no scheme) and checks they agree with the registered copy.
func TestSelfContained(t *testing.T) {
	ctx := context.Background()
	matcher := field.ErrorMatcher{}.ByField().ByDetailExact()

	// Shared types must match the registered copy.
	obj := &types.T1{List: []types.T2{{}, {}}}
	ext := Validate_T1(ctx, operation.Operation{}, nil, obj, nil)
	if len(ext) == 0 {
		t.Fatalf("expected validation errors from Validate_T1, got none")
	}
	reg := registered.Validate_T1(ctx, operation.Operation{}, nil, obj, nil)
	matcher.Test(t, reg, ext)

	// T3 is selected only by external; registered has no Validate_T3.
	got := Validate_T3(ctx, operation.Operation{}, nil, &types.T3{}, nil)
	want := field.ErrorList{
		field.Invalid(field.NewPath("s"), nil, "forced failure: field T3.S"),
	}
	matcher.Test(t, want, got)
}
