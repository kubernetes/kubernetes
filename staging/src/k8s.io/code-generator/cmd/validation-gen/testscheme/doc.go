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

// Package testscheme provides a scheme implementation and test utilities
// useful for writing output_tests for validation-gen.
//
// For an output test to use this scheme, it should be located in a dedicated go package.
// The go package should have validation-gen and this test scheme enabled and must declare a
// 'localSchemeBuilder' using the 'testscheme.New()' function. For example:
//
//	// +k8s:validation-gen-scheme-registry=k8s.io/code-generator/cmd/validation-gen/testscheme.Scheme
//	// +k8s:validation-gen
//	package example
//	import "k8s.io/code-generator/cmd/validation-gen/testscheme"
//	var localSchemeBuilder = testscheme.New()
//
// This is sufficient for validation-gen to generate a `zz_generated.validations.go` for the types
// in the package that compile.
//
// With the scheme enabled. An output test may be tested either by handwritten test code or
// by generated test fixtures.
//
// For example, to test by hand. The testschema provides utilities to create a value and assert
// that the expected errors are returned when the value is validated:
//
//	 func Test(t *testing.T) {
//		  st := localSchemeBuilder.Test(t)
//		  st.Value(&T1{
//			E0:  "x",
//			PE0: pointer.To(E0("y")),
//		  }).ExpectInvalid(
//			field.NotSupported(field.NewPath("e0"), "x", []string{EnumValue1, EnumValue2}),
//			field.NotSupported(field.NewPath("pe0"), "y", []string{EnumValue1, EnumValue2}))}
//	 }
//
// Tests fixtures can also be enabled. For example:
//
//	// ...
//	// +k8s:validation-gen-test-fixture=validateFalse
//	// package example
//
// When a test fixture is enabled, a `zz_generated.validations_test.go` file will be generated
// test for all the registered types in the package according to the behavior of the named test
// fixture(s).
//
// Test Fixtures:
//
// `validateFalse` - This test fixture executes validation of each registered type and accumulates
// all `validateFalse` validation errors. For example:
//
//	type T1 struct {
//		// +k8s:validateFalse="field T1.S"
//		S string `json:"s"`
//		// +k8s:validateFalse="field T1.T"
//		T T2 `json:"t"`
//	}
//
// The above `example.T1` test type has two validated fields: `s` and 't'. The fields are named
// according to the Go "json" field tag, and each has a validation error identifier
// provided by `+k8s:validateFalse=<identifier>`.
//
// The `validateFalse` test fixture will validate an instance of `example.T1`, generated using fuzzed
// data, and then group the validation errors by field name. Represented in JSON like:
//
//	{
//	  "*example.T1": {
//	    "s": [
//	      "field T1.S"
//	    ],
//	    "t": [
//	      "field T1.T"
//	    ]
//	  }
//	}
//
// This validation error data contains an object for each registered type, keyed by type name.
// For each registered type, the validation errors from `+k8s:validateFalse` tags are grouped
// by field path with all validation error identifiers collected into a list.
//
// This data is compared with the expected validation results that are defined in
// a `testdata/validate-false.json` file in the same package, and any differences are
// reported as test errors.
//
// `testdata/validate-false.json` can be generated automatically by setting the
// `UPDATE_VALIDATION_GEN_FIXTURE_DATA=true` environment variable to true when running the tests.
//
// Test authors that generated `testdata/validate-false.json` are expected to ensure that file
// is correct before checking it in to source control.
//
// The fuzzed data is generated pseudo-randomly with a consistent seed, with all nilable fields se
// to a value, and with a single entry for each map  and a single element for each slice.
package testscheme
