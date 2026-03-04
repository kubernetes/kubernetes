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

package testscheme

import (
	"bytes"
	stdcmp "cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"         // nolint:depguard // this package provides test utilities
	"github.com/google/go-cmp/cmp/cmpopts" // nolint:depguard // this package provides test utilities

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/randfill"
)

// Scheme is similar to runtime.Scheme, but for validation testing purposes. Scheme only supports validation,
// supports registration of any type (not just runtime.Object) and implements Register directly, allowing it
// to also be used as a scheme builder.
// Must only be used with tests that perform all registration before calls to validate.
type Scheme struct {
	validationFuncs    map[reflect.Type]func(ctx context.Context, op operation.Operation, object, oldObject interface{}) field.ErrorList
	registrationErrors field.ErrorList
}

// New creates a new Scheme.
func New() *Scheme {
	return &Scheme{validationFuncs: map[reflect.Type]func(ctx context.Context, op operation.Operation, object interface{}, oldObject interface{}) field.ErrorList{}}
}

// AddValidationFunc registers a validation function.
// Last writer wins.
func (s *Scheme) AddValidationFunc(srcType any, fn func(ctx context.Context, op operation.Operation, object, oldObject interface{}) field.ErrorList) {
	s.validationFuncs[reflect.TypeOf(srcType)] = fn
}

// Validate validates an object using the registered validation function.
func (s *Scheme) Validate(ctx context.Context, options []string, object any, subresources ...string) field.ErrorList {
	if len(s.registrationErrors) > 0 {
		return s.registrationErrors // short circuit with registration errors if any are present
	}
	if fn, ok := s.validationFuncs[reflect.TypeOf(object)]; ok {
		return fn(ctx, operation.Operation{Type: operation.Create, Request: operation.Request{Subresources: subresources}, Options: options}, object, nil)
	}
	return nil
}

// ValidateUpdate validates an update to an object using the registered validation function.
func (s *Scheme) ValidateUpdate(ctx context.Context, options []string, object, oldObject any, subresources ...string) field.ErrorList {
	if len(s.registrationErrors) > 0 {
		return s.registrationErrors // short circuit with registration errors if any are present
	}
	if fn, ok := s.validationFuncs[reflect.TypeOf(object)]; ok {
		return fn(ctx, operation.Operation{Type: operation.Update, Request: operation.Request{Subresources: subresources}, Options: options}, object, oldObject)
	}
	return nil
}

// Register adds a scheme setup function to the list.
func (s *Scheme) Register(funcs ...func(*Scheme) error) {
	for _, f := range funcs {
		err := f(s)
		if err != nil {
			s.registrationErrors = append(s.registrationErrors, toRegistrationError(err))
		}
	}
}

func toRegistrationError(err error) *field.Error {
	return field.InternalError(nil, fmt.Errorf("registration error: %w", err))
}

// Test returns a ValidationTestBuilder for this scheme.
func (s *Scheme) Test(t *testing.T) *ValidationTestBuilder {
	return &ValidationTestBuilder{t, s}
}

// ValidationTestBuilder provides convenience functions to build
// validation tests.
type ValidationTestBuilder struct {
	*testing.T
	s *Scheme
}

const fixtureEnvVar = "UPDATE_VALIDATION_GEN_FIXTURE_DATA"

// ValidateFixtures ensures that the validation errors of all registered types match what is expected by the test fixture files.
// For each registered type, a value is created for the type, and populated by fuzzing the value, before validating the type.
// See ValueFuzzed for details.
//
// If the UPDATE_VALIDATION_GEN_FIXTURE_DATA=true environment variable is set, test fixture files are created or overridden.
//
// Fixtures:
//   - validate-false.json: defines a map of registered type to a map of field path to  +validateFalse validations args
//     that are expected to be returned as errors when the type is validated.
func (s *ValidationTestBuilder) ValidateFixtures() {
	s.T.Helper()

	flag := os.Getenv(fixtureEnvVar)
	// Run validation
	got := map[string]map[string][]string{}
	for t := range s.s.validationFuncs {
		var v any
		// TODO: this should handle maps and slices
		if t.Kind() == reflect.Ptr {
			v = reflect.New(t.Elem()).Interface()
		} else {
			v = reflect.Indirect(reflect.New(t)).Interface()
		}
		if reflect.TypeOf(v).Kind() != reflect.Ptr {
			v = &v
		}
		s.ValueFuzzed(v)
		vt := &ValidationTester{ValidationTestBuilder: s, value: v}
		byPath := vt.validateFalseArgsByPath()
		got[t.String()] = byPath
	}

	testdataFilename := "testdata/validate-false.json"
	if flag == "true" {
		// Generate fixture file
		if err := os.MkdirAll(path.Dir(testdataFilename), os.FileMode(0755)); err != nil {
			s.Fatal("error making directory", err)
		}
		data, err := json.MarshalIndent(got, "", "  ")
		if err != nil {
			s.Fatal(err)
		}
		err = os.WriteFile(testdataFilename, data, os.FileMode(0644))
		if err != nil {
			s.Fatal(err)
		}
	} else {
		// Load fixture file
		testdataFile, err := os.Open(testdataFilename)
		if errors.Is(err, os.ErrNotExist) {
			s.Fatalf("%s test fixture data not found. Run go test with the environment variable %s=true to create test fixture data.",
				testdataFilename, fixtureEnvVar)
		} else if err != nil {
			s.Fatal(err)
		}
		defer func() {
			err := testdataFile.Close()
			if err != nil {
				s.Fatal(err)
			}
		}()

		byteValue, err := io.ReadAll(testdataFile)
		if err != nil {
			s.Fatal(err)
		}
		testdata := map[string]map[string][]string{}
		err = json.Unmarshal(byteValue, &testdata)
		if err != nil {
			s.Fatal(err)
		}
		// Compare fixture with validation results
		expectedKeys := sets.New[string]()
		gotKeys := sets.New[string]()
		for k := range got {
			gotKeys.Insert(k)
		}
		hasErrors := false
		for k, expectedForType := range testdata {
			expectedKeys.Insert(k)
			gotForType, ok := got[k]
			s.T.Run(k, func(t *testing.T) {
				t.Helper()

				if !ok {
					t.Errorf("%q has expected validateFalse args in %s but got no validation errors.", k, testdataFilename)
					hasErrors = true
				} else if !cmp.Equal(gotForType, expectedForType) {
					t.Errorf("validateFalse args, grouped by field path, differed from %s:\n%s\n",
						testdataFilename, cmp.Diff(gotForType, expectedForType, cmpopts.SortMaps(stdcmp.Less[string])))
					hasErrors = true
				}
			})
		}
		for unexpectedType := range gotKeys.Difference(expectedKeys) {
			s.T.Run(unexpectedType, func(t *testing.T) {
				t.Helper()

				t.Errorf("%q got unexpected validateFalse args, grouped by field path:\n%s\n",
					unexpectedType, cmp.Diff(nil, got[unexpectedType], cmpopts.SortMaps(stdcmp.Less[string])))
				hasErrors = true
			})
		}
		if hasErrors {
			s.T.Logf("If the test expectations have changed, run go test with the environment variable %s=true", fixtureEnvVar)
		}
	}
}

func randfiller() *randfill.Filler {
	// Ensure that lists and maps are not empty and use a deterministic seed.
	// But also, don't recurse infinitely.
	return randfill.New().NilChance(0.0).NumElements(2, 2).MaxDepth(8).RandSource(rand.NewSource(0))
}

// ValueFuzzed automatically populates the given value using a deterministic filler.
// The filler sets pointers to values and always includes a two map keys and slice elements.
func (s *ValidationTestBuilder) ValueFuzzed(value any) *ValidationTester {
	randfiller().Fill(value)
	return &ValidationTester{ValidationTestBuilder: s, value: value}
}

// Value returns a ValidationTester for the given value. The value
// must be a registered with the scheme for validation.
func (s *ValidationTestBuilder) Value(value any) *ValidationTester {
	return &ValidationTester{ValidationTestBuilder: s, value: value}
}

// ValidationTester provides convenience functions to define validation
// tests for a validatable value.
type ValidationTester struct {
	*ValidationTestBuilder
	value        any
	oldValue     any
	isUpdate     bool
	options      []string
	subresources []string
}

// OldValue sets the oldValue for this ValidationTester. When oldValue is set,
// update validation will be used to test validation.
// oldValue must be the same type as value.
// Returns ValidationTester to support call chaining.
func (v *ValidationTester) OldValue(oldValue any) *ValidationTester {
	v.oldValue = oldValue
	v.isUpdate = true
	return v
}

// OldValueFuzzed automatically populates the given value using a deterministic filler.
// The filler sets pointers to values and always includes a two map keys and slice elements.
func (v *ValidationTester) OldValueFuzzed(oldValue any) *ValidationTester {
	randfiller().Fill(oldValue)
	v.oldValue = oldValue
	v.isUpdate = true
	return v
}

// Opts sets the ValidationOpts to use.
func (v *ValidationTester) Opts(options []string) *ValidationTester {
	v.options = options
	return v
}

// Subresource sets the ValidationOpts to use.
func (v *ValidationTester) Subresources(subresources []string) *ValidationTester {
	v.subresources = subresources
	return v
}

func multiline(errs field.ErrorList) string {
	if len(errs) == 0 {
		return "<no errors>"
	}
	if len(errs) == 1 {
		return errs[0].Error()
	}

	var buf bytes.Buffer
	for _, err := range errs {
		buf.WriteString("\n")
		buf.WriteString(err.Error())
	}
	return buf.String()
}

// ExpectValid validates the value and calls t.Errorf if any validation errors are returned.
// Returns ValidationTester to support call chaining.
func (v *ValidationTester) ExpectValid() *ValidationTester {
	v.T.Helper()

	v.T.Run(fmt.Sprintf("%T", v.value), func(t *testing.T) {
		t.Helper()

		errs := v.validate()
		if len(errs) > 0 {
			t.Errorf("want no errors, got: %v", multiline(errs))
		}
	})
	return v
}

// ExpectValidateFalseByPath validates the value and looks for the errors
// specifically produced by `+k8s:validateFalse` tags. Each field (the map key)
// can have multiple error strings (the map value). Test which are trying
// to prove that the validation logic itself (e.g. validation-gen) produces the
// expected errors should use this method.
func (v *ValidationTester) ExpectValidateFalseByPath(expectedByPath map[string][]string) *ValidationTester {
	v.T.Helper()

	v.T.Run(fmt.Sprintf("%T", v.value), func(t *testing.T) {
		t.Helper()

		actualByPath := v.validateFalseArgsByPath()
		// ensure args are sorted
		for _, args := range expectedByPath {
			sort.Strings(args)
		}
		if !cmp.Equal(expectedByPath, actualByPath) {
			t.Errorf("validateFalse args, grouped by field path, differed from expected:\n%s\n", cmp.Diff(expectedByPath, actualByPath, cmpopts.SortMaps(stdcmp.Less[string])))
		}

	})
	return v
}

func (v *ValidationTester) validateFalseArgsByPath() map[string][]string {
	byPath := map[string][]string{}
	errs := v.validate()
	for _, e := range errs {
		if strings.HasPrefix(e.Detail, "forced failure: ") {
			arg := strings.TrimPrefix(e.Detail, "forced failure: ")
			f := e.Field
			if f == "<nil>" {
				f = ""
			}
			byPath[f] = append(byPath[f], arg)
		}
	}
	// ensure args are sorted
	for _, args := range byPath {
		sort.Strings(args)
	}
	return byPath
}

// ExpectMatches compares the expected errors with the actual errors returned
// by the validation, using the provided ErrorMatcher. Tests which are trying
// to prove that a use-case of validation (e.g. testing pod validation)
// produces the expected errors should use this method.
func (v *ValidationTester) ExpectMatches(matcher field.ErrorMatcher, expected field.ErrorList) *ValidationTester {
	v.Helper()

	v.Run(fmt.Sprintf("%T", v.value), func(t *testing.T) {
		t.Helper()
		actual := v.validate()
		matcher.Test(t, expected, actual)
	})
	return v
}

func (v *ValidationTester) validate() field.ErrorList {
	if v.isUpdate {
		return v.s.ValidateUpdate(context.Background(), v.options, v.value, v.oldValue, v.subresources...)
	}
	return v.s.Validate(context.Background(), v.options, v.value, v.subresources...)
}
