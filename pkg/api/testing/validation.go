/*
Copyright 2025 The Kubernetes Authors.

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

package testing

import (
	"bytes"
	"context"
	"sort"
	"strconv"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimetest "k8s.io/apimachinery/pkg/runtime/testing"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

// ValidateFunc is a function that runs validation.
type ValidateFunc func(ctx context.Context, obj runtime.Object) field.ErrorList

// ValidateUpdateFunc is a function that runs update validation.
type ValidateUpdateFunc func(ctx context.Context, obj, old runtime.Object) field.ErrorList

// VerifyVersionedValidationEquivalence tests that all versions of an API return equivalent validation errors.
// It accepts optional configuration to handle path normalization across API versions where structures differ.
func VerifyVersionedValidationEquivalence(t *testing.T, obj, old runtime.Object, testConfigs ...ValidationTestConfig) {
	t.Helper()

	opts := &validationOption{}
	for _, testcfg := range testConfigs {
		testcfg(opts)
	}

	// Accumulate errors from all versioned validation, per version.
	all := map[string]field.ErrorList{}
	accumulate := func(t *testing.T, gv string, errs field.ErrorList) {
		// If normalization rules are provided, apply them to the field paths of generated errors.
		// This allows comparing errors between API versions that have structural differences
		// (e.g. flattened vs nested fields).
		// We must normalize in place before sorting.
		for i := range errs {
			currentPath := errs[i].Field
			for _, rule := range opts.NormalizationRules {
				normalized := rule.Regexp.ReplaceAllString(currentPath, rule.Replacement)
				if normalized != currentPath {
					errs[i].Field = normalized
					// Apply only the first matching rule per error
					break
				}
			}
		}
		// Re-sort the error list based primarily on the normalized field paths
		// to ensure errors align correctly during index-by-index comparison,
		// regardless of their original structure.
		sort.Slice(errs, func(i, j int) bool {
			if errs[i].Field != errs[j].Field {
				return errs[i].Field < errs[j].Field
			}
			// Secondary sort by full error string for determinism when fields are equal
			return errs[i].Error() < errs[j].Error()
		})
		all[gv] = errs
	}
	// Convert versioned object to internal format before validation.
	// runtimetest.RunValidationForEachVersion requires unversioned (internal) objects as input.
	internalObj, err := convertToInternal(t, legacyscheme.Scheme, obj)
	if err != nil {
		t.Fatal(err)
	}
	if internalObj == nil {
		return
	}
	if old == nil {
		runtimetest.RunValidationForEachVersion(t, legacyscheme.Scheme, []string{}, internalObj, accumulate, opts.IgnoreObjectConversionErrors, opts.SubResources...)
	} else {
		// Convert old versioned object to internal format before validation.
		// runtimetest.RunUpdateValidationForEachVersion requires unversioned (internal) objects as input.
		internalOld, err := convertToInternal(t, legacyscheme.Scheme, old)
		if err != nil {
			t.Fatal(err)
		}
		if internalOld == nil {
			return
		}
		runtimetest.RunUpdateValidationForEachVersion(t, legacyscheme.Scheme, []string{}, internalObj, internalOld, accumulate, opts.IgnoreObjectConversionErrors, opts.SubResources...)
	}

	// Make a copy so we can modify it.
	other := map[string]field.ErrorList{}
	// Index for nicer output.
	keys := []string{}
	for k, v := range all {
		other[k] = v
		keys = append(keys, k)
	}
	sort.Strings(keys)

	// Compare each lhs to each rhs.
	for _, lk := range keys {
		lv := all[lk]
		// remove lk since to prevent comparison to itself and because this
		// iteration will compare it to any version it has not yet been
		// compared to. e.g. [1, 2, 3] vs. [1, 2, 3] yields:
		//   1 vs. 2
		//   1 vs. 3
		//   2 vs. 3
		delete(other, lk)
		// don't compare to ourself
		for _, rk := range keys {
			rv, found := other[rk]
			if !found {
				continue // done already
			}
			if len(lv) != len(rv) {
				t.Errorf("different error count (%d vs. %d)\n%s: %v\n%s: %v", len(lv), len(rv), lk, fmtErrs(lv), rk, fmtErrs(rv))
				continue
			}
			next := false
			for i := range lv {
				// We don't use reflect.DeepEqual here because unversioned and versioned
				// validation might have different bad values (e.g. pointer vs value).
				// We also don't use ErrorMatcher here because it doesn't handle re-sorting
				// required after normalization in multi-error scenarios within this specific loop structure.
				l, r := lv[i], rv[i]
				// Compare field (already normalized), type, detail, and origin.
				if l.Type != r.Type || l.Field != r.Field || l.Detail != r.Detail || l.Origin != r.Origin {
					t.Errorf("different errors at index %d\n%s: %v\n%s: %v", i, lk, l.Error(), rk, r.Error())
					next = true
				}
			}
			if next {
				t.Errorf("complete error lists for context:\n%s: %v\n%s: %v", lk, fmtErrs(lv), rk, fmtErrs(rv))
				continue
			}
		}
	}
}

// helper for nicer output
func fmtErrs(errs field.ErrorList) string {
	if len(errs) == 0 {
		return "<no errors>"
	}
	if len(errs) == 1 {
		return strconv.Quote(errs[0].Error())
	}
	buf := bytes.Buffer{}
	for _, e := range errs {
		buf.WriteString("\n\t")
		buf.WriteString(strconv.Quote(e.Error()))
	}

	return buf.String()
}

func convertToInternal(t *testing.T, scheme *runtime.Scheme, obj runtime.Object) (runtime.Object, error) {
	t.Helper()

	gvks, _, err := scheme.ObjectKinds(obj)
	if err != nil {
		t.Fatal(err)
	}
	if len(gvks) == 0 {
		t.Fatal("no GVKs found for object")
	}
	gvk := gvks[0]
	if gvk.Version == runtime.APIVersionInternal {
		return obj, nil
	}
	gvk.Version = runtime.APIVersionInternal
	if !scheme.Recognizes(gvk) {
		t.Logf("no internal object found for GroupKind %s", gvk.GroupKind().String())
		return nil, nil
	}
	return scheme.ConvertToVersion(obj, schema.GroupVersion{Group: gvk.Group, Version: runtime.APIVersionInternal})
}

type ValidationTestConfig func(*validationOption)

// validationOptions encapsulates optional parameters for validation equivalence tests.
type validationOption struct {
	// SubResources are the subresources to validate.
	SubResources []string
	// NormalizationRules are the rules to apply to field paths before comparison.
	NormalizationRules []field.NormalizationRule

	// IgnoreObjectConversions skips the tests if the conversion between object fails.
	IgnoreObjectConversionErrors bool
}

func WithSubResources(subResources ...string) ValidationTestConfig {
	return func(o *validationOption) {
		o.SubResources = subResources
	}
}

func WithNormalizationRules(rules ...field.NormalizationRule) ValidationTestConfig {
	return func(o *validationOption) {
		o.NormalizationRules = rules
	}
}

func WithIgnoreObjectConversionErrors() ValidationTestConfig {
	return func(o *validationOption) {
		o.IgnoreObjectConversionErrors = true
	}
}

// VerifyValidationEquivalence provides a helper for testing the migration from
// hand-written imperative validation to declarative validation. It ensures that
// the validation logic remains consistent before and after the feature is enabled.
//
// The function operates by running the provided validation function under two scenarios:
//  1. With DeclarativeValidation and DeclarativeValidationTakeover feature gates disabled,
//     simulating the legacy hand-written validation.
//  2. With both feature gates enabled, using the new declarative validation rules.
//
// It then asserts that the validation errors produced in both scenarios are equivalent,
// guaranteeing a safe migration. It also checks the errors against an expected set.
// It compares errors by field, origin and type; all three should match to be called equivalent.
// It also make sure all versions of the given API returns equivalent errors.
func VerifyValidationEquivalence(t *testing.T, ctx context.Context, obj runtime.Object, validateFn ValidateFunc, expectedErrs field.ErrorList, testConfigs ...ValidationTestConfig) {
	t.Helper()
	opts := &validationOption{}
	for _, testcfg := range testConfigs {
		testcfg(opts)
	}
	verifyValidationEquivalence(t, expectedErrs, func() field.ErrorList {
		return validateFn(ctx, obj)
	}, opts)
	VerifyVersionedValidationEquivalence(t, obj, nil, testConfigs...)
}

// VerifyUpdateValidationEquivalence provides a helper for testing the migration from
// hand-written imperative validation to declarative validation for update operations.
// It ensures that the validation logic remains consistent before and after the feature is enabled.
//
// The function operates by running the provided validation function under two scenarios:
//  1. With DeclarativeValidation and DeclarativeValidationTakeover feature gates disabled,
//     simulating the legacy hand-written validation.
//  2. With both feature gates enabled, using the new declarative validation rules.
//
// It then asserts that the validation errors produced in both scenarios are equivalent,
// guaranteeing a safe migration. It also checks the errors against an expected set.
// It compares errors by field, origin and type; all three should match to be called equivalent.
// It also make sure all versions of the given API returns equivalent errors.
func VerifyUpdateValidationEquivalence(t *testing.T, ctx context.Context, obj, old runtime.Object, validateUpdateFn ValidateUpdateFunc, expectedErrs field.ErrorList, testConfigs ...ValidationTestConfig) {
	t.Helper()
	opts := &validationOption{}
	for _, testcfg := range testConfigs {
		testcfg(opts)
	}
	verifyValidationEquivalence(t, expectedErrs, func() field.ErrorList {
		return validateUpdateFn(ctx, obj, old)
	}, opts)
	VerifyVersionedValidationEquivalence(t, obj, old, testConfigs...)
}

// verifyValidationEquivalence is a generic helper that verifies validation equivalence with and without declarative validation.
func verifyValidationEquivalence(t *testing.T, expectedErrs field.ErrorList, runValidations func() field.ErrorList, opt *validationOption) {
	t.Helper()
	var declarativeTakeoverErrs field.ErrorList
	var imperativeErrs field.ErrorList

	// The errOutputMatcher is used to verify the output matches the expected errors in test cases.
	errOutputMatcher := field.ErrorMatcher{}.ByType().ByOrigin().ByFieldNormalized(opt.NormalizationRules)

	// We only need to test both gate enabled and disabled together, because
	// 1) the DeclarativeValidationTakeover won't take effect if DeclarativeValidation is disabled.
	// 2) the validation output, when only DeclarativeValidation is enabled, is the same as when both gates are disabled.
	t.Run("with declarative validation", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.DeclarativeValidation:         true,
			features.DeclarativeValidationTakeover: true,
		})
		declarativeTakeoverErrs = runValidations()

		if len(expectedErrs) > 0 {
			errOutputMatcher.Test(t, expectedErrs, declarativeTakeoverErrs)
		} else if len(declarativeTakeoverErrs) != 0 {
			t.Errorf("expected no errors, but got: %v", declarativeTakeoverErrs)
		}
	})

	t.Run("hand written validation", func(t *testing.T) {
		featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
			features.DeclarativeValidationTakeover: false,
			features.DeclarativeValidation:         false,
		})
		imperativeErrs = runValidations()

		if len(expectedErrs) > 0 {
			errOutputMatcher.Test(t, expectedErrs, imperativeErrs)
		} else if len(imperativeErrs) != 0 {
			t.Errorf("expected no errors, but got: %v", imperativeErrs)
		}
	})

	if t.Failed() {
		// There is no point in moving forward, if any of above tests failed for any reason. Running follow up tests will return noise.
		t.SkipNow()
	}

	// The equivalenceMatcher is used to verify the output errors from hand-written imperative validation
	// are equivalent to the output errors when DeclarativeValidationTakeover is enabled.
	equivalenceMatcher := field.ErrorMatcher{}.ByType().ByOrigin()
	if len(opt.NormalizationRules) > 0 {
		equivalenceMatcher = equivalenceMatcher.ByFieldNormalized(opt.NormalizationRules)
	} else {
		equivalenceMatcher = equivalenceMatcher.ByField()
	}

	// The imperative validation may produce duplicate errors, which is not supported by the ErrorMatcher.
	// TODO: remove this once ErrorMatcher has been extended to handle this form of deduplication.
	imperativeErrs = deDuplicateErrors(imperativeErrs, equivalenceMatcher)

	equivalenceMatcher.Test(t, imperativeErrs, declarativeTakeoverErrs)
}

// deDuplicateErrors removes duplicate errors from an ErrorList based on the provided matcher.
func deDuplicateErrors(errs field.ErrorList, matcher field.ErrorMatcher) field.ErrorList {
	var deduped field.ErrorList
	for _, err := range errs {
		found := false
		for _, existingErr := range deduped {
			if matcher.Matches(existingErr, err) {
				found = true
				break
			}
		}
		if !found {
			deduped = append(deduped, err)
		}
	}
	return deduped
}
