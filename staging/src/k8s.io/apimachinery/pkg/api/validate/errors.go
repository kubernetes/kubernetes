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

package validate

import (
	"context"
	"errors"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

type allDeclarativeEnforcedKeyType struct{}

var allDeclarativeEnforcedKey = allDeclarativeEnforcedKeyType{}

// WithAllDeclarativeEnforcedForTest returns a copy of parent context with allDeclarativeEnforcedKey set to true.
// This is used for testing to expose all declarative validation errors and filter all handwritten validation errors
// that are covered by declarative validation, regardless of the feature gate or maturity level.
//
// NOTE: This function is intended for testing purposes only and should not be used in production code.
func WithAllDeclarativeEnforcedForTest(ctx context.Context) context.Context {
	return context.WithValue(ctx, allDeclarativeEnforcedKey, true)
}

// AllDeclarativeEnforced returns true if the context contains allDeclarativeEnforcedKey set to true.
func AllDeclarativeEnforced(ctx context.Context) bool {
	if ctx == nil {
		return false
	}
	return ctx.Value(allDeclarativeEnforcedKey) == true
}

// FilterCoveredHandwrittenErrors removes a CoveredByDeclarative handwritten error when a matching enforced
// beta declarative error exists (matched by type, field, and origin). In AllDeclarativeEnforced
// (testing-only) mode every covered handwritten error is removed.
func FilterCoveredHandwrittenErrors(ctx context.Context, imperativeErrs, enforcedDeclarativeErrs field.ErrorList, betaEnabled bool, rules ...field.NormalizationRule) field.ErrorList {
	matcher := field.ErrorMatcher{}.ByType().ByOrigin().RequireOriginWhenInvalid().ByFieldNormalized(rules)
	allDeclarativeEnforced := AllDeclarativeEnforced(ctx)
	return imperativeErrs.Filter(func(e error) bool {
		var fe *field.Error
		if !errors.As(e, &fe) || !fe.CoveredByDeclarative {
			return false
		}
		if allDeclarativeEnforced {
			return true
		}
		for _, dErr := range enforcedDeclarativeErrs {
			if dErr.IsBeta() && matcher.Matches(fe, dErr) {
				return true
			}
		}
		return false
	})
}

// FilterEnforcedDeclarativeErrors collects the declarative errors that are enforced (i.e. surfaced to the user) in the
// current mode. A declarative error is enforced when any of the following holds:
//   - AllDeclarativeEnforced is set (testing): every declarative error is enforced.
//   - It is an internal error: always enforced, regardless of lifecycle.
//   - It is a beta error and BetaEnabled is true.
//   - It is a standard (unprefixed) error: always enforced.
//
// Alpha errors are never enforced; they remain shadowed by handwritten validation.
func FilterEnforcedDeclarativeErrors(ctx context.Context, declarativeErrs field.ErrorList, betaEnabled bool) field.ErrorList {
	enforcedDeclarativeErrs := make(field.ErrorList, 0, len(declarativeErrs))
	allDeclarativeEnforced := AllDeclarativeEnforced(ctx)
	for _, dvErr := range declarativeErrs {
		switch {
		case allDeclarativeEnforced:
			enforcedDeclarativeErrs = append(enforcedDeclarativeErrs, dvErr)
		case dvErr.Type == field.ErrorTypeInternal:
			enforcedDeclarativeErrs = append(enforcedDeclarativeErrs, dvErr)
		case dvErr.IsBeta():
			if betaEnabled {
				enforcedDeclarativeErrs = append(enforcedDeclarativeErrs, dvErr)
			}
		case !dvErr.IsAlpha():
			enforcedDeclarativeErrs = append(enforcedDeclarativeErrs, dvErr) // Standard
		}
	}
	return enforcedDeclarativeErrs
}
