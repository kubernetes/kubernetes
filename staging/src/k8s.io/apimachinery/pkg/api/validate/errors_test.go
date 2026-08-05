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
	"testing"

	"k8s.io/apimachinery/pkg/util/validation/field"
)

func TestFilterCoveredHandwrittenErrors(t *testing.T) {
	originRule := "test-origin"

	handwrittenCovered := field.Invalid(field.NewPath("spec", "name"), "invalid-value", "must be valid").WithOrigin(originRule).MarkCoveredByDeclarative()

	handwrittenNotCovered := field.Invalid(field.NewPath("spec", "name"), "invalid-value", "must be valid").WithOrigin(originRule)

	matchingBetaDeclarative := field.Invalid(field.NewPath("spec", "name"), "invalid-value", "must be valid").WithOrigin(originRule).MarkBeta()

	matchingAlphaDeclarative := field.Invalid(field.NewPath("spec", "name"), "invalid-value", "must be valid").WithOrigin(originRule).MarkAlpha()

	tests := []struct {
		name                   string
		errs                   field.ErrorList
		enforcedDeclarative    field.ErrorList
		allDeclarativeEnforced bool
		betaEnabled            bool
		rules                  []field.NormalizationRule
		expectedLen            int
	}{
		{
			name:                   "uncovered handwritten error is preserved",
			errs:                   field.ErrorList{handwrittenNotCovered},
			enforcedDeclarative:    field.ErrorList{matchingBetaDeclarative},
			allDeclarativeEnforced: false,
			expectedLen:            1,
		},
		{
			name:                   "covered handwritten error filtered when matching beta declarative error exists",
			errs:                   field.ErrorList{handwrittenCovered},
			enforcedDeclarative:    field.ErrorList{matchingBetaDeclarative},
			allDeclarativeEnforced: false,
			expectedLen:            0,
		},
		{
			name:                   "covered handwritten error not filtered when matching declarative is alpha",
			errs:                   field.ErrorList{handwrittenCovered},
			enforcedDeclarative:    field.ErrorList{matchingAlphaDeclarative},
			allDeclarativeEnforced: false,
			expectedLen:            1,
		},
		{
			name:                   "covered handwritten error filtered in allDeclarativeEnforced mode regardless of matching declarative error",
			errs:                   field.ErrorList{handwrittenCovered},
			enforcedDeclarative:    nil,
			allDeclarativeEnforced: true,
			expectedLen:            0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			if tt.allDeclarativeEnforced {
				ctx = WithAllDeclarativeEnforcedForTest(ctx)
			}
			res := FilterCoveredHandwrittenErrors(ctx, tt.errs, tt.enforcedDeclarative, tt.betaEnabled, tt.rules...)
			if len(res) != tt.expectedLen {
				t.Errorf("FilterCoveredHandwrittenErrors() returned %d errors, expected %d", len(res), tt.expectedLen)
			}
		})
	}
}

func TestFilterEnforcedDeclarativeErrors(t *testing.T) {
	internalErr := field.InternalError(field.NewPath("spec"), errors.New("internal error"))
	alphaErr := field.Invalid(field.NewPath("spec"), "val", "alpha").MarkAlpha()
	betaErr := field.Invalid(field.NewPath("spec"), "val", "beta").MarkBeta()
	stdErr := field.Invalid(field.NewPath("spec"), "val", "standard")

	tests := []struct {
		name                   string
		declarativeErrs        field.ErrorList
		betaEnabled            bool
		allDeclarativeEnforced bool
		expectedLen            int
	}{
		{
			name:                   "all errors enforced in allDeclarativeEnforced mode",
			declarativeErrs:        field.ErrorList{internalErr, alphaErr, betaErr, stdErr},
			betaEnabled:            false,
			allDeclarativeEnforced: true,
			expectedLen:            4,
		},
		{
			name:                   "beta disabled ignores alpha and beta",
			declarativeErrs:        field.ErrorList{internalErr, alphaErr, betaErr, stdErr},
			betaEnabled:            false,
			allDeclarativeEnforced: false,
			expectedLen:            2, // internal + std
		},
		{
			name:                   "beta enabled includes beta but excludes alpha",
			declarativeErrs:        field.ErrorList{internalErr, alphaErr, betaErr, stdErr},
			betaEnabled:            true,
			allDeclarativeEnforced: false,
			expectedLen:            3, // internal + beta + std
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			if tt.allDeclarativeEnforced {
				ctx = WithAllDeclarativeEnforcedForTest(ctx)
			}
			res := FilterEnforcedDeclarativeErrors(ctx, tt.declarativeErrs, tt.betaEnabled)
			if len(res) != tt.expectedLen {
				t.Errorf("FilterEnforcedDeclarativeErrors() returned %d errors, expected %d", len(res), tt.expectedLen)
			}
		})
	}
}
