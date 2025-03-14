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

package rest

import (
	"context"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	validationmetrics "k8s.io/apiserver/pkg/validation"
	"k8s.io/klog/v2"
)

// ValidateDeclaratively validates obj against declarative validation tags
// defined in its Go type. It uses the API version extracted from ctx and the
// provided scheme for validation.
//
// The ctx MUST contain requestInfo, which determines the target API for
// validation. The obj is converted to the API version using the provided scheme
// before validation occurs. The scheme MUST have the declarative validation
// registered for the requested resource/subresource.
//
// option should contain any validation options that the declarative validation
// tags expect.
//
// Returns a field.ErrorList containing any validation errors. An internal error
// is included if requestInfo is missing from the context or if version
// conversion fails.
func ValidateDeclaratively(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj runtime.Object) field.ErrorList {
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		versionedObj, err := scheme.ConvertToVersion(obj, groupVersion)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error converting to versioned type: %w", err))}
		}
		subresources, err := parseSubresourcePath(requestInfo.Subresource)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error parsing subresource path: %w", err))}
		}
		return scheme.Validate(ctx, options, versionedObj, subresources...)
	} else {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("could not find requestInfo in context"))}
	}
}

// ValidateUpdateDeclaratively validates obj and oldObj against declarative
// validation tags defined in its Go type. It uses the API version extracted from
// ctx and the provided scheme for validation.
//
// The ctx MUST contain requestInfo, which determines the target API for
// validation. The obj is converted to the API version using the provided scheme
// before validation occurs. The scheme MUST have the declarative validation
// registered for the requested resource/subresource.
//
// option should contain any validation options that the declarative validation
// tags expect.
//
// Returns a field.ErrorList containing any validation errors. An internal error
// is included if requestInfo is missing from the context or if version
// conversion fails.
func ValidateUpdateDeclaratively(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj, oldObj runtime.Object) field.ErrorList {
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		versionedObj, err := scheme.ConvertToVersion(obj, groupVersion)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error converting to versioned type: %w", err))}
		}
		versionedOldObj, err := scheme.ConvertToVersion(oldObj, groupVersion)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error converting to versioned type: %w", err))}
		}
		subresources, err := parseSubresourcePath(requestInfo.Subresource)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error parsing subresource path: %w", err))}
		}
		return scheme.ValidateUpdate(ctx, options, versionedObj, versionedOldObj, subresources...)
	} else {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("could not find requestInfo in context"))}
	}
}

func parseSubresourcePath(subresourcePath string) ([]string, error) {
	if len(subresourcePath) == 0 {
		return nil, nil
	}
	parts := strings.Split(subresourcePath, "/")
	for _, part := range parts {
		if len(part) == 0 {
			return nil, fmt.Errorf("invalid subresource path: %s", subresourcePath)
		}
	}
	return parts, nil
}

// CompareDeclarativeErrorsAndEmitMismatches checks for mismatches between imperative and declarative validation
// and logs + emits metrics when inconsistencies are found
func CompareDeclarativeErrorsAndEmitMismatches(ctx context.Context, imperativeErrs, declarativeErrs field.ErrorList, takeover bool) {
	logger := klog.FromContext(ctx)
	mismatchDetails := gatherDeclarativeValidationMismatches(imperativeErrs, declarativeErrs, takeover)
	for _, detail := range mismatchDetails {
		// Log information about the mismatch using contextual logger
		logger.Error(nil, detail)

		// Increment the metric for the mismatch
		validationmetrics.Metrics.IncDeclarativeValidationMismatchMetric()
	}
}

// gatherDeclarativeValidationMismatches compares imperative and declarative validation errors
// and returns detailed information about any mismatches found. Errors are compared via type, field, and origin
func gatherDeclarativeValidationMismatches(imperativeErrs, declarativeErrs field.ErrorList, takeover bool) []string {
	var mismatchDetails []string
	// short circuit here to minimize allocs for usual case of 0 validation errors
	if len(imperativeErrs) == 0 && len(declarativeErrs) == 0 {
		return mismatchDetails
	}
	// recommendation based on takeover status
	recommendation := "This difference should not affect system operation since hand written validation is authoritative."
	if takeover {
		recommendation = "Consider disabling the DeclarativeValidationTakeover feature gate to keep data persisted in etcd consistent with prior versions of Kubernetes."
	}
	fuzzyMatcher := field.ErrorMatcher{}.ByType().ByField().ByOrigin().RequireOriginWhenInvalid()
	exactMatcher := field.ErrorMatcher{}.Exactly()

	// Dedupe imperative errors of exact error matches as they are
	// not intended and come from (buggy) duplicate validation calls
	// This is necessary as without deduping we could get unmatched
	// imperative errors for cases that are correct (matching)
	dedupedImperativeErrs := field.ErrorList{}
	for _, err := range imperativeErrs {
		found := false
		for _, existingErr := range dedupedImperativeErrs {
			if exactMatcher.Matches(existingErr, err) {
				found = true
				break
			}
		}
		if !found {
			dedupedImperativeErrs = append(dedupedImperativeErrs, err)
		}
	}
	imperativeErrs = dedupedImperativeErrs

	// Create a copy of declarative errors to track remaining ones
	remaining := make(field.ErrorList, len(declarativeErrs))
	copy(remaining, declarativeErrs)

	// Match each "covered" imperative error to declarative errors.
	// We use a fuzzy matching approach to find corresponding declarative errors
	// for each imperative error marked as CoveredByDeclarative.
	// As matches are found, they're removed from the 'remaining' list.
	// They are removed from `remaining` with a "1:many" mapping: for a given
	// imperative error we mark as matched all matching declarative errors
	// This allows us to:
	// 1. Detect imperative errors that should have matching declarative errors but don't
	// 2. Identify extra declarative errors with no imperative counterpart
	// Both cases indicate issues with the declarative validation implementation.
	for _, iErr := range imperativeErrs {
		if !iErr.CoveredByDeclarative {
			continue
		}

		tmp := make(field.ErrorList, 0, len(remaining))
		matchCount := 0

		for _, dErr := range remaining {
			if fuzzyMatcher.Matches(iErr, dErr) {
				matchCount++
			} else {
				tmp = append(tmp, dErr)
			}
		}

		if matchCount == 0 {
			mismatchDetails = append(mismatchDetails,
				fmt.Sprintf(
					"Unexpected difference between hand written validation and declarative validation error results, unmatched error(s) found %s. "+
						"This indicates an issue with declarative validation. %s",
					fuzzyMatcher.Render(iErr),
					recommendation,
				),
			)
		}

		remaining = tmp
	}

	// Any remaining unmatched declarative errors are considered "extra"
	for _, dErr := range remaining {
		mismatchDetails = append(mismatchDetails,
			fmt.Sprintf(
				"Unexpected difference between hand written validation and declarative validation error results, extra error(s) found %s. "+
					"This indicates an issue with declarative validation. %s",
				fuzzyMatcher.Render(dErr),
				recommendation,
			),
		)
	}

	return mismatchDetails
}

// createDeclarativeValidationPanicHandler returns a function with panic recovery logic
// that will increment the panic metric and either log or append errors based on the takeover parameter.
func createDeclarativeValidationPanicHandler(ctx context.Context, errs *field.ErrorList, takeover bool) func() {
	logger := klog.FromContext(ctx)
	return func() {
		if r := recover(); r != nil {
			// Increment the panic metric counter
			validationmetrics.Metrics.IncDeclarativeValidationPanicMetric()

			const errorFmt = "panic during declarative validation: %v"
			if takeover {
				// If takeover is enabled, output as a validation error as authoritative validator panicked and validation should error
				*errs = append(*errs, field.InternalError(nil, fmt.Errorf(errorFmt, r)))
			} else {
				// if takeover not enabled, log the panic as an error message
				logger.Error(nil, fmt.Sprintf(errorFmt, r))
			}
		}
	}
}

// panicSafeValidateFunc wraps a validation function with panic recovery logic.
// It takes a validation function with the ValidateDeclaratively signature
// and returns a function with the same signature.
// The returned function will execute the wrapped function and handle any panics by
// incrementing the panic metric, and logging an error message
// if takeover=false, and adding a validation error if takeover=true.
func panicSafeValidateFunc(
	validateFunc func(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj runtime.Object) field.ErrorList,
	takeover bool,
) func(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj runtime.Object) field.ErrorList {
	return func(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj runtime.Object) (errs field.ErrorList) {
		defer createDeclarativeValidationPanicHandler(ctx, &errs, takeover)()

		return validateFunc(ctx, options, scheme, obj)
	}
}

// panicSafeValidateUpdateFunc wraps an update validation function with panic recovery logic.
// It takes a validation function with the ValidateUpdateDeclaratively signature
// and returns a function with the same signature.
// The returned function will execute the wrapped function and handle any panics by
// incrementing the panic metric, and logging an error message
// if takeover=false, and adding a validation error if takeover=true.
func panicSafeValidateUpdateFunc(
	validateUpdateFunc func(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj, oldObj runtime.Object) field.ErrorList,
	takeover bool,
) func(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj, oldObj runtime.Object) field.ErrorList {
	return func(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj, oldObj runtime.Object) (errs field.ErrorList) {
		defer createDeclarativeValidationPanicHandler(ctx, &errs, takeover)()

		return validateUpdateFunc(ctx, options, scheme, obj, oldObj)
	}
}

// ValidateDeclarativelyWithRecovery validates obj against declarative validation tags
// with panic recovery logic. It uses the API version extracted from ctx and the
// provided scheme for validation.
//
// The ctx MUST contain requestInfo, which determines the target API for
// validation. The obj is converted to the API version using the provided scheme
// before validation occurs. The scheme MUST have the declarative validation
// registered for the requested resource/subresource.
//
// option should contain any validation options that the declarative validation
// tags expect.
//
// takeover determines if panic recovery should return validation errors (true) or
// just log warnings (false).
//
// Returns a field.ErrorList containing any validation errors. An internal error
// is included if requestInfo is missing from the context, if version
// conversion fails, or if a panic occurs during validation when
// takeover is true.
func ValidateDeclarativelyWithRecovery(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj runtime.Object, takeover bool) field.ErrorList {
	return panicSafeValidateFunc(ValidateDeclaratively, takeover)(ctx, options, scheme, obj)
}

// ValidateUpdateDeclarativelyWithRecovery validates obj and oldObj against declarative
// validation tags with panic recovery logic. It uses the API version extracted from
// ctx and the provided scheme for validation.
//
// The ctx MUST contain requestInfo, which determines the target API for
// validation. The obj is converted to the API version using the provided scheme
// before validation occurs. The scheme MUST have the declarative validation
// registered for the requested resource/subresource.
//
// option should contain any validation options that the declarative validation
// tags expect.
//
// takeover determines if panic recovery should return validation errors (true) or
// just log warnings (false).
//
// Returns a field.ErrorList containing any validation errors. An internal error
// is included if requestInfo is missing from the context, if version
// conversion fails, or if a panic occurs during validation when
// takeover is true.
func ValidateUpdateDeclarativelyWithRecovery(ctx context.Context, options sets.Set[string], scheme *runtime.Scheme, obj, oldObj runtime.Object, takeover bool) field.ErrorList {
	return panicSafeValidateUpdateFunc(ValidateUpdateDeclaratively, takeover)(ctx, options, scheme, obj, oldObj)
}
