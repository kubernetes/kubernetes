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
	"errors"
	"fmt"
	"slices"
	"strings"

	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	validationmetrics "k8s.io/apiserver/pkg/validation"
	"k8s.io/klog/v2"
)

// ValidationConfig defines how a declarative validation request may be configured.
type ValidationConfig func(*validationConfigOption)

// WithOptions sets the validation options.
// Options should contain any validation options that the declarative validation
// tags expect. These often correspond to feature gates.
func WithOptions(options []string) ValidationConfig {
	return func(config *validationConfigOption) {
		config.options = options
	}
}

// WithSubresourceMapper sets the subresource mapper for validation.
// This should be used when registering validation for polymorphic subresources like /scale.
//
// For example, the deployments/scale subresource mapper might map from:
//
//	group: apps, version: v1, subresource=scale
//
// to a target of:
//
//	group: autoscaling, version: v1, kind=Scale
//
// When set, the group version in the requestInfo of the ctx provided to a declarative validation
// request will be passed to the subresource mapper to find the group version kind of the subresource.
// Declarative validation will then convert the object to the subresource group version kind and validate it.
//
// Note that the target of the mapping contains no subresource part since the mapper is expected to
// map to the group version kind of the subresource.
func WithSubresourceMapper(subresourceMapper GroupVersionKindProvider) ValidationConfig {
	return func(config *validationConfigOption) {
		config.subresourceGVKMapper = subresourceMapper
	}
}

// WithNormalizationRules sets the normalization rules for validation.
func WithNormalizationRules(rules []field.NormalizationRule) ValidationConfig {
	return func(config *validationConfigOption) {
		config.normalizationRules = rules
	}
}

// WithDeclarativeEnforcement marks the validation configuration to indicate that it includes
// declarative validations that should follow the fine-grained Validation Lifecycle.
// When set, declarative validation is always executed regardless of feature gates.
// Authority is determined by individual tag prefixes (+k8s:alpha, +k8s:beta) and the
// DeclarativeValidationBeta safety switch.
func WithDeclarativeEnforcement() ValidationConfig {
	return func(config *validationConfigOption) {
		config.declarativeEnforcement = true
	}
}

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

type validationConfigOption struct {
	opType                 operation.Type
	options                []string
	subresourceGVKMapper   GroupVersionKindProvider
	validationIdentifier   string
	normalizationRules     []field.NormalizationRule
	declarativeEnforcement bool
}

// validateDeclaratively validates obj and oldObj against declarative
// validation tags defined in its Go type. It uses the API version extracted from
// ctx and the provided scheme for validation.
//
// The ctx MUST contain requestInfo, which determines the target API for
// validation. The obj is converted to the API version using the provided scheme
// before validation occurs. The scheme MUST have the declarative validation
// registered for the requested resource/subresource.
//
// Returns a field.ErrorList containing any validation errors. An internal error
// is included if requestInfo is missing from the context or if version
// conversion fails.
func validateDeclaratively(ctx context.Context, scheme *runtime.Scheme, obj, oldObj runtime.Object, o *validationConfigOption) field.ErrorList {
	// Find versionedGroupVersion, which identifies the API version to use for declarative validation.
	versionedGroupVersion, subresources, err := requestInfo(ctx, o.subresourceGVKMapper)
	if err != nil {
		return field.ErrorList{field.InternalError(nil, err)}
	}
	versionedObj, err := scheme.ConvertToVersion(obj, versionedGroupVersion)
	if err != nil {
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error converting to versioned type: %w", err))}
	}
	var versionedOldObj runtime.Object

	switch o.opType {
	case operation.Create:
		return scheme.Validate(ctx, o.options, versionedObj, subresources...)
	case operation.Update:
		versionedOldObj, err = scheme.ConvertToVersion(oldObj, versionedGroupVersion)
		if err != nil {
			return field.ErrorList{field.InternalError(nil, fmt.Errorf("unexpected error converting to versioned type: %w", err))}
		}
		return scheme.ValidateUpdate(ctx, o.options, versionedObj, versionedOldObj, subresources...)
	default:
		return field.ErrorList{field.InternalError(nil, fmt.Errorf("unknown operation type: %v", o.opType))}
	}
}

func requestInfo(ctx context.Context, subresourceMapper GroupVersionKindProvider) (schema.GroupVersion, []string, error) {
	requestInfo, found := genericapirequest.RequestInfoFrom(ctx)
	if !found {
		return schema.GroupVersion{}, nil, fmt.Errorf("could not find requestInfo in context")
	}
	groupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
	if subresourceMapper != nil {
		groupVersion = subresourceMapper.GroupVersionKind(groupVersion).GroupVersion()
	}
	subresources, err := parseSubresourcePath(requestInfo.Subresource)
	if err != nil {
		return schema.GroupVersion{}, nil, fmt.Errorf("unexpected error parsing subresource path: %w", err)
	}
	return groupVersion, subresources, nil

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

// compareDeclarativeErrorsAndEmitMismatches checks for mismatches between imperative and declarative validation
// and logs + emits metrics when inconsistencies are found
func compareDeclarativeErrorsAndEmitMismatches(ctx context.Context, imperativeErrs, declarativeErrs field.ErrorList, enforced bool, validationIdentifier string, normalizationRules []field.NormalizationRule) {
	logger := klog.FromContext(ctx)
	mismatchDetails := gatherDeclarativeValidationMismatches(imperativeErrs, declarativeErrs, enforced, normalizationRules)
	for _, detail := range mismatchDetails {
		// Log information about the mismatch using contextual logger
		logger.Error(nil, detail)

		// Increment the metric for the mismatch
		validationmetrics.Metrics.IncDeclarativeValidationMismatchMetric(validationIdentifier)
	}
}

// gatherDeclarativeValidationMismatches compares imperative and declarative validation errors
// and returns detailed information about any mismatches found. Errors are compared via type, field, and origin
func gatherDeclarativeValidationMismatches(imperativeErrs, declarativeErrs field.ErrorList, enforced bool, normalizationRules []field.NormalizationRule) []string {
	var mismatchDetails []string
	// short circuit here to minimize allocs for usual case of 0 validation errors
	if len(imperativeErrs) == 0 && len(declarativeErrs) == 0 {
		return mismatchDetails
	}
	// default recommendation based on enforcement status
	const (
		authoritativeMsg = "This difference should not affect system operation since hand written validation is authoritative."
		disableBetaMsg   = "Consider disabling the DeclarativeValidationBeta feature gate to keep data persisted in etcd consistent with prior versions of Kubernetes."
	)

	defaultRecommendation := authoritativeMsg
	if enforced {
		defaultRecommendation = disableBetaMsg
	}

	fuzzyMatcher := field.ErrorMatcher{}.ByType().ByOrigin().RequireOriginWhenInvalid().ByFieldNormalized(normalizationRules)

	// Dedupe imperative errors using the fuzzy matcher (type, field, and origin) as they are
	// not intended and come from (buggy) duplicate validation calls.
	// This is necessary as without deduping we could get unmatched
	// imperative errors for cases that are correct (matching).
	dedupedImperativeErrs := field.ErrorList{}
	for _, err := range imperativeErrs {
		found := false
		for _, existingErr := range dedupedImperativeErrs {
			if fuzzyMatcher.Matches(existingErr, err) {
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
			rec := defaultRecommendation
			// If the imperative error is explicitly Alpha, it is never enforced, so HV is authoritative.
			if iErr.IsAlpha() {
				rec = authoritativeMsg
			}

			mismatchDetails = append(mismatchDetails,
				fmt.Sprintf(
					"Unexpected difference between hand written validation and declarative validation error results, unmatched error(s) found %s. "+
						"This indicates an issue with declarative validation. %s",
					fuzzyMatcher.Render(iErr),
					rec,
				),
			)
		}

		remaining = tmp
	}

	// Any remaining unmatched declarative errors are considered "extra"
	for _, dErr := range remaining {
		rec := defaultRecommendation
		// If the declarative error is Alpha, it is never enforced (shadowed), so HV is authoritative.
		if dErr.IsAlpha() {
			rec = authoritativeMsg
		}

		mismatchDetails = append(mismatchDetails,
			fmt.Sprintf(
				"Unexpected difference between hand written validation and declarative validation error results, extra error(s) found %s. "+
					"This indicates an issue with declarative validation. %s",
				fuzzyMatcher.Render(dErr),
				rec,
			),
		)
	}

	return mismatchDetails
}

// createDeclarativeValidationPanicHandler returns a function with panic recovery logic
// that will increment the panic metric and either log or append errors based on the shouldFail parameter.
func createDeclarativeValidationPanicHandler(ctx context.Context, errs *field.ErrorList, shouldFail bool, validationIdentifier string) func() {
	logger := klog.FromContext(ctx)
	return func() {
		if r := recover(); r != nil {
			// Increment the panic metric counter
			validationmetrics.Metrics.IncDeclarativeValidationPanicMetric(validationIdentifier)

			const errorFmt = "panic during declarative validation: %v"
			if shouldFail {
				// If shouldFail is enabled, output as a validation error as authoritative validator panicked and validation should error
				*errs = append(*errs, field.InternalError(nil, fmt.Errorf(errorFmt, r)))
			} else {
				// if shouldFail not enabled, log the panic as an error message
				logger.Error(nil, fmt.Sprintf(errorFmt, r))
			}
		}
	}
}

// panicSafeValidateFunc wraps an validation function with panic recovery logic.
// The returned function will execute the wrapped function and handle any panics by
// incrementing the panic metric, and logging an error message
// if shouldFail=false, and adding a validation error if shouldFail=true.
func panicSafeValidateFunc(
	validateFunc func(ctx context.Context, scheme *runtime.Scheme, obj, oldObj runtime.Object, o *validationConfigOption) field.ErrorList,
	shouldFail bool, validationIdentifier string,
) func(ctx context.Context, scheme *runtime.Scheme, obj, oldObj runtime.Object, o *validationConfigOption) field.ErrorList {
	return func(ctx context.Context, scheme *runtime.Scheme, obj, oldObj runtime.Object, o *validationConfigOption) (errs field.ErrorList) {
		defer createDeclarativeValidationPanicHandler(ctx, &errs, shouldFail, validationIdentifier)()

		return validateFunc(ctx, scheme, obj, oldObj, o)
	}
}

func metricIdentifier(ctx context.Context, scheme *runtime.Scheme, obj runtime.Object, opType operation.Type) (string, error) {
	var errs error
	var identifier string

	identifier = "unknown_resource"
	// Use kind for identifier.
	if obj != nil && scheme != nil {
		gvks, _, err := scheme.ObjectKinds(obj)
		if err != nil {
			errs = errors.Join(errs, err)
		}
		if len(gvks) > 0 {
			identifier = strings.ToLower(gvks[0].Kind)
		}
	}

	// Use requestInfo for subresource.
	requestInfo, found := genericapirequest.RequestInfoFrom(ctx)
	if !found {
		errs = errors.Join(errs, fmt.Errorf("could not find requestInfo in context"))
	} else if len(requestInfo.Subresource) > 0 {
		// subresource can be a path, so replace '/' with '_'
		identifier += "_" + strings.ReplaceAll(requestInfo.Subresource, "/", "_")
	}

	switch opType {
	case operation.Create:
		identifier += "_create"
	case operation.Update:
		identifier += "_update"
	default:
		errs = errors.Join(errs, fmt.Errorf("unknown operation type: %v", opType))
		identifier += "_unknown_op"
	}
	return identifier, errs
}

// ValidateDeclarativelyWithMigrationChecks executes declarative validation and implements the Validation Lifecycle strategy.
// It manages the transition from handwritten (HV) to declarative (DV) validation by controlling enforcement:
//   - Standard: Enforced if declarativeEnforcement is set. HV counterparts are expected to be deleted from source.
//   - Beta: Enforced if declarativeEnforcement is set AND DeclarativeValidationBeta feature gate is enabled.
//     When enforced, corresponding HV errors are filtered out. Otherwise, DV is shadowed.
//   - Alpha: Always shadowed; HV remains authoritative.
//
// Mismatches between HV and DV are logged if the DeclarativeValidation gate is enabled.
// Mismatch checking is limited to Alpha and Beta stages when explicit enforcement is active.
//
// For testing purposes, WithAllDeclarativeEnforcedForTest can be used to enforce all declarative validations
// regardless of feature gates and filter all covered handwritten validations.
func ValidateDeclarativelyWithMigrationChecks(ctx context.Context, scheme *runtime.Scheme, obj, oldObj runtime.Object, errs field.ErrorList, opType operation.Type, configOpts ...ValidationConfig) field.ErrorList {
	declarativeValidationEnabled := utilfeature.DefaultFeatureGate.Enabled(features.DeclarativeValidation)
	betaEnabled := utilfeature.DefaultFeatureGate.Enabled(features.DeclarativeValidationBeta)
	// allDeclarativeEnforced indicates that we should check all declarative errors for testing purposes.
	allDeclarativeEnforced := ctx.Value(allDeclarativeEnforcedKey) == true

	validationIdentifier, err := metricIdentifier(ctx, scheme, obj, opType)
	if err != nil {
		// Log the error, but continue with the best-effort identifier.
		klog.FromContext(ctx).Error(err, "failed to generate complete validation identifier for declarative validation")
	}

	// Directly create the config and call the core validation logic.
	cfg := &validationConfigOption{
		opType:               opType,
		validationIdentifier: validationIdentifier,
	}
	for _, opt := range configOpts {
		opt(cfg)
	}

	// Short-circuit if neither DeclarativeValidation is enabled nor the object is explicitly configured for declarative enforcement.
	if !declarativeValidationEnabled && !cfg.declarativeEnforcement && !allDeclarativeEnforced {
		return errs
	}

	// Call the panic-safe wrapper with the real validation function.
	// We should fail if validation is enforced.
	declarativeErrs := panicSafeValidateFunc(validateDeclaratively, cfg.declarativeEnforcement, cfg.validationIdentifier)(ctx, scheme, obj, oldObj, cfg)

	if declarativeValidationEnabled {
		// Log mismatches.
		// When explicit strategy is used (declarativeEnforcement), Standard errors are authoritative
		// and may not have handwritten counterparts (e.g., in new APIs).
		// We only mismatch check Alpha and Beta errors in this mode.
		mismatchCandidateErrs := declarativeErrs
		if cfg.declarativeEnforcement {
			mismatchCandidateErrs = nil
			for _, err := range declarativeErrs {
				if err.IsAlpha() || err.IsBeta() {
					mismatchCandidateErrs = append(mismatchCandidateErrs, err)
				}
			}
		}

		// We pass betaEnabled (and enforcement) as the takeover flag to avoid changing logic elsewhere for now.
		compareDeclarativeErrorsAndEmitMismatches(ctx, errs, mismatchCandidateErrs, cfg.declarativeEnforcement && betaEnabled, validationIdentifier, cfg.normalizationRules)
	}

	if !cfg.declarativeEnforcement && !allDeclarativeEnforced {
		// If enforcement is not enabled, we shadow declarative errors with hand-written ones, so we return early here.
		return errs
	}

	// Filter HV errors
	errs = filterHandwrittenErrors(errs, allDeclarativeEnforced, betaEnabled)

	// Append Enforced DV errors
	for _, dvErr := range declarativeErrs {
		if allDeclarativeEnforced {
			errs = append(errs, dvErr)
			continue
		}
		switch {
		case dvErr.Type == field.ErrorTypeInternal:
			errs = append(errs, dvErr)
		case dvErr.IsBeta():
			if betaEnabled {
				errs = append(errs, dvErr)
			}
		case !dvErr.IsAlpha():
			errs = append(errs, dvErr) // Standard
		}
	}

	return errs
}

func filterHandwrittenErrors(errs field.ErrorList, allDeclarativeEnforced, betaEnabled bool) field.ErrorList {
	// We remove HV errors that are covered by declarative validation AND are enforced.
	return errs.Filter(func(e error) bool {
		var fe *field.Error
		if !errors.As(e, &fe) || !fe.CoveredByDeclarative {
			return false
		}

		if allDeclarativeEnforced {
			return true
		}

		// Explicit Strategy
		if fe.IsBeta() {
			// Beta validations are enforced only if the Beta feature gate is enabled.
			return betaEnabled
		}
		// For Standard validations, we keep the handwritten error for now to avoid losing coverage
		// before it is deleted from source. Alpha validations are always shadowed (kept).
		return false
	})
}

// RecordDuplicateValidationErrors increments a metric and log the error when duplicate validation errors are found.
func RecordDuplicateValidationErrors(ctx context.Context, qualifiedKind schema.GroupKind, errs field.ErrorList) {
	logger := klog.FromContext(ctx)
	seenErrs := make([]string, 0, len(errs))

	for _, err := range errs {
		errStr := fmt.Sprintf("%v", err)

		if slices.Contains(seenErrs, errStr) {
			logger.Info("Found duplicate validation error", "kind", qualifiedKind.String(), "error", errStr)
			validationmetrics.Metrics.IncDuplicateValidationErrorMetric()
		} else {
			seenErrs = append(seenErrs, errStr)
		}
	}
}
