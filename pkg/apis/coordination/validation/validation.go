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

package validation

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/blang/semver/v4"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate"
	"k8s.io/apimachinery/pkg/api/validate/content"
	"k8s.io/apimachinery/pkg/api/validation"
	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	validation2 "k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/util/sets"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/kubernetes/pkg/apis/core"
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/utils/clock"
)

var validLeaseStrategies = []coordination.CoordinatedLeaseStrategy{coordination.OldestEmulationVersion}

// ValidateLease validates a Lease.
func ValidateLease(lease *coordination.Lease) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&lease.ObjectMeta, true, validation.NameIsDNSSubdomain, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLeaseSpec(&lease.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateLeaseUpdate validates an update of Lease object.
func ValidateLeaseUpdate(lease, oldLease *coordination.Lease) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&lease.ObjectMeta, &oldLease.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLeaseSpec(&lease.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateLeaseSpec validates spec of Lease.
func ValidateLeaseSpec(spec *coordination.LeaseSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if spec.LeaseDurationSeconds != nil && *spec.LeaseDurationSeconds <= 0 {
		fld := fldPath.Child("leaseDurationSeconds")
		allErrs = append(allErrs, field.Invalid(fld, spec.LeaseDurationSeconds, "must be greater than 0"))
	}
	if spec.LeaseTransitions != nil && *spec.LeaseTransitions < 0 {
		fld := fldPath.Child("leaseTransitions")
		allErrs = append(allErrs, field.Invalid(fld, spec.LeaseTransitions, "must be greater than or equal to 0"))
	}
	if spec.Strategy != nil {
		allErrs = append(allErrs, ValidateCoordinatedLeaseStrategy(*spec.Strategy, fldPath.Child("strategy"))...)
	}
	if spec.PreferredHolder != nil && *spec.PreferredHolder != "" && (spec.Strategy == nil || *spec.Strategy == "") {
		allErrs = append(allErrs, field.Forbidden(fldPath.Child("preferredHolder"), "may only be specified if `strategy` is defined"))
	}
	// spec.RenewTime is a MicroTime and doesn't need further validation
	return allErrs
}

// ValidateLeaseCandidate validates a LeaseCandidate.
func ValidateLeaseCandidate(lease *coordination.LeaseCandidate) field.ErrorList {
	allErrs := validation.ValidateObjectMeta(&lease.ObjectMeta, true, ValidLeaseCandidateName, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLeaseCandidateSpec(&lease.Spec, field.NewPath("spec"))...)
	return allErrs
}

func ValidLeaseCandidateName(name string, prefix bool) []string {
	// prefix is already handled by IsConfigMapKey, a trailing - is permitted.
	return utilvalidation.IsConfigMapKey(name)
}

func ValidateLeaseCandidateSpecUpdate(leaseCandidateSpec, oldLeaseCandidateSpec *coordination.LeaseCandidateSpec) field.ErrorList {
	allErrs := field.ErrorList{}
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(leaseCandidateSpec.LeaseName, oldLeaseCandidateSpec.LeaseName, field.NewPath("spec").Child("leaseName"))...)
	return allErrs
}

// ValidateLeaseCandidateUpdate validates an update of LeaseCandidate object.
func ValidateLeaseCandidateUpdate(leaseCandidate, oldLeaseCandidate *coordination.LeaseCandidate) field.ErrorList {
	allErrs := validation.ValidateObjectMetaUpdate(&leaseCandidate.ObjectMeta, &oldLeaseCandidate.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateLeaseCandidateSpec(&leaseCandidate.Spec, field.NewPath("spec"))...)
	allErrs = append(allErrs, ValidateLeaseCandidateSpecUpdate(&leaseCandidate.Spec, &oldLeaseCandidate.Spec)...)
	return allErrs
}

// ValidateLeaseCandidateSpec validates spec of LeaseCandidate.
func ValidateLeaseCandidateSpec(spec *coordination.LeaseCandidateSpec, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(spec.LeaseName) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("leaseName"), ""))
	}

	ev := semver.Version{}
	if spec.EmulationVersion != "" {
		var err error
		ev, err = semver.Parse(spec.EmulationVersion)
		if err != nil {
			fld := fldPath.Child("emulationVersion")
			allErrs = append(allErrs, field.Invalid(fld, spec.EmulationVersion, "must be a valid semantic version"))
		}
	}
	bv := semver.Version{}
	if spec.BinaryVersion == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("binaryVersion"), ""))
	} else {
		var err error
		bv, err = semver.Parse(spec.BinaryVersion)
		if err != nil {
			fld := fldPath.Child("binaryVersion")
			allErrs = append(allErrs, field.Invalid(fld, spec.BinaryVersion, "must be a valid semantic version"))
		}
	}
	if spec.BinaryVersion != "" && spec.EmulationVersion != "" && bv.LT(ev) {
		fld := fldPath.Child("binaryVersion")
		allErrs = append(allErrs, field.Invalid(fld, spec.BinaryVersion, "must be greater than or equal to `emulationVersion`"))
	}

	if spec.Strategy == "" {
		allErrs = append(allErrs, field.Required(fldPath.Child("strategy"), ""))
	} else {
		fld := fldPath.Child("strategy")
		if spec.Strategy == coordination.OldestEmulationVersion {
			zeroVersion := semver.Version{}
			if ev.EQ(zeroVersion) {
				allErrs = append(allErrs, field.Required(fldPath.Child("emulationVersion"), "must be specified when `strategy` is 'OldestEmulationVersion'"))
			}
		}

		allErrs = append(allErrs, ValidateCoordinatedLeaseStrategy(spec.Strategy, fld)...)
	}
	// spec.PingTime is a MicroTime and doesn't need further validation
	// spec.RenewTime is a MicroTime and doesn't need further validation
	return allErrs
}

// ValidateCoordinatedLeaseStrategy validates the Strategy field in both the Lease and LeaseCandidate
func ValidateCoordinatedLeaseStrategy(strategy coordination.CoordinatedLeaseStrategy, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	parts := strings.Split(string(strategy), "/")
	switch len(parts) {
	case 1:
		// Must be a Kubernetes-defined name.
		if !slices.Contains(validLeaseStrategies, coordination.CoordinatedLeaseStrategy(parts[0])) {
			allErrs = append(allErrs, field.NotSupported(fldPath.Child("strategy"), strategy, validLeaseStrategies))
		}
	default:
		if msgs := utilvalidation.IsQualifiedName(string(strategy)); len(msgs) > 0 {
			for _, msg := range msgs {
				allErrs = append(allErrs, field.Invalid(fldPath.Child("strategy"), strategy, msg))
			}
		}
	}
	return allErrs
}

type EvictionRequestSpecValidationOptions struct {
	// Name of the eviction request
	EvictionRequestName string
}

// ValidateEvictionRequest validates an EvictionRequest.
func ValidateEvictionRequest(evictionRequest *coordination.EvictionRequest) field.ErrorList {
	var allErrs field.ErrorList
	metadataFldPath := field.NewPath("metadata")
	if len(evictionRequest.GenerateName) != 0 {
		allErrs = append(allErrs, field.Forbidden(metadataFldPath.Child("generateName"), "").MarkCoveredByDeclarative())
	}
	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&evictionRequest.ObjectMeta, true, func(name string, prefix bool) []string {
		if prefix {
			// no need for uuid check and an error because generateName is forbidden
			return nil
		}
		return apivalidation.IsUUID(name)
	}, metadataFldPath)...)
	specOpts := EvictionRequestSpecValidationOptions{EvictionRequestName: evictionRequest.Name}
	allErrs = append(allErrs, ValidateEvictionRequestSpec(&evictionRequest.Spec, field.NewPath("spec"), specOpts)...)
	return allErrs
}

// ValidateEvictionRequestSpec validates an ValidateEvictionRequestSpec.
func ValidateEvictionRequestSpec(evictionRequestSpec *coordination.EvictionRequestSpec, fldPath *field.Path, opts EvictionRequestSpecValidationOptions) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, ValidateEvictionTarget(evictionRequestSpec.Target, fldPath.Child("target"), opts)...)
	allErrs = append(allErrs, ValidateRequesters(evictionRequestSpec.Requesters, nil, fldPath.Child("requesters"), operation.Create)...)
	return allErrs
}

func ValidateEvictionTarget(evictionTarget coordination.EvictionTarget, fldPath *field.Path, opts EvictionRequestSpecValidationOptions) field.ErrorList {
	var allErrs field.ErrorList
	// union
	var unionMembership = validate.NewUnionMembership(validate.NewUnionMember("pod"))
	allErrs = append(allErrs, validate.Union(context.TODO(), operation.Operation{Type: operation.Create}, fldPath, &evictionTarget, nil, unionMembership, func(obj *coordination.EvictionTarget) bool {
		if obj == nil {
			return false
		}
		return obj.Pod != nil
	}).MarkCoveredByDeclarative()...)
	if evictionTarget.Pod != nil {
		allErrs = append(allErrs, ValidateLocalTargetReference(*evictionTarget.Pod, fldPath.Child("pod"), opts)...)
	}
	return allErrs
}

func ValidateLocalTargetReference(localTargetReference coordination.LocalTargetReference, fldPath *field.Path, opts EvictionRequestSpecValidationOptions) field.ErrorList {
	var allErrs field.ErrorList
	// name
	if len(localTargetReference.Name) == 0 {
		allErrs = append(allErrs, field.Required(fldPath.Child("name"), "")).MarkCoveredByDeclarative()
	}
	// uid
	uidFldPath := fldPath.Child("uid")
	if len(localTargetReference.UID) == 0 {
		allErrs = append(allErrs, field.Required(uidFldPath, "")).MarkCoveredByDeclarative()
	} else {
		if errs := apivalidation.IsUUID(localTargetReference.UID); len(errs) > 0 {
			allErrs = append(allErrs, field.Invalid(uidFldPath, localTargetReference.UID, strings.Join(errs, ", ")))
		}
		if localTargetReference.UID != opts.EvictionRequestName {
			msg := fmt.Sprintf("must be the same value as %s", uidFldPath.String())
			allErrs = append(allErrs, field.Forbidden(field.NewPath("metadata").Child("name"), msg))
		}
	}
	return allErrs
}

func ValidateRequesters(requesters, oldRequesters []coordination.Requester, fldPath *field.Path, operationType operation.Type) field.ErrorList {
	var allErrs field.ErrorList
	if operationType == operation.Create && len(requesters) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, "must have at least one requester on EvictionRequest creation"))
	}
	if operationType == operation.Update && len(requesters) > 0 && len(oldRequesters) == 0 {
		// eviction request cancellation - should be picked up by the controller
		allErrs = append(allErrs, field.Invalid(fldPath, requesters, validation.FieldImmutableErrorMsg).WithOrigin("immutable"))
	}
	if maximum := 100; len(requesters) > maximum {
		return field.ErrorList{field.TooMany(fldPath, len(requesters), maximum).WithOrigin("maxItems")}.MarkCoveredByDeclarative()
	}
	allErrs = append(allErrs, validate.Unique(context.TODO(), operation.Operation{Type: operationType}, fldPath, requesters, oldRequesters,
		func(a coordination.Requester, b coordination.Requester) bool { return a.Name == b.Name }).MarkCoveredByDeclarative()...)

	for i, requester := range requesters {
		allErrs = append(allErrs, ValidateRequester(requester, fldPath.Index(i))...)
	}
	return allErrs
}

func ValidateRequester(requester coordination.Requester, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	namePath := fldPath.Child("name")
	if len(requester.Name) == 0 {
		allErrs = append(allErrs, field.Required(namePath, "")).MarkCoveredByDeclarative()
	} else {
		allErrs = append(allErrs, apivalidation.ValidateEvictionRequestParticipantName(namePath, requester.Name, apivalidation.EvictionRequestParticipantReservedSuffixes)...)
	}
	return allErrs
}

// ValidateEvictionRequestUpdate validates an EvictionRequest.
func ValidateEvictionRequestUpdate(evictionRequest, oldEvictionRequest *coordination.EvictionRequest) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&evictionRequest.ObjectMeta, &oldEvictionRequest.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateEvictionRequestSpecUpdate(&evictionRequest.Spec, &oldEvictionRequest.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateEvictionRequestSpec validates an ValidateEvictionRequestSpec.
func ValidateEvictionRequestSpecUpdate(evictionRequestSpec, oldEvictionRequestSpec *coordination.EvictionRequestSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, apivalidation.ValidateImmutableField(evictionRequestSpec.Target, oldEvictionRequestSpec.Target, fldPath.Child("target")).WithOrigin("immutable").MarkCoveredByDeclarative()...)
	allErrs = append(allErrs, ValidateRequesters(evictionRequestSpec.Requesters, oldEvictionRequestSpec.Requesters, fldPath.Child("requesters"), operation.Update)...)
	return allErrs
}

type EvictionRequestStatusValidationOptions struct {
	Clock clock.PassiveClock
}

// ValidateEvictionRequestStatusUpdate validates an EvictionRequest.
func ValidateEvictionRequestStatusUpdate(evictionRequest, oldEvictionRequest *coordination.EvictionRequest, opts EvictionRequestStatusValidationOptions) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&evictionRequest.ObjectMeta, &oldEvictionRequest.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateEvictionRequestStatus(&evictionRequest.Status, &oldEvictionRequest.Status, field.NewPath("status"), opts)...)
	return allErrs
}

// ValidateEvictionRequestStatus validates an EvictionRequestStatus.
func ValidateEvictionRequestStatus(status, oldStatus *coordination.EvictionRequestStatus, fldPath *field.Path, opts EvictionRequestStatusValidationOptions) field.ErrorList {
	var allErrs field.ErrorList

	// observedGeneration
	observedGenerationPath := fldPath.Child("observedGeneration")
	if minimum := int64(0); status.ObservedGeneration < minimum {
		allErrs = append(allErrs, field.Invalid(observedGenerationPath, status.ObservedGeneration, content.MinError(minimum)).WithOrigin("minimum")).MarkCoveredByDeclarative()
	} else if status.ObservedGeneration < oldStatus.ObservedGeneration {
		allErrs = append(allErrs, field.Invalid(observedGenerationPath, status.ObservedGeneration, "cannot decrement, "+content.MinError(oldStatus.ObservedGeneration)))
	}

	// conditions
	conditionsPath := fldPath.Child("conditions")
	allErrs = append(allErrs, validate.Unique(context.TODO(), operation.Operation{}, conditionsPath, status.Conditions, nil,
		func(a v1.Condition, b v1.Condition) bool { return a.Type == b.Type }).MarkCoveredByDeclarative()...)
	for i, condition := range status.Conditions {
		allErrs = append(allErrs, validation2.ValidateCondition(condition, conditionsPath.Index(i))...)
	}
	isEvicted := meta.IsStatusConditionTrue(status.Conditions, string(coordination.EvictionRequestConditionEvicted))
	isCanceled := meta.IsStatusConditionTrue(status.Conditions, string(coordination.EvictionRequestConditionCanceled))
	for _, oldCondition := range oldStatus.Conditions {
		if oldCondition.Type == string(coordination.EvictionRequestConditionEvicted) || oldCondition.Type == string(coordination.EvictionRequestConditionCanceled) {
			newCondtion := meta.FindStatusCondition(status.Conditions, oldCondition.Type)
			if oldCondition.Status == v1.ConditionTrue && !validate.SemanticDeepEqual(&oldCondition, newCondtion) {
				allErrs = append(allErrs, field.Invalid(conditionsPath, status.Conditions, fmt.Sprintf("%s condition is immutable", oldCondition.Type)))
				if oldCondition.Type == string(coordination.EvictionRequestConditionEvicted) {
					isEvicted = true // do not use invalid condition state for next validations
				}
				if oldCondition.Type == string(coordination.EvictionRequestConditionCanceled) {
					isCanceled = true // do not use invalid condition state for next validations
				}
			}
		}
	}

	// targetInterceptors, activeInterceptors, processedInterceptors, and interceptors
	allErrs = append(allErrs, ValidateAllEvictionRequestStatusInterceptorFields(status, oldStatus, fldPath, EvictionRequestStatusInterceptorsValidationOptions{
		Clock:      opts.Clock,
		IsEvicted:  isEvicted,
		IsCanceled: isCanceled,
	})...)

	return allErrs
}

type EvictionRequestStatusInterceptorsValidationOptions struct {
	Clock      clock.PassiveClock
	IsEvicted  bool
	IsCanceled bool
}

// ValidateAllEvictionRequestStatusInterceptorFields validates .status.targetInterceptors, .status.activeInterceptors,
// .status.processedInterceptors, and .status.interceptors
//
// Multiple actors are expected to update the status (evictionrequest-controller, interceptors and requesters updating conditions).
// We have to emulate the evictionrequest-controller behavior, to prevent invalid updates by other misbehaving/malicious actors.
//  1. .status.targetInterceptors and .status.interceptors should be set first.
//  2. Interceptors are copied from .status.targetInterceptors -> .status.activeInterceptors gradually in the order
//     they appear in the first list. This is not reversible.
//  3. Interceptors are moved from .status.activeInterceptors -> .status.processedInterceptors when
//     .status.interceptors[].completionTime is set or when heartbeat is exceeded.
//     Items cannot be removed from .status.processedInterceptors.
//  4. .status.interceptors items cannot be removed once set. Only active interceptors can mutate it.
//  5. The controller can mark an EvictionRequest as Evicted or Canceled via conditions. We then have to allow removal
//     from activeInterceptors and prevent any new ones.
func ValidateAllEvictionRequestStatusInterceptorFields(status, oldStatus *coordination.EvictionRequestStatus, fldPath *field.Path, opts EvictionRequestStatusInterceptorsValidationOptions) field.ErrorList {
	var allErrs field.ErrorList
	if len(status.TargetInterceptors) == 0 {
		allErrs = append(allErrs, apivalidation.ValidateImmutableField(status.ActiveInterceptors, oldStatus.ActiveInterceptors, fldPath.Child("activeInterceptors")).WithOrigin("immutable")...)
		allErrs = append(allErrs, apivalidation.ValidateImmutableField(status.ProcessedInterceptors, oldStatus.ProcessedInterceptors, fldPath.Child("processedInterceptors")).WithOrigin("immutable")...)
		allErrs = append(allErrs, apivalidation.ValidateImmutableField(status.Interceptors, oldStatus.Interceptors, fldPath.Child("interceptors")).WithOrigin("immutable")...)
		// all interceptors' fields shouldn't be allowed to be set, until we have target interceptors
		return allErrs
	}
	defaultInterceptorCount := 1 // EvictionInterceptorImperativeEviction
	maxInterceptors := 15 + defaultInterceptorCount

	heartbeatDeadline := time.Minute * 20
	heartbeatDeadlineMinDurationBetweenUpdates := time.Minute
	allowedTimeSkew := time.Second * 30
	allowedMaxExpectedCompletionTime := time.Hour * 24 * 365 * 10 // 10 years

	// from this point we can depend on the target interceptors to be set
	// define dependencies between targetInterceptors, activeInterceptors, interceptors, and processedInterceptors validation steps
	targetInterceptors := targetInterceptorsToNames(status.TargetInterceptors)
	activeInterceptors := sets.New(status.ActiveInterceptors...)
	statusInterceptors := append(make([]coordination.InterceptorStatus, 0, len(status.Interceptors)), status.Interceptors...)

	// targetInterceptors
	if errs := ValidateEvictionRequestTargetInterceptors(status.TargetInterceptors, oldStatus.TargetInterceptors, fldPath.Child("targetInterceptors"), ValidateEvictionRequestTargetInterceptorsOptions{
		MaxInterceptors: maxInterceptors,
	}); len(errs) > 0 {
		allErrs = append(allErrs, errs...)
		targetInterceptors = targetInterceptorsToNames(oldStatus.TargetInterceptors) // do not use invalid data for next validation steps
	}

	// activeInterceptors
	if errs := ValidateEvictionRequestActiveInterceptors(status.ActiveInterceptors, oldStatus.ActiveInterceptors, fldPath.Child("activeInterceptors"), ValidateEvictionRequestActiveInterceptorsOptions{
		IsCanceled:            opts.IsCanceled,
		IsEvicted:             opts.IsEvicted,
		TargetInterceptors:    targetInterceptors,
		ProcessedInterceptors: sets.New[string](status.ProcessedInterceptors...),
	}); len(errs) > 0 {
		allErrs = append(allErrs, errs...)
		activeInterceptors = activeInterceptors.Clear().Insert(oldStatus.ActiveInterceptors...) // do not use invalid data for next validation steps
	}

	// interceptors
	if errs := ValidateEvictionRequestStatusInterceptors(status.Interceptors, oldStatus.Interceptors, fldPath.Child("interceptors"), ValidateEvictionRequestStatusInterceptorsOptions{
		Clock:              opts.Clock,
		TargetInterceptors: targetInterceptors,
		ActiveInterceptors: activeInterceptors,
		MaxInterceptors:    maxInterceptors,
		AllowedTimeSkew:    allowedTimeSkew,
		HeartbeatDeadlineMinDurationBetweenUpdates: heartbeatDeadlineMinDurationBetweenUpdates,
		MaxExpectedCompletionTime:                  allowedMaxExpectedCompletionTime,
	}); len(errs) > 0 {
		allErrs = append(allErrs, errs...)
		statusInterceptors = append(make([]coordination.InterceptorStatus, 0, len(oldStatus.Interceptors)), oldStatus.Interceptors...) // do not use invalid data for next validation steps
	}

	// processedInterceptors
	allErrs = append(allErrs, ValidateEvictionRequestProcessedInterceptors(status.ProcessedInterceptors, oldStatus.ProcessedInterceptors, fldPath.Child("processedInterceptors"), ValidateEvictionRequestProcessedInterceptorsOptions{
		Clock:                 opts.Clock,
		IsCanceled:            opts.IsCanceled,
		IsEvicted:             opts.IsEvicted,
		TargetInterceptors:    sets.New[string](targetInterceptors...),
		OldActiveInterceptors: sets.New[string](oldStatus.ActiveInterceptors...),
		StatusInterceptors:    statusInterceptors,
		MaxInterceptors:       maxInterceptors,
		HeartbeatDeadline:     heartbeatDeadline,
		AllowedTimeSkew:       allowedTimeSkew,
	})...)

	return allErrs
}

func targetInterceptorsToNames(targetInterceptors []core.EvictionInterceptor) []string {
	targetInterceptorNames := make([]string, 0, len(targetInterceptors))
	for _, interceptor := range targetInterceptors {
		targetInterceptorNames = append(targetInterceptorNames, interceptor.Name)
	}
	return targetInterceptorNames
}

type ValidateEvictionRequestTargetInterceptorsOptions struct {
	MaxInterceptors int
}

func ValidateEvictionRequestTargetInterceptors(targetInterceptors, oldTargetInterceptors []core.EvictionInterceptor, fldPath *field.Path, opts ValidateEvictionRequestTargetInterceptorsOptions) field.ErrorList {
	var allErrs field.ErrorList
	if len(oldTargetInterceptors) != 0 {
		return append(allErrs, apivalidation.ValidateImmutableField(targetInterceptors, oldTargetInterceptors, fldPath).WithOrigin("immutable")...)
	}
	allErrs = append(allErrs, apivalidation.ValidateEvictionInterceptors(targetInterceptors, fldPath, apivalidation.EvictionInterceptorValidationOptions{
		MaxItems:                     opts.MaxInterceptors, // more than .pod.spec.evictionInterceptors to account for default interceptors
		ForbiddenReservedSuffixes:    nil,                  // unlike in a pod, there are no forbidden prefixes in the status
		MarkDeclarativeErrorsCovered: true,
	})...)
	return allErrs
}

type ValidateEvictionRequestActiveInterceptorsOptions struct {
	IsEvicted             bool
	IsCanceled            bool
	TargetInterceptors    []string
	ProcessedInterceptors sets.Set[string]
}

func ValidateEvictionRequestActiveInterceptors(activeInterceptors, oldActiveInterceptors []string, fldPath *field.Path, opts ValidateEvictionRequestActiveInterceptorsOptions) field.ErrorList {
	var allErrs field.ErrorList
	targetInterceptorsSet := sets.New(opts.TargetInterceptors...)
	activeInterceptorsSet := sets.New(activeInterceptors...)
	oldActiveInterceptorsSet := sets.New(oldActiveInterceptors...)
	// there can be only 1 active interceptor
	maxActiveInterceptors := 1
	nextInterceptor := ""
	for _, targetInterceptor := range opts.TargetInterceptors {
		if !opts.ProcessedInterceptors.Has(targetInterceptor) {
			nextInterceptor = targetInterceptor
			break
		}
	}

	// do not allow new active once we are evicted or canceled
	if activeInterceptorsSet.Difference(oldActiveInterceptorsSet).Len() > 0 && (opts.IsEvicted || opts.IsCanceled) {
		return append(allErrs, field.Invalid(fldPath, activeInterceptors, validation.FieldImmutableErrorMsg).WithOrigin("immutable"))
	}
	// in case we completed all active, but evicted or canceled is not set yet
	if len(activeInterceptors) > 0 && len(nextInterceptor) == 0 {
		return append(allErrs, field.Forbidden(fldPath, "must not be set because all interceptors have been processed"))
	}
	// guard against removal of active interceptors
	for _, interceptor := range oldActiveInterceptors {
		if !activeInterceptorsSet.Has(interceptor) && !opts.ProcessedInterceptors.Has(interceptor) {
			return append(allErrs, field.Forbidden(fldPath, "items cannot be removed, unless they are added to status.processedInterceptors"))
		}
	}
	if len(activeInterceptors) > maxActiveInterceptors {
		return append(allErrs, field.TooMany(fldPath, len(activeInterceptors), maxActiveInterceptors).WithOrigin("maxItems").MarkCoveredByDeclarative())
	}
	allErrs = append(allErrs, validate.Unique(context.TODO(), operation.Operation{Type: operation.Update}, fldPath, activeInterceptors, oldActiveInterceptors, validate.DirectEqual).MarkCoveredByDeclarative()...)

	for i, activeInterceptor := range activeInterceptors {
		if i >= maxActiveInterceptors {
			break // rest is caught with TooMany
		}
		activeInterceptorPath := fldPath.Index(i)
		if !targetInterceptorsSet.Has(activeInterceptor) {
			allErrs = append(allErrs, field.Invalid(activeInterceptorPath, activeInterceptor, "is not a valid interceptor from status.targetInterceptors"))
			continue
		}
		if len(nextInterceptor) > 0 && activeInterceptor != nextInterceptor {
			allErrs = append(allErrs, field.Forbidden(activeInterceptorPath, fmt.Sprintf("must be %q because this interceptor is next in line", nextInterceptor)))
		}

	}
	return allErrs
}

type ValidateEvictionRequestStatusInterceptorsOptions struct {
	Clock                                      clock.PassiveClock
	TargetInterceptors                         []string
	ActiveInterceptors                         sets.Set[string]
	MaxInterceptors                            int
	AllowedTimeSkew                            time.Duration
	HeartbeatDeadlineMinDurationBetweenUpdates time.Duration
	MaxExpectedCompletionTime                  time.Duration
}

func ValidateEvictionRequestStatusInterceptors(statusInterceptors, oldStatusInterceptors []coordination.InterceptorStatus, fldPath *field.Path, opts ValidateEvictionRequestStatusInterceptorsOptions) field.ErrorList {
	var allErrs field.ErrorList
	targetInterceptorsSet := sets.NewString(opts.TargetInterceptors...)

	// become required when .status.targetInterceptors are set
	if len(statusInterceptors) == 0 {
		return append(allErrs, field.Required(fldPath, ""))
	}
	if len(opts.TargetInterceptors) != len(statusInterceptors) {
		return append(allErrs, field.Invalid(fldPath, statusInterceptors, "should be the same length as status.targetInterceptors and contain the same keys in the same order"))
	} else {
		for i, targetInterceptor := range opts.TargetInterceptors {
			if targetInterceptor != statusInterceptors[i].Name {
				return append(allErrs, field.Invalid(fldPath, statusInterceptors, "should contain the same keys in the same order as status.targetInterceptors"))
			}
		}
	}
	if maximum := opts.MaxInterceptors; len(statusInterceptors) > maximum {
		return append(allErrs, field.TooMany(fldPath, len(statusInterceptors), maximum).WithOrigin("maxItems").MarkCoveredByDeclarative())
	}
	allErrs = append(allErrs, validate.Unique(context.TODO(), operation.Operation{Type: operation.Update}, fldPath, statusInterceptors, oldStatusInterceptors, func(a coordination.InterceptorStatus, b coordination.InterceptorStatus) bool {
		return a.Name == b.Name
	}).MarkCoveredByDeclarative()...)

	for i, interceptor := range statusInterceptors {
		interceptorPath := fldPath.Index(i)
		var oldInterceptorStatus *coordination.InterceptorStatus
		if i < len(oldStatusInterceptors) {
			oldInterceptorStatus = &oldStatusInterceptors[i] // +k8s:verify-mutation:reason=clone
		}
		allErrs = append(allErrs, ValidateEvictionRequestStatusInterceptor(&interceptor, oldInterceptorStatus, interceptorPath, EvictionRequestStatusInterceptorValidationOptions{
			Clock:               opts.Clock,
			IsActiveInterceptor: opts.ActiveInterceptors.Has(interceptor.Name),
			IsTargetInterceptor: targetInterceptorsSet.Has(interceptor.Name),
			AllowedTimeSkew:     opts.AllowedTimeSkew,
			HeartbeatDeadlineMinDurationBetweenUpdates: opts.HeartbeatDeadlineMinDurationBetweenUpdates,
			MaxExpectedCompletionTime:                  opts.MaxExpectedCompletionTime,
		})...)
	}
	return allErrs
}

type EvictionRequestStatusInterceptorValidationOptions struct {
	IsActiveInterceptor                        bool
	IsTargetInterceptor                        bool
	AllowedTimeSkew                            time.Duration
	HeartbeatDeadlineMinDurationBetweenUpdates time.Duration
	MaxExpectedCompletionTime                  time.Duration
	Clock                                      clock.PassiveClock
}

func ValidateEvictionRequestStatusInterceptor(status, oldStatus *coordination.InterceptorStatus, fldPath *field.Path, opts EvictionRequestStatusInterceptorValidationOptions) field.ErrorList {
	var allErrs field.ErrorList
	oldDefaultedStatus := coordination.InterceptorStatus{}
	if oldStatus != nil {
		oldDefaultedStatus = *oldStatus // +k8s:verify-mutation:reason=clone
	}

	if oldStatus != nil && !validate.SemanticDeepEqual(status, oldStatus) && !opts.IsActiveInterceptor {
		// immutable; changes to the InterceptorStatus are only allowed by the active interceptor or during initialization
		return append(allErrs, field.Invalid(fldPath, status, validation.FieldImmutableErrorMsg).WithOrigin("immutable"))
	}

	// name
	namePath := fldPath.Child("name")
	// immutable after the first initialization - short circuited by targetInterceptors key order
	if oldStatus != nil && status.Name != oldStatus.Name {
		return append(allErrs, field.Invalid(namePath, status.Name, validation.FieldImmutableErrorMsg).WithOrigin("immutable"))
	}
	if len(status.Name) == 0 {
		allErrs = append(allErrs, field.Required(namePath, "")).MarkCoveredByDeclarative()
	} else if !opts.IsTargetInterceptor {
		allErrs = append(allErrs, field.Invalid(namePath, status.Name, "is not a valid target interceptor"))
	}

	// startTime
	startTimePath := fldPath.Child("startTime")
	// immutable once set
	if oldDefaultedStatus.StartTime != nil && !oldDefaultedStatus.StartTime.Equal(status.StartTime) {
		allErrs = append(allErrs, field.Invalid(startTimePath, status.StartTime, validation.FieldImmutableErrorMsg).WithOrigin("immutable"))
	} else if status.StartTime == nil && opts.IsActiveInterceptor {
		allErrs = append(allErrs, field.Required(startTimePath, "is required for an active interceptor"))
	} else if status.StartTime != nil && !oldDefaultedStatus.StartTime.Equal(status.StartTime) && !timeNear(status.StartTime.Time, opts.Clock.Now(), opts.AllowedTimeSkew) {
		allErrs = append(allErrs, field.Invalid(startTimePath, status.StartTime, "should be set to the present time"))
	}

	// heartbeatTime
	heartbeatTimePath := fldPath.Child("heartbeatTime")
	if oldDefaultedStatus.HeartbeatTime != nil && status.HeartbeatTime == nil {
		allErrs = append(allErrs, field.Required(heartbeatTimePath, "is required once set"))
	}
	if !oldDefaultedStatus.HeartbeatTime.Equal(status.HeartbeatTime) && status.HeartbeatTime != nil {
		if status.StartTime == nil {
			allErrs = append(allErrs, field.Invalid(heartbeatTimePath, status.HeartbeatTime, fmt.Sprintf("cannot be set before %s is set", startTimePath.String())))
		} else if status.HeartbeatTime.Before(oldDefaultedStatus.HeartbeatTime) {
			// this could still happen since we allow for the skew
			allErrs = append(allErrs, field.Invalid(heartbeatTimePath, status.HeartbeatTime, "cannot be decreased"))
		} else if status.HeartbeatTime.Before(status.StartTime) {
			// this could still happen since we allow for the skew
			allErrs = append(allErrs, field.Invalid(heartbeatTimePath, status.HeartbeatTime, fmt.Sprintf("must occur after %s", startTimePath.String())))
		} else if oldDefaultedStatus.HeartbeatTime != nil && opts.Clock.Now().Add(-opts.HeartbeatDeadlineMinDurationBetweenUpdates).Before(oldDefaultedStatus.HeartbeatTime.Time) {
			allErrs = append(allErrs, field.Invalid(heartbeatTimePath, status.HeartbeatTime, fmt.Sprintf("there must be at least %s increments during subsequent updates", opts.HeartbeatDeadlineMinDurationBetweenUpdates.String())))
		} else if !timeNear(status.HeartbeatTime.Time, opts.Clock.Now(), opts.AllowedTimeSkew) {
			allErrs = append(allErrs, field.Invalid(heartbeatTimePath, status.HeartbeatTime, "should be set to the present time"))
		}
	}

	// expectedCompletionTime
	expectedCompletionTimePath := fldPath.Child("expectedCompletionTime")
	if !oldDefaultedStatus.ExpectedCompletionTime.Equal(status.ExpectedCompletionTime) && status.ExpectedCompletionTime != nil {
		if status.StartTime == nil {
			allErrs = append(allErrs, field.Invalid(expectedCompletionTimePath, status.ExpectedCompletionTime, fmt.Sprintf("cannot be set before %s is set", startTimePath.String())))
		} else if status.ExpectedCompletionTime.Before(status.StartTime) {
			// this could still happen since we allow for the skew
			allErrs = append(allErrs, field.Invalid(expectedCompletionTimePath, status.ExpectedCompletionTime, fmt.Sprintf("must occur after %s", startTimePath.String())))
		} else if status.ExpectedCompletionTime.Time.Before(opts.Clock.Now().Add(-opts.AllowedTimeSkew)) {
			allErrs = append(allErrs, field.Invalid(expectedCompletionTimePath, status.ExpectedCompletionTime, "cannot be set to the past time"))
		} else if status.ExpectedCompletionTime.Time.After(opts.Clock.Now().Add(opts.MaxExpectedCompletionTime)) {
			allErrs = append(allErrs, field.Invalid(expectedCompletionTimePath, status.ExpectedCompletionTime, "must complete within 10 years")) // sanity check
		}
	}

	// completionTime
	completionTimePath := fldPath.Child("completionTime")
	// immutable once set
	if oldDefaultedStatus.CompletionTime != nil && !oldDefaultedStatus.CompletionTime.Equal(status.CompletionTime) {
		allErrs = append(allErrs, field.Invalid(completionTimePath, status.CompletionTime, validation.FieldImmutableErrorMsg).WithOrigin("immutable"))
	} else if status.CompletionTime != nil {
		if status.StartTime == nil {
			allErrs = append(allErrs, field.Invalid(completionTimePath, status.CompletionTime, fmt.Sprintf("cannot be set before %s is set", startTimePath.String())))
		} else if status.CompletionTime.Before(status.StartTime) {
			allErrs = append(allErrs, field.Invalid(completionTimePath, status.CompletionTime, fmt.Sprintf("must occur after %s", startTimePath.String())))
		} else if !oldDefaultedStatus.CompletionTime.Equal(status.CompletionTime) && !timeNear(status.CompletionTime.Time, opts.Clock.Now(), opts.AllowedTimeSkew) {
			allErrs = append(allErrs, field.Invalid(completionTimePath, status.CompletionTime, "should be set to the present time"))
		}
	}

	// message
	// truncated instead of validation
	return allErrs
}

type ValidateEvictionRequestProcessedInterceptorsOptions struct {
	Clock                 clock.PassiveClock
	IsEvicted             bool
	IsCanceled            bool
	TargetInterceptors    sets.Set[string]
	OldActiveInterceptors sets.Set[string]
	StatusInterceptors    []coordination.InterceptorStatus
	MaxInterceptors       int
	HeartbeatDeadline     time.Duration
	AllowedTimeSkew       time.Duration
}

func ValidateEvictionRequestProcessedInterceptors(processedInterceptors, oldProcessedInterceptors []string, fldPath *field.Path, opts ValidateEvictionRequestProcessedInterceptorsOptions) field.ErrorList {
	var allErrs field.ErrorList

	if len(processedInterceptors) < len(oldProcessedInterceptors) {
		return append(allErrs, field.Forbidden(fldPath, "items cannot be removed"))
	}
	if len(processedInterceptors) > len(oldProcessedInterceptors)+1 {
		return append(allErrs, field.Forbidden(fldPath, "items can only be added one at a time"))
	}
	if maximum := opts.MaxInterceptors; len(processedInterceptors) > maximum {
		return append(allErrs, field.TooMany(fldPath, len(processedInterceptors), maximum).WithOrigin("maxItems").MarkCoveredByDeclarative())
	}
	allErrs = append(allErrs, validate.Unique(context.TODO(), operation.Operation{Type: operation.Update}, fldPath, processedInterceptors, oldProcessedInterceptors, validate.DirectEqual).MarkCoveredByDeclarative()...)

	for i, interceptor := range processedInterceptors {
		processedInterceptorPath := fldPath.Index(i)
		statusInterceptorsPath := field.NewPath("status", "interceptors")
		statusInterceptorPath := statusInterceptorsPath.Index(i)
		// items cannot be changed, only added
		if i < len(oldProcessedInterceptors) {
			// old interceptor
			allErrs = append(allErrs, apivalidation.ValidateImmutableField(interceptor, oldProcessedInterceptors[i], processedInterceptorPath).WithOrigin("immutable")...)
			continue
		}
		// new processed interceptor
		// must be present in targetInterceptors
		if !opts.TargetInterceptors.Has(interceptor) {
			allErrs = append(allErrs, field.Forbidden(processedInterceptorPath, "is not a valid interceptor from status.targetInterceptors"))
			continue
		}
		// must be present in status interceptors
		if i >= len(opts.StatusInterceptors) {
			msg := fmt.Sprintf("is immutable because a %q has to be tracked in %s first", interceptor, statusInterceptorsPath.String())
			allErrs = append(allErrs, field.Forbidden(processedInterceptorPath, msg))
			continue
		}
		interceptorStatus := opts.StatusInterceptors[i]
		if interceptorStatus.Name != interceptor {
			msg := fmt.Sprintf("is immutable because a %q does not have a matching name", statusInterceptorPath.String())
			allErrs = append(allErrs, field.Forbidden(processedInterceptorPath, msg))
			continue
		}
		// should have been active
		if !opts.OldActiveInterceptors.Has(interceptor) {
			allErrs = append(allErrs, field.Forbidden(processedInterceptorPath, fmt.Sprintf("%q should have been active and present in status.activeInterceptors", interceptor)))
			continue
		}
		// check the status tracking of this interceptor (.status.interceptors)
		// check the heartbeat deadline
		if interceptorStatus.CompletionTime == nil && !opts.IsEvicted && !opts.IsCanceled {
			if interceptorStatus.StartTime == nil {
				msg := fmt.Sprintf("is immutable because %q interceptor should have %s", interceptor, statusInterceptorPath.Child("startTime"))
				allErrs = append(allErrs, field.Forbidden(processedInterceptorPath, msg))
				continue
			}
			heartbeat := interceptorStatus.StartTime
			if interceptorStatus.HeartbeatTime != nil {
				heartbeat = interceptorStatus.HeartbeatTime
			}
			if opts.Clock.Now().Before(heartbeat.Add(opts.HeartbeatDeadline).Add(-opts.AllowedTimeSkew)) {
				msg := fmt.Sprintf("is immutable because %q interceptor is in progress and it should report %s or %s", interceptor, statusInterceptorPath.Child("heartbeatTime"), statusInterceptorPath.Child("completionTime"))
				allErrs = append(allErrs, field.Forbidden(processedInterceptorPath, msg))
			}
		}
	}
	return allErrs
}

func timeNear(a, b time.Time, skew time.Duration) bool {
	return a.After(b.Add(-skew)) && a.Before(b.Add(skew))
}
