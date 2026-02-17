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
	"k8s.io/utils/ptr"

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
	apivalidation "k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/utils/clock"
)

const defaultEvictionRequestResponderCount = 1 // EvictionResponderImperativeEviction
// more than .pod.spec.evictionResponders to account for default responders
const maxEvictionRequestResponders = apivalidation.MaxPodEvictionResponders + defaultEvictionRequestResponderCount

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
	// generateName
	// validate.ForbiddenValue

	// name
	// validate.UUID covered by DV

	allErrs = append(allErrs, apivalidation.ValidateObjectMeta(&evictionRequest.ObjectMeta, true, func(name string, prefix bool) []string {
		// validation is done declaratively by the validate.UUID
		return nil
	}, metadataFldPath)...)
	specOpts := EvictionRequestSpecValidationOptions{EvictionRequestName: evictionRequest.Name}
	allErrs = append(allErrs, ValidateEvictionRequestSpec(&evictionRequest.Spec, field.NewPath("spec"), specOpts)...)
	return allErrs
}

// ValidateEvictionRequestSpec validates an EvictionRequest spec.
func ValidateEvictionRequestSpec(evictionRequestSpec *coordination.EvictionRequestSpec, fldPath *field.Path, opts EvictionRequestSpecValidationOptions) field.ErrorList {
	var allErrs field.ErrorList
	allErrs = append(allErrs, ValidateEvictionTarget(evictionRequestSpec.Target, fldPath.Child("target"), opts)...)
	allErrs = append(allErrs, ValidateRequesters(evictionRequestSpec.Requesters, fldPath.Child("requesters"))...)
	return allErrs
}

func ValidateEvictionTarget(evictionTarget coordination.EvictionTarget, fldPath *field.Path, opts EvictionRequestSpecValidationOptions) field.ErrorList {
	var allErrs field.ErrorList
	// validate.Union covered by DV
	if evictionTarget.Pod != nil {
		allErrs = append(allErrs, ValidatePodReference(*evictionTarget.Pod, fldPath.Child("pod"), opts)...)
	}
	return allErrs
}

func ValidatePodReference(podReference coordination.PodReference, fldPath *field.Path, opts EvictionRequestSpecValidationOptions) field.ErrorList {
	var allErrs field.ErrorList
	// name
	// validate.RequiredValue covered by DV
	// validate.LongName covered by DV
	// uid
	uidFldPath := fldPath.Child("uid")
	// validate.RequiredValue covered by DV
	if len(podReference.UID) != 0 {
		if string(podReference.UID) != opts.EvictionRequestName {
			msg := fmt.Sprintf("must be the same value as %s", uidFldPath.String())
			allErrs = append(allErrs, field.Invalid(field.NewPath("metadata").Child("name"), podReference.UID, msg))
		}
	}
	return allErrs
}

func ValidateRequesters(requesters []coordination.Requester, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// validate.Required covered by DV
	// validate.MaxItems covered by DV
	// validate.Unique covered by DV
	if hasAllWithdrawnIntents(requesters) {
		allErrs = append(allErrs, field.Invalid(fldPath, requesters, fmt.Sprintf("must have at least one requester with an intent that is not %q on EvictionRequest creation", coordination.RequesterIntentWithdrawn)))
	}
	for i, requester := range requesters {
		allErrs = append(allErrs, ValidateRequester(requester, fldPath.Index(i))...)
	}
	return allErrs
}

func hasAllWithdrawnIntents(requesters []coordination.Requester) bool {
	for _, requester := range requesters {
		if requester.Intent != coordination.RequesterIntentWithdrawn {
			return false
		}
	}
	return true
}

func ValidateRequestersUpdate(requesters, oldRequesters []coordination.Requester, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// validate.MinItems covered by DV
	// validate.MaxItems covered by DV
	// validate.Unique covered by DV
	newRequestersSet := sets.New[string]()
	for _, requester := range requesters {
		newRequestersSet.Insert(requester.Name)
	}
	for _, oldRequester := range oldRequesters {
		if !newRequestersSet.Has(oldRequester.Name) {
			return append(allErrs, field.Invalid(fldPath, requesters, "requesters cannot be removed"))
		}
	}

	for i, requester := range requesters {
		allErrs = append(allErrs, ValidateRequester(requester, fldPath.Index(i))...)
	}
	return allErrs
}

// ValidateEvictionRequestUpdate validates an EvictionRequest.
func ValidateEvictionRequestUpdate(evictionRequest, oldEvictionRequest *coordination.EvictionRequest) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&evictionRequest.ObjectMeta, &oldEvictionRequest.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateEvictionRequestSpecUpdate(&evictionRequest.Spec, &oldEvictionRequest.Spec, field.NewPath("spec"))...)
	return allErrs
}

// ValidateEvictionRequestSpecUpdate validates an EvictionRequest Spec update.
func ValidateEvictionRequestSpecUpdate(evictionRequestSpec, oldEvictionRequestSpec *coordination.EvictionRequestSpec, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList
	// target
	// validate.Immutable covered by DV

	// requesters
	allErrs = append(allErrs, ValidateRequestersUpdate(evictionRequestSpec.Requesters, oldEvictionRequestSpec.Requesters, fldPath.Child("requesters"))...)
	return allErrs
}

func ValidateRequester(requester coordination.Requester, fldPath *field.Path) field.ErrorList {
	var allErrs field.ErrorList

	// name
	namePath := fldPath.Child("name")
	// validate.RequiredValue covered by DV
	if len(requester.Name) != 0 {
		allErrs = append(allErrs, utilvalidation.IsDomainPrefixedKey(namePath, requester.Name)...)
	}

	// intent
	// validate.RequiredValue covered by DV
	// validate.Enum covered by DV
	return allErrs
}

type EvictionRequestStatusValidationOptions struct {
	Clock clock.PassiveClock
}

// ValidateEvictionRequestStatusUpdate validates an EvictionRequest Status update.
func ValidateEvictionRequestStatusUpdate(evictionRequest, oldEvictionRequest *coordination.EvictionRequest, opts EvictionRequestStatusValidationOptions) field.ErrorList {
	allErrs := apivalidation.ValidateObjectMetaUpdate(&evictionRequest.ObjectMeta, &oldEvictionRequest.ObjectMeta, field.NewPath("metadata"))
	allErrs = append(allErrs, ValidateEvictionRequestStatus(&evictionRequest.Status, &oldEvictionRequest.Status, field.NewPath("status"), opts)...)
	return allErrs
}

// ValidateEvictionRequestStatus validates an EvictionRequest Status.
func ValidateEvictionRequestStatus(status, oldStatus *coordination.EvictionRequestStatus, fldPath *field.Path, opts EvictionRequestStatusValidationOptions) field.ErrorList {
	var allErrs field.ErrorList

	// observedGeneration
	observedGenerationPath := fldPath.Child("observedGeneration")
	// validate.Minimum covered by DV
	shouldCheckObservedGen := ptr.Deref(status.ObservedGeneration, 0) >= 1 || (status.ObservedGeneration == nil && oldStatus.ObservedGeneration != nil) // the rest is handled by validate.Minimum
	if shouldCheckObservedGen && ptr.Deref(status.ObservedGeneration, 0) < ptr.Deref(oldStatus.ObservedGeneration, 0) {
		allErrs = append(allErrs, field.Invalid(observedGenerationPath, status.ObservedGeneration, "cannot decrement, "+content.MinError(ptr.Deref(oldStatus.ObservedGeneration, 1))))
	}

	// conditions
	conditionsPath := fldPath.Child("conditions")
	// validate.MaxItems covered by DV
	// validate.Unique covered by DV
	for i, condition := range status.Conditions {
		allErrs = append(allErrs, validation2.ValidateCondition(condition, conditionsPath.Index(i))...)
	}
	isEvicted := meta.IsStatusConditionTrue(status.Conditions, string(coordination.EvictionRequestConditionEvicted))
	isFailed := meta.IsStatusConditionTrue(status.Conditions, string(coordination.EvictionRequestConditionFailed))
	for _, oldCondition := range oldStatus.Conditions {
		if oldCondition.Type == string(coordination.EvictionRequestConditionEvicted) || oldCondition.Type == string(coordination.EvictionRequestConditionFailed) {
			newCondition := meta.FindStatusCondition(status.Conditions, oldCondition.Type)
			if oldCondition.Status == v1.ConditionTrue && !validate.SemanticDeepEqual(&oldCondition, newCondition) {
				allErrs = append(allErrs, field.Invalid(conditionsPath, status.Conditions, fmt.Sprintf("%s condition status cannot be reverted", oldCondition.Type)))
				if oldCondition.Type == string(coordination.EvictionRequestConditionEvicted) {
					isEvicted = true // do not use invalid condition state for next validations
				}
				if oldCondition.Type == string(coordination.EvictionRequestConditionFailed) {
					isFailed = true // do not use invalid condition state for next validations
				}
			}
		}
	}

	// targetResponders, and responders
	allErrs = append(allErrs, ValidateAllEvictionRequestStatusResponderFields(status, oldStatus, fldPath, EvictionRequestStatusRespondersValidationOptions{
		Clock:     opts.Clock,
		IsEvicted: isEvicted,
		IsFailed:  isFailed,
	})...)

	return allErrs
}

type EvictionRequestStatusRespondersValidationOptions struct {
	Clock     clock.PassiveClock
	IsEvicted bool
	IsFailed  bool
}

// ValidateAllEvictionRequestStatusResponderFields validates .status.targetResponders and .status.responders
//
// Multiple actors are expected to update the status (evictionrequest-controller, responders and requesters updating conditions).
// We have to emulate the evictionrequest-controller behavior, to prevent invalid updates by other misbehaving/malicious actors.
//  1. .status.targetResponders and .status.responders should be set first.
//  2. Responders' state transitions from Inactive to Active, and from Active to Interrupted, Canceled or, Complete.
//     gradually in the order they appear in the first list. This is not reversible.
//  3. Responders' state transition from Active to Interrupted, Canceled or, Complete when
//     .status.responders[].completionTime is set or when heartbeat is exceeded. This is not reversible.
//  4. .status.responders items cannot be removed once set. Only active responders can mutate it.
//  5. The controller can mark an EvictionRequest as Evicted or Canceled via conditions. We then have to allow removal
//     from activeResponders and prevent any new ones.
func ValidateAllEvictionRequestStatusResponderFields(status, oldStatus *coordination.EvictionRequestStatus, fldPath *field.Path, opts EvictionRequestStatusRespondersValidationOptions) field.ErrorList {
	var allErrs field.ErrorList
	heartbeatDeadline := time.Minute * 20
	allowedTimeSkew := time.Second * 30
	allowedMaxExpectedCompletionTime := time.Hour * 24 * 365 * 10 // 10 years

	// from this point we can depend on the target responders to be set
	// define dependencies between targetResponders,and responders validation steps
	targetResponders := status.TargetResponders
	statusResponders := append(make([]coordination.ResponderStatus, 0, len(status.Responders)), status.Responders...)

	// targetResponders
	if errs := ValidateEvictionRequestTargetResponders(status.TargetResponders, oldStatus.TargetResponders, fldPath.Child("targetResponders"), ValidateEvictionRequestTargetRespondersOptions{
		Clock:             opts.Clock,
		IsFailed:          opts.IsFailed,
		IsEvicted:         opts.IsEvicted,
		StatusResponders:  statusResponders,
		HeartbeatDeadline: heartbeatDeadline,
		AllowedTimeSkew:   allowedTimeSkew,
	}); len(errs) > 0 {
		// Declarative errors are used to check if we have a valid data. Filter them as we will get them in declarative validation again.
		allErrs = append(allErrs, filterOutDeclarativeErrors(errs)...)
		// do not use invalid data for next validation steps
		targetResponders = oldStatus.TargetResponders // +k8s:verify-mutation:reason=clone
	}

	// responders
	allErrs = append(allErrs, ValidateEvictionRequestStatusResponders(status.Responders, oldStatus.Responders, fldPath.Child("responders"), ValidateEvictionRequestStatusRespondersOptions{
		Clock:                     opts.Clock,
		TargetResponders:          targetResponders,
		AllowedTimeSkew:           allowedTimeSkew,
		MaxExpectedCompletionTime: allowedMaxExpectedCompletionTime,
	})...)

	return allErrs
}

func filterOutDeclarativeErrors(errs field.ErrorList) field.ErrorList {
	var allErrs field.ErrorList
	for _, err := range errs {
		if !err.CoveredByDeclarative {
			allErrs = append(allErrs, err)
		}
	}
	return allErrs
}

type ValidateEvictionRequestTargetRespondersOptions struct {
	Clock             clock.PassiveClock
	IsEvicted         bool
	IsFailed          bool
	StatusResponders  []coordination.ResponderStatus
	HeartbeatDeadline time.Duration
	AllowedTimeSkew   time.Duration
}

func ValidateEvictionRequestTargetResponders(targetResponders, oldTargetResponders []coordination.TargetResponder, fldPath *field.Path, opts ValidateEvictionRequestTargetRespondersOptions) field.ErrorList {
	var allErrs field.ErrorList
	if len(oldTargetResponders) != 0 {
		if len(targetResponders) != len(oldTargetResponders) {
			return append(allErrs, field.Invalid(fldPath, targetResponders, "must preserve the same length and the same keys in the same order"))
		}
		for i, oldTargetResponder := range oldTargetResponders {
			if oldTargetResponder.Name != targetResponders[i].Name {
				return append(allErrs, field.Invalid(fldPath, targetResponders, "must preserve the same keys in the same order"))
			}
		}
	}
	if len(targetResponders) == 0 {
		// not initialized yet
		return allErrs
	}

	if len(targetResponders) > maxEvictionRequestResponders {
		// simulate declarative error for code flow control and further data validation
		return append(allErrs, field.TooMany(fldPath, len(targetResponders), maxEvictionRequestResponders).WithOrigin("maxItems").MarkCoveredByDeclarative())
	}

	// simulate declarative error for further data validation
	uniqueErrors := validate.Unique(context.TODO(), operation.Operation{}, fldPath, targetResponders, nil,
		func(a coordination.TargetResponder, b coordination.TargetResponder) bool { return a.Name == b.Name }).MarkCoveredByDeclarative()
	allErrs = append(allErrs, uniqueErrors...)
	if len(uniqueErrors) > 0 {
		// does not make sense to check for state transition with duplicates
		return allErrs
	}

	lastActiveIdx := -1
	for i, responder := range oldTargetResponders {
		if responder.State == coordination.ResponderStateActive {
			lastActiveIdx = i
			break
		}
	}
	hasNeverBeenActive := len(oldTargetResponders) == 0 // we should Activate during the first sync
	hasFoundLastActive := lastActiveIdx != -1
	activeChanged := hasFoundLastActive && targetResponders[lastActiveIdx].State != coordination.ResponderStateActive
	isFinal := opts.IsEvicted || opts.IsFailed

	for i, responder := range targetResponders {
		expectedStates := sets.New[coordination.ResponderStateType]()

		expectedStatesReason := ""
		switch {
		case i < lastActiveIdx ||
			(!hasFoundLastActive && isFinal):
			// Processed responders must preserve their state.
			expectedStates.Insert(responder.State)
			expectedStatesReason = "final state is immutable"
		case i == lastActiveIdx && activeChanged:
			// If Active responder changes, it must have a final state.
			expectedStates.Insert(coordination.ResponderStateInterrupted,
				coordination.ResponderStateCanceled,
				coordination.ResponderStateCompleted)
			expectedStatesReason = "this responder must reach a final state"
		case i == lastActiveIdx:
			// Unchanged Active responders can stay Active.
			expectedStates.Insert(coordination.ResponderStateActive)
			expectedStatesReason = "the eviction request stays active"
			if isFinal {
				// Last Responder must finish before setting the final condition.
				expectedStates.Insert(coordination.ResponderStateInterrupted,
					coordination.ResponderStateCanceled,
					coordination.ResponderStateCompleted)
				expectedStatesReason = "the eviction request has finished processing"
			}
		case i == 0 && hasNeverBeenActive,
			i == lastActiveIdx+1 && activeChanged:
			// Next responder must move to an Active state, when the old one is final.
			expectedStates.Insert(coordination.ResponderStateActive)
			expectedStatesReason = "this responder is next in line"
			if isFinal {
				// Do not active next responder if we have finished.
				expectedStates.Insert(coordination.ResponderStateInactive)
				expectedStatesReason = "the eviction request has finished processing"
			}
		default:
			expectedStates.Insert(coordination.ResponderStateInactive)
			expectedStatesReason = "the previously active has not finished processing"
		}

		var responderStatus *coordination.ResponderStatus
		if i < len(opts.StatusResponders) && opts.StatusResponders[i].Name == responder.Name {
			responderStatus = &opts.StatusResponders[i]
		}

		allErrs = append(allErrs, ValidateTargetResponder(responder, fldPath.Index(i), ValidateEvictionRequestTargetResponderOptions{
			Clock:                opts.Clock,
			IsEvicted:            opts.IsEvicted,
			IsFailed:             opts.IsFailed,
			responderStatus:      responderStatus,
			expectedStates:       expectedStates,
			expectedStatesReason: expectedStatesReason,
			HeartbeatDeadline:    opts.HeartbeatDeadline,
			AllowedTimeSkew:      opts.AllowedTimeSkew,
		})...)
	}
	return allErrs
}

type ValidateEvictionRequestTargetResponderOptions struct {
	Clock                clock.PassiveClock
	IsEvicted            bool
	IsFailed             bool
	responderStatus      *coordination.ResponderStatus
	expectedStates       sets.Set[coordination.ResponderStateType]
	expectedStatesReason string
	HeartbeatDeadline    time.Duration
	AllowedTimeSkew      time.Duration
}

func ValidateTargetResponder(evictionResponder coordination.TargetResponder, fldPath *field.Path, opts ValidateEvictionRequestTargetResponderOptions) field.ErrorList {
	var allErrs field.ErrorList

	// name
	namePath := fldPath.Child("name")
	if len(evictionResponder.Name) == 0 {
		// simulate declarative error for further data validation
		allErrs = append(allErrs, field.Required(namePath, "")).MarkCoveredByDeclarative()
	} else {
		allErrs = append(allErrs, utilvalidation.IsDomainPrefixedKey(namePath, evictionResponder.Name)...)
	}

	statusResponderPath := field.NewPath("status", "responders")
	if opts.responderStatus == nil {
		msg := fmt.Sprintf("%q has to be tracked in %s first", evictionResponder.Name, statusResponderPath)
		return append(allErrs, field.Invalid(fldPath, evictionResponder, msg))
	}

	// state
	statePath := fldPath.Child("state")
	// validate.RequiredValue covered by DV
	// validate.Enum covered by DV

	// check that the state transition is allowed
	if !opts.expectedStates.Has(evictionResponder.State) {
		var expectedValues []string
		for _, ev := range sets.List(opts.expectedStates) {
			expectedValues = append(expectedValues, string(ev))
		}
		msg := fmt.Sprintf("must be one of: %s", strings.Join(expectedValues, ", "))
		if len(opts.expectedStatesReason) > 0 {
			msg = fmt.Sprintf("%s, because %s", msg, opts.expectedStatesReason)
		}
		allErrs = append(allErrs, field.Invalid(statePath, evictionResponder.State, msg))
	}

	// must be present in status responders

	// check the heartbeat deadline if the responder is not active anymore (.status.responders)
	// we can skip the check if the EvictionRequest is final (failed or evicted).
	// StartTime presence is validated in ValidateEvictionRequestStatusResponder
	if !opts.expectedStates.Has(coordination.ResponderStateActive) && opts.responderStatus.StartTime != nil &&
		opts.responderStatus.CompletionTime == nil && !opts.IsEvicted && !opts.IsFailed {
		heartbeat := opts.responderStatus.StartTime
		if opts.responderStatus.HeartbeatTime != nil {
			heartbeat = opts.responderStatus.HeartbeatTime
		}
		if opts.Clock.Now().Before(heartbeat.Add(opts.HeartbeatDeadline).Add(-opts.AllowedTimeSkew)) {
			msg := fmt.Sprintf("must stay Active because the responder is in progress and it should report %s or %s", statusResponderPath.Child("heartbeatTime"), statusResponderPath.Child("completionTime"))
			allErrs = append(allErrs, field.Forbidden(statePath, msg))
		}
	}
	return allErrs
}

type ValidateEvictionRequestActiveRespondersOptions struct {
	IsEvicted           bool
	IsFailed            bool
	TargetResponders    []string
	ProcessedResponders sets.Set[string]
}

type ValidateEvictionRequestStatusRespondersOptions struct {
	Clock                     clock.PassiveClock
	TargetResponders          []coordination.TargetResponder
	AllowedTimeSkew           time.Duration
	MaxExpectedCompletionTime time.Duration
}

func ValidateEvictionRequestStatusResponders(statusResponders, oldStatusResponders []coordination.ResponderStatus, fldPath *field.Path, opts ValidateEvictionRequestStatusRespondersOptions) field.ErrorList {
	var allErrs field.ErrorList

	if len(opts.TargetResponders) != len(statusResponders) {
		return append(allErrs, field.Invalid(fldPath, statusResponders, "must be the same length as status.targetResponders and contain the same keys in the same order"))
	}
	for i, targetResponder := range opts.TargetResponders {
		if targetResponder.Name != statusResponders[i].Name {
			return append(allErrs, field.Invalid(fldPath, statusResponders, "must contain the same keys in the same order as status.targetResponders"))
		}
	}
	if len(statusResponders) == 0 {
		// statusResponders and TargetResponders are not initialized yet
		return allErrs
	}

	if len(statusResponders) > maxEvictionRequestResponders {
		// TooMany is handled by declarative validation - detect early return
		return allErrs
	}
	// validate.Unique covered by DV - simulate an error
	uniqueErrors := validate.Unique(context.TODO(), operation.Operation{}, fldPath, statusResponders, nil,
		func(a coordination.ResponderStatus, b coordination.ResponderStatus) bool { return a.Name == b.Name })
	if len(uniqueErrors) > 0 {
		// does not make sense to check each responder status with duplicates as we depend on the target responder and an oldStatusResponder
		return allErrs
	}

	for i, responder := range statusResponders {
		targetResponder := opts.TargetResponders[i] // index bounds checked above
		responderPath := fldPath.Index(i)
		var oldResponderStatus *coordination.ResponderStatus
		if i < len(oldStatusResponders) {
			oldResponderStatus = &oldStatusResponders[i] // +k8s:verify-mutation:reason=clone
		}
		allErrs = append(allErrs, ValidateEvictionRequestStatusResponder(&responder, oldResponderStatus, responderPath, EvictionRequestStatusResponderValidationOptions{
			Clock:                     opts.Clock,
			responderState:            targetResponder.State,
			AllowedTimeSkew:           opts.AllowedTimeSkew,
			MaxExpectedCompletionTime: opts.MaxExpectedCompletionTime,
		})...)
	}
	return allErrs
}

type EvictionRequestStatusResponderValidationOptions struct {
	responderState            coordination.ResponderStateType
	AllowedTimeSkew           time.Duration
	MaxExpectedCompletionTime time.Duration
	Clock                     clock.PassiveClock
}

func ValidateEvictionRequestStatusResponder(status, oldStatus *coordination.ResponderStatus, fldPath *field.Path, opts EvictionRequestStatusResponderValidationOptions) field.ErrorList {
	var allErrs field.ErrorList
	oldDefaultedStatus := coordination.ResponderStatus{}
	if oldStatus != nil {
		oldDefaultedStatus = *oldStatus // +k8s:verify-mutation:reason=clone
	}

	if oldStatus != nil && !validate.SemanticDeepEqual(status, oldStatus) && opts.responderState != coordination.ResponderStateActive {
		// immutable; changes to the ResponderStatus are only allowed by the active responder or during initialization
		return append(allErrs, field.Invalid(fldPath, status, validation.FieldImmutableErrorMsg).WithOrigin("immutable"))
	}

	// name
	// validate.Required covered by DV
	// The existence of targetResponder with the same name is done in ValidateEvictionRequestStatusResponders

	// startTime
	startTimePath := fldPath.Child("startTime")
	// immutable once set
	if oldDefaultedStatus.StartTime != nil && !oldDefaultedStatus.StartTime.Equal(status.StartTime) {
		// validate.NoUnset and validate.NoModify checks covered by DV
	} else if status.StartTime == nil && opts.responderState == coordination.ResponderStateActive {
		allErrs = append(allErrs, field.Required(startTimePath, "is required for an active responder"))
	} else if status.StartTime != nil && !oldDefaultedStatus.StartTime.Equal(status.StartTime) && !timeNear(status.StartTime.Time, opts.Clock.Now(), opts.AllowedTimeSkew) {
		allErrs = append(allErrs, field.Invalid(startTimePath, status.StartTime, "must be set to the present time"))
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
		} else if !timeNear(status.HeartbeatTime.Time, opts.Clock.Now(), opts.AllowedTimeSkew) {
			allErrs = append(allErrs, field.Invalid(heartbeatTimePath, status.HeartbeatTime, "must be set to the present time"))
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
		// validate.NoUnset and validate.NoModify checks covered by DV
	} else if status.CompletionTime != nil {
		if status.StartTime == nil {
			allErrs = append(allErrs, field.Invalid(completionTimePath, status.CompletionTime, fmt.Sprintf("cannot be set before %s is set", startTimePath.String())))
		} else if status.CompletionTime.Before(status.StartTime) {
			allErrs = append(allErrs, field.Invalid(completionTimePath, status.CompletionTime, fmt.Sprintf("must occur after %s", startTimePath.String())))
		} else if !oldDefaultedStatus.CompletionTime.Equal(status.CompletionTime) && !timeNear(status.CompletionTime.Time, opts.Clock.Now(), opts.AllowedTimeSkew) {
			allErrs = append(allErrs, field.Invalid(completionTimePath, status.CompletionTime, "must be set to the present time"))
		}
	}

	// message
	// validate.MaxLength covered by DV

	return allErrs
}

func timeNear(a, b time.Time, skew time.Duration) bool {
	return a.After(b.Add(-skew)) && a.Before(b.Add(skew))
}
