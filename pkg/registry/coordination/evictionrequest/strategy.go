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

package evictionrequest

import (
	"context"
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/coordination"
	"k8s.io/kubernetes/pkg/apis/coordination/validation"
	"k8s.io/utils/clock"
)

// evictionRequestStrategy is the default logic that applies when creating and updating EvictionRequest objects.
type evictionRequestStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
	authorizer authorizer.Authorizer
	clock      clock.PassiveClock
}

// noopNameGenerator does not generate names, it just returns the base.
type noopNameGenerator struct{}

func (noopNameGenerator) GenerateName(base string) string {
	return base
}

func NewStrategy(authorizer authorizer.Authorizer, clock clock.PassiveClock) *evictionRequestStrategy {
	return &evictionRequestStrategy{
		legacyscheme.Scheme,
		noopNameGenerator{},
		authorizer,
		clock,
	}
}

var _ = rest.ResetFieldsStrategy(&evictionRequestStrategy{})

func (*evictionRequestStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (*evictionRequestStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"coordination/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}
	return fields
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (*evictionRequestStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	evictionRequest := obj.(*coordination.EvictionRequest)
	evictionRequest.Status = coordination.EvictionRequestStatus{}
	evictionRequest.Generation = 1
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (*evictionRequestStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	oldEvictionRequest := old.(*coordination.EvictionRequest)
	newEvictionRequest := obj.(*coordination.EvictionRequest)
	newEvictionRequest.Status = oldEvictionRequest.Status

	// Spec updates bump the generation. Labels are periodically synced with the target's
	// and should not affect the generation.
	if !apiequality.Semantic.DeepEqual(oldEvictionRequest.Spec, newEvictionRequest.Spec) {
		newEvictionRequest.Generation = oldEvictionRequest.Generation + 1
	}
}

// Validate validates a new EvictionRequest.
func (s *evictionRequestStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	evictionRequest := obj.(*coordination.EvictionRequest)
	unauthorizedMustFail, errs := s.isTargetDeletionAuthorized(ctx, evictionRequest, field.NewPath(""))
	if errs != nil {
		return errs
	}
	allErrs := validation.ValidateEvictionRequest(evictionRequest)
	allErrs = rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, nil, allErrs, operation.Create, rest.WithDeclarativeEnforcement())
	if len(allErrs) == 0 && unauthorizedMustFail {
		allErrs = append(allErrs, field.InternalError(field.NewPath("spec", "target"), fmt.Errorf("unknown target type, authorization support not implemented")))
	}
	return allErrs
}
func (*evictionRequestStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*evictionRequestStrategy) Canonicalize(obj runtime.Object) {
}

func (*evictionRequestStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (s *evictionRequestStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	var allErrs field.ErrorList
	evictionRequest := obj.(*coordination.EvictionRequest)
	oldEvictionRequest := old.(*coordination.EvictionRequest)
	unauthorizedMustFail := false
	// each new requester with an eviction intent (any but not withdrawn) must be added by a user with pod delete privileges
	if hasRequestersIntentChangedExcludingWithdrawal(evictionRequest.Spec.Requesters, oldEvictionRequest.Spec.Requesters) {
		if unauthorizedMustFail, allErrs = s.isTargetDeletionAuthorized(ctx, evictionRequest, field.NewPath("spec", "requesters")); allErrs != nil {
			return allErrs
		}
	}
	allErrs = validation.ValidateEvictionRequestUpdate(evictionRequest, oldEvictionRequest)
	allErrs = rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, evictionRequest, oldEvictionRequest, allErrs, operation.Update, rest.WithDeclarativeEnforcement())
	if len(allErrs) == 0 && unauthorizedMustFail {
		allErrs = append(allErrs, field.InternalError(field.NewPath("spec", "target"), fmt.Errorf("unknown target type, authorization support not implemented")))
	}
	return allErrs
}

func (*evictionRequestStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*evictionRequestStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (s *evictionRequestStrategy) isTargetDeletionAuthorized(ctx context.Context, evictionRequest *coordination.EvictionRequest, fldPath *field.Path) (bool, field.ErrorList) {
	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return true, field.ErrorList{
			field.InternalError(field.NewPath(""), fmt.Errorf("cannot determine calling user to perform \"authorization\" check")),
		}
	}
	resource := ""
	if evictionRequest.Spec.Target.Pod != nil {
		resource = "pods"
	}
	if len(resource) != 0 {
		attr := authorizer.AttributesRecord{
			User:            user,
			Namespace:       evictionRequest.Namespace,
			Verb:            "delete",
			APIGroup:        "",
			APIVersion:      "*",
			Resource:        resource,
			ResourceRequest: true,
		}
		if decision, _, _ := s.authorizer.Authorize(ctx, attr); decision != authorizer.DecisionAllow {
			return true, field.ErrorList{
				field.Forbidden(fldPath, fmt.Sprintf("User %q must have permission to delete pods in %q namespace when %s is set", user.GetName(), evictionRequest.Namespace, field.NewPath("spec", "target", "pod").String())),
			}
		}
	} else {
		// If there is no resource set on the target, then validation will fail, so we don't need to handle it here.
		return true, nil
	}
	return false, nil
}

func hasRequestersIntentChangedExcludingWithdrawal(requesters, oldRequesters []coordination.Requester) bool {
	oldIntents := map[string]coordination.RequesterIntent{}
	for _, requester := range oldRequesters {
		oldIntents[requester.Name] = requester.Intent
	}
	for _, requester := range requesters {
		// Any change that is not withdrawn is privileged.
		if requester.Intent == "" ||
			(oldIntents[requester.Name] != requester.Intent && requester.Intent != coordination.RequesterIntentWithdrawn) {
			// Check for invalid requester removals later.
			delete(oldIntents, requester.Name)
			return true
		}
		// Fallback to auth check if somebody adds duplicates.
		// Also check for invalid requester removals later.
		delete(oldIntents, requester.Name)
	}
	if len(oldIntents) > 0 {
		// All old requesters must be preserved.
		return true
	}
	return false
}

// evictionRequestStatusStrategy is the default logic invoked when updating object status.
type evictionRequestStatusStrategy struct {
	*evictionRequestStrategy
}

var _ = rest.ResetFieldsStrategy(&evictionRequestStatusStrategy{})

func NewStatusStrategy(strategy *evictionRequestStrategy) *evictionRequestStatusStrategy {
	return &evictionRequestStatusStrategy{
		strategy,
	}
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (*evictionRequestStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return map[fieldpath.APIVersion]*fieldpath.Set{
		"coordination/v1alpha1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
			fieldpath.MakePathOrDie("metadata"),
		),
	}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (*evictionRequestStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newEvictionRequest := obj.(*coordination.EvictionRequest)
	oldEvictionRequest := old.(*coordination.EvictionRequest)
	// Status updates should not include metadata update privileges, also,
	// the evictionrequest-controller should be responsible for the labels
	// and not the responders - let's not promote label updates.
	metav1.ResetObjectMetaForStatus(&newEvictionRequest.ObjectMeta, &oldEvictionRequest.ObjectMeta)
	newEvictionRequest.Spec = oldEvictionRequest.Spec
}

func (*evictionRequestStatusStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user updating status
func (s *evictionRequestStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	allErrs := validation.ValidateEvictionRequestStatusUpdate(obj.(*coordination.EvictionRequest), old.(*coordination.EvictionRequest), validation.EvictionRequestStatusValidationOptions{
		Clock: s.clock,
	})
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, old, allErrs, operation.Update, rest.WithDeclarativeEnforcement())
}

// WarningsOnUpdate returns warnings for the given update.
func (*evictionRequestStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*evictionRequestStatusStrategy) AllowUnconditionalUpdate() bool {
	return false
}
