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

	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
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
	if errs := s.isTargetDeletionAuthorized(ctx, evictionRequest, field.NewPath("")); errs != nil {
		return errs
	}
	allErrs := validation.ValidateEvictionRequest(evictionRequest)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, nil, allErrs, operation.Create)

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
	evictionRequest := obj.(*coordination.EvictionRequest)
	oldEvictionRequest := old.(*coordination.EvictionRequest)
	// each new requester must be added by a user with pod delete privileges
	if hasNewRequester(evictionRequest.Spec.Requesters, oldEvictionRequest.Spec.Requesters) {
		if errs := s.isTargetDeletionAuthorized(ctx, evictionRequest, field.NewPath("spec", "requesters")); errs != nil {
			return errs
		}
	}
	allErrs := validation.ValidateEvictionRequestUpdate(evictionRequest, oldEvictionRequest)
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, evictionRequest, oldEvictionRequest, allErrs, operation.Update)
}

func (*evictionRequestStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*evictionRequestStrategy) AllowUnconditionalUpdate() bool {
	return false
}

func (s *evictionRequestStrategy) isTargetDeletionAuthorized(ctx context.Context, evictionRequest *coordination.EvictionRequest, fldPath *field.Path) field.ErrorList {
	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return field.ErrorList{
			field.InternalError(field.NewPath(""), fmt.Errorf("cannot determine calling user to perform \"authorization\" check")),
		}
	}
	resource := ""
	if evictionRequest.Spec.Target.Pod != nil {
		resource = "pods"
	}
	targetPath := field.NewPath("spec", "target")
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
			return field.ErrorList{
				field.Forbidden(fldPath, fmt.Sprintf("User %q must have permission to delete pods in %q namespace when %s is set", user.GetName(), evictionRequest.Namespace, targetPath.Child("pod").String())),
			}
		}
	} else {
		// If there is no resource set on the target, then we have to ensure that the target validation is failing on create
		targetValidationErr := validation.ValidateEvictionTarget(evictionRequest.Spec.Target, targetPath, validation.EvictionRequestSpecValidationOptions{EvictionRequestName: evictionRequest.Name})
		if targetValidationErr == nil {
			// If the validation passes, fail to ensure we don't forget to add auth support for the new target type.
			return field.ErrorList{field.InternalError(targetPath, fmt.Errorf("unknown target type, authorization support not implemented"))}
		}
		// ValidateEvictionTarget will be resolved again later in ValidateEvictionRequest
	}
	return nil
}

func hasNewRequester(requesters, oldRequesters []coordination.Requester) bool {
	if len(requesters) > len(oldRequesters) {
		return true
	}
	oldRequesterNames := sets.New[string]()
	for _, requester := range oldRequesters {
		oldRequesterNames.Insert(requester.Name)
	}
	for _, requester := range requesters {
		if !oldRequesterNames.Has(requester.Name) {
			return true
		}
		// fallback to auth check if somebody adds duplicates
		oldRequesterNames.Delete(requester.Name)
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
			fieldpath.MakePathOrDie("metadata", "labels"),
		),
	}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (*evictionRequestStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newEvictionRequest := obj.(*coordination.EvictionRequest)
	oldEvictionRequest := old.(*coordination.EvictionRequest)
	newEvictionRequest.Spec = oldEvictionRequest.Spec
	// eviction-request-controller should be responsible for the labels
	// and not the interceptors - let's not promote label updates.
	newEvictionRequest.Labels = oldEvictionRequest.Labels
	// Ensure that the interceptors are not setting overly long messages. Truncation is preferable rather than returning
	// validation errors which could block heartbeat updates if the limit is exceeded.
	messageTruncationLimit := 4000
	for i, interceptor := range newEvictionRequest.Status.Interceptors {
		if len(interceptor.Message) > messageTruncationLimit {
			newEvictionRequest.Status.Interceptors[i].Message = newEvictionRequest.Status.Interceptors[i].Message[:messageTruncationLimit]
		}
	}
}

func (*evictionRequestStatusStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user updating status
func (s *evictionRequestStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	allErrs := validation.ValidateEvictionRequestStatusUpdate(obj.(*coordination.EvictionRequest), old.(*coordination.EvictionRequest), validation.EvictionRequestStatusValidationOptions{
		Clock: s.clock,
	})
	return rest.ValidateDeclarativelyWithMigrationChecks(ctx, legacyscheme.Scheme, obj, old, allErrs, operation.Update)
}

// WarningsOnUpdate returns warnings for the given update.
func (*evictionRequestStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*evictionRequestStatusStrategy) AllowUnconditionalUpdate() bool {
	return false
}
