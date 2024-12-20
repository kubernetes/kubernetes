/*
Copyright 2022 The Kubernetes Authors.

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

package resourceclaimtemplate

import (
	"context"
	"errors"

	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
	"k8s.io/kubernetes/pkg/features"
	resourceutils "k8s.io/kubernetes/pkg/registry/resource"
)

// resourceClaimTemplateStrategy implements behavior for ResourceClaimTemplate objects
type resourceClaimTemplateStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
	nsClient v1.NamespaceInterface
}

// NewStrategy is the default logic that applies when creating and updating ResourceClaimTemplate objects.
func NewStrategy(nsClient v1.NamespaceInterface) *resourceClaimTemplateStrategy {
	return &resourceClaimTemplateStrategy{
		legacyscheme.Scheme,
		names.SimpleNameGenerator,
		nsClient,
	}
}

func (*resourceClaimTemplateStrategy) NamespaceScoped() bool {
	return true
}

func (*resourceClaimTemplateStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	claimTemplate := obj.(*resource.ResourceClaimTemplate)
	dropDisabledFields(claimTemplate, nil)
}

func (s *resourceClaimTemplateStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	resourceClaimTemplate := obj.(*resource.ResourceClaimTemplate)
	allErrs := resourceutils.AuthorizedForAdmin(ctx, resourceClaimTemplate.Spec.Spec.Devices.Requests, resourceClaimTemplate.Namespace, s.nsClient)
	return append(allErrs, validation.ValidateResourceClaimTemplate(resourceClaimTemplate)...)
}

func (*resourceClaimTemplateStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (*resourceClaimTemplateStrategy) Canonicalize(obj runtime.Object) {
}

func (*resourceClaimTemplateStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (*resourceClaimTemplateStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	claimTemplate, oldClaimTemplate := obj.(*resource.ResourceClaimTemplate), old.(*resource.ResourceClaimTemplate)
	dropDisabledFields(claimTemplate, oldClaimTemplate)
}

func (s *resourceClaimTemplateStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	// AuthorizedForAdmin isn't needed here because the spec is immutable.
	errorList := validation.ValidateResourceClaimTemplate(obj.(*resource.ResourceClaimTemplate))
	return append(errorList, validation.ValidateResourceClaimTemplateUpdate(obj.(*resource.ResourceClaimTemplate), old.(*resource.ResourceClaimTemplate))...)
}

func (*resourceClaimTemplateStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (*resourceClaimTemplateStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	template, ok := obj.(*resource.ResourceClaimTemplate)
	if !ok {
		return nil, nil, errors.New("not a resourceclaimtemplate")
	}
	return labels.Set(template.Labels), toSelectableFields(template), nil
}

// toSelectableFields returns a field set that represents the object
func toSelectableFields(template *resource.ResourceClaimTemplate) fields.Set {
	fields := generic.ObjectMetaFieldsSet(&template.ObjectMeta, true)
	return fields
}

func dropDisabledFields(newClaimTemplate, oldClaimTemplate *resource.ResourceClaimTemplate) {
	dropDisabledDRAPrioritizedListFields(newClaimTemplate, oldClaimTemplate)
	dropDisabledDRAAdminAccessFields(newClaimTemplate, oldClaimTemplate)
	dropDisabledDRAResourceClaimConsumableCapacityFields(newClaimTemplate, oldClaimTemplate)
}

func dropDisabledDRAPrioritizedListFields(newClaimTemplate, oldClaimTemplate *resource.ResourceClaimTemplate) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAPrioritizedList) {
		return
	}
	if draPrioritizedListFeatureInUse(oldClaimTemplate) {
		return
	}

	for i := range newClaimTemplate.Spec.Spec.Devices.Requests {
		newClaimTemplate.Spec.Spec.Devices.Requests[i].FirstAvailable = nil
	}
}

func draPrioritizedListFeatureInUse(claimTemplate *resource.ResourceClaimTemplate) bool {
	if claimTemplate == nil {
		return false
	}

	for _, request := range claimTemplate.Spec.Spec.Devices.Requests {
		if len(request.FirstAvailable) > 0 {
			return true
		}
	}

	return false
}

func dropDisabledDRAAdminAccessFields(newClaimTemplate, oldClaimTemplate *resource.ResourceClaimTemplate) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAAdminAccess) {
		// No need to drop anything.
		return
	}
	if draAdminAccessFeatureInUse(oldClaimTemplate) {
		// If anything was set in the past, then fields must not get
		// dropped on potentially unrelated updates.
		return
	}

	for i := range newClaimTemplate.Spec.Spec.Devices.Requests {
		if newClaimTemplate.Spec.Spec.Devices.Requests[i].Exactly != nil {
			newClaimTemplate.Spec.Spec.Devices.Requests[i].Exactly.AdminAccess = nil
		}
	}
}

func draAdminAccessFeatureInUse(claimTemplate *resource.ResourceClaimTemplate) bool {
	if claimTemplate == nil {
		return false
	}

	for _, request := range claimTemplate.Spec.Spec.Devices.Requests {
		if request.Exactly != nil && request.Exactly.AdminAccess != nil {
			return true
		}
	}

	return false
}

func draConsumableCapacityFeatureInUse(claimTemplate *resource.ResourceClaimTemplate) bool {
	if claimTemplate == nil {
		return false
	}

	for _, constaint := range claimTemplate.Spec.Spec.Devices.Constraints {
		if constaint.DistinctAttribute != nil {
			return true
		}
	}

	for _, request := range claimTemplate.Spec.Spec.Devices.Requests {
		if request.Exactly != nil && request.Exactly.Capacity != nil {
			return true
		}
		for _, subRequest := range request.FirstAvailable {
			if subRequest.Capacity != nil {
				return true
			}
		}
	}

	return false
}

// dropDisabledDRAResourceClaimConsumableCapacityFields drops any new feature field
// from the newClaimTemplate if they were not used in the oldClaimTemplate.
func dropDisabledDRAResourceClaimConsumableCapacityFields(newClaimTemplate, oldClaimTemplate *resource.ResourceClaimTemplate) {
	if utilfeature.DefaultFeatureGate.Enabled(features.DRAConsumableCapacity) ||
		draConsumableCapacityFeatureInUse(oldClaimTemplate) {
		// No need to drop anything.
		return
	}

	for _, constaint := range newClaimTemplate.Spec.Spec.Devices.Constraints {
		constaint.DistinctAttribute = nil
	}

	for i := range newClaimTemplate.Spec.Spec.Devices.Requests {
		if newClaimTemplate.Spec.Spec.Devices.Requests[i].Exactly != nil {
			newClaimTemplate.Spec.Spec.Devices.Requests[i].Exactly.Capacity = nil
		}
		request := newClaimTemplate.Spec.Spec.Devices.Requests[i]
		for j := range request.FirstAvailable {
			newClaimTemplate.Spec.Spec.Devices.Requests[i].FirstAvailable[j].Capacity = nil
		}
	}
}
