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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/apis/resource/validation"
	"k8s.io/kubernetes/pkg/features"
)

const (
	DRAAdminNamespaceLabel = "kubernetes.io/dra-admin-access"
)

type NamespaceGetter interface {
	Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error)
}

// resourceClaimTemplateStrategy implements behavior for ResourceClaimTemplate objects
type resourceClaimTemplateStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
	nsClient v1.NamespaceInterface
}

// NewStrategy is the default logic that applies when creating and updating ResourceClaimTemplate objects.
func NewStrategy(ro runtime.ObjectTyper, ng names.NameGenerator, nsClient v1.NamespaceInterface) *resourceClaimTemplateStrategy {
	return &resourceClaimTemplateStrategy{
		ro,
		ng,
		nsClient,
	}
}

func (resourceClaimTemplateStrategy) NamespaceScoped() bool {
	return true
}

func (resourceClaimTemplateStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	claimTemplate := obj.(*resource.ResourceClaimTemplate)
	dropDisabledFields(claimTemplate, nil)
}

func (s resourceClaimTemplateStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	resourceClaimTemplate := obj.(*resource.ResourceClaimTemplate)
	allErrs := authorizedForAdmin(ctx, resourceClaimTemplate, s.nsClient)
	return append(allErrs, validation.ValidateResourceClaimTemplate(resourceClaimTemplate)...)
}

func (resourceClaimTemplateStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string {
	return nil
}

func (resourceClaimTemplateStrategy) Canonicalize(obj runtime.Object) {
}

func (resourceClaimTemplateStrategy) AllowCreateOnUpdate() bool {
	return false
}

func (resourceClaimTemplateStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	claimTemplate, oldClaimTemplate := obj.(*resource.ResourceClaimTemplate), old.(*resource.ResourceClaimTemplate)
	dropDisabledFields(claimTemplate, oldClaimTemplate)
}

func (s resourceClaimTemplateStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	claimTemplate, oldClaimTemplate := obj.(*resource.ResourceClaimTemplate), old.(*resource.ResourceClaimTemplate)

	allErrs := authorizedForAdmin(ctx, claimTemplate, s.nsClient)
	allErrs = append(allErrs, validation.ValidateResourceClaimTemplate(claimTemplate)...)
	return append(allErrs, validation.ValidateResourceClaimTemplateUpdate(claimTemplate, oldClaimTemplate)...)
}

func (resourceClaimTemplateStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (resourceClaimTemplateStrategy) AllowUnconditionalUpdate() bool {
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

// authorizedForAdmin checks if the request is authorized to get admin access to devices
// based on namespace label
func authorizedForAdmin(ctx context.Context, template *resource.ResourceClaimTemplate, nsClient v1.NamespaceInterface) field.ErrorList {
	allErrs := field.ErrorList{}
	adminRequested := false

	if !utilfeature.DefaultFeatureGate.Enabled(features.DRAAdminAccess) {
		// No need to validate unless feature gate is enabled
		return allErrs
	}

	for i := range template.Spec.Spec.Devices.Requests {
		value := template.Spec.Spec.Devices.Requests[i].AdminAccess
		if value != nil && *value {
			adminRequested = true
			break
		}
	}
	if !adminRequested {
		// No need to validate unless admin access is requested
		return allErrs
	}
	if nsClient == nil {
		return append(allErrs, field.Forbidden(field.NewPath(""), "nsClient is nil"))
	}

	namespaceName := template.Namespace
	// Retrieve the namespace object from the store
	ns, err := nsClient.Get(ctx, namespaceName, metav1.GetOptions{ResourceVersion: "0"})
	if err != nil {
		return append(allErrs, field.Forbidden(field.NewPath(""), "namespace object cannot be retrieved"))
	}
	if value, exists := ns.Labels[DRAAdminNamespaceLabel]; !(exists && value == "true") {
		return append(allErrs, field.Forbidden(field.NewPath(""), "admin access to devices is not allowed in namespace without DRA Admin Access label"))
	}

	return allErrs
}

func dropDisabledFields(newClaimTemplate, oldClaimTemplate *resource.ResourceClaimTemplate) {
	dropDisabledDRAAdminAccessFields(newClaimTemplate, oldClaimTemplate)
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
		newClaimTemplate.Spec.Spec.Devices.Requests[i].AdminAccess = nil
	}
}

func draAdminAccessFeatureInUse(claimTemplate *resource.ResourceClaimTemplate) bool {
	if claimTemplate == nil {
		return false
	}

	for _, request := range claimTemplate.Spec.Spec.Devices.Requests {
		if request.AdminAccess != nil {
			return true
		}
	}

	return false
}
