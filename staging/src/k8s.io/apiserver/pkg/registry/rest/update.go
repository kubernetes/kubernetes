/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	genericvalidation "k8s.io/apimachinery/pkg/api/validation"
	"k8s.io/apimachinery/pkg/api/validation/path"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// RESTUpdateStrategy defines the minimum validation, accepted input, and
// name generation behavior to update an object that follows Kubernetes
// API conventions. A resource may have many UpdateStrategies, depending on
// the call pattern in use.
type RESTUpdateStrategy interface {
	runtime.ObjectTyper
	// NamespaceScoped returns true if the object must be within a namespace.
	NamespaceScoped() bool
	// AllowCreateOnUpdate returns true if the object can be created by a PUT.
	AllowCreateOnUpdate() bool
	// PrepareForUpdate is invoked on update before validation to normalize
	// the object.  For example: remove fields that are not to be persisted,
	// sort order-insensitive list fields, etc.  This should not remove fields
	// whose presence would be considered a validation error.
	PrepareForUpdate(ctx context.Context, obj, old runtime.Object)
	// ValidateUpdate is invoked after default fields in the object have been
	// filled in before the object is persisted.  This method should not mutate
	// the object.
	ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList
	// Canonicalize allows an object to be mutated into a canonical form. This
	// ensures that code that operates on these objects can rely on the common
	// form for things like comparison.  Canonicalize is invoked after
	// validation has succeeded but before the object has been persisted.
	// This method may mutate the object.
	Canonicalize(obj runtime.Object)
	// AllowUnconditionalUpdate returns true if the object can be updated
	// unconditionally (irrespective of the latest resource version), when
	// there is no resource version specified in the object.
	AllowUnconditionalUpdate() bool
}

// TODO: add other common fields that require global validation.
func validateCommonFields(obj, old runtime.Object, strategy RESTUpdateStrategy) (field.ErrorList, error) {
	allErrs := field.ErrorList{}
	objectMeta, err := meta.Accessor(obj)
	if err != nil {
		return nil, fmt.Errorf("failed to get new object metadata: %v", err)
	}
	oldObjectMeta, err := meta.Accessor(old)
	if err != nil {
		return nil, fmt.Errorf("failed to get old object metadata: %v", err)
	}
	allErrs = append(allErrs, genericvalidation.ValidateObjectMetaAccessor(objectMeta, strategy.NamespaceScoped(), path.ValidatePathSegmentName, field.NewPath("metadata"))...)
	allErrs = append(allErrs, genericvalidation.ValidateObjectMetaAccessorUpdate(objectMeta, oldObjectMeta, field.NewPath("metadata"))...)

	return allErrs, nil
}

// BeforeUpdate ensures that common operations for all resources are performed on update. It only returns
// errors that can be converted to api.Status. It will invoke update validation with the provided existing
// and updated objects.
// It sets zero values only if the object does not have a zero value for the respective field.
func BeforeUpdate(strategy RESTUpdateStrategy, ctx context.Context, obj, old runtime.Object) error {
	objectMeta, kind, kerr := objectMetaAndKind(strategy, obj)
	if kerr != nil {
		return kerr
	}
	if strategy.NamespaceScoped() {
		if !ValidNamespace(ctx, objectMeta) {
			return errors.NewBadRequest("the namespace of the provided object does not match the namespace sent on the request")
		}
	} else if len(objectMeta.GetNamespace()) > 0 {
		objectMeta.SetNamespace(metav1.NamespaceNone)
	}

	// Ensure requests cannot update generation
	oldMeta, err := meta.Accessor(old)
	if err != nil {
		return err
	}
	objectMeta.SetGeneration(oldMeta.GetGeneration())

	// Initializers are a deprecated alpha field and should not be saved
	oldMeta.SetInitializers(nil)
	objectMeta.SetInitializers(nil)

	// Ensure managedFields state is removed unless ServerSideApply is enabled
	if !utilfeature.DefaultFeatureGate.Enabled(features.ServerSideApply) {
		oldMeta.SetManagedFields(nil)
		objectMeta.SetManagedFields(nil)
	}

	strategy.PrepareForUpdate(ctx, obj, old)

	// ClusterName is ignored and should not be saved
	if len(objectMeta.GetClusterName()) > 0 {
		objectMeta.SetClusterName("")
	}
	// Use the existing UID if none is provided
	if len(objectMeta.GetUID()) == 0 {
		objectMeta.SetUID(oldMeta.GetUID())
	}
	// ignore changes to timestamp
	if oldCreationTime := oldMeta.GetCreationTimestamp(); !oldCreationTime.IsZero() {
		objectMeta.SetCreationTimestamp(oldMeta.GetCreationTimestamp())
	}
	// an update can never remove/change a deletion timestamp
	if !oldMeta.GetDeletionTimestamp().IsZero() {
		objectMeta.SetDeletionTimestamp(oldMeta.GetDeletionTimestamp())
	}
	// an update can never remove/change grace period seconds
	if oldMeta.GetDeletionGracePeriodSeconds() != nil && objectMeta.GetDeletionGracePeriodSeconds() == nil {
		objectMeta.SetDeletionGracePeriodSeconds(oldMeta.GetDeletionGracePeriodSeconds())
	}

	// Ensure some common fields, like UID, are validated for all resources.
	errs, err := validateCommonFields(obj, old, strategy)
	if err != nil {
		return errors.NewInternalError(err)
	}

	errs = append(errs, strategy.ValidateUpdate(ctx, obj, old)...)
	if len(errs) > 0 {
		return errors.NewInvalid(kind.GroupKind(), objectMeta.GetName(), errs)
	}

	strategy.Canonicalize(obj)

	return nil
}

// TransformFunc is a function to transform and return newObj
type TransformFunc func(ctx context.Context, newObj runtime.Object, oldObj runtime.Object) (transformedNewObj runtime.Object, err error)

// defaultUpdatedObjectInfo implements UpdatedObjectInfo
type defaultUpdatedObjectInfo struct {
	// obj is the updated object
	obj runtime.Object

	// transformers is an optional list of transforming functions that modify or
	// replace obj using information from the context, old object, or other sources.
	transformers []TransformFunc
}

// DefaultUpdatedObjectInfo returns an UpdatedObjectInfo impl based on the specified object.
func DefaultUpdatedObjectInfo(obj runtime.Object, transformers ...TransformFunc) UpdatedObjectInfo {
	return &defaultUpdatedObjectInfo{obj, transformers}
}

// Preconditions satisfies the UpdatedObjectInfo interface.
func (i *defaultUpdatedObjectInfo) Preconditions() *metav1.Preconditions {
	// Attempt to get the UID out of the object
	accessor, err := meta.Accessor(i.obj)
	if err != nil {
		// If no UID can be read, no preconditions are possible
		return nil
	}

	// If empty, no preconditions needed
	uid := accessor.GetUID()
	if len(uid) == 0 {
		return nil
	}

	return &metav1.Preconditions{UID: &uid}
}

// UpdatedObject satisfies the UpdatedObjectInfo interface.
// It returns a copy of the held obj, passed through any configured transformers.
func (i *defaultUpdatedObjectInfo) UpdatedObject(ctx context.Context, oldObj runtime.Object) (runtime.Object, error) {
	var err error
	// Start with the configured object
	newObj := i.obj

	// If the original is non-nil (might be nil if the first transformer builds the object from the oldObj), make a copy,
	// so we don't return the original. BeforeUpdate can mutate the returned object, doing things like clearing ResourceVersion.
	// If we're re-called, we need to be able to return the pristine version.
	if newObj != nil {
		newObj = newObj.DeepCopyObject()
	}

	// Allow any configured transformers to update the new object
	for _, transformer := range i.transformers {
		newObj, err = transformer(ctx, newObj, oldObj)
		if err != nil {
			return nil, err
		}
	}

	return newObj, nil
}

// wrappedUpdatedObjectInfo allows wrapping an existing objInfo and
// chaining additional transformations/checks on the result of UpdatedObject()
type wrappedUpdatedObjectInfo struct {
	// obj is the updated object
	objInfo UpdatedObjectInfo

	// transformers is an optional list of transforming functions that modify or
	// replace obj using information from the context, old object, or other sources.
	transformers []TransformFunc
}

// WrapUpdatedObjectInfo returns an UpdatedObjectInfo impl that delegates to
// the specified objInfo, then calls the passed transformers
func WrapUpdatedObjectInfo(objInfo UpdatedObjectInfo, transformers ...TransformFunc) UpdatedObjectInfo {
	return &wrappedUpdatedObjectInfo{objInfo, transformers}
}

// Preconditions satisfies the UpdatedObjectInfo interface.
func (i *wrappedUpdatedObjectInfo) Preconditions() *metav1.Preconditions {
	return i.objInfo.Preconditions()
}

// UpdatedObject satisfies the UpdatedObjectInfo interface.
// It delegates to the wrapped objInfo and passes the result through any configured transformers.
func (i *wrappedUpdatedObjectInfo) UpdatedObject(ctx context.Context, oldObj runtime.Object) (runtime.Object, error) {
	newObj, err := i.objInfo.UpdatedObject(ctx, oldObj)
	if err != nil {
		return newObj, err
	}

	// Allow any configured transformers to update the new object or error
	for _, transformer := range i.transformers {
		newObj, err = transformer(ctx, newObj, oldObj)
		if err != nil {
			return nil, err
		}
	}

	return newObj, nil
}

// AdmissionToValidateObjectUpdateFunc converts validating admission to a rest validate object update func
func AdmissionToValidateObjectUpdateFunc(admit admission.Interface, staticAttributes admission.Attributes, o admission.ObjectInterfaces) ValidateObjectUpdateFunc {
	validatingAdmission, ok := admit.(admission.ValidationInterface)
	if !ok {
		return func(obj, old runtime.Object) error { return nil }
	}
	return func(obj, old runtime.Object) error {
		finalAttributes := admission.NewAttributesRecord(
			obj,
			old,
			staticAttributes.GetKind(),
			staticAttributes.GetNamespace(),
			staticAttributes.GetName(),
			staticAttributes.GetResource(),
			staticAttributes.GetSubresource(),
			staticAttributes.GetOperation(),
			staticAttributes.GetOperationOptions(),
			staticAttributes.IsDryRun(),
			staticAttributes.GetUserInfo(),
		)
		if !validatingAdmission.Handles(finalAttributes.GetOperation()) {
			return nil
		}
		return validatingAdmission.Validate(finalAttributes, o)
	}
}
