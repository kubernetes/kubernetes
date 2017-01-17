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
	"fmt"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation/genericvalidation"
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
	PrepareForUpdate(ctx genericapirequest.Context, obj, old runtime.Object)
	// ValidateUpdate is invoked after default fields in the object have been
	// filled in before the object is persisted.  This method should not mutate
	// the object.
	ValidateUpdate(ctx genericapirequest.Context, obj, old runtime.Object) field.ErrorList
	// Canonicalize is invoked after validation has succeeded but before the
	// object has been persisted.  This method may mutate the object.
	Canonicalize(obj runtime.Object)
	// AllowUnconditionalUpdate returns true if the object can be updated
	// unconditionally (irrespective of the latest resource version), when
	// there is no resource version specified in the object.
	AllowUnconditionalUpdate() bool
}

// TODO: add other common fields that require global validation.
func validateCommonFields(obj, old runtime.Object) (field.ErrorList, error) {
	allErrs := field.ErrorList{}
	objectMeta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return nil, fmt.Errorf("failed to get new object metadata: %v", err)
	}
	oldObjectMeta, err := api.ObjectMetaFor(old)
	if err != nil {
		return nil, fmt.Errorf("failed to get old object metadata: %v", err)
	}
	allErrs = append(allErrs, genericvalidation.ValidateObjectMetaUpdate(objectMeta, oldObjectMeta, field.NewPath("metadata"))...)

	return allErrs, nil
}

// BeforeUpdate ensures that common operations for all resources are performed on update. It only returns
// errors that can be converted to api.Status. It will invoke update validation with the provided existing
// and updated objects.
func BeforeUpdate(strategy RESTUpdateStrategy, ctx genericapirequest.Context, obj, old runtime.Object) error {
	objectMeta, kind, kerr := objectMetaAndKind(strategy, obj)
	if kerr != nil {
		return kerr
	}
	if strategy.NamespaceScoped() {
		if !ValidNamespace(ctx, objectMeta) {
			return errors.NewBadRequest("the namespace of the provided object does not match the namespace sent on the request")
		}
	} else {
		objectMeta.Namespace = api.NamespaceNone
	}
	// Ensure requests cannot update generation
	oldMeta, err := api.ObjectMetaFor(old)
	if err != nil {
		return err
	}
	objectMeta.Generation = oldMeta.Generation

	strategy.PrepareForUpdate(ctx, obj, old)

	// ClusterName is ignored and should not be saved
	objectMeta.ClusterName = ""

	// Ensure some common fields, like UID, are validated for all resources.
	errs, err := validateCommonFields(obj, old)
	if err != nil {
		return errors.NewInternalError(err)
	}

	errs = append(errs, strategy.ValidateUpdate(ctx, obj, old)...)
	if len(errs) > 0 {
		return errors.NewInvalid(kind.GroupKind(), objectMeta.Name, errs)
	}

	strategy.Canonicalize(obj)

	return nil
}

// TransformFunc is a function to transform and return newObj
type TransformFunc func(ctx genericapirequest.Context, newObj runtime.Object, oldObj runtime.Object) (transformedNewObj runtime.Object, err error)

// defaultUpdatedObjectInfo implements UpdatedObjectInfo
type defaultUpdatedObjectInfo struct {
	// obj is the updated object
	obj runtime.Object

	// copier makes a copy of the object before returning it.
	// this allows repeated calls to UpdatedObject() to return
	// pristine data, even if the returned value is mutated.
	copier runtime.ObjectCopier

	// transformers is an optional list of transforming functions that modify or
	// replace obj using information from the context, old object, or other sources.
	transformers []TransformFunc
}

// DefaultUpdatedObjectInfo returns an UpdatedObjectInfo impl based on the specified object.
func DefaultUpdatedObjectInfo(obj runtime.Object, copier runtime.ObjectCopier, transformers ...TransformFunc) UpdatedObjectInfo {
	return &defaultUpdatedObjectInfo{obj, copier, transformers}
}

// Preconditions satisfies the UpdatedObjectInfo interface.
func (i *defaultUpdatedObjectInfo) Preconditions() *api.Preconditions {
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

	return &api.Preconditions{UID: &uid}
}

// UpdatedObject satisfies the UpdatedObjectInfo interface.
// It returns a copy of the held obj, passed through any configured transformers.
func (i *defaultUpdatedObjectInfo) UpdatedObject(ctx genericapirequest.Context, oldObj runtime.Object) (runtime.Object, error) {
	var err error
	// Start with the configured object
	newObj := i.obj

	// If the original is non-nil (might be nil if the first transformer builds the object from the oldObj), make a copy,
	// so we don't return the original. BeforeUpdate can mutate the returned object, doing things like clearing ResourceVersion.
	// If we're re-called, we need to be able to return the pristine version.
	if newObj != nil {
		newObj, err = i.copier.Copy(newObj)
		if err != nil {
			return nil, err
		}
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
func (i *wrappedUpdatedObjectInfo) Preconditions() *api.Preconditions {
	return i.objInfo.Preconditions()
}

// UpdatedObject satisfies the UpdatedObjectInfo interface.
// It delegates to the wrapped objInfo and passes the result through any configured transformers.
func (i *wrappedUpdatedObjectInfo) UpdatedObject(ctx genericapirequest.Context, oldObj runtime.Object) (runtime.Object, error) {
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
