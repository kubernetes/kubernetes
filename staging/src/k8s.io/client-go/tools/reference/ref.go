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

package reference

import (
	"errors"
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

var (
	// Errors that could be returned by GetReference.
	ErrNilObject = errors.New("can't reference a nil object")
)

// GetReference returns an ObjectReference which refers to the given
// object, or an error if the object doesn't follow the conventions
// that would allow this.
// TODO: should take a meta.Interface see https://issue.k8s.io/7127
func GetReference(scheme *runtime.Scheme, obj runtime.Object) (*v1.ObjectReference, error) {
	if obj == nil {
		return nil, ErrNilObject
	}
	if ref, ok := obj.(*v1.ObjectReference); ok {
		// Don't make a reference to a reference.
		return ref, nil
	}

	// An object that implements only List has enough metadata to build a reference
	var listMeta metav1.Common
	objectMeta, err := meta.Accessor(obj)
	if err != nil {
		listMeta, err = meta.CommonAccessor(obj)
		if err != nil {
			return nil, err
		}
	} else {
		listMeta = objectMeta
	}

	gvk := obj.GetObjectKind().GroupVersionKind()

	// If object meta doesn't contain data about kind and/or version,
	// we are falling back to scheme.
	//
	// TODO: This doesn't work for CRDs, which are not registered in scheme.
	if gvk.Empty() {
		if scheme == nil {
			return nil, errors.New("scheme is required to look up gvk")
		}
		gvks, _, err := scheme.ObjectKinds(obj)
		if err != nil {
			return nil, err
		}
		if len(gvks) == 0 || gvks[0].Empty() {
			return nil, fmt.Errorf("unexpected gvks registered for object %T: %v", obj, gvks)
		}
		// TODO: The same object can be registered for multiple group versions
		// (although in practise this doesn't seem to be used).
		// In such case, the version set may not be correct.
		gvk = gvks[0]
	}

	kind := gvk.Kind
	version := gvk.GroupVersion().String()

	// only has list metadata
	if objectMeta == nil {
		return &v1.ObjectReference{
			Kind:            kind,
			APIVersion:      version,
			ResourceVersion: listMeta.GetResourceVersion(),
		}, nil
	}

	return &v1.ObjectReference{
		Kind:            kind,
		APIVersion:      version,
		Name:            objectMeta.GetName(),
		Namespace:       objectMeta.GetNamespace(),
		UID:             objectMeta.GetUID(),
		ResourceVersion: objectMeta.GetResourceVersion(),
	}, nil
}

// GetPartialReference is exactly like GetReference, but allows you to set the FieldPath.
func GetPartialReference(scheme *runtime.Scheme, obj runtime.Object, fieldPath string) (*v1.ObjectReference, error) {
	ref, err := GetReference(scheme, obj)
	if err != nil {
		return nil, err
	}
	ref.FieldPath = fieldPath
	return ref, nil
}
