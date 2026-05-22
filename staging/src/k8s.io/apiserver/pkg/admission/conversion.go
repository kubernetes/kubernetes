/*
Copyright 2017 The Kubernetes Authors.

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

package admission

import (
	"reflect"

	"github.com/google/cel-go/common/types"
	"github.com/google/cel-go/common/types/ref"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// VersionedAttributes is a wrapper around the original admission attributes, adding versioned
// variants of the object and old object.
type VersionedAttributes struct {
	// Attributes holds the original admission attributes
	Attributes
	// VersionedOldObject holds Attributes.OldObject (if non-nil), converted to VersionedKind and encapsulated in a LazyObject.
	// It must never be mutated.
	VersionedOldObject LazyObject
	// VersionedObject holds Attributes.Object (if non-nil), converted to VersionedKind and encapsulated in a LazyObject.
	// If mutated, Dirty must be set to true by the mutator.
	VersionedObject LazyObject

	// VersionedKind holds the fully qualified kind
	VersionedKind schema.GroupVersionKind
	// Dirty indicates the inner object in VersionedObject has been modified since being converted from Attributes.Object
	Dirty bool
}

// UpdateObject updates the VersionedObject and clears the cached CEL representation.
func (v *VersionedAttributes) UpdateObject(obj runtime.Object) {
	v.Dirty = true
	v.VersionedObject.Set(obj)
}

// GetObject overrides the Attributes.GetObject()
func (v *VersionedAttributes) GetObject() runtime.Object {
	if v.VersionedObject.object != nil {
		return v.VersionedObject.object
	}
	return v.Attributes.GetObject()
}

// ConvertToGVK converts object to the desired gvk.
func ConvertToGVK(obj runtime.Object, gvk schema.GroupVersionKind, o ObjectInterfaces) (runtime.Object, error) {
	// Unlike other resources, custom resources do not have internal version, so
	// if obj is a custom resource, it should not need conversion.
	if obj.GetObjectKind().GroupVersionKind() == gvk {
		return obj, nil
	}
	out, err := o.GetObjectCreater().New(gvk)
	if err != nil {
		return nil, err
	}
	err = o.GetObjectConvertor().Convert(obj, out, nil)
	if err != nil {
		return nil, err
	}
	// Explicitly set the GVK
	out.GetObjectKind().SetGroupVersionKind(gvk)
	return out, nil
}

// NewVersionedAttributes returns versioned attributes with the old and new object (if non-nil) converted to the requested kind
func NewVersionedAttributes(attr Attributes, gvk schema.GroupVersionKind, o ObjectInterfaces) (*VersionedAttributes, error) {
	// convert the old and new objects to the requested version
	versionedAttr := &VersionedAttributes{
		Attributes:    attr,
		VersionedKind: gvk,
	}
	if oldObj := attr.GetOldObject(); oldObj != nil {
		out, err := ConvertToGVK(oldObj, gvk, o)
		if err != nil {
			return nil, err
		}
		versionedAttr.VersionedOldObject.Set(out)
	}
	if obj := attr.GetObject(); obj != nil {
		out, err := ConvertToGVK(obj, gvk, o)
		if err != nil {
			return nil, err
		}
		versionedAttr.VersionedObject.Set(out)
	}

	return versionedAttr, nil
}

// LazyObject encapsulates a versioned runtime.Object and its lazily-evaluated Common Expression Language (CEL) representation.
type LazyObject struct {
	object runtime.Object
	celVal ref.Val
}

// NewLazyObject returns a new LazyObject wrapping the provided runtime.Object.
func NewLazyObject(obj runtime.Object) LazyObject {
	return LazyObject{object: obj}
}

// Object returns the underlying runtime.Object.
func (l *LazyObject) Object() runtime.Object {
	return l.object
}

func (l *LazyObject) Set(obj runtime.Object) {
	l.object = obj
	l.celVal = nil
}

func (l *LazyObject) CELValue() (ref.Val, error) {
	if l.celVal != nil {
		return l.celVal, nil
	}
	if l.object == nil {
		return nil, nil
	}
	// TODO: Eventually use TypedToVal instead of unstructured object conversion.
	unstructuredObj, err := ConvertObjectToUnstructured(l.object)
	if err != nil {
		return nil, err
	}
	l.celVal = types.DefaultTypeAdapter.NativeToValue(unstructuredObj.Object)
	return l.celVal, nil
}

// ConvertVersionedAttributes converts VersionedObject and VersionedOldObject to the specified kind, if needed.
// If attr.VersionedKind already matches the requested kind, no conversion is performed.
// If conversion is required:
// * attr.VersionedObject.Object() is used as the source for the new object if Dirty=true (and is round-tripped through attr.Attributes.Object, clearing Dirty in the process)
// * attr.Attributes.Object is used as the source for the new object if Dirty=false
// * attr.Attributes.OldObject is used as the source for the old object
func ConvertVersionedAttributes(attr *VersionedAttributes, gvk schema.GroupVersionKind, o ObjectInterfaces) error {
	// we already have the desired kind, we're done
	if attr.VersionedKind == gvk {
		return nil
	}

	// convert the original old object to the desired GVK
	if oldObj := attr.Attributes.GetOldObject(); oldObj != nil {
		out, err := ConvertToGVK(oldObj, gvk, o)
		if err != nil {
			return err
		}
		attr.VersionedOldObject.Set(out)
	}

	if attr.VersionedObject.object != nil {
		// convert the existing versioned object to internal
		if attr.Dirty {
			err := o.GetObjectConvertor().Convert(attr.VersionedObject.object, attr.Attributes.GetObject(), nil)
			if err != nil {
				return err
			}
		}

		// and back to external
		out, err := ConvertToGVK(attr.Attributes.GetObject(), gvk, o)
		if err != nil {
			return err
		}
		attr.VersionedObject.Set(out)
	}

	// Remember we converted to this version
	attr.VersionedKind = gvk
	attr.Dirty = false

	return nil
}

// ConvertObjectToUnstructured converts an object to an unstructured representation.
func ConvertObjectToUnstructured(obj interface{}) (*unstructured.Unstructured, error) {
	if obj == nil || reflect.ValueOf(obj).IsNil() {
		return &unstructured.Unstructured{Object: nil}, nil
	}
	ret, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		return nil, err
	}
	return &unstructured.Unstructured{Object: ret}, nil
}
