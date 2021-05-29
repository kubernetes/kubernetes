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

package generic

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

// ConvertToGVK converts object to the desired gvk.
func ConvertToGVK(obj runtime.Object, gvk schema.GroupVersionKind, o admission.ObjectInterfaces) (runtime.Object, error) {
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
func NewVersionedAttributes(attr admission.Attributes, gvk schema.GroupVersionKind, o admission.ObjectInterfaces) (*VersionedAttributes, error) {
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
		versionedAttr.VersionedOldObject = out
	}
	if obj := attr.GetObject(); obj != nil {
		out, err := ConvertToGVK(obj, gvk, o)
		if err != nil {
			return nil, err
		}
		versionedAttr.VersionedObject = out
	}
	return versionedAttr, nil
}

// ConvertVersionedAttributes converts VersionedObject and VersionedOldObject to the specified kind, if needed.
// If attr.VersionedKind already matches the requested kind, no conversion is performed.
// If conversion is required:
// * attr.VersionedObject is used as the source for the new object if Dirty=true (and is round-tripped through attr.Attributes.Object, clearing Dirty in the process)
// * attr.Attributes.Object is used as the source for the new object if Dirty=false
// * attr.Attributes.OldObject is used as the source for the old object
func ConvertVersionedAttributes(attr *VersionedAttributes, gvk schema.GroupVersionKind, o admission.ObjectInterfaces) error {
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
		attr.VersionedOldObject = out
	}

	if attr.VersionedObject != nil {
		// convert the existing versioned object to internal
		if attr.Dirty {
			err := o.GetObjectConvertor().Convert(attr.VersionedObject, attr.Attributes.GetObject(), nil)
			if err != nil {
				return err
			}
		}

		// and back to external
		out, err := ConvertToGVK(attr.Attributes.GetObject(), gvk, o)
		if err != nil {
			return err
		}
		attr.VersionedObject = out
	}

	// Remember we converted to this version
	attr.VersionedKind = gvk
	attr.Dirty = false

	return nil
}
