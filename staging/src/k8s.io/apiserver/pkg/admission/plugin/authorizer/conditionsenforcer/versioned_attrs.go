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

package conditionsenforcer

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

// newVersionedAttributes returns versioned attributes with the old and new object (if non-nil) converted to the requested kind
// TODO(luxas): This should be admission.NewVersionedAttributes, but that implementation by error does not override GetOldObject().
// That should be fixed, but as a separate change (probably feature-gated).
// Thus is versionedAttributes temporarily re-implemented here with the fix.
func newVersionedAttributes(attr admission.Attributes, gvk schema.GroupVersionKind, o admission.ObjectInterfaces) (*versionedAttributes, error) {
	// convert the old and new objects to the requested version
	versionedAttr := &versionedAttributes{
		Attributes: attr,
	}
	if oldObj := attr.GetOldObject(); oldObj != nil {
		out, err := admission.ConvertToGVK(oldObj, gvk, o)
		if err != nil {
			return nil, err
		}
		versionedAttr.VersionedOldObject = out
	}
	if obj := attr.GetObject(); obj != nil {
		out, err := admission.ConvertToGVK(obj, gvk, o)
		if err != nil {
			return nil, err
		}
		versionedAttr.VersionedObject = out
	}
	return versionedAttr, nil
}

// versionedAttributes is a wrapper around the original admission attributes, adding versioned
// variants of the object and old object.
type versionedAttributes struct {
	// Attributes holds the original admission attributes
	admission.Attributes
	// VersionedOldObject holds Attributes.OldObject (if non-nil), converted to VersionedKind.
	// It must never be mutated.
	VersionedOldObject runtime.Object
	// VersionedObject holds Attributes.Object (if non-nil), converted to VersionedKind.
	// If mutated, Dirty must be set to true by the mutator.
	VersionedObject runtime.Object
}

// GetObject overrides the Attributes.GetObject()
func (v *versionedAttributes) GetObject() runtime.Object {
	if v.VersionedObject != nil {
		return v.VersionedObject
	}
	return v.Attributes.GetObject()
}

// GetOldObject overrides the Attributes.GetOldObject()
func (v *versionedAttributes) GetOldObject() runtime.Object {
	if v.VersionedOldObject != nil {
		return v.VersionedOldObject
	}
	return v.Attributes.GetOldObject()
}
