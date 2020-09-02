/*
Copyright 2020 The Kubernetes Authors.

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

package storage

import (
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
)

// Preconditions must be fulfilled before an operation (update, delete, etc.) is carried out.
type Preconditions struct {
	// Specifies the target UID.
	// +optional
	UID *types.UID `json:"uid,omitempty"`
	// Specifies the target ResourceVersion
	// +optional
	ResourceVersion *string `json:"resourceVersion,omitempty"`
}

// NewPreconditions returns a Preconditions with UID and ResourceVersion set.
func NewPreconditions(uid string, rv string) *Preconditions {
	u := types.UID(uid)
	return &Preconditions{
		UID:             &u,
		ResourceVersion: &rv,
	}
}

// NewUIDPreconditions returns a Preconditions with UID set.
func NewUIDPreconditions(uid string) *Preconditions {
	u := types.UID(uid)
	return &Preconditions{UID: &u}
}

// NewResourceVersionPreconditions returns a Preconditions with ResourceVersion
// set.
func NewResourceVersionPreconditions(rv string) *Preconditions {
	return &Preconditions{ResourceVersion: &rv}
}

// Check is deprecated because it can produce confusing error messages if `obj`
// does not exist. Details at https://github.com/kubernetes/kubernetes/issues/89985.
// Use "SafeCheck" instead.
func (p *Preconditions) Check(key string, obj runtime.Object) error {
	if p == nil {
		return nil
	}
	objMeta, err := meta.Accessor(obj)
	if err != nil {
		return NewInternalErrorf(
			"can't enforce preconditions %v on un-introspectable object %v, got error: %v",
			*p,
			obj,
			err)
	}
	if p.UID != nil && *p.UID != objMeta.GetUID() {
		err := fmt.Sprintf(
			"Precondition failed: UID in precondition: %v, UID in object meta: %v",
			*p.UID,
			objMeta.GetUID())
		return NewInvalidObjError(key, err)
	}
	if p.ResourceVersion != nil && *p.ResourceVersion != objMeta.GetResourceVersion() {
		err := fmt.Sprintf(
			"Precondition failed: ResourceVersion in precondition: %v, ResourceVersion in object meta: %v",
			*p.ResourceVersion,
			objMeta.GetResourceVersion())
		return NewInvalidObjError(key, err)
	}
	return nil
}

// SafeCheck checks whether `obj` meets the preconditions `p`. It is "safe" in
// the sense that it takes into account the case where the preconditions fail
// because `obj` does not exist (`objExists` is false) and produces a clearer
// error message.
func (p *Preconditions) SafeCheck(key string, obj runtime.Object, objExists bool) error {
	if p == nil {
		return nil
	}
	objMeta, err := meta.Accessor(obj)
	if err != nil {
		return NewInternalErrorf(
			"can't enforce preconditions %v on un-introspectable object %v, got error: %v",
			*p,
			obj,
			err)
	}
	if p.UID != nil && *p.UID != objMeta.GetUID() {
		return newPreconditionFailError("UID", string(*p.UID), string(objMeta.GetUID()), key, objExists)
	}
	if p.ResourceVersion != nil && *p.ResourceVersion != objMeta.GetResourceVersion() {
		return newPreconditionFailError("ResourceVersion", *p.ResourceVersion, objMeta.GetResourceVersion(), key, objExists)
	}
	return nil
}

func newPreconditionFailError(preconditionField, preconditionVal, objVal, objName string, objExists bool) error {
	var msg string
	if objExists {
		msg = fmt.Sprintf(
			"Precondition failed: %s in precondition: %s, %s in object meta: %s",
			preconditionField,
			preconditionVal,
			preconditionField,
			objVal)
	} else {
		msg = fmt.Sprintf(
			"Precondition on %s (must be equal to %s) failed because the object does not exist",
			preconditionField,
			preconditionVal)
	}
	return NewInvalidObjError(objName, msg)
}
