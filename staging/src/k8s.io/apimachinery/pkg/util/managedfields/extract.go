/*
Copyright 2021 The Kubernetes Authors.

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

package managedfields

import (
	"bytes"
	"fmt"

	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/typed"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

// ExtractInto extracts the applied configuration state from object for fieldManager
// into applyConfiguration. If no managed fields are found for the given fieldManager,
// no error is returned, but applyConfiguration is left unpopulated. It is possible
// that no managed fields were found for the fieldManager because other field managers
// have taken ownership of all the fields previously owned by the fieldManager. It is
// also possible the fieldManager never owned fields.
//
// The provided object MUST bo a root resource object since subresource objects
// do not contain their own managed fields. For example, an autoscaling.Scale
// object read from a "scale" subresource does not have any managed fields and so
// cannot be used as the object.
//
// If the fields of a subresource are a subset of the fields of the root object,
// and their field paths and types are exactly the same, then ExtractInto can be
// called with the root resource as the object and the subresource as the
// applyConfiguration. This works for "status", obviously, because status is
// represented by the exact same object as the root resource. This this does NOT
// work, for example, with the "scale" subresources of Deployment, ReplicaSet and
// StatefulSet. While the spec.replicas, status.replicas fields are in the same
// exact field path locations as they are in autoscaling.Scale, the selector
// fields are in different locations, and are a different type.
func ExtractInto(object runtime.Object, objectType typed.ParseableType, fieldManager string, applyConfiguration interface{}, subresource string) error {
	typedObj, err := toTyped(object, objectType)
	if err != nil {
		return fmt.Errorf("error converting obj to typed: %w", err)
	}

	accessor, err := meta.Accessor(object)
	if err != nil {
		return fmt.Errorf("error accessing metadata: %w", err)
	}
	fieldsEntry, ok := findManagedFields(accessor, fieldManager, subresource)
	if !ok {
		return nil
	}
	fieldset := &fieldpath.Set{}
	err = fieldset.FromJSON(bytes.NewReader(fieldsEntry.FieldsV1.Raw))
	if err != nil {
		return fmt.Errorf("error marshalling FieldsV1 to JSON: %w", err)
	}

	u := typedObj.ExtractItems(fieldset.Leaves()).AsValue().Unstructured()
	m, ok := u.(map[string]interface{})
	if !ok {
		return fmt.Errorf("unable to convert managed fields for %s to unstructured, expected map, got %T", fieldManager, u)
	}

	// set the type meta manually if it doesn't exist to avoid missing kind errors
	// when decoding from unstructured JSON
	if _, ok := m["kind"]; !ok && object.GetObjectKind().GroupVersionKind().Kind != "" {
		m["kind"] = object.GetObjectKind().GroupVersionKind().Kind
		m["apiVersion"] = object.GetObjectKind().GroupVersionKind().GroupVersion().String()
	}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(m, applyConfiguration); err != nil {
		return fmt.Errorf("error extracting into obj from unstructured: %w", err)
	}
	return nil
}

func findManagedFields(accessor metav1.Object, fieldManager string, subresource string) (metav1.ManagedFieldsEntry, bool) {
	objManagedFields := accessor.GetManagedFields()
	for _, mf := range objManagedFields {
		if mf.Manager == fieldManager && mf.Operation == metav1.ManagedFieldsOperationApply && mf.Subresource == subresource {
			return mf, true
		}
	}
	return metav1.ManagedFieldsEntry{}, false
}

func toTyped(obj runtime.Object, objectType typed.ParseableType) (*typed.TypedValue, error) {
	switch o := obj.(type) {
	case *unstructured.Unstructured:
		return objectType.FromUnstructured(o.Object)
	default:
		return objectType.FromStructured(o)
	}
}
