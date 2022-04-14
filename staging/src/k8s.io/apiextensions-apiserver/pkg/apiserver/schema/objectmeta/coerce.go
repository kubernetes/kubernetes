/*
Copyright 2019 The Kubernetes Authors.

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

package objectmeta

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	utiljson "k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/validation/field"
	kjson "sigs.k8s.io/json"
)

// GetObjectMeta calls GetObjectMetaWithOptions without returning unknown field paths.
func GetObjectMeta(obj map[string]interface{}, dropMalformedFields bool) (*metav1.ObjectMeta, bool, error) {
	meta, found, _, err := GetObjectMetaWithOptions(obj, ObjectMetaOptions{
		DropMalformedFields: dropMalformedFields,
	})
	return meta, found, err
}

// ObjectMetaOptions provides the options for how GetObjectMeta should retrieve the object meta.
type ObjectMetaOptions struct {
	// DropMalformedFields discards malformed serialized metadata fields that
	// cannot be successfully decoded to the corresponding ObjectMeta field.
	// This only applies to fields that are recognized as part of the schema,
	// but of an invalid type (i.e. cause an error when unmarshaling, rather
	// than being dropped or causing a strictErr).
	DropMalformedFields bool
	// ReturnUnknownFieldPaths will return the paths to fields that are not
	// recognized as part of the schema.
	ReturnUnknownFieldPaths bool
	// ParentPath provides the current path up to the given ObjectMeta.
	// If nil, the metadata is assumed to be at the root of the object.
	ParentPath *field.Path
}

// GetObjectMetaWithOptions  does conversion of JSON to ObjectMeta.
// It first tries json.Unmarshal into a metav1.ObjectMeta
// type. If that does not work and opts.DropMalformedFields is true, it does field-by-field best-effort conversion
// throwing away fields which lead to errors.
// If opts.ReturnedUnknownFields is true, it will UnmarshalStrict instead, returning the paths of any unknown fields
// it encounters (i.e. paths returned as strict errs from UnmarshalStrict)
func GetObjectMetaWithOptions(obj map[string]interface{}, opts ObjectMetaOptions) (*metav1.ObjectMeta, bool, []string, error) {
	metadata, found := obj["metadata"]
	if !found {
		return nil, false, nil, nil
	}

	// round-trip through JSON first, hoping that unmarshalling just works
	objectMeta := &metav1.ObjectMeta{}
	metadataBytes, err := utiljson.Marshal(metadata)
	if err != nil {
		return nil, false, nil, err
	}
	var unmarshalErr error
	if opts.ReturnUnknownFieldPaths {
		var strictErrs []error
		strictErrs, unmarshalErr = kjson.UnmarshalStrict(metadataBytes, objectMeta)
		if unmarshalErr == nil {
			if len(strictErrs) > 0 {
				unknownPaths := []string{}
				prefix := opts.ParentPath.Child("metadata").String()
				for _, err := range strictErrs {
					if fieldPathErr, ok := err.(kjson.FieldError); ok {
						unknownPaths = append(unknownPaths, prefix+"."+fieldPathErr.FieldPath())
					}
				}
				return objectMeta, true, unknownPaths, nil
			}
			return objectMeta, true, nil, nil
		}
	} else {
		if unmarshalErr = utiljson.Unmarshal(metadataBytes, objectMeta); unmarshalErr == nil {
			// if successful, return
			return objectMeta, true, nil, nil
		}
	}
	if !opts.DropMalformedFields {
		// if we're not trying to drop malformed fields, return the error
		return nil, true, nil, unmarshalErr
	}

	metadataMap, ok := metadata.(map[string]interface{})
	if !ok {
		return nil, false, nil, fmt.Errorf("invalid metadata: expected object, got %T", metadata)
	}

	// Go field by field accumulating into the metadata object.
	// This takes advantage of the fact that you can repeatedly unmarshal individual fields into a single struct,
	// each iteration preserving the old key-values.
	accumulatedObjectMeta := &metav1.ObjectMeta{}
	testObjectMeta := &metav1.ObjectMeta{}
	var unknownFields []string
	for k, v := range metadataMap {
		// serialize a single field
		if singleFieldBytes, err := utiljson.Marshal(map[string]interface{}{k: v}); err == nil {
			// do a test unmarshal
			if utiljson.Unmarshal(singleFieldBytes, testObjectMeta) == nil {
				// if that succeeds, unmarshal for real
				if opts.ReturnUnknownFieldPaths {
					strictErrs, _ := kjson.UnmarshalStrict(singleFieldBytes, accumulatedObjectMeta)
					if len(strictErrs) > 0 {
						prefix := opts.ParentPath.Child("metadata").String()
						for _, err := range strictErrs {
							if fieldPathErr, ok := err.(kjson.FieldError); ok {
								unknownFields = append(unknownFields, prefix+"."+fieldPathErr.FieldPath())
							}
						}
					}
				} else {
					utiljson.Unmarshal(singleFieldBytes, accumulatedObjectMeta)
				}
			}
		}
	}

	return accumulatedObjectMeta, true, unknownFields, nil
}

// SetObjectMeta writes back ObjectMeta into a JSON data structure.
func SetObjectMeta(obj map[string]interface{}, objectMeta *metav1.ObjectMeta) error {
	if objectMeta == nil {
		unstructured.RemoveNestedField(obj, "metadata")
		return nil
	}

	metadata, err := runtime.DefaultUnstructuredConverter.ToUnstructured(objectMeta)
	if err != nil {
		return err
	}

	obj["metadata"] = metadata
	return nil
}
