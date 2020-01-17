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
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

var encodingjson = json.CaseSensitiveJsonIterator()

// GetObjectMeta does conversion of JSON to ObjectMeta. It first tries json.Unmarshal into a metav1.ObjectMeta
// type. If that does not work and dropMalformedFields is true, it does field-by-field best-effort conversion
// throwing away fields which lead to errors.
func GetObjectMeta(obj map[string]interface{}, dropMalformedFields bool) (*metav1.ObjectMeta, bool, error) {
	metadata, found := obj["metadata"]
	if !found {
		return nil, false, nil
	}

	// round-trip through JSON first, hoping that unmarshalling just works
	objectMeta := &metav1.ObjectMeta{}
	metadataBytes, err := encodingjson.Marshal(metadata)
	if err != nil {
		return nil, false, err
	}
	if err = encodingjson.Unmarshal(metadataBytes, objectMeta); err == nil {
		// if successful, return
		return objectMeta, true, nil
	}
	if !dropMalformedFields {
		// if we're not trying to drop malformed fields, return the error
		return nil, true, err
	}

	metadataMap, ok := metadata.(map[string]interface{})
	if !ok {
		return nil, false, fmt.Errorf("invalid metadata: expected object, got %T", metadata)
	}

	// Go field by field accumulating into the metadata object.
	// This takes advantage of the fact that you can repeatedly unmarshal individual fields into a single struct,
	// each iteration preserving the old key-values.
	accumulatedObjectMeta := &metav1.ObjectMeta{}
	testObjectMeta := &metav1.ObjectMeta{}
	for k, v := range metadataMap {
		// serialize a single field
		if singleFieldBytes, err := encodingjson.Marshal(map[string]interface{}{k: v}); err == nil {
			// do a test unmarshal
			if encodingjson.Unmarshal(singleFieldBytes, testObjectMeta) == nil {
				// if that succeeds, unmarshal for real
				encodingjson.Unmarshal(singleFieldBytes, accumulatedObjectMeta)
			}
		}
	}

	return accumulatedObjectMeta, true, nil
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
