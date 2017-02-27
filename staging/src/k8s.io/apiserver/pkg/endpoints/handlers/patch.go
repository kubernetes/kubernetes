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

package handlers

import (
	"encoding/json"
	"fmt"

	"k8s.io/apimachinery/pkg/conversion/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"

	"github.com/evanphx/json-patch"
)

// patchObjectJSON patches the <originalObject> with <patchJS> and stores
// the result in <objToUpdate>.
// Currently it also returns the original and patched objects serialized to
// JSONs (this may not be needed once we can apply patches at the
// map[string]interface{} level).
func patchObjectJSON(
	patchType types.PatchType,
	codec runtime.Codec,
	originalObject runtime.Object,
	patchJS []byte,
	objToUpdate runtime.Object,
	versionedObj runtime.Object,
) (originalObjJS []byte, patchedObjJS []byte, retErr error) {
	js, err := runtime.Encode(codec, originalObject)
	if err != nil {
		return nil, nil, err
	}
	originalObjJS = js

	switch patchType {
	case types.JSONPatchType:
		patchObj, err := jsonpatch.DecodePatch(patchJS)
		if err != nil {
			return nil, nil, err
		}
		if patchedObjJS, err = patchObj.Apply(originalObjJS); err != nil {
			return nil, nil, err
		}
	case types.MergePatchType:
		if patchedObjJS, err = jsonpatch.MergePatch(originalObjJS, patchJS); err != nil {
			return nil, nil, err
		}
	case types.StrategicMergePatchType:
		if patchedObjJS, err = strategicpatch.StrategicMergePatch(originalObjJS, patchJS, versionedObj); err != nil {
			return nil, nil, err
		}
	default:
		// only here as a safety net - go-restful filters content-type
		return nil, nil, fmt.Errorf("unknown Content-Type header for patch: %v", patchType)
	}
	if err := runtime.DecodeInto(codec, patchedObjJS, objToUpdate); err != nil {
		return nil, nil, err
	}
	return
}

// strategicPatchObject applies a strategic merge patch of <patchJS> to
// <originalObject> and stores the result in <objToUpdate>.
// It additionally returns the map[string]interface{} representation of the
// <originalObject> and <patchJS>.
// NOTE: Both <originalObject> and <objToUpdate> are supposed to be versioned.
func strategicPatchObject(
	codec runtime.Codec,
	originalObject runtime.Object,
	patchJS []byte,
	objToUpdate runtime.Object,
	versionedObj runtime.Object,
) (originalObjMap map[string]interface{}, patchMap map[string]interface{}, retErr error) {
	originalObjMap = make(map[string]interface{})
	if err := unstructured.DefaultConverter.ToUnstructured(originalObject, &originalObjMap); err != nil {
		return nil, nil, err
	}

	patchMap = make(map[string]interface{})
	if err := json.Unmarshal(patchJS, &patchMap); err != nil {
		return nil, nil, err
	}

	if err := applyPatchToObject(codec, originalObjMap, patchMap, objToUpdate, versionedObj); err != nil {
		return nil, nil, err
	}
	return
}

// applyPatchToObject applies a strategic merge patch of <patchMap> to
// <originalMap> and stores the result in <objToUpdate>, though it operates
// on versioned map[string]interface{} representations.
func applyPatchToObject(
	codec runtime.Codec,
	originalMap map[string]interface{},
	patchMap map[string]interface{},
	objToUpdate runtime.Object,
	versionedObj runtime.Object,
) error {
	patchedObjMap, err := strategicpatch.StrategicMergeMapPatch(originalMap, patchMap, versionedObj)
	if err != nil {
		return err
	}

	return unstructured.DefaultConverter.FromUnstructured(patchedObjMap, objToUpdate)
}
