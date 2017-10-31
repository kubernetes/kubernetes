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
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/evanphx/json-patch"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/conversion/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
)

// PatchResource returns a function that will handle a resource patch
// TODO: Eventually PatchResource should just use GuaranteedUpdate and this routine should be a bit cleaner
func PatchResource(r rest.Patcher, scope RequestScope, admit admission.Interface, converter runtime.ObjectConvertor) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// TODO: we either want to remove timeout or document it (if we
		// document, move timeout out of this function and declare it in
		// api_installer)
		timeout := parseTimeout(req.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)

		versionedObj, err := converter.ConvertToVersion(r.New(), scope.Kind.GroupVersion())
		if err != nil {
			scope.err(err, w, req)
			return
		}

		// TODO: handle this in negotiation
		contentType := req.Header.Get("Content-Type")
		// Remove "; charset=" if included in header.
		if idx := strings.Index(contentType, ";"); idx > 0 {
			contentType = contentType[:idx]
		}
		patchType := types.PatchType(contentType)

		patchJS, err := readBody(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		ae := request.AuditEventFrom(ctx)
		audit.LogRequestPatch(ae, patchJS)

		s, ok := runtime.SerializerInfoForMediaType(scope.Serializer.SupportedMediaTypes(), runtime.ContentTypeJSON)
		if !ok {
			scope.err(fmt.Errorf("no serializer defined for JSON"), w, req)
			return
		}
		gv := scope.Kind.GroupVersion()
		codec := runtime.NewCodec(
			scope.Serializer.EncoderForVersion(s.Serializer, gv),
			scope.Serializer.DecoderToVersion(s.Serializer, schema.GroupVersion{Group: gv.Group, Version: runtime.APIVersionInternal}),
		)

		updateAdmit := func(updatedObject runtime.Object, currentObject runtime.Object) error {
			if admit != nil && admit.Handles(admission.Update) {
				userInfo, _ := request.UserFrom(ctx)
				return admit.Admit(admission.NewAttributesRecord(updatedObject, currentObject, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Update, userInfo))
			}

			return nil
		}

		result, err := patchResource(ctx, updateAdmit, timeout, versionedObj, r, name, patchType, patchJS,
			scope.Namer, scope.Creater, scope.Defaulter, scope.UnsafeConvertor, scope.Kind, scope.Resource, codec)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		requestInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			scope.err(fmt.Errorf("missing requestInfo"), w, req)
			return
		}
		if err := setSelfLink(result, requestInfo, scope.Namer); err != nil {
			scope.err(err, w, req)
			return
		}

		transformResponseObject(ctx, scope, req, w, http.StatusOK, result)
	}
}

type updateAdmissionFunc func(updatedObject runtime.Object, currentObject runtime.Object) error

// patchResource divides PatchResource for easier unit testing
func patchResource(
	ctx request.Context,
	admit updateAdmissionFunc,
	timeout time.Duration,
	versionedObj runtime.Object,
	patcher rest.Patcher,
	name string,
	patchType types.PatchType,
	patchJS []byte,
	namer ScopeNamer,
	creater runtime.ObjectCreater,
	defaulter runtime.ObjectDefaulter,
	unsafeConvertor runtime.ObjectConvertor,
	kind schema.GroupVersionKind,
	resource schema.GroupVersionResource,
	codec runtime.Codec,
) (runtime.Object, error) {

	namespace := request.NamespaceValue(ctx)

	var (
		originalObjJS           []byte
		originalPatchedObjJS    []byte
		originalObjMap          map[string]interface{}
		getOriginalPatchMap     func() (map[string]interface{}, error)
		lastConflictErr         error
		originalResourceVersion string
	)

	// applyPatch is called every time GuaranteedUpdate asks for the updated object,
	// and is given the currently persisted object as input.
	applyPatch := func(_ request.Context, _, currentObject runtime.Object) (runtime.Object, error) {
		// Make sure we actually have a persisted currentObject
		if hasUID, err := hasUID(currentObject); err != nil {
			return nil, err
		} else if !hasUID {
			return nil, errors.NewNotFound(resource.GroupResource(), name)
		}

		currentResourceVersion := ""
		if currentMetadata, err := meta.Accessor(currentObject); err == nil {
			currentResourceVersion = currentMetadata.GetResourceVersion()
		}

		switch {
		case originalObjJS == nil && originalObjMap == nil:
			// first time through,
			// 1. apply the patch
			// 2. save the original and patched to detect whether there were conflicting changes on retries

			originalResourceVersion = currentResourceVersion
			objToUpdate := patcher.New()

			// For performance reasons, in case of strategicpatch, we avoid json
			// marshaling and unmarshaling and operate just on map[string]interface{}.
			// In case of other patch types, we still have to operate on JSON
			// representations.
			switch patchType {
			case types.JSONPatchType, types.MergePatchType:
				originalJS, patchedJS, err := patchObjectJSON(patchType, codec, currentObject, patchJS, objToUpdate, versionedObj)
				if err != nil {
					return nil, interpretPatchError(err)
				}
				originalObjJS, originalPatchedObjJS = originalJS, patchedJS

				// Make a getter that can return a fresh strategic patch map if needed for conflict retries
				// We have to rebuild it each time we need it, because the map gets mutated when being applied
				var originalPatchBytes []byte
				getOriginalPatchMap = func() (map[string]interface{}, error) {
					if originalPatchBytes == nil {
						// Compute once
						originalPatchBytes, err = strategicpatch.CreateTwoWayMergePatch(originalObjJS, originalPatchedObjJS, versionedObj)
						if err != nil {
							return nil, interpretPatchError(err)
						}
					}
					// Return a fresh map every time
					originalPatchMap := make(map[string]interface{})
					if err := json.Unmarshal(originalPatchBytes, &originalPatchMap); err != nil {
						return nil, errors.NewBadRequest(err.Error())
					}
					return originalPatchMap, nil
				}

			case types.StrategicMergePatchType:
				// Since the patch is applied on versioned objects, we need to convert the
				// current object to versioned representation first.
				currentVersionedObject, err := unsafeConvertor.ConvertToVersion(currentObject, kind.GroupVersion())
				if err != nil {
					return nil, err
				}
				versionedObjToUpdate, err := creater.New(kind)
				if err != nil {
					return nil, err
				}
				// Capture the original object map and patch for possible retries.
				originalMap, err := unstructured.DefaultConverter.ToUnstructured(currentVersionedObject)
				if err != nil {
					return nil, err
				}
				if err := strategicPatchObject(codec, defaulter, currentVersionedObject, patchJS, versionedObjToUpdate, versionedObj); err != nil {
					return nil, err
				}
				// Convert the object back to unversioned.
				gvk := kind.GroupKind().WithVersion(runtime.APIVersionInternal)
				unversionedObjToUpdate, err := unsafeConvertor.ConvertToVersion(versionedObjToUpdate, gvk.GroupVersion())
				if err != nil {
					return nil, err
				}
				objToUpdate = unversionedObjToUpdate
				// Store unstructured representation for possible retries.
				originalObjMap = originalMap
				// Make a getter that can return a fresh strategic patch map if needed for conflict retries
				// We have to rebuild it each time we need it, because the map gets mutated when being applied
				getOriginalPatchMap = func() (map[string]interface{}, error) {
					patchMap := make(map[string]interface{})
					if err := json.Unmarshal(patchJS, &patchMap); err != nil {
						return nil, errors.NewBadRequest(err.Error())
					}
					return patchMap, nil
				}
			}
			if err := checkName(objToUpdate, name, namespace, namer); err != nil {
				return nil, err
			}
			return objToUpdate, nil

		default:
			// on a conflict,
			// 1. build a strategic merge patch from originalJS and the patchedJS.  Different patch types can
			//    be specified, but a strategic merge patch should be expressive enough handle them.  Build the
			//    patch with this type to handle those cases.
			// 2. build a strategic merge patch from originalJS and the currentJS
			// 3. ensure no conflicts between the two patches
			// 4. apply the #1 patch to the currentJS object

			// Since the patch is applied on versioned objects, we need to convert the
			// current object to versioned representation first.
			currentVersionedObject, err := unsafeConvertor.ConvertToVersion(currentObject, kind.GroupVersion())
			if err != nil {
				return nil, err
			}
			currentObjMap, err := unstructured.DefaultConverter.ToUnstructured(currentVersionedObject)
			if err != nil {
				return nil, err
			}

			var currentPatchMap map[string]interface{}
			if originalObjMap != nil {
				var err error
				currentPatchMap, err = strategicpatch.CreateTwoWayMergeMapPatch(originalObjMap, currentObjMap, versionedObj)
				if err != nil {
					return nil, interpretPatchError(err)
				}
			} else {
				// Compute current patch.
				currentObjJS, err := runtime.Encode(codec, currentObject)
				if err != nil {
					return nil, err
				}
				currentPatch, err := strategicpatch.CreateTwoWayMergePatch(originalObjJS, currentObjJS, versionedObj)
				if err != nil {
					return nil, interpretPatchError(err)
				}
				currentPatchMap = make(map[string]interface{})
				if err := json.Unmarshal(currentPatch, &currentPatchMap); err != nil {
					return nil, errors.NewBadRequest(err.Error())
				}
			}

			// Get a fresh copy of the original strategic patch each time through, since applying it mutates the map
			originalPatchMap, err := getOriginalPatchMap()
			if err != nil {
				return nil, err
			}

			hasConflicts, err := mergepatch.HasConflicts(originalPatchMap, currentPatchMap)
			if err != nil {
				return nil, err
			}

			if hasConflicts {
				diff1, _ := json.Marshal(currentPatchMap)
				diff2, _ := json.Marshal(originalPatchMap)
				patchDiffErr := fmt.Errorf("there is a meaningful conflict (firstResourceVersion: %q, currentResourceVersion: %q):\n diff1=%v\n, diff2=%v\n", originalResourceVersion, currentResourceVersion, string(diff1), string(diff2))
				glog.V(4).Infof("patchResource failed for resource %s, because there is a meaningful conflict(firstResourceVersion: %q, currentResourceVersion: %q):\n diff1=%v\n, diff2=%v\n", name, originalResourceVersion, currentResourceVersion, string(diff1), string(diff2))

				// Return the last conflict error we got if we have one
				if lastConflictErr != nil {
					return nil, lastConflictErr
				}
				// Otherwise manufacture one of our own
				return nil, errors.NewConflict(resource.GroupResource(), name, patchDiffErr)
			}

			versionedObjToUpdate, err := creater.New(kind)
			if err != nil {
				return nil, err
			}
			if err := applyPatchToObject(codec, defaulter, currentObjMap, originalPatchMap, versionedObjToUpdate, versionedObj); err != nil {
				return nil, err
			}
			// Convert the object back to unversioned.
			gvk := kind.GroupKind().WithVersion(runtime.APIVersionInternal)
			objToUpdate, err := unsafeConvertor.ConvertToVersion(versionedObjToUpdate, gvk.GroupVersion())
			if err != nil {
				return nil, err
			}

			return objToUpdate, nil
		}
	}

	// applyAdmission is called every time GuaranteedUpdate asks for the updated object,
	// and is given the currently persisted object and the patched object as input.
	applyAdmission := func(ctx request.Context, patchedObject runtime.Object, currentObject runtime.Object) (runtime.Object, error) {
		return patchedObject, admit(patchedObject, currentObject)
	}

	updatedObjectInfo := rest.DefaultUpdatedObjectInfo(nil, applyPatch, applyAdmission)

	return finishRequest(timeout, func() (runtime.Object, error) {
		updateObject, _, updateErr := patcher.Update(ctx, name, updatedObjectInfo)
		for i := 0; i < MaxRetryWhenPatchConflicts && (errors.IsConflict(updateErr)); i++ {
			lastConflictErr = updateErr
			updateObject, _, updateErr = patcher.Update(ctx, name, updatedObjectInfo)
		}
		return updateObject, updateErr
	})
}

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
	defaulter runtime.ObjectDefaulter,
	originalObject runtime.Object,
	patchJS []byte,
	objToUpdate runtime.Object,
	versionedObj runtime.Object,
) error {
	originalObjMap, err := unstructured.DefaultConverter.ToUnstructured(originalObject)
	if err != nil {
		return err
	}

	patchMap := make(map[string]interface{})
	if err := json.Unmarshal(patchJS, &patchMap); err != nil {
		return errors.NewBadRequest(err.Error())
	}

	if err := applyPatchToObject(codec, defaulter, originalObjMap, patchMap, objToUpdate, versionedObj); err != nil {
		return err
	}
	return nil
}

// applyPatchToObject applies a strategic merge patch of <patchMap> to
// <originalMap> and stores the result in <objToUpdate>.
// NOTE: <objToUpdate> must be a versioned object.
func applyPatchToObject(
	codec runtime.Codec,
	defaulter runtime.ObjectDefaulter,
	originalMap map[string]interface{},
	patchMap map[string]interface{},
	objToUpdate runtime.Object,
	versionedObj runtime.Object,
) error {
	patchedObjMap, err := strategicpatch.StrategicMergeMapPatch(originalMap, patchMap, versionedObj)
	if err != nil {
		return err
	}

	// Rather than serialize the patched map to JSON, then decode it to an object, we go directly from a map to an object
	if err := unstructured.DefaultConverter.FromUnstructured(patchedObjMap, objToUpdate); err != nil {
		return err
	}
	// Decoding from JSON to a versioned object would apply defaults, so we do the same here
	defaulter.Default(objToUpdate)

	return nil
}

// interpretPatchError interprets the error type and returns an error with appropriate HTTP code.
func interpretPatchError(err error) error {
	switch err {
	case mergepatch.ErrBadJSONDoc, mergepatch.ErrBadPatchFormatForPrimitiveList, mergepatch.ErrBadPatchFormatForRetainKeys, mergepatch.ErrBadPatchFormatForSetElementOrderList:
		return errors.NewBadRequest(err.Error())
	case mergepatch.ErrNoListOfLists, mergepatch.ErrPatchContentNotMatchRetainKeys:
		return errors.NewGenericServerResponse(http.StatusUnprocessableEntity, "", schema.GroupResource{}, "", err.Error(), 0, false)
	default:
		return err
	}
}
