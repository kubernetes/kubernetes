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
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/evanphx/json-patch"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
)

// PatchResource returns a function that will handle a resource patch.
func PatchResource(r rest.Patcher, scope RequestScope, admit admission.Interface, converter runtime.ObjectConvertor, patchTypes []string) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("Patch " + req.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		// Do this first, otherwise name extraction can fail for unrecognized content types
		// TODO: handle this in negotiation
		contentType := req.Header.Get("Content-Type")
		// Remove "; charset=" if included in header.
		if idx := strings.Index(contentType, ";"); idx > 0 {
			contentType = contentType[:idx]
		}
		patchType := types.PatchType(contentType)

		// Ensure the patchType is one we support
		if !sets.NewString(patchTypes...).Has(contentType) {
			scope.err(negotiation.NewUnsupportedMediaTypeError(patchTypes), w, req)
			return
		}

		// TODO: we either want to remove timeout or document it (if we
		// document, move timeout out of this function and declare it in
		// api_installer)
		timeout := parseTimeout(req.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		ctx := req.Context()
		ctx = request.WithNamespace(ctx, namespace)

		// TODO: this is NOT using the scope's convertor [sic]. Figure
		// out if this is intentional or not. Perhaps it matters on
		// subresources? Rename this parameter if this is purposful and
		// delete it (using scope.Convertor instead) otherwise.
		//
		// Already some tests set this converter but apparently not the
		// scope's unsafeConvertor.
		schemaReferenceObj, err := converter.ConvertToVersion(r.New(), scope.Kind.GroupVersion())
		if err != nil {
			scope.err(err, w, req)
			return
		}

		patchJS, err := readBody(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		ae := request.AuditEventFrom(ctx)
		audit.LogRequestPatch(ae, patchJS)
		trace.Step("Recorded the audit event")

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

		userInfo, _ := request.UserFrom(ctx)
		staticAdmissionAttributes := admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Update, userInfo)
		admissionCheck := func(updatedObject runtime.Object, currentObject runtime.Object) error {
			if mutatingAdmission, ok := admit.(admission.MutationInterface); ok && admit.Handles(admission.Update) {
				return mutatingAdmission.Admit(admission.NewAttributesRecord(updatedObject, currentObject, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Update, userInfo))
			}
			return nil
		}

		p := patcher{
			namer:           scope.Namer,
			creater:         scope.Creater,
			defaulter:       scope.Defaulter,
			unsafeConvertor: scope.UnsafeConvertor,
			kind:            scope.Kind,
			resource:        scope.Resource,

			createValidation: rest.AdmissionToValidateObjectFunc(admit, staticAdmissionAttributes),
			updateValidation: rest.AdmissionToValidateObjectUpdateFunc(admit, staticAdmissionAttributes),
			admissionCheck:   admissionCheck,

			codec: codec,

			timeout: timeout,

			schemaReferenceObj: schemaReferenceObj,

			restPatcher: r,
			name:        name,
			patchType:   patchType,
			patchJS:     patchJS,

			trace: trace,
		}

		result, err := p.patchResource(ctx)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Object stored in database")

		requestInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			scope.err(fmt.Errorf("missing requestInfo"), w, req)
			return
		}
		if err := setSelfLink(result, requestInfo, scope.Namer); err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Self-link added")

		transformResponseObject(ctx, scope, req, w, http.StatusOK, result)
	}
}

type mutateObjectUpdateFunc func(obj, old runtime.Object) error

// patcher breaks the process of patch application and retries into smaller
// pieces of functionality.
// TODO: Use builder pattern to construct this object?
// TODO: As part of that effort, some aspects of PatchResource above could be
// moved into this type.
type patcher struct {
	// Pieces of RequestScope
	namer           ScopeNamer
	creater         runtime.ObjectCreater
	defaulter       runtime.ObjectDefaulter
	unsafeConvertor runtime.ObjectConvertor
	resource        schema.GroupVersionResource
	kind            schema.GroupVersionKind

	// Validation functions
	createValidation rest.ValidateObjectFunc
	updateValidation rest.ValidateObjectUpdateFunc
	admissionCheck   mutateObjectUpdateFunc

	codec runtime.Codec

	timeout time.Duration

	// Schema
	schemaReferenceObj runtime.Object

	// Operation information
	restPatcher rest.Patcher
	name        string
	patchType   types.PatchType
	patchJS     []byte

	trace *utiltrace.Trace

	// Set at invocation-time (by applyPatch) and immutable thereafter
	namespace         string
	updatedObjectInfo rest.UpdatedObjectInfo
	mechanism         patchMechanism

	// Set on first iteration, currently only used to construct error messages
	originalResourceVersion string

	// Modified each iteration
	iterationCount  int
	lastConflictErr error
}

func (p *patcher) toUnversioned(versionedObj runtime.Object) (runtime.Object, error) {
	gvk := p.kind.GroupKind().WithVersion(runtime.APIVersionInternal)
	return p.unsafeConvertor.ConvertToVersion(versionedObj, gvk.GroupVersion())
}

type patchMechanism interface {
	firstPatchAttempt(currentObject runtime.Object, currentResourceVersion string) (runtime.Object, error)
	subsequentPatchAttempt(currentObject runtime.Object, currentResourceVersion string) (runtime.Object, error)
}

type jsonPatcher struct {
	*patcher

	// set by firstPatchAttempt
	originalObjJS        []byte
	originalPatchedObjJS []byte

	// State for originalStrategicMergePatch
	originalPatchBytes []byte
}

func (p *jsonPatcher) firstPatchAttempt(currentObject runtime.Object, currentResourceVersion string) (runtime.Object, error) {
	// Encode will convert & return a versioned object in JSON.
	// Store this JS for future use.
	originalObjJS, err := runtime.Encode(p.codec, currentObject)
	if err != nil {
		return nil, err
	}

	// Apply the patch. Store patched result for future use.
	originalPatchedObjJS, err := p.applyJSPatch(originalObjJS)
	if err != nil {
		return nil, interpretPatchError(err)
	}

	// Since both succeeded, store the results. (This shouldn't be
	// necessary since neither of the above items can return conflict
	// errors, but it also doesn't hurt.)
	p.originalObjJS = originalObjJS
	p.originalPatchedObjJS = originalPatchedObjJS

	// Construct the resulting typed, unversioned object.
	objToUpdate := p.restPatcher.New()
	if err := runtime.DecodeInto(p.codec, p.originalPatchedObjJS, objToUpdate); err != nil {
		return nil, err
	}

	return objToUpdate, nil
}

// patchJS applies the patch. Input and output objects must both have
// the external version, since that is what the patch must have been constructed against.
func (p *jsonPatcher) applyJSPatch(versionedJS []byte) (patchedJS []byte, retErr error) {
	switch p.patchType {
	case types.JSONPatchType:
		patchObj, err := jsonpatch.DecodePatch(p.patchJS)
		if err != nil {
			return nil, err
		}
		return patchObj.Apply(versionedJS)
	case types.MergePatchType:
		return jsonpatch.MergePatch(versionedJS, p.patchJS)
	default:
		// only here as a safety net - go-restful filters content-type
		return nil, fmt.Errorf("unknown Content-Type header for patch: %v", p.patchType)
	}
}

func (p *jsonPatcher) subsequentPatchAttempt(currentObject runtime.Object, currentResourceVersion string) (runtime.Object, error) {
	if len(p.originalObjJS) == 0 || len(p.originalPatchedObjJS) == 0 {
		return nil, errors.NewInternalError(fmt.Errorf("unexpected error on patch retry: this indicates a bug in the server; remaking the patch against the most recent version of the object might succeed"))
	}
	return subsequentPatchLogic(p.patcher, p, currentObject, currentResourceVersion)
}

// Return a fresh strategic patch map if needed for conflict retries. We have
// to rebuild it each time we need it, because the map gets mutated when being
// applied.
func (p *jsonPatcher) originalStrategicMergePatch() (map[string]interface{}, error) {
	if p.originalPatchBytes == nil {
		// Compute once. Compute here instead of in the first patch
		// attempt because this isn't needed unless there's actually a
		// conflict.
		var err error
		p.originalPatchBytes, err = strategicpatch.CreateTwoWayMergePatch(p.originalObjJS, p.originalPatchedObjJS, p.schemaReferenceObj)
		if err != nil {
			return nil, interpretPatchError(err)
		}
	}

	// Return a fresh map every time
	originalPatchMap := make(map[string]interface{})
	if err := json.Unmarshal(p.originalPatchBytes, &originalPatchMap); err != nil {
		return nil, errors.NewBadRequest(err.Error())
	}
	return originalPatchMap, nil
}

// TODO: this will totally fail for CR types today, as no schema is available.
// The interface should be changed.
func (p *jsonPatcher) computeStrategicMergePatch(currentObject runtime.Object, _ map[string]interface{}) (map[string]interface{}, error) {
	// Compute current patch.
	currentObjJS, err := runtime.Encode(p.codec, currentObject)
	if err != nil {
		return nil, err
	}
	currentPatch, err := strategicpatch.CreateTwoWayMergePatch(p.originalObjJS, currentObjJS, p.schemaReferenceObj)
	if err != nil {
		return nil, interpretPatchError(err)
	}
	currentPatchMap := make(map[string]interface{})
	if err := json.Unmarshal(currentPatch, &currentPatchMap); err != nil {
		return nil, errors.NewBadRequest(err.Error())
	}

	return currentPatchMap, nil
}

type smpPatcher struct {
	*patcher
	originalObjMap map[string]interface{}
}

func (p *smpPatcher) firstPatchAttempt(currentObject runtime.Object, currentResourceVersion string) (runtime.Object, error) {
	// first time through,
	// 1. apply the patch
	// 2. save the original and patched to detect whether there were conflicting changes on retries

	// For performance reasons, in case of strategicpatch, we avoid json
	// marshaling and unmarshaling and operate just on map[string]interface{}.
	// In case of other patch types, we still have to operate on JSON
	// representations.

	// Since the patch is applied on versioned objects, we need to convert the
	// current object to versioned representation first.
	currentVersionedObject, err := p.unsafeConvertor.ConvertToVersion(currentObject, p.kind.GroupVersion())
	if err != nil {
		return nil, err
	}
	versionedObjToUpdate, err := p.creater.New(p.kind)
	if err != nil {
		return nil, err
	}
	// Capture the original object map and patch for possible retries.
	originalObjMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(currentVersionedObject)
	if err != nil {
		return nil, err
	}
	// Store only after success. (This shouldn't be necessary since neither
	// of the above items can return conflict errors, but it also doesn't
	// hurt.)
	p.originalObjMap = originalObjMap
	if err := strategicPatchObject(p.codec, p.defaulter, currentVersionedObject, p.patchJS, versionedObjToUpdate, p.schemaReferenceObj); err != nil {
		return nil, err
	}
	// Convert the object back to unversioned (aka internal version).
	unversionedObjToUpdate, err := p.toUnversioned(versionedObjToUpdate)
	if err != nil {
		return nil, err
	}

	return unversionedObjToUpdate, nil
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
	schemaReferenceObj runtime.Object,
) error {
	originalObjMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(originalObject)
	if err != nil {
		return err
	}

	patchMap := make(map[string]interface{})
	if err := json.Unmarshal(patchJS, &patchMap); err != nil {
		return errors.NewBadRequest(err.Error())
	}

	if err := applyPatchToObject(codec, defaulter, originalObjMap, patchMap, objToUpdate, schemaReferenceObj); err != nil {
		return err
	}
	return nil
}

func (p *smpPatcher) subsequentPatchAttempt(currentObject runtime.Object, currentResourceVersion string) (runtime.Object, error) {
	if p.originalObjMap == nil {
		return nil, errors.NewInternalError(fmt.Errorf("unexpected error on (SMP) patch retry: this indicates a bug in the server; remaking the patch against the most recent version of the object might succeed"))
	}
	return subsequentPatchLogic(p.patcher, p, currentObject, currentResourceVersion)
}

// Return a fresh strategic patch map if needed for conflict retries.  We have
// to rebuild it each time we need it, because the map gets mutated when being
// applied.
func (p *smpPatcher) originalStrategicMergePatch() (map[string]interface{}, error) {
	patchMap := make(map[string]interface{})
	if err := json.Unmarshal(p.patchJS, &patchMap); err != nil {
		return nil, errors.NewBadRequest(err.Error())
	}
	return patchMap, nil
}

func (p *smpPatcher) computeStrategicMergePatch(_ runtime.Object, currentObjMap map[string]interface{}) (map[string]interface{}, error) {
	o, err := strategicpatch.CreateTwoWayMergeMapPatch(p.originalObjMap, currentObjMap, p.schemaReferenceObj)
	if err != nil {
		return nil, interpretPatchError(err)
	}
	return o, nil
}

// patchSource lets you get two SMPs, an original and a current. These can be
// compared for conflicts.
//
// TODO: Instead of computing two 2-way merges and comparing them for
// conflicts, we could do one 3-way merge, which can detect the same
// conflicts. This would likely be more readable and more efficient,
// and should be logically exactly the same operation.
//
// TODO: Currently, the user gets this behavior whether or not they
// specified a RV. I believe we can stop doing this if the user did not
// specify an RV, and that would not be a breaking change.
//
type patchSource interface {
	// originalStrategicMergePatch must reconstruct this map each time,
	// because it is consumed when it is used.
	originalStrategicMergePatch() (map[string]interface{}, error)
	computeStrategicMergePatch(unversionedObject runtime.Object, currentVersionedObjMap map[string]interface{}) (map[string]interface{}, error)
}

func subsequentPatchLogic(p *patcher, ps patchSource, currentObject runtime.Object, currentResourceVersion string) (runtime.Object, error) {
	// on a conflict (which is the only reason to have more than one attempt),
	// 1. build a strategic merge patch from originalJS and the patchedJS.  Different patch types can
	//    be specified, but a strategic merge patch should be expressive enough handle them.  Build the
	//    patch with this type to handle those cases.
	// 2. build a strategic merge patch from originalJS and the currentJS
	// 3. ensure no conflicts between the two patches
	// 4. apply the #1 patch to the currentJS object

	// Since the patch is applied on versioned objects, we need to convert the
	// current object to versioned representation first.
	currentVersionedObject, err := p.unsafeConvertor.ConvertToVersion(currentObject, p.kind.GroupVersion())
	if err != nil {
		return nil, err
	}
	currentObjMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(currentVersionedObject)
	if err != nil {
		return nil, err
	}

	currentPatchMap, err := ps.computeStrategicMergePatch(currentObject, currentObjMap)
	if err != nil {
		return nil, err
	}

	// Get a fresh copy of the original strategic patch each time through, since applying it mutates the map
	originalPatchMap, err := ps.originalStrategicMergePatch()
	if err != nil {
		return nil, err
	}

	patchMetaFromStruct, err := strategicpatch.NewPatchMetaFromStruct(p.schemaReferenceObj)
	if err != nil {
		return nil, err
	}
	hasConflicts, err := strategicpatch.MergingMapsHaveConflicts(originalPatchMap, currentPatchMap, patchMetaFromStruct)
	if err != nil {
		return nil, err
	}

	if hasConflicts {
		diff1, _ := json.Marshal(currentPatchMap)
		diff2, _ := json.Marshal(originalPatchMap)
		patchDiffErr := fmt.Errorf("there is a meaningful conflict (firstResourceVersion: %q, currentResourceVersion: %q):\n diff1=%v\n, diff2=%v\n", p.originalResourceVersion, currentResourceVersion, string(diff1), string(diff2))
		glog.V(4).Infof("patchResource failed for resource %s, because there is a meaningful conflict(firstResourceVersion: %q, currentResourceVersion: %q):\n diff1=%v\n, diff2=%v\n", p.name, p.originalResourceVersion, currentResourceVersion, string(diff1), string(diff2))

		// Return the last conflict error we got if we have one
		if p.lastConflictErr != nil {
			return nil, p.lastConflictErr
		}
		// Otherwise manufacture one of our own
		return nil, errors.NewConflict(p.resource.GroupResource(), p.name, patchDiffErr)
	}

	versionedObjToUpdate, err := p.creater.New(p.kind)
	if err != nil {
		return nil, err
	}
	if err := applyPatchToObject(p.codec, p.defaulter, currentObjMap, originalPatchMap, versionedObjToUpdate, p.schemaReferenceObj); err != nil {
		return nil, err
	}
	// Convert the object back to unversioned.
	return p.toUnversioned(versionedObjToUpdate)
}

// applyPatch is called every time GuaranteedUpdate asks for the updated object,
// and is given the currently persisted object as input.
func (p *patcher) applyPatch(_ context.Context, _, currentObject runtime.Object) (runtime.Object, error) {
	// Make sure we actually have a persisted currentObject
	p.trace.Step("About to apply patch")
	if hasUID, err := hasUID(currentObject); err != nil {
		return nil, err
	} else if !hasUID {
		return nil, errors.NewNotFound(p.resource.GroupResource(), p.name)
	}

	currentResourceVersion := ""
	if currentMetadata, err := meta.Accessor(currentObject); err == nil {
		currentResourceVersion = currentMetadata.GetResourceVersion()
	}

	p.iterationCount++

	if p.iterationCount == 1 {
		p.originalResourceVersion = currentResourceVersion
		objToUpdate, err := p.mechanism.firstPatchAttempt(currentObject, currentResourceVersion)
		if err != nil {
			return nil, err
		}
		if err := checkName(objToUpdate, p.name, p.namespace, p.namer); err != nil {
			return nil, err
		}
		return objToUpdate, nil
	}
	return p.mechanism.subsequentPatchAttempt(currentObject, currentResourceVersion)
}

// applyAdmission is called every time GuaranteedUpdate asks for the updated object,
// and is given the currently persisted object and the patched object as input.
func (p *patcher) applyAdmission(ctx context.Context, patchedObject runtime.Object, currentObject runtime.Object) (runtime.Object, error) {
	p.trace.Step("About to check admission control")
	return patchedObject, p.admissionCheck(patchedObject, currentObject)
}

func (p *patcher) requestLoop(ctx context.Context) func() (runtime.Object, error) {
	// return a function to catch ctx in a closure, until finishRequest
	// starts handling the context.
	return func() (runtime.Object, error) {
		updateObject, _, updateErr := p.restPatcher.Update(ctx, p.name, p.updatedObjectInfo, p.createValidation, p.updateValidation)
		for i := 0; i < MaxRetryWhenPatchConflicts && (errors.IsConflict(updateErr)); i++ {
			p.lastConflictErr = updateErr
			updateObject, _, updateErr = p.restPatcher.Update(ctx, p.name, p.updatedObjectInfo, p.createValidation, p.updateValidation)
		}
		return updateObject, updateErr
	}
}

// patchResource divides PatchResource for easier unit testing
func (p *patcher) patchResource(ctx context.Context) (runtime.Object, error) {
	p.namespace = request.NamespaceValue(ctx)
	switch p.patchType {
	case types.JSONPatchType, types.MergePatchType:
		p.mechanism = &jsonPatcher{patcher: p}
	case types.StrategicMergePatchType:
		p.mechanism = &smpPatcher{patcher: p}
	default:
		return nil, fmt.Errorf("%v: unimplemented patch type", p.patchType)
	}
	p.updatedObjectInfo = rest.DefaultUpdatedObjectInfo(nil, p.applyPatch, p.applyAdmission)
	return finishRequest(p.timeout, p.requestLoop(ctx))
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
	schemaReferenceObj runtime.Object,
) error {
	patchedObjMap, err := strategicpatch.StrategicMergeMapPatch(originalMap, patchMap, schemaReferenceObj)
	if err != nil {
		return interpretPatchError(err)
	}

	// Rather than serialize the patched map to JSON, then decode it to an object, we go directly from a map to an object
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(patchedObjMap, objToUpdate); err != nil {
		return err
	}
	// Decoding from JSON to a versioned object would apply defaults, so we do the same here
	defaulter.Default(objToUpdate)

	return nil
}

// interpretPatchError interprets the error type and returns an error with appropriate HTTP code.
func interpretPatchError(err error) error {
	switch err {
	case mergepatch.ErrBadJSONDoc, mergepatch.ErrBadPatchFormatForPrimitiveList, mergepatch.ErrBadPatchFormatForRetainKeys, mergepatch.ErrBadPatchFormatForSetElementOrderList, mergepatch.ErrUnsupportedStrategicMergePatchFormat:
		return errors.NewBadRequest(err.Error())
	case mergepatch.ErrNoListOfLists, mergepatch.ErrPatchContentNotMatchRetainKeys:
		return errors.NewGenericServerResponse(http.StatusUnprocessableEntity, "", schema.GroupResource{}, "", err.Error(), 0, false)
	default:
		return err
	}
}
