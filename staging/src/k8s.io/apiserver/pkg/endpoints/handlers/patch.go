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

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/validation"
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
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/util/dryrun"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
)

// PatchResource returns a function that will handle a resource patch.
func PatchResource(r rest.Patcher, scope RequestScope, admit admission.Interface, patchTypes []string) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("Patch " + req.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		if isDryRun(req.URL) && !utilfeature.DefaultFeatureGate.Enabled(features.DryRun) {
			scope.err(errors.NewBadRequest("the dryRun alpha feature is disabled"), w, req)
			return
		}

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

		patchJS, err := readBody(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		options := &metav1.UpdateOptions{}
		if err := metainternalversion.ParameterCodec.DecodeParameters(req.URL.Query(), scope.MetaGroupVersion, options); err != nil {
			err = errors.NewBadRequest(err.Error())
			scope.err(err, w, req)
			return
		}
		if errs := validation.ValidateUpdateOptions(options); len(errs) > 0 {
			err := errors.NewInvalid(schema.GroupKind{Group: metav1.GroupName, Kind: "UpdateOptions"}, "", errs)
			scope.err(err, w, req)
			return
		}

		ae := request.AuditEventFrom(ctx)
		admit = admission.WithAudit(admit, ae)

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
		staticAdmissionAttributes := admission.NewAttributesRecord(
			nil,
			nil,
			scope.Kind,
			namespace,
			name,
			scope.Resource,
			scope.Subresource,
			admission.Update,
			dryrun.IsDryRun(options.DryRun),
			userInfo,
		)
		admissionCheck := func(updatedObject runtime.Object, currentObject runtime.Object) error {
			// if we allow create-on-patch, we have this TODO: call the mutating admission chain with the CREATE verb instead of UPDATE
			if mutatingAdmission, ok := admit.(admission.MutationInterface); ok && admit.Handles(admission.Update) {
				return mutatingAdmission.Admit(admission.NewAttributesRecord(
					updatedObject,
					currentObject,
					scope.Kind,
					namespace,
					name,
					scope.Resource,
					scope.Subresource,
					admission.Update,
					dryrun.IsDryRun(options.DryRun),
					userInfo,
				))
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
			options: options,

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
	options *metav1.UpdateOptions

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
}

func (p *patcher) toUnversioned(versionedObj runtime.Object) (runtime.Object, error) {
	gvk := p.kind.GroupKind().WithVersion(runtime.APIVersionInternal)
	return p.unsafeConvertor.ConvertToVersion(versionedObj, gvk.GroupVersion())
}

type patchMechanism interface {
	applyPatchToCurrentObject(currentObject runtime.Object) (runtime.Object, error)
}

type jsonPatcher struct {
	*patcher
}

func (p *jsonPatcher) applyPatchToCurrentObject(currentObject runtime.Object) (runtime.Object, error) {
	// Encode will convert & return a versioned object in JSON.
	currentObjJS, err := runtime.Encode(p.codec, currentObject)
	if err != nil {
		return nil, err
	}

	// Apply the patch.
	patchedObjJS, err := p.applyJSPatch(currentObjJS)
	if err != nil {
		return nil, interpretPatchError(err)
	}

	// Construct the resulting typed, unversioned object.
	objToUpdate := p.restPatcher.New()
	if err := runtime.DecodeInto(p.codec, patchedObjJS, objToUpdate); err != nil {
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

type smpPatcher struct {
	*patcher

	// Schema
	schemaReferenceObj runtime.Object
}

func (p *smpPatcher) applyPatchToCurrentObject(currentObject runtime.Object) (runtime.Object, error) {
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
	if err := strategicPatchObject(p.defaulter, currentVersionedObject, p.patchJS, versionedObjToUpdate, p.schemaReferenceObj); err != nil {
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

	if err := applyPatchToObject(defaulter, originalObjMap, patchMap, objToUpdate, schemaReferenceObj); err != nil {
		return err
	}
	return nil
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

	objToUpdate, err := p.mechanism.applyPatchToCurrentObject(currentObject)
	if err != nil {
		return nil, err
	}
	if err := checkName(objToUpdate, p.name, p.namespace, p.namer); err != nil {
		return nil, err
	}
	return objToUpdate, nil
}

// applyAdmission is called every time GuaranteedUpdate asks for the updated object,
// and is given the currently persisted object and the patched object as input.
func (p *patcher) applyAdmission(ctx context.Context, patchedObject runtime.Object, currentObject runtime.Object) (runtime.Object, error) {
	p.trace.Step("About to check admission control")
	return patchedObject, p.admissionCheck(patchedObject, currentObject)
}

// patchResource divides PatchResource for easier unit testing
func (p *patcher) patchResource(ctx context.Context) (runtime.Object, error) {
	p.namespace = request.NamespaceValue(ctx)
	switch p.patchType {
	case types.JSONPatchType, types.MergePatchType:
		p.mechanism = &jsonPatcher{patcher: p}
	case types.StrategicMergePatchType:
		schemaReferenceObj, err := p.unsafeConvertor.ConvertToVersion(p.restPatcher.New(), p.kind.GroupVersion())
		if err != nil {
			return nil, err
		}
		p.mechanism = &smpPatcher{patcher: p, schemaReferenceObj: schemaReferenceObj}
	default:
		return nil, fmt.Errorf("%v: unimplemented patch type", p.patchType)
	}
	p.updatedObjectInfo = rest.DefaultUpdatedObjectInfo(nil, p.applyPatch, p.applyAdmission)
	return finishRequest(p.timeout, func() (runtime.Object, error) {
		updateObject, _, updateErr := p.restPatcher.Update(ctx, p.name, p.updatedObjectInfo, p.createValidation, p.updateValidation, false, p.options)
		return updateObject, updateErr
	})
}

// applyPatchToObject applies a strategic merge patch of <patchMap> to
// <originalMap> and stores the result in <objToUpdate>.
// NOTE: <objToUpdate> must be a versioned object.
func applyPatchToObject(
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
