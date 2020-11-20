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

	jsonpatch "github.com/evanphx/json-patch"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversionscheme "k8s.io/apimachinery/pkg/apis/meta/internalversion/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/util/dryrun"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utiltrace "k8s.io/utils/trace"
)

const (
	// maximum number of operations a single json patch may contain.
	maxJSONPatchOperations = 10000
)

// PatchResource returns a function that will handle a resource patch.
func PatchResource(r rest.Patcher, scope *RequestScope, admit admission.Interface, patchTypes []string) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("Patch", utiltrace.Field{Key: "url", Value: req.URL.Path}, utiltrace.Field{Key: "user-agent", Value: &lazyTruncatedUserAgent{req}}, utiltrace.Field{Key: "client", Value: &lazyClientIP{req}})
		defer trace.LogIfLong(500 * time.Millisecond)

		if isDryRun(req.URL) && !utilfeature.DefaultFeatureGate.Enabled(features.DryRun) {
			scope.err(errors.NewBadRequest("the dryRun feature is disabled"), w, req)
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

		ctx, cancel := context.WithTimeout(req.Context(), timeout)
		defer cancel()
		ctx = request.WithNamespace(ctx, namespace)

		outputMediaType, _, err := negotiation.NegotiateOutputMediaType(req, scope.Serializer, scope)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		patchBytes, err := limitedReadBody(req, scope.MaxRequestBodyBytes)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		options := &metav1.PatchOptions{}
		if err := metainternalversionscheme.ParameterCodec.DecodeParameters(req.URL.Query(), scope.MetaGroupVersion, options); err != nil {
			err = errors.NewBadRequest(err.Error())
			scope.err(err, w, req)
			return
		}
		if errs := validation.ValidatePatchOptions(options, patchType); len(errs) > 0 {
			err := errors.NewInvalid(schema.GroupKind{Group: metav1.GroupName, Kind: "PatchOptions"}, "", errs)
			scope.err(err, w, req)
			return
		}
		options.TypeMeta.SetGroupVersionKind(metav1.SchemeGroupVersion.WithKind("PatchOptions"))

		ae := request.AuditEventFrom(ctx)
		admit = admission.WithAudit(admit, ae)

		audit.LogRequestPatch(ae, patchBytes)
		trace.Step("Recorded the audit event")

		baseContentType := runtime.ContentTypeJSON
		if patchType == types.ApplyPatchType {
			baseContentType = runtime.ContentTypeYAML
		}
		s, ok := runtime.SerializerInfoForMediaType(scope.Serializer.SupportedMediaTypes(), baseContentType)
		if !ok {
			scope.err(fmt.Errorf("no serializer defined for %v", baseContentType), w, req)
			return
		}
		gv := scope.Kind.GroupVersion()

		codec := runtime.NewCodec(
			scope.Serializer.EncoderForVersion(s.Serializer, gv),
			scope.Serializer.DecoderToVersion(s.Serializer, scope.HubGroupVersion),
		)

		userInfo, _ := request.UserFrom(ctx)
		staticCreateAttributes := admission.NewAttributesRecord(
			nil,
			nil,
			scope.Kind,
			namespace,
			name,
			scope.Resource,
			scope.Subresource,
			admission.Create,
			patchToCreateOptions(options),
			dryrun.IsDryRun(options.DryRun),
			userInfo)
		staticUpdateAttributes := admission.NewAttributesRecord(
			nil,
			nil,
			scope.Kind,
			namespace,
			name,
			scope.Resource,
			scope.Subresource,
			admission.Update,
			patchToUpdateOptions(options),
			dryrun.IsDryRun(options.DryRun),
			userInfo,
		)

		mutatingAdmission, _ := admit.(admission.MutationInterface)
		createAuthorizerAttributes := authorizer.AttributesRecord{
			User:            userInfo,
			ResourceRequest: true,
			Path:            req.URL.Path,
			Verb:            "create",
			APIGroup:        scope.Resource.Group,
			APIVersion:      scope.Resource.Version,
			Resource:        scope.Resource.Resource,
			Subresource:     scope.Subresource,
			Namespace:       namespace,
			Name:            name,
		}

		p := patcher{
			namer:           scope.Namer,
			creater:         scope.Creater,
			defaulter:       scope.Defaulter,
			typer:           scope.Typer,
			unsafeConvertor: scope.UnsafeConvertor,
			kind:            scope.Kind,
			resource:        scope.Resource,
			subresource:     scope.Subresource,
			dryRun:          dryrun.IsDryRun(options.DryRun),

			objectInterfaces: scope,

			hubGroupVersion: scope.HubGroupVersion,

			createValidation: withAuthorization(rest.AdmissionToValidateObjectFunc(admit, staticCreateAttributes, scope), scope.Authorizer, createAuthorizerAttributes),
			updateValidation: rest.AdmissionToValidateObjectUpdateFunc(admit, staticUpdateAttributes, scope),
			admissionCheck:   mutatingAdmission,

			codec: codec,

			timeout: timeout,
			options: options,

			restPatcher: r,
			name:        name,
			patchType:   patchType,
			patchBytes:  patchBytes,
			userAgent:   req.UserAgent(),

			trace: trace,
		}

		result, wasCreated, err := p.patchResource(ctx, scope)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Object stored in database")

		if err := setObjectSelfLink(ctx, result, req, scope.Namer); err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Self-link added")

		status := http.StatusOK
		if wasCreated {
			status = http.StatusCreated
		}
		transformResponseObject(ctx, scope, trace, req, w, status, outputMediaType, result)
	}
}

type mutateObjectUpdateFunc func(ctx context.Context, obj, old runtime.Object) error

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
	typer           runtime.ObjectTyper
	unsafeConvertor runtime.ObjectConvertor
	resource        schema.GroupVersionResource
	kind            schema.GroupVersionKind
	subresource     string
	dryRun          bool

	objectInterfaces admission.ObjectInterfaces

	hubGroupVersion schema.GroupVersion

	// Validation functions
	createValidation rest.ValidateObjectFunc
	updateValidation rest.ValidateObjectUpdateFunc
	admissionCheck   admission.MutationInterface

	codec runtime.Codec

	timeout time.Duration
	options *metav1.PatchOptions

	// Operation information
	restPatcher rest.Patcher
	name        string
	patchType   types.PatchType
	patchBytes  []byte
	userAgent   string

	trace *utiltrace.Trace

	// Set at invocation-time (by applyPatch) and immutable thereafter
	namespace         string
	updatedObjectInfo rest.UpdatedObjectInfo
	mechanism         patchMechanism
	forceAllowCreate  bool
}

type patchMechanism interface {
	applyPatchToCurrentObject(currentObject runtime.Object) (runtime.Object, error)
	createNewObject() (runtime.Object, error)
}

type jsonPatcher struct {
	*patcher

	fieldManager *fieldmanager.FieldManager
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
		return nil, err
	}

	// Construct the resulting typed, unversioned object.
	objToUpdate := p.restPatcher.New()
	if err := runtime.DecodeInto(p.codec, patchedObjJS, objToUpdate); err != nil {
		return nil, errors.NewInvalid(schema.GroupKind{}, "", field.ErrorList{
			field.Invalid(field.NewPath("patch"), string(patchedObjJS), err.Error()),
		})
	}

	if p.fieldManager != nil {
		objToUpdate = p.fieldManager.UpdateNoErrors(currentObject, objToUpdate, managerOrUserAgent(p.options.FieldManager, p.userAgent))
	}
	return objToUpdate, nil
}

func (p *jsonPatcher) createNewObject() (runtime.Object, error) {
	return nil, errors.NewNotFound(p.resource.GroupResource(), p.name)
}

// applyJSPatch applies the patch. Input and output objects must both have
// the external version, since that is what the patch must have been constructed against.
func (p *jsonPatcher) applyJSPatch(versionedJS []byte) (patchedJS []byte, retErr error) {
	switch p.patchType {
	case types.JSONPatchType:
		// sanity check potentially abusive patches
		// TODO(liggitt): drop this once golang json parser limits stack depth (https://github.com/golang/go/issues/31789)
		if len(p.patchBytes) > 1024*1024 {
			v := []interface{}{}
			if err := json.Unmarshal(p.patchBytes, &v); err != nil {
				return nil, errors.NewBadRequest(fmt.Sprintf("error decoding patch: %v", err))
			}
		}

		patchObj, err := jsonpatch.DecodePatch(p.patchBytes)
		if err != nil {
			return nil, errors.NewBadRequest(err.Error())
		}
		if len(patchObj) > maxJSONPatchOperations {
			return nil, errors.NewRequestEntityTooLargeError(
				fmt.Sprintf("The allowed maximum operations in a JSON patch is %d, got %d",
					maxJSONPatchOperations, len(patchObj)))
		}
		patchedJS, err := patchObj.Apply(versionedJS)
		if err != nil {
			return nil, errors.NewGenericServerResponse(http.StatusUnprocessableEntity, "", schema.GroupResource{}, "", err.Error(), 0, false)
		}
		return patchedJS, nil
	case types.MergePatchType:
		// sanity check potentially abusive patches
		// TODO(liggitt): drop this once golang json parser limits stack depth (https://github.com/golang/go/issues/31789)
		if len(p.patchBytes) > 1024*1024 {
			v := map[string]interface{}{}
			if err := json.Unmarshal(p.patchBytes, &v); err != nil {
				return nil, errors.NewBadRequest(fmt.Sprintf("error decoding patch: %v", err))
			}
		}

		return jsonpatch.MergePatch(versionedJS, p.patchBytes)
	default:
		// only here as a safety net - go-restful filters content-type
		return nil, fmt.Errorf("unknown Content-Type header for patch: %v", p.patchType)
	}
}

type smpPatcher struct {
	*patcher

	// Schema
	schemaReferenceObj runtime.Object
	fieldManager       *fieldmanager.FieldManager
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
	if err := strategicPatchObject(p.defaulter, currentVersionedObject, p.patchBytes, versionedObjToUpdate, p.schemaReferenceObj); err != nil {
		return nil, err
	}
	// Convert the object back to the hub version
	newObj, err := p.unsafeConvertor.ConvertToVersion(versionedObjToUpdate, p.hubGroupVersion)
	if err != nil {
		return nil, err
	}

	if p.fieldManager != nil {
		newObj = p.fieldManager.UpdateNoErrors(currentObject, newObj, managerOrUserAgent(p.options.FieldManager, p.userAgent))
	}
	return newObj, nil
}

func (p *smpPatcher) createNewObject() (runtime.Object, error) {
	return nil, errors.NewNotFound(p.resource.GroupResource(), p.name)
}

type applyPatcher struct {
	patch        []byte
	options      *metav1.PatchOptions
	creater      runtime.ObjectCreater
	kind         schema.GroupVersionKind
	fieldManager *fieldmanager.FieldManager
	userAgent    string
}

func (p *applyPatcher) applyPatchToCurrentObject(obj runtime.Object) (runtime.Object, error) {
	force := false
	if p.options.Force != nil {
		force = *p.options.Force
	}
	if p.fieldManager == nil {
		panic("FieldManager must be installed to run apply")
	}

	patchObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal(p.patch, &patchObj.Object); err != nil {
		return nil, errors.NewBadRequest(fmt.Sprintf("error decoding YAML: %v", err))
	}

	return p.fieldManager.Apply(obj, patchObj, p.options.FieldManager, force)
}

func (p *applyPatcher) createNewObject() (runtime.Object, error) {
	obj, err := p.creater.New(p.kind)
	if err != nil {
		return nil, fmt.Errorf("failed to create new object: %v", err)
	}
	return p.applyPatchToCurrentObject(obj)
}

// strategicPatchObject applies a strategic merge patch of <patchBytes> to
// <originalObject> and stores the result in <objToUpdate>.
// It additionally returns the map[string]interface{} representation of the
// <originalObject> and <patchBytes>.
// NOTE: Both <originalObject> and <objToUpdate> are supposed to be versioned.
func strategicPatchObject(
	defaulter runtime.ObjectDefaulter,
	originalObject runtime.Object,
	patchBytes []byte,
	objToUpdate runtime.Object,
	schemaReferenceObj runtime.Object,
) error {
	originalObjMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(originalObject)
	if err != nil {
		return err
	}

	patchMap := make(map[string]interface{})
	if err := json.Unmarshal(patchBytes, &patchMap); err != nil {
		return errors.NewBadRequest(err.Error())
	}

	if err := applyPatchToObject(defaulter, originalObjMap, patchMap, objToUpdate, schemaReferenceObj); err != nil {
		return err
	}
	return nil
}

// applyPatch is called every time GuaranteedUpdate asks for the updated object,
// and is given the currently persisted object as input.
// TODO: rename this function because the name implies it is related to applyPatcher
func (p *patcher) applyPatch(_ context.Context, _, currentObject runtime.Object) (objToUpdate runtime.Object, patchErr error) {
	// Make sure we actually have a persisted currentObject
	p.trace.Step("About to apply patch")
	currentObjectHasUID, err := hasUID(currentObject)
	if err != nil {
		return nil, err
	} else if !currentObjectHasUID {
		objToUpdate, patchErr = p.mechanism.createNewObject()
	} else {
		objToUpdate, patchErr = p.mechanism.applyPatchToCurrentObject(currentObject)
	}

	if patchErr != nil {
		return nil, patchErr
	}

	objToUpdateHasUID, err := hasUID(objToUpdate)
	if err != nil {
		return nil, err
	}
	if objToUpdateHasUID && !currentObjectHasUID {
		accessor, err := meta.Accessor(objToUpdate)
		if err != nil {
			return nil, err
		}
		return nil, errors.NewConflict(p.resource.GroupResource(), p.name, fmt.Errorf("uid mismatch: the provided object specified uid %s, and no existing object was found", accessor.GetUID()))
	}

	if err := checkName(objToUpdate, p.name, p.namespace, p.namer); err != nil {
		return nil, err
	}
	return objToUpdate, nil
}

func (p *patcher) admissionAttributes(ctx context.Context, updatedObject runtime.Object, currentObject runtime.Object, operation admission.Operation, operationOptions runtime.Object) admission.Attributes {
	userInfo, _ := request.UserFrom(ctx)
	return admission.NewAttributesRecord(updatedObject, currentObject, p.kind, p.namespace, p.name, p.resource, p.subresource, operation, operationOptions, p.dryRun, userInfo)
}

// applyAdmission is called every time GuaranteedUpdate asks for the updated object,
// and is given the currently persisted object and the patched object as input.
// TODO: rename this function because the name implies it is related to applyPatcher
func (p *patcher) applyAdmission(ctx context.Context, patchedObject runtime.Object, currentObject runtime.Object) (runtime.Object, error) {
	p.trace.Step("About to check admission control")
	var operation admission.Operation
	var options runtime.Object
	if hasUID, err := hasUID(currentObject); err != nil {
		return nil, err
	} else if !hasUID {
		operation = admission.Create
		currentObject = nil
		options = patchToCreateOptions(p.options)
	} else {
		operation = admission.Update
		options = patchToUpdateOptions(p.options)
	}
	if p.admissionCheck != nil && p.admissionCheck.Handles(operation) {
		attributes := p.admissionAttributes(ctx, patchedObject, currentObject, operation, options)
		return patchedObject, p.admissionCheck.Admit(ctx, attributes, p.objectInterfaces)
	}
	return patchedObject, nil
}

// patchResource divides PatchResource for easier unit testing
func (p *patcher) patchResource(ctx context.Context, scope *RequestScope) (runtime.Object, bool, error) {
	p.namespace = request.NamespaceValue(ctx)
	switch p.patchType {
	case types.JSONPatchType, types.MergePatchType:
		p.mechanism = &jsonPatcher{
			patcher:      p,
			fieldManager: scope.FieldManager,
		}
	case types.StrategicMergePatchType:
		schemaReferenceObj, err := p.unsafeConvertor.ConvertToVersion(p.restPatcher.New(), p.kind.GroupVersion())
		if err != nil {
			return nil, false, err
		}
		p.mechanism = &smpPatcher{
			patcher:            p,
			schemaReferenceObj: schemaReferenceObj,
			fieldManager:       scope.FieldManager,
		}
	// this case is unreachable if ServerSideApply is not enabled because we will have already rejected the content type
	case types.ApplyPatchType:
		p.mechanism = &applyPatcher{
			fieldManager: scope.FieldManager,
			patch:        p.patchBytes,
			options:      p.options,
			creater:      p.creater,
			kind:         p.kind,
			userAgent:    p.userAgent,
		}
		p.forceAllowCreate = true
	default:
		return nil, false, fmt.Errorf("%v: unimplemented patch type", p.patchType)
	}
	dedupOwnerReferencesTransformer := func(_ context.Context, obj, _ runtime.Object) (runtime.Object, error) {
		// Dedup owner references after mutating admission happens
		dedupOwnerReferencesAndAddWarning(obj, ctx, true)
		return obj, nil
	}

	wasCreated := false
	p.updatedObjectInfo = rest.DefaultUpdatedObjectInfo(nil, p.applyPatch, p.applyAdmission, dedupOwnerReferencesTransformer)
	requestFunc := func() (runtime.Object, error) {
		// Pass in UpdateOptions to override UpdateStrategy.AllowUpdateOnCreate
		options := patchToUpdateOptions(p.options)
		updateObject, created, updateErr := p.restPatcher.Update(ctx, p.name, p.updatedObjectInfo, p.createValidation, p.updateValidation, p.forceAllowCreate, options)
		wasCreated = created
		return updateObject, updateErr
	}
	result, err := finishRequest(p.timeout, func() (runtime.Object, error) {
		result, err := requestFunc()
		// If the object wasn't committed to storage because it's serialized size was too large,
		// it is safe to remove managedFields (which can be large) and try again.
		if isTooLargeError(err) && p.patchType != types.ApplyPatchType {
			if _, accessorErr := meta.Accessor(p.restPatcher.New()); accessorErr == nil {
				p.updatedObjectInfo = rest.DefaultUpdatedObjectInfo(nil,
					p.applyPatch,
					p.applyAdmission,
					dedupOwnerReferencesTransformer,
					func(_ context.Context, obj, _ runtime.Object) (runtime.Object, error) {
						accessor, _ := meta.Accessor(obj)
						accessor.SetManagedFields(nil)
						return obj, nil
					})
				result, err = requestFunc()
			}
		}
		return result, err
	})
	return result, wasCreated, err
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
		return interpretStrategicMergePatchError(err)
	}

	// Rather than serialize the patched map to JSON, then decode it to an object, we go directly from a map to an object
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(patchedObjMap, objToUpdate); err != nil {
		return errors.NewInvalid(schema.GroupKind{}, "", field.ErrorList{
			field.Invalid(field.NewPath("patch"), fmt.Sprintf("%+v", patchMap), err.Error()),
		})
	}
	// Decoding from JSON to a versioned object would apply defaults, so we do the same here
	defaulter.Default(objToUpdate)

	return nil
}

// interpretStrategicMergePatchError interprets the error type and returns an error with appropriate HTTP code.
func interpretStrategicMergePatchError(err error) error {
	switch err {
	case mergepatch.ErrBadJSONDoc, mergepatch.ErrBadPatchFormatForPrimitiveList, mergepatch.ErrBadPatchFormatForRetainKeys, mergepatch.ErrBadPatchFormatForSetElementOrderList, mergepatch.ErrUnsupportedStrategicMergePatchFormat:
		return errors.NewBadRequest(err.Error())
	case mergepatch.ErrNoListOfLists, mergepatch.ErrPatchContentNotMatchRetainKeys:
		return errors.NewGenericServerResponse(http.StatusUnprocessableEntity, "", schema.GroupResource{}, "", err.Error(), 0, false)
	default:
		return err
	}
}

// patchToUpdateOptions creates an UpdateOptions with the same field values as the provided PatchOptions.
func patchToUpdateOptions(po *metav1.PatchOptions) *metav1.UpdateOptions {
	if po == nil {
		return nil
	}
	uo := &metav1.UpdateOptions{
		DryRun:       po.DryRun,
		FieldManager: po.FieldManager,
	}
	uo.TypeMeta.SetGroupVersionKind(metav1.SchemeGroupVersion.WithKind("UpdateOptions"))
	return uo
}

// patchToCreateOptions creates an CreateOptions with the same field values as the provided PatchOptions.
func patchToCreateOptions(po *metav1.PatchOptions) *metav1.CreateOptions {
	if po == nil {
		return nil
	}
	co := &metav1.CreateOptions{
		DryRun:       po.DryRun,
		FieldManager: po.FieldManager,
	}
	co.TypeMeta.SetGroupVersionKind(metav1.SchemeGroupVersion.WithKind("CreateOptions"))
	return co
}
