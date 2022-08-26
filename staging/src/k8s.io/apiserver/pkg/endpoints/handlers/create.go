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
	"bytes"
	"context"
	"fmt"
	"net/http"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversionscheme "k8s.io/apimachinery/pkg/apis/meta/internalversion/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/apiserver/pkg/endpoints/handlers/finisher"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/util/dryrun"
	"k8s.io/klog/v2"
	utiltrace "k8s.io/utils/trace"
)

var namespaceGVR = schema.GroupVersionResource{Group: "", Version: "v1", Resource: "namespaces"}

func createHandler(r rest.NamedCreater, scope *RequestScope, admit admission.Interface, includeName bool) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("Create", traceFields(req)...)
		defer trace.LogIfLong(500 * time.Millisecond)

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			if includeName {
				// name was required, return
				scope.err(err, w, req)
				return
			}

			// otherwise attempt to look up the namespace
			namespace, err = scope.Namer.Namespace(req)
			if err != nil {
				scope.err(err, w, req)
				return
			}
		}

		// enforce a timeout of at most requestTimeoutUpperBound (34s) or less if the user-provided
		// timeout inside the parent context is lower than requestTimeoutUpperBound.
		ctx, cancel := context.WithTimeout(req.Context(), requestTimeoutUpperBound)
		defer cancel()
		outputMediaType, _, err := negotiation.NegotiateOutputMediaType(req, scope.Serializer, scope)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		gv := scope.Kind.GroupVersion()
		s, err := negotiation.NegotiateInputSerializer(req, false, scope.Serializer)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		body, err := limitedReadBody(req, scope.MaxRequestBodyBytes)
		trace.Step("limitedReadBody done", utiltrace.Field{"len", len(body)}, utiltrace.Field{"err", err})
		if err != nil {
			scope.err(err, w, req)
			return
		}

		options := &metav1.CreateOptions{}
		values := req.URL.Query()
		if err := metainternalversionscheme.ParameterCodec.DecodeParameters(values, scope.MetaGroupVersion, options); err != nil {
			err = errors.NewBadRequest(err.Error())
			scope.err(err, w, req)
			return
		}
		if errs := validation.ValidateCreateOptions(options); len(errs) > 0 {
			err := errors.NewInvalid(schema.GroupKind{Group: metav1.GroupName, Kind: "CreateOptions"}, "", errs)
			scope.err(err, w, req)
			return
		}
		options.TypeMeta.SetGroupVersionKind(metav1.SchemeGroupVersion.WithKind("CreateOptions"))

		defaultGVK := scope.Kind
		original := r.New()

		validationDirective := fieldValidation(options.FieldValidation)
		decodeSerializer := s.Serializer
		if validationDirective == metav1.FieldValidationWarn || validationDirective == metav1.FieldValidationStrict {
			decodeSerializer = s.StrictSerializer
		}

		decoder := scope.Serializer.DecoderToVersion(decodeSerializer, scope.HubGroupVersion)
		trace.Step("About to convert to expected version")
		obj, gvk, err := decoder.Decode(body, &defaultGVK, original)
		if err != nil {
			strictError, isStrictError := runtime.AsStrictDecodingError(err)
			switch {
			case isStrictError && obj != nil && validationDirective == metav1.FieldValidationWarn:
				addStrictDecodingWarnings(req.Context(), strictError.Errors())
			case isStrictError && validationDirective == metav1.FieldValidationIgnore:
				klog.Warningf("unexpected strict error when field validation is set to ignore")
				fallthrough
			default:
				err = transformDecodeError(scope.Typer, err, original, gvk, body)
				scope.err(err, w, req)
				return
			}
		}

		objGV := gvk.GroupVersion()
		if !scope.AcceptsGroupVersion(objGV) {
			err = errors.NewBadRequest(fmt.Sprintf("the API version in the data (%s) does not match the expected API version (%v)", objGV.String(), gv.String()))
			scope.err(err, w, req)
			return
		}
		trace.Step("Conversion done")

		// On create, get name from new object if unset
		if len(name) == 0 {
			_, name, _ = scope.Namer.ObjectName(obj)
		}
		if len(namespace) == 0 && scope.Resource == namespaceGVR {
			namespace = name
		}
		ctx = request.WithNamespace(ctx, namespace)

		admit = admission.WithAudit(admit)
		audit.LogRequestObject(req.Context(), obj, objGV, scope.Resource, scope.Subresource, scope.Serializer)

		userInfo, _ := request.UserFrom(ctx)

		if objectMeta, err := meta.Accessor(obj); err == nil {
			// Wipe fields which cannot take user-provided values
			rest.WipeObjectMetaSystemFields(objectMeta)

			// ensure namespace on the object is correct, or error if a conflicting namespace was set in the object
			if err := rest.EnsureObjectNamespaceMatchesRequestNamespace(rest.ExpectedNamespaceForResource(namespace, scope.Resource), objectMeta); err != nil {
				scope.err(err, w, req)
				return
			}
		}

		trace.Step("About to store object in database")
		admissionAttributes := admission.NewAttributesRecord(obj, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Create, options, dryrun.IsDryRun(options.DryRun), userInfo)
		requestFunc := func() (runtime.Object, error) {
			return r.Create(
				ctx,
				name,
				obj,
				rest.AdmissionToValidateObjectFunc(admit, admissionAttributes, scope),
				options,
			)
		}
		// Dedup owner references before updating managed fields
		dedupOwnerReferencesAndAddWarning(obj, req.Context(), false)
		result, err := finisher.FinishRequest(ctx, func() (runtime.Object, error) {
			if scope.FieldManager != nil {
				liveObj, err := scope.Creater.New(scope.Kind)
				if err != nil {
					return nil, fmt.Errorf("failed to create new object (Create for %v): %v", scope.Kind, err)
				}
				obj = scope.FieldManager.UpdateNoErrors(liveObj, obj, managerOrUserAgent(options.FieldManager, req.UserAgent()))
				admit = fieldmanager.NewManagedFieldsValidatingAdmissionController(admit)
			}
			if mutatingAdmission, ok := admit.(admission.MutationInterface); ok && mutatingAdmission.Handles(admission.Create) {
				if err := mutatingAdmission.Admit(ctx, admissionAttributes, scope); err != nil {
					return nil, err
				}
			}
			// Dedup owner references again after mutating admission happens
			dedupOwnerReferencesAndAddWarning(obj, req.Context(), true)
			result, err := requestFunc()
			// If the object wasn't committed to storage because it's serialized size was too large,
			// it is safe to remove managedFields (which can be large) and try again.
			if isTooLargeError(err) {
				if accessor, accessorErr := meta.Accessor(obj); accessorErr == nil {
					accessor.SetManagedFields(nil)
					result, err = requestFunc()
				}
			}
			return result, err
		})
		trace.Step("Write to database call finished", utiltrace.Field{"len", len(body)}, utiltrace.Field{"err", err})
		if err != nil {
			scope.err(err, w, req)
			return
		}

		code := http.StatusCreated
		status, ok := result.(*metav1.Status)
		if ok && status.Code == 0 {
			status.Code = int32(code)
		}

		trace.Step("About to write a response")
		defer trace.Step("Writing http response done")
		transformResponseObject(ctx, scope, trace, req, w, code, outputMediaType, result)
	}
}

// CreateNamedResource returns a function that will handle a resource creation with name.
func CreateNamedResource(r rest.NamedCreater, scope *RequestScope, admission admission.Interface) http.HandlerFunc {
	return createHandler(r, scope, admission, true)
}

// CreateResource returns a function that will handle a resource creation.
func CreateResource(r rest.Creater, scope *RequestScope, admission admission.Interface) http.HandlerFunc {
	return createHandler(&namedCreaterAdapter{r}, scope, admission, false)
}

type namedCreaterAdapter struct {
	rest.Creater
}

func (c *namedCreaterAdapter) Create(ctx context.Context, name string, obj runtime.Object, createValidatingAdmission rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	return c.Creater.Create(ctx, obj, createValidatingAdmission, options)
}

// manager is assumed to be already a valid value, we need to make
// userAgent into a valid value too.
func managerOrUserAgent(manager, userAgent string) string {
	if manager != "" {
		return manager
	}
	return prefixFromUserAgent(userAgent)
}

// prefixFromUserAgent takes the characters preceding the first /, quote
// unprintable character and then trim what's beyond the
// FieldManagerMaxLength limit.
func prefixFromUserAgent(u string) string {
	m := strings.Split(u, "/")[0]
	buf := bytes.NewBuffer(nil)
	for _, r := range m {
		// Ignore non-printable characters
		if !unicode.IsPrint(r) {
			continue
		}
		// Only append if we have room for it
		if buf.Len()+utf8.RuneLen(r) > validation.FieldManagerMaxLength {
			break
		}
		buf.WriteRune(r)
	}
	return buf.String()
}
