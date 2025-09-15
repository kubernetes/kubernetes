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
	"sync"
	"time"

	"go.opentelemetry.io/otel/attribute"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversionscheme "k8s.io/apimachinery/pkg/apis/meta/internalversion/scheme"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/apiserver/pkg/endpoints/handlers/finisher"
	requestmetrics "k8s.io/apiserver/pkg/endpoints/handlers/metrics"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/util/dryrun"
	"k8s.io/component-base/tracing"
	"k8s.io/klog/v2"
)

// UpdateResource returns a function that will handle a resource update
func UpdateResource(r rest.Updater, scope *RequestScope, admit admission.Interface) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		// For performance tracking purposes.
		ctx, span := tracing.Start(ctx, "Update", traceFields(req)...)
		req = req.WithContext(ctx)
		defer span.End(500 * time.Millisecond)

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		// enforce a timeout of at most requestTimeoutUpperBound (34s) or less if the user-provided
		// timeout inside the parent context is lower than requestTimeoutUpperBound.
		ctx, cancel := context.WithTimeout(ctx, requestTimeoutUpperBound)
		defer cancel()

		ctx = request.WithNamespace(ctx, namespace)

		outputMediaType, _, err := negotiation.NegotiateOutputMediaType(req, scope.Serializer, scope)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		body, err := limitedReadBodyWithRecordMetric(ctx, req, scope.MaxRequestBodyBytes, scope.Resource.GroupResource(), requestmetrics.Update)
		if err != nil {
			span.AddEvent("limitedReadBody failed", attribute.Int("len", len(body)), attribute.String("err", err.Error()))
			scope.err(err, w, req)
			return
		}
		span.AddEvent("limitedReadBody succeeded", attribute.Int("len", len(body)))

		options := &metav1.UpdateOptions{}
		if err := metainternalversionscheme.ParameterCodec.DecodeParameters(req.URL.Query(), scope.MetaGroupVersion, options); err != nil {
			err = errors.NewBadRequest(err.Error())
			scope.err(err, w, req)
			return
		}
		if errs := validation.ValidateUpdateOptions(options); len(errs) > 0 {
			err := errors.NewInvalid(schema.GroupKind{Group: metav1.GroupName, Kind: "UpdateOptions"}, "", errs)
			scope.err(err, w, req)
			return
		}
		options.TypeMeta.SetGroupVersionKind(metav1.SchemeGroupVersion.WithKind("UpdateOptions"))

		s, err := negotiation.NegotiateInputSerializer(req, false, scope.Serializer)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		defaultGVK := scope.Kind
		original := r.New()

		validationDirective := fieldValidation(options.FieldValidation)
		decodeSerializer := s.Serializer
		if validationDirective == metav1.FieldValidationWarn || validationDirective == metav1.FieldValidationStrict {
			decodeSerializer = s.StrictSerializer
		}

		decoder := scope.Serializer.DecoderToVersion(decodeSerializer, scope.HubGroupVersion)
		span.AddEvent("About to convert to expected version")
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
			err = errors.NewBadRequest(fmt.Sprintf("the API version in the data (%s) does not match the expected API version (%s)", objGV, defaultGVK.GroupVersion()))
			scope.err(err, w, req)
			return
		}
		span.AddEvent("Conversion done")

		audit.LogRequestObject(req.Context(), obj, objGV, scope.Resource, scope.Subresource, scope.Serializer)
		admit = admission.WithAudit(admit)

		// if this object supports namespace info
		if objectMeta, err := meta.Accessor(obj); err == nil {
			// ensure namespace on the object is correct, or error if a conflicting namespace was set in the object
			if err := rest.EnsureObjectNamespaceMatchesRequestNamespace(rest.ExpectedNamespaceForResource(namespace, scope.Resource), objectMeta); err != nil {
				scope.err(err, w, req)
				return
			}
		}

		if err := checkName(obj, name, namespace, scope.Namer); err != nil {
			scope.err(err, w, req)
			return
		}

		userInfo, _ := request.UserFrom(ctx)
		transformers := []rest.TransformFunc{}

		// allows skipping managedFields update if the resulting object is too big
		shouldUpdateManagedFields := true
		admit = fieldmanager.NewManagedFieldsValidatingAdmissionController(admit)
		transformers = append(transformers, func(_ context.Context, newObj, liveObj runtime.Object) (runtime.Object, error) {
			if shouldUpdateManagedFields {
				return scope.FieldManager.UpdateNoErrors(liveObj, newObj, managerOrUserAgent(options.FieldManager, req.UserAgent())), nil
			}
			return newObj, nil
		})

		if mutatingAdmission, ok := admit.(admission.MutationInterface); ok {
			transformers = append(transformers, func(ctx context.Context, newObj, oldObj runtime.Object) (runtime.Object, error) {
				isNotZeroObject, err := hasUID(oldObj)
				if err != nil {
					return nil, fmt.Errorf("unexpected error when extracting UID from oldObj: %v", err.Error())
				} else if !isNotZeroObject {
					if mutatingAdmission.Handles(admission.Create) {
						return newObj, mutatingAdmission.Admit(ctx, admission.NewAttributesRecord(newObj, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Create, updateToCreateOptions(options), dryrun.IsDryRun(options.DryRun), userInfo), scope)
					}
				} else {
					if mutatingAdmission.Handles(admission.Update) {
						return newObj, mutatingAdmission.Admit(ctx, admission.NewAttributesRecord(newObj, oldObj, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Update, options, dryrun.IsDryRun(options.DryRun), userInfo), scope)
					}
				}
				return newObj, nil
			})
			transformers = append(transformers, func(ctx context.Context, newObj, oldObj runtime.Object) (runtime.Object, error) {
				// Dedup owner references again after mutating admission happens
				dedupOwnerReferencesAndAddWarning(newObj, req.Context(), true)
				return newObj, nil
			})
		}

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

		span.AddEvent("About to store object in database")
		wasCreated := false
		requestFunc := func() (runtime.Object, error) {
			obj, created, err := r.Update(
				ctx,
				name,
				rest.DefaultUpdatedObjectInfo(obj, transformers...),
				withAuthorization(rest.AdmissionToValidateObjectFunc(
					admit,
					admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Create, updateToCreateOptions(options), dryrun.IsDryRun(options.DryRun), userInfo), scope),
					scope.Authorizer, createAuthorizerAttributes),
				rest.AdmissionToValidateObjectUpdateFunc(
					admit,
					admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Update, options, dryrun.IsDryRun(options.DryRun), userInfo), scope),
				false,
				options,
			)
			wasCreated = created
			return obj, err
		}
		// Dedup owner references before updating managed fields
		dedupOwnerReferencesAndAddWarning(obj, req.Context(), false)
		result, err := finisher.FinishRequest(ctx, func() (runtime.Object, error) {
			result, err := requestFunc()
			// If the object wasn't committed to storage because it's serialized size was too large,
			// it is safe to remove managedFields (which can be large) and try again.
			if isTooLargeError(err) {
				if accessor, accessorErr := meta.Accessor(obj); accessorErr == nil {
					accessor.SetManagedFields(nil)
					shouldUpdateManagedFields = false
					result, err = requestFunc()
				}
			}
			return result, err
		})
		if err != nil {
			span.AddEvent("Write to database call failed", attribute.Int("len", len(body)), attribute.String("err", err.Error()))
			scope.err(err, w, req)
			return
		}
		span.AddEvent("Write to database call succeeded", attribute.Int("len", len(body)))

		status := http.StatusOK
		if wasCreated {
			status = http.StatusCreated
		}

		span.AddEvent("About to write a response")
		defer span.AddEvent("Writing http response done")
		transformResponseObject(ctx, scope, req, w, status, outputMediaType, result)
	}
}

func withAuthorization(validate rest.ValidateObjectFunc, a authorizer.Authorizer, attributes authorizer.Attributes) rest.ValidateObjectFunc {
	var once sync.Once
	var authorizerDecision authorizer.Decision
	var authorizerReason string
	var authorizerErr error
	return func(ctx context.Context, obj runtime.Object) error {
		if a == nil {
			return errors.NewInternalError(fmt.Errorf("no authorizer provided, unable to authorize a create on update"))
		}
		once.Do(func() {
			authorizerDecision, authorizerReason, authorizerErr = a.Authorize(ctx, attributes)
		})
		// an authorizer like RBAC could encounter evaluation errors and still allow the request, so authorizer decision is checked before error here.
		if authorizerDecision == authorizer.DecisionAllow {
			// Continue to validating admission
			return validate(ctx, obj)
		}
		if authorizerErr != nil {
			return errors.NewInternalError(authorizerErr)
		}

		// The user is not authorized to perform this action, so we need to build the error response
		return responsewriters.ForbiddenStatusError(attributes, authorizerReason)
	}
}

// updateToCreateOptions creates a CreateOptions with the same field values as the provided UpdateOptions.
func updateToCreateOptions(uo *metav1.UpdateOptions) *metav1.CreateOptions {
	if uo == nil {
		return nil
	}
	co := &metav1.CreateOptions{
		DryRun:          uo.DryRun,
		FieldManager:    uo.FieldManager,
		FieldValidation: uo.FieldValidation,
	}
	co.TypeMeta.SetGroupVersionKind(metav1.SchemeGroupVersion.WithKind("CreateOptions"))
	return co
}
