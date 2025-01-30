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
	"time"

	"go.opentelemetry.io/otel/attribute"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metainternalversionscheme "k8s.io/apimachinery/pkg/apis/meta/internalversion/scheme"
	metainternalversionvalidation "k8s.io/apimachinery/pkg/apis/meta/internalversion/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/finisher"
	requestmetrics "k8s.io/apiserver/pkg/endpoints/handlers/metrics"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/util/apihelpers"
	"k8s.io/apiserver/pkg/util/dryrun"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/tracing"

	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

// DeleteResource returns a function that will handle a resource deletion
// TODO admission here becomes solely validating admission
func DeleteResource(r rest.GracefulDeleter, allowsOptions bool, scope *RequestScope, admit admission.Interface) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		// For performance tracking purposes.
		ctx, span := tracing.Start(ctx, "Delete", traceFields(req)...)
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
		admit = admission.WithAudit(admit)

		outputMediaType, _, err := negotiation.NegotiateOutputMediaType(req, scope.Serializer, scope)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		options := &metav1.DeleteOptions{}
		if allowsOptions {
			body, err := limitedReadBodyWithRecordMetric(ctx, req, scope.MaxRequestBodyBytes, scope.Resource.GroupResource().String(), requestmetrics.Delete)
			if err != nil {
				span.AddEvent("limitedReadBody failed", attribute.Int("len", len(body)), attribute.String("err", err.Error()))
				scope.err(err, w, req)
				return
			}
			span.AddEvent("limitedReadBody succeeded", attribute.Int("len", len(body)))
			if len(body) > 0 {
				s, err := negotiation.NegotiateInputSerializer(req, false, apihelpers.GetMetaInternalVersionCodecs())
				if err != nil {
					scope.err(err, w, req)
					return
				}
				// For backwards compatibility, we need to allow existing clients to submit per group DeleteOptions
				// It is also allowed to pass a body with meta.k8s.io/v1.DeleteOptions
				defaultGVK := scope.MetaGroupVersion.WithKind("DeleteOptions")
				obj, gvk, err := apihelpers.GetMetaInternalVersionCodecs().DecoderToVersion(s.Serializer, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, options)
				if err != nil {
					scope.err(err, w, req)
					return
				}
				if obj != options {
					scope.err(fmt.Errorf("decoded object cannot be converted to DeleteOptions"), w, req)
					return
				}
				span.AddEvent("Decoded delete options")

				objGV := gvk.GroupVersion()
				audit.LogRequestObject(req.Context(), obj, objGV, scope.Resource, scope.Subresource, apihelpers.GetMetaInternalVersionCodecs())
				span.AddEvent("Recorded the audit event")
			} else {
				if err := metainternalversionscheme.ParameterCodec.DecodeParameters(req.URL.Query(), scope.MetaGroupVersion, options); err != nil {
					err = errors.NewBadRequest(err.Error())
					scope.err(err, w, req)
					return
				}
			}
		}
		if !utilfeature.DefaultFeatureGate.Enabled(features.AllowUnsafeMalformedObjectDeletion) && options != nil {
			options.IgnoreStoreReadErrorWithClusterBreakingPotential = nil
		}
		if errs := validation.ValidateDeleteOptions(options); len(errs) > 0 {
			err := errors.NewInvalid(schema.GroupKind{Group: metav1.GroupName, Kind: "DeleteOptions"}, "", errs)
			scope.err(err, w, req)
			return
		}
		options.TypeMeta.SetGroupVersionKind(metav1.SchemeGroupVersion.WithKind("DeleteOptions"))

		userInfo, _ := request.UserFrom(ctx)
		staticAdmissionAttrs := admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Delete, options, dryrun.IsDryRun(options.DryRun), userInfo)

		if utilfeature.DefaultFeatureGate.Enabled(features.AllowUnsafeMalformedObjectDeletion) {
			if options != nil && ptr.Deref(options.IgnoreStoreReadErrorWithClusterBreakingPotential, false) {
				// let's make sure that the audit will reflect that this delete request
				// was tried with ignoreStoreReadErrorWithClusterBreakingPotential enabled
				audit.AddAuditAnnotation(ctx, "apiserver.k8s.io/unsafe-delete-ignore-read-error", "")

				p, ok := r.(rest.CorruptObjectDeleterProvider)
				if !ok || p.GetCorruptObjDeleter() == nil {
					// this is a developer error
					scope.err(errors.NewInternalError(fmt.Errorf("no unsafe deleter provided, can not honor ignoreStoreReadErrorWithClusterBreakingPotential")), w, req)
					return
				}
				if scope.Authorizer == nil {
					scope.err(errors.NewInternalError(fmt.Errorf("no authorizer provided, unable to authorize unsafe delete")), w, req)
					return
				}
				if err := authorizeUnsafeDelete(ctx, staticAdmissionAttrs, scope.Authorizer); err != nil {
					scope.err(err, w, req)
					return
				}

				r = p.GetCorruptObjDeleter()
			}
		}

		span.AddEvent("About to delete object from database")
		wasDeleted := true
		result, err := finisher.FinishRequest(ctx, func() (runtime.Object, error) {
			obj, deleted, err := r.Delete(ctx, name, rest.AdmissionToValidateObjectDeleteFunc(admit, staticAdmissionAttrs, scope), options)
			wasDeleted = deleted
			return obj, err
		})
		if err != nil {
			scope.err(err, w, req)
			return
		}
		span.AddEvent("Object deleted from database")

		status := http.StatusOK
		// Return http.StatusAccepted if the resource was not deleted immediately and
		// user requested cascading deletion by setting OrphanDependents=false.
		// Note: We want to do this always if resource was not deleted immediately, but
		// that will break existing clients.
		// Other cases where resource is not instantly deleted are: namespace deletion
		// and pod graceful deletion.
		//nolint:staticcheck // SA1019 backwards compatibility
		//nolint: staticcheck
		if !wasDeleted && options.OrphanDependents != nil && !*options.OrphanDependents {
			status = http.StatusAccepted
		}
		// if the rest.Deleter returns a nil object, fill out a status. Callers may return a valid
		// object with the response.
		if result == nil {
			result = &metav1.Status{
				Status: metav1.StatusSuccess,
				Code:   int32(status),
				Details: &metav1.StatusDetails{
					Name: name,
					Kind: scope.Kind.Kind,
				},
			}
		}

		span.AddEvent("About to write a response")
		defer span.AddEvent("Writing http response done")
		transformResponseObject(ctx, scope, req, w, status, outputMediaType, result)
	}
}

// DeleteCollection returns a function that will handle a collection deletion
func DeleteCollection(r rest.CollectionDeleter, checkBody bool, scope *RequestScope, admit admission.Interface) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		ctx := req.Context()
		ctx, span := tracing.Start(ctx, "Delete", traceFields(req)...)
		req = req.WithContext(ctx)
		defer span.End(500 * time.Millisecond)

		namespace, err := scope.Namer.Namespace(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		// DELETECOLLECTION can be a lengthy operation,
		// we should not impose any 34s timeout here.
		// NOTE: This is similar to LIST which does not enforce a 34s timeout.
		ctx = request.WithNamespace(ctx, namespace)

		outputMediaType, _, err := negotiation.NegotiateOutputMediaType(req, scope.Serializer, scope)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		listOptions := metainternalversion.ListOptions{}
		if err := metainternalversionscheme.ParameterCodec.DecodeParameters(req.URL.Query(), scope.MetaGroupVersion, &listOptions); err != nil {
			err = errors.NewBadRequest(err.Error())
			scope.err(err, w, req)
			return
		}

		metainternalversion.SetListOptionsDefaults(&listOptions, utilfeature.DefaultFeatureGate.Enabled(features.WatchList))
		if errs := metainternalversionvalidation.ValidateListOptions(&listOptions, utilfeature.DefaultFeatureGate.Enabled(features.WatchList)); len(errs) > 0 {
			err := errors.NewInvalid(schema.GroupKind{Group: metav1.GroupName, Kind: "ListOptions"}, "", errs)
			scope.err(err, w, req)
			return
		}

		// transform fields
		// TODO: DecodeParametersInto should do this.
		if listOptions.FieldSelector != nil {
			fn := func(label, value string) (newLabel, newValue string, err error) {
				return scope.Convertor.ConvertFieldLabel(scope.Kind, label, value)
			}
			if listOptions.FieldSelector, err = listOptions.FieldSelector.Transform(fn); err != nil {
				// TODO: allow bad request to set field causes based on query parameters
				err = errors.NewBadRequest(err.Error())
				scope.err(err, w, req)
				return
			}
		}

		options := &metav1.DeleteOptions{}
		if checkBody {
			body, err := limitedReadBodyWithRecordMetric(ctx, req, scope.MaxRequestBodyBytes, scope.Resource.GroupResource().String(), requestmetrics.DeleteCollection)
			if err != nil {
				span.AddEvent("limitedReadBody failed", attribute.Int("len", len(body)), attribute.String("err", err.Error()))
				scope.err(err, w, req)
				return
			}
			span.AddEvent("limitedReadBody succeeded", attribute.Int("len", len(body)))
			if len(body) > 0 {
				s, err := negotiation.NegotiateInputSerializer(req, false, apihelpers.GetMetaInternalVersionCodecs())
				if err != nil {
					scope.err(err, w, req)
					return
				}
				// For backwards compatibility, we need to allow existing clients to submit per group DeleteOptions
				// It is also allowed to pass a body with meta.k8s.io/v1.DeleteOptions
				defaultGVK := scope.MetaGroupVersion.WithKind("DeleteOptions")
				obj, gvk, err := apihelpers.GetMetaInternalVersionCodecs().DecoderToVersion(s.Serializer, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, options)
				if err != nil {
					scope.err(err, w, req)
					return
				}
				if obj != options {
					scope.err(fmt.Errorf("decoded object cannot be converted to DeleteOptions"), w, req)
					return
				}

				objGV := gvk.GroupVersion()
				audit.LogRequestObject(req.Context(), obj, objGV, scope.Resource, scope.Subresource, apihelpers.GetMetaInternalVersionCodecs())
			} else {
				if err := metainternalversionscheme.ParameterCodec.DecodeParameters(req.URL.Query(), scope.MetaGroupVersion, options); err != nil {
					err = errors.NewBadRequest(err.Error())
					scope.err(err, w, req)
					return
				}
			}
		}
		if !utilfeature.DefaultFeatureGate.Enabled(features.AllowUnsafeMalformedObjectDeletion) && options != nil {
			options.IgnoreStoreReadErrorWithClusterBreakingPotential = nil
		}
		if errs := validation.ValidateDeleteOptions(options); len(errs) > 0 {
			err := errors.NewInvalid(schema.GroupKind{Group: metav1.GroupName, Kind: "DeleteOptions"}, "", errs)
			scope.err(err, w, req)
			return
		}

		if utilfeature.DefaultFeatureGate.Enabled(features.AllowUnsafeMalformedObjectDeletion) {
			if options != nil && ptr.Deref(options.IgnoreStoreReadErrorWithClusterBreakingPotential, false) {
				fieldErrList := field.ErrorList{
					field.Invalid(field.NewPath("ignoreStoreReadErrorWithClusterBreakingPotential"), true, "is not allowed with DELETECOLLECTION, try again after removing the option"),
				}
				err := errors.NewInvalid(schema.GroupKind{Group: metav1.GroupName, Kind: "DeleteOptions"}, "", fieldErrList)
				scope.err(err, w, req)
				return
			}
		}

		options.TypeMeta.SetGroupVersionKind(metav1.SchemeGroupVersion.WithKind("DeleteOptions"))

		admit = admission.WithAudit(admit)
		userInfo, _ := request.UserFrom(ctx)
		staticAdmissionAttrs := admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, "", scope.Resource, scope.Subresource, admission.Delete, options, dryrun.IsDryRun(options.DryRun), userInfo)
		result, err := finisher.FinishRequest(ctx, func() (runtime.Object, error) {
			return r.DeleteCollection(ctx, rest.AdmissionToValidateObjectDeleteFunc(admit, staticAdmissionAttrs, scope), options, &listOptions)
		})
		if err != nil {
			scope.err(err, w, req)
			return
		}

		// if the rest.Deleter returns a nil object, fill out a status. Callers may return a valid
		// object with the response.
		if result == nil {
			result = &metav1.Status{
				Status: metav1.StatusSuccess,
				Code:   http.StatusOK,
				Details: &metav1.StatusDetails{
					Kind: scope.Kind.Kind,
				},
			}
		}

		span.AddEvent("About to write a response")
		defer span.AddEvent("Writing http response done")
		transformResponseObject(ctx, scope, req, w, http.StatusOK, outputMediaType, result)
	}
}

// authorizeUnsafeDelete ensures that the user has permission to do
// 'unsafe-delete-ignore-read-errors' on the resource being deleted when
// ignoreStoreReadErrorWithClusterBreakingPotential is enabled
func authorizeUnsafeDelete(ctx context.Context, attr admission.Attributes, authz authorizer.Authorizer) (err error) {
	if attr.GetOperation() != admission.Delete || attr.GetOperationOptions() == nil {
		return nil
	}
	options, ok := attr.GetOperationOptions().(*metav1.DeleteOptions)
	if !ok {
		return errors.NewInternalError(fmt.Errorf("expected an option of type: %T, but got: %T", &metav1.DeleteOptions{}, attr.GetOperationOptions()))
	}
	if !ptr.Deref(options.IgnoreStoreReadErrorWithClusterBreakingPotential, false) {
		return nil
	}

	requestInfo, found := request.RequestInfoFrom(ctx)
	if !found {
		return admission.NewForbidden(attr, fmt.Errorf("no RequestInfo found in the context"))
	}
	if !requestInfo.IsResourceRequest || len(attr.GetSubresource()) > 0 {
		return admission.NewForbidden(attr, fmt.Errorf("ignoreStoreReadErrorWithClusterBreakingPotential delete option is not allowed on a subresource or non-resource request"))
	}

	// if we are here, IgnoreStoreReadErrorWithClusterBreakingPotential
	// is set to true in the delete options, the user must have permission
	// to do 'unsafe-delete-ignore-read-errors' on the given resource.
	record := authorizer.AttributesRecord{
		User:            attr.GetUserInfo(),
		Verb:            "unsafe-delete-ignore-read-errors",
		Namespace:       attr.GetNamespace(),
		Name:            attr.GetName(),
		APIGroup:        attr.GetResource().Group,
		APIVersion:      attr.GetResource().Version,
		Resource:        attr.GetResource().Resource,
		ResourceRequest: true,
	}
	// TODO: can't use ResourceAttributesFrom from k8s.io/kubernetes/pkg/registry/authorization/util
	// due to prevent staging --> k8s.io/kubernetes dep issue
	if utilfeature.DefaultFeatureGate.Enabled(features.AuthorizeWithSelectors) {
		if len(requestInfo.FieldSelector) > 0 {
			fieldSelector, err := fields.ParseSelector(requestInfo.FieldSelector)
			if err != nil {
				record.FieldSelectorRequirements, record.FieldSelectorParsingErr = nil, err
			} else {
				if requirements := fieldSelector.Requirements(); len(requirements) > 0 {
					record.FieldSelectorRequirements, record.FieldSelectorParsingErr = fieldSelector.Requirements(), nil
				}
			}
		}
		if len(requestInfo.LabelSelector) > 0 {
			labelSelector, err := labels.Parse(requestInfo.LabelSelector)
			if err != nil {
				record.LabelSelectorRequirements, record.LabelSelectorParsingErr = nil, err
			} else {
				if requirements, _ /*selectable*/ := labelSelector.Requirements(); len(requirements) > 0 {
					record.LabelSelectorRequirements, record.LabelSelectorParsingErr = requirements, nil
				}
			}
		}
	}

	decision, reason, err := authz.Authorize(ctx, record)
	if err != nil {
		err = fmt.Errorf("error while checking permission for %q, %w", record.Verb, err)
		klog.FromContext(ctx).V(1).Error(err, "failed to authorize")
		return admission.NewForbidden(attr, err)
	}
	if decision == authorizer.DecisionAllow {
		return nil
	}

	return admission.NewForbidden(attr, fmt.Errorf("not permitted to do %q, reason: %s", record.Verb, reason))
}
