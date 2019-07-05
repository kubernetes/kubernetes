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

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/validation"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/util/dryrun"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	utiltrace "k8s.io/utils/trace"
)

// UpdateResource returns a function that will handle a resource update
func UpdateResource(r rest.Updater, scope *RequestScope, admit admission.Interface) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("Update " + req.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		if isDryRun(req.URL) && !utilfeature.DefaultFeatureGate.Enabled(features.DryRun) {
			scope.err(errors.NewBadRequest("the dryRun alpha feature is disabled"), w, req)
			return
		}

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		ctx := req.Context()
		ctx = request.WithNamespace(ctx, namespace)

		outputMediaType, _, err := negotiation.NegotiateOutputMediaType(req, scope.Serializer, scope)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		body, err := limitedReadBody(req, scope.MaxRequestBodyBytes)
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
		options.TypeMeta.SetGroupVersionKind(metav1.SchemeGroupVersion.WithKind("UpdateOptions"))

		s, err := negotiation.NegotiateInputSerializer(req, false, scope.Serializer)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		defaultGVK := scope.Kind
		original := r.New()

		trace.Step("About to convert to expected version")
		decoder := scope.Serializer.DecoderToVersion(s.Serializer, scope.HubGroupVersion)
		obj, gvk, err := decoder.Decode(body, &defaultGVK, original)
		if err != nil {
			err = transformDecodeError(scope.Typer, err, original, gvk, body)
			scope.err(err, w, req)
			return
		}
		if gvk.GroupVersion() != defaultGVK.GroupVersion() {
			err = errors.NewBadRequest(fmt.Sprintf("the API version in the data (%s) does not match the expected API version (%s)", gvk.GroupVersion(), defaultGVK.GroupVersion()))
			scope.err(err, w, req)
			return
		}
		trace.Step("Conversion done")

		ae := request.AuditEventFrom(ctx)
		audit.LogRequestObject(ae, obj, scope.Resource, scope.Subresource, scope.Serializer)
		admit = admission.WithAudit(admit, ae)

		if err := checkName(obj, name, namespace, scope.Namer); err != nil {
			scope.err(err, w, req)
			return
		}

		userInfo, _ := request.UserFrom(ctx)
		transformers := []rest.TransformFunc{}
		if scope.FieldManager != nil {
			transformers = append(transformers, func(_ context.Context, newObj, liveObj runtime.Object) (runtime.Object, error) {
				obj, err := scope.FieldManager.Update(liveObj, newObj, managerOrUserAgent(options.FieldManager, req.UserAgent()))
				if err != nil {
					return nil, fmt.Errorf("failed to update object (Update for %v) managed fields: %v", scope.Kind, err)
				}
				return obj, nil
			})
		}
		if mutatingAdmission, ok := admit.(admission.MutationInterface); ok {
			transformers = append(transformers, func(ctx context.Context, newObj, oldObj runtime.Object) (runtime.Object, error) {
				isNotZeroObject, err := hasUID(oldObj)
				if err != nil {
					return nil, fmt.Errorf("unexpected error when extracting UID from oldObj: %v", err.Error())
				} else if !isNotZeroObject {
					if mutatingAdmission.Handles(admission.Create) {
						return newObj, mutatingAdmission.Admit(admission.NewAttributesRecord(newObj, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Create, updateToCreateOptions(options), dryrun.IsDryRun(options.DryRun), userInfo), scope)
					}
				} else {
					if mutatingAdmission.Handles(admission.Update) {
						return newObj, mutatingAdmission.Admit(admission.NewAttributesRecord(newObj, oldObj, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Update, options, dryrun.IsDryRun(options.DryRun), userInfo), scope)
					}
				}
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

		trace.Step("About to store object in database")
		wasCreated := false
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
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
		})
		if err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Object stored in database")

		status := http.StatusOK
		if wasCreated {
			status = http.StatusCreated
		}

		transformResponseObject(ctx, scope, trace, req, w, status, outputMediaType, result)
	}
}

func withAuthorization(validate rest.ValidateObjectFunc, a authorizer.Authorizer, attributes authorizer.Attributes) rest.ValidateObjectFunc {
	var once sync.Once
	var authorizerDecision authorizer.Decision
	var authorizerReason string
	var authorizerErr error
	return func(obj runtime.Object) error {
		if a == nil {
			return errors.NewInternalError(fmt.Errorf("no authorizer provided, unable to authorize a create on update"))
		}
		once.Do(func() {
			authorizerDecision, authorizerReason, authorizerErr = a.Authorize(attributes)
		})
		// an authorizer like RBAC could encounter evaluation errors and still allow the request, so authorizer decision is checked before error here.
		if authorizerDecision == authorizer.DecisionAllow {
			// Continue to validating admission
			return validate(obj)
		}
		if authorizerErr != nil {
			return errors.NewInternalError(authorizerErr)
		}

		// The user is not authorized to perform this action, so we need to build the error response
		gr := schema.GroupResource{
			Group:    attributes.GetAPIGroup(),
			Resource: attributes.GetResource(),
		}
		name := attributes.GetName()
		err := fmt.Errorf("%v", authorizerReason)
		return errors.NewForbidden(gr, name, err)
	}
}

// updateToCreateOptions creates a CreateOptions with the same field values as the provided UpdateOptions.
func updateToCreateOptions(uo *metav1.UpdateOptions) *metav1.CreateOptions {
	if uo == nil {
		return nil
	}
	co := &metav1.CreateOptions{
		DryRun:       uo.DryRun,
		FieldManager: uo.FieldManager,
	}
	co.TypeMeta.SetGroupVersionKind(metav1.SchemeGroupVersion.WithKind("CreateOptions"))
	return co
}
