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
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
)

// DeleteResource returns a function that will handle a resource deletion
// TODO admission here becomes solely validating admission
func DeleteResource(r rest.GracefulDeleter, allowsOptions bool, scope RequestScope, admit admission.Interface) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("Delete " + req.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		if isDryRun(req.URL) {
			scope.err(errors.NewBadRequest("dryRun is not supported yet"), w, req)
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
		ae := request.AuditEventFrom(ctx)
		admit = admission.WithAudit(admit, ae)

		options := &metav1.DeleteOptions{}
		if allowsOptions {
			body, err := readBody(req)
			if err != nil {
				scope.err(err, w, req)
				return
			}
			if len(body) > 0 {
				s, err := negotiation.NegotiateInputSerializer(req, false, metainternalversion.Codecs)
				if err != nil {
					scope.err(err, w, req)
					return
				}
				// For backwards compatibility, we need to allow existing clients to submit per group DeleteOptions
				// It is also allowed to pass a body with meta.k8s.io/v1.DeleteOptions
				defaultGVK := scope.MetaGroupVersion.WithKind("DeleteOptions")
				obj, _, err := metainternalversion.Codecs.DecoderToVersion(s.Serializer, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, options)
				if err != nil {
					scope.err(err, w, req)
					return
				}
				if obj != options {
					scope.err(fmt.Errorf("decoded object cannot be converted to DeleteOptions"), w, req)
					return
				}
				trace.Step("Decoded delete options")

				ae := request.AuditEventFrom(ctx)
				audit.LogRequestObject(ae, obj, scope.Resource, scope.Subresource, scope.Serializer)
				trace.Step("Recorded the audit event")
			} else {
				if values := req.URL.Query(); len(values) > 0 {
					if err := metainternalversion.ParameterCodec.DecodeParameters(values, scope.MetaGroupVersion, options); err != nil {
						err = errors.NewBadRequest(err.Error())
						scope.err(err, w, req)
						return
					}
				}
			}
		}

		trace.Step("About to check admission control")
		if admit != nil && admit.Handles(admission.Delete) {
			userInfo, _ := request.UserFrom(ctx)
			attrs := admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Delete, userInfo)
			if mutatingAdmission, ok := admit.(admission.MutationInterface); ok {
				if err := mutatingAdmission.Admit(attrs); err != nil {
					scope.err(err, w, req)
					return
				}
			}
			if validatingAdmission, ok := admit.(admission.ValidationInterface); ok {
				if err := validatingAdmission.Validate(attrs); err != nil {
					scope.err(err, w, req)
					return
				}
			}
		}

		trace.Step("About to delete object from database")
		wasDeleted := true
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			obj, deleted, err := r.Delete(ctx, name, options)
			wasDeleted = deleted
			return obj, err
		})
		if err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Object deleted from database")

		status := http.StatusOK
		// Return http.StatusAccepted if the resource was not deleted immediately and
		// user requested cascading deletion by setting OrphanDependents=false.
		// Note: We want to do this always if resource was not deleted immediately, but
		// that will break existing clients.
		// Other cases where resource is not instantly deleted are: namespace deletion
		// and pod graceful deletion.
		if !wasDeleted && options.OrphanDependents != nil && *options.OrphanDependents == false {
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
		} else {
			// when a non-status response is returned, set the self link
			requestInfo, ok := request.RequestInfoFrom(ctx)
			if !ok {
				scope.err(fmt.Errorf("missing requestInfo"), w, req)
				return
			}
			if _, ok := result.(*metav1.Status); !ok {
				if err := setSelfLink(result, requestInfo, scope.Namer); err != nil {
					scope.err(err, w, req)
					return
				}
			}
		}

		transformResponseObject(ctx, scope, req, w, status, result)
	}
}

// DeleteCollection returns a function that will handle a collection deletion
func DeleteCollection(r rest.CollectionDeleter, checkBody bool, scope RequestScope, admit admission.Interface) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		if isDryRun(req.URL) {
			scope.err(errors.NewBadRequest("dryRun is not supported yet"), w, req)
			return
		}

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.URL.Query().Get("timeout"))

		namespace, err := scope.Namer.Namespace(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		ctx := req.Context()
		ctx = request.WithNamespace(ctx, namespace)
		ae := request.AuditEventFrom(ctx)
		admit = admission.WithAudit(admit, ae)

		if admit != nil && admit.Handles(admission.Delete) {
			userInfo, _ := request.UserFrom(ctx)
			attrs := admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, "", scope.Resource, scope.Subresource, admission.Delete, userInfo)
			if mutatingAdmission, ok := admit.(admission.MutationInterface); ok {
				err = mutatingAdmission.Admit(attrs)
				if err != nil {
					scope.err(err, w, req)
					return
				}
			}

			if validatingAdmission, ok := admit.(admission.ValidationInterface); ok {
				err = validatingAdmission.Validate(attrs)
				if err != nil {
					scope.err(err, w, req)
					return
				}
			}
		}

		listOptions := metainternalversion.ListOptions{}
		if err := metainternalversion.ParameterCodec.DecodeParameters(req.URL.Query(), scope.MetaGroupVersion, &listOptions); err != nil {
			err = errors.NewBadRequest(err.Error())
			scope.err(err, w, req)
			return
		}

		// transform fields
		// TODO: DecodeParametersInto should do this.
		if listOptions.FieldSelector != nil {
			fn := func(label, value string) (newLabel, newValue string, err error) {
				return scope.Convertor.ConvertFieldLabel(scope.Kind.GroupVersion().String(), scope.Kind.Kind, label, value)
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
			body, err := readBody(req)
			if err != nil {
				scope.err(err, w, req)
				return
			}
			if len(body) > 0 {
				s, err := negotiation.NegotiateInputSerializer(req, false, scope.Serializer)
				if err != nil {
					scope.err(err, w, req)
					return
				}
				defaultGVK := scope.Kind.GroupVersion().WithKind("DeleteOptions")
				obj, _, err := scope.Serializer.DecoderToVersion(s.Serializer, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, options)
				if err != nil {
					scope.err(err, w, req)
					return
				}
				if obj != options {
					scope.err(fmt.Errorf("decoded object cannot be converted to DeleteOptions"), w, req)
					return
				}

				ae := request.AuditEventFrom(ctx)
				audit.LogRequestObject(ae, obj, scope.Resource, scope.Subresource, scope.Serializer)
			}
		}

		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			return r.DeleteCollection(ctx, options, &listOptions)
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
		} else {
			// when a non-status response is returned, set the self link
			if _, ok := result.(*metav1.Status); !ok {
				if _, err := setListSelfLink(result, ctx, req, scope.Namer); err != nil {
					scope.err(err, w, req)
					return
				}
			}
		}

		transformResponseObject(ctx, scope, req, w, http.StatusOK, result)
	}
}
