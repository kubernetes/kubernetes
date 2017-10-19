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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
)

// UpdateResource returns a function that will handle a resource update
func UpdateResource(r rest.Updater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("Update " + req.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)

		body, err := readBody(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		s, err := negotiation.NegotiateInputSerializer(req, scope.Serializer)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		defaultGVK := scope.Kind
		original := r.New()
		trace.Step("About to convert to expected version")
		decoder := scope.Serializer.DecoderToVersion(s.Serializer, schema.GroupVersion{Group: defaultGVK.Group, Version: runtime.APIVersionInternal})
		obj, gvk, err := decoder.Decode(body, &defaultGVK, original)
		if err != nil {
			err = transformDecodeError(typer, err, original, gvk, body)
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

		if err := checkName(obj, name, namespace, scope.Namer); err != nil {
			scope.err(err, w, req)
			return
		}

		var transformers []rest.TransformFunc
		if admit != nil && admit.Handles(admission.Update) {
			transformers = append(transformers, func(ctx request.Context, newObj, oldObj runtime.Object) (runtime.Object, error) {
				userInfo, _ := request.UserFrom(ctx)
				return newObj, admit.Admit(admission.NewAttributesRecord(newObj, oldObj, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Update, userInfo))
			})
		}

		trace.Step("About to store object in database")
		wasCreated := false
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			obj, created, err := r.Update(ctx, name, rest.DefaultUpdatedObjectInfo(obj, transformers...))
			wasCreated = created
			return obj, err
		})
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

		status := http.StatusOK
		if wasCreated {
			status = http.StatusCreated
		}

		transformResponseObject(ctx, scope, req, w, status, result)
	}
}
