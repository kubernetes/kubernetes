/*
Copyright 2014 The Kubernetes Authors.

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
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	goruntime "runtime"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	openapiproto "k8s.io/kube-openapi/pkg/util/proto"
)

// RequestScope encapsulates common fields across all RESTful handler methods.
type RequestScope struct {
	Namer ScopeNamer

	Serializer runtime.NegotiatedSerializer
	runtime.ParameterCodec

	Creater         runtime.ObjectCreater
	Convertor       runtime.ObjectConvertor
	Defaulter       runtime.ObjectDefaulter
	Typer           runtime.ObjectTyper
	UnsafeConvertor runtime.ObjectConvertor

	TableConvertor rest.TableConvertor
	OpenAPISchema  openapiproto.Schema

	Resource    schema.GroupVersionResource
	Kind        schema.GroupVersionKind
	Subresource string

	MetaGroupVersion schema.GroupVersion
}

func (scope *RequestScope) err(err error, w http.ResponseWriter, req *http.Request) {
	responsewriters.ErrorNegotiated(err, scope.Serializer, scope.Kind.GroupVersion(), w, req)
}

func (scope *RequestScope) AllowsConversion(gvk schema.GroupVersionKind) bool {
	// TODO: this is temporary, replace with an abstraction calculated at endpoint installation time
	if gvk.GroupVersion() == metav1beta1.SchemeGroupVersion {
		switch gvk.Kind {
		case "Table":
			return scope.TableConvertor != nil
		case "PartialObjectMetadata", "PartialObjectMetadataList":
			// TODO: should delineate between lists and non-list endpoints
			return true
		default:
			return false
		}
	}
	return false
}

func (scope *RequestScope) AllowsServerVersion(version string) bool {
	return version == scope.MetaGroupVersion.Version
}

func (scope *RequestScope) AllowsStreamSchema(s string) bool {
	return s == "watch"
}

// ConnectResource returns a function that handles a connect request on a rest.Storage object.
func ConnectResource(connecter rest.Connecter, scope RequestScope, admit admission.Interface, restPath string, isSubresource bool) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		ctx := req.Context()
		ctx = request.WithNamespace(ctx, namespace)
		ae := request.AuditEventFrom(ctx)
		admit = admission.WithAudit(admit, ae)

		opts, subpath, subpathKey := connecter.NewConnectOptions()
		if err := getRequestOptions(req, scope, opts, subpath, subpathKey, isSubresource); err != nil {
			err = errors.NewBadRequest(err.Error())
			scope.err(err, w, req)
			return
		}
		if admit != nil && admit.Handles(admission.Connect) {
			connectRequest := &rest.ConnectRequest{
				Name:         name,
				Options:      opts,
				ResourcePath: restPath,
			}
			userInfo, _ := request.UserFrom(ctx)
			// TODO: remove the mutating admission here as soon as we have ported all plugin that handle CONNECT
			if mutatingAdmission, ok := admit.(admission.MutationInterface); ok {
				err = mutatingAdmission.Admit(admission.NewAttributesRecord(connectRequest, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Connect, userInfo))
				if err != nil {
					scope.err(err, w, req)
					return
				}
			}
			if validatingAdmission, ok := admit.(admission.ValidationInterface); ok {
				err = validatingAdmission.Validate(admission.NewAttributesRecord(connectRequest, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Connect, userInfo))
				if err != nil {
					scope.err(err, w, req)
					return
				}
			}
		}
		requestInfo, _ := request.RequestInfoFrom(ctx)
		metrics.RecordLongRunning(req, requestInfo, func() {
			handler, err := connecter.Connect(ctx, name, opts, &responder{scope: scope, req: req, w: w})
			if err != nil {
				scope.err(err, w, req)
				return
			}
			handler.ServeHTTP(w, req)
		})
	}
}

// responder implements rest.Responder for assisting a connector in writing objects or errors.
type responder struct {
	scope RequestScope
	req   *http.Request
	w     http.ResponseWriter
}

func (r *responder) Object(statusCode int, obj runtime.Object) {
	responsewriters.WriteObject(statusCode, r.scope.Kind.GroupVersion(), r.scope.Serializer, obj, r.w, r.req)
}

func (r *responder) Error(err error) {
	r.scope.err(err, r.w, r.req)
}

// resultFunc is a function that returns a rest result and can be run in a goroutine
type resultFunc func() (runtime.Object, error)

// finishRequest makes a given resultFunc asynchronous and handles errors returned by the response.
// An api.Status object with status != success is considered an "error", which interrupts the normal response flow.
func finishRequest(timeout time.Duration, fn resultFunc) (result runtime.Object, err error) {
	// these channels need to be buffered to prevent the goroutine below from hanging indefinitely
	// when the select statement reads something other than the one the goroutine sends on.
	ch := make(chan runtime.Object, 1)
	errCh := make(chan error, 1)
	panicCh := make(chan interface{}, 1)
	go func() {
		// panics don't cross goroutine boundaries, so we have to handle ourselves
		defer func() {
			panicReason := recover()
			if panicReason != nil {
				const size = 64 << 10
				buf := make([]byte, size)
				buf = buf[:goruntime.Stack(buf, false)]
				panicReason = strings.TrimSuffix(fmt.Sprintf("%v\n%s", panicReason, string(buf)), "\n")
				// Propagate to parent goroutine
				panicCh <- panicReason
			}
		}()

		if result, err := fn(); err != nil {
			errCh <- err
		} else {
			ch <- result
		}
	}()

	select {
	case result = <-ch:
		if status, ok := result.(*metav1.Status); ok {
			if status.Status != metav1.StatusSuccess {
				return nil, errors.FromObject(status)
			}
		}
		return result, nil
	case err = <-errCh:
		return nil, err
	case p := <-panicCh:
		panic(p)
	case <-time.After(timeout):
		return nil, errors.NewTimeoutError("request did not complete within allowed duration", 0)
	}
}

// transformDecodeError adds additional information when a decode fails.
func transformDecodeError(typer runtime.ObjectTyper, baseErr error, into runtime.Object, gvk *schema.GroupVersionKind, body []byte) error {
	objGVKs, _, err := typer.ObjectKinds(into)
	if err != nil {
		return err
	}
	objGVK := objGVKs[0]
	if gvk != nil && len(gvk.Kind) > 0 {
		return errors.NewBadRequest(fmt.Sprintf("%s in version %q cannot be handled as a %s: %v", gvk.Kind, gvk.Version, objGVK.Kind, baseErr))
	}
	summary := summarizeData(body, 30)
	return errors.NewBadRequest(fmt.Sprintf("the object provided is unrecognized (must be of type %s): %v (%s)", objGVK.Kind, baseErr, summary))
}

// setSelfLink sets the self link of an object (or the child items in a list) to the base URL of the request
// plus the path and query generated by the provided linkFunc
func setSelfLink(obj runtime.Object, requestInfo *request.RequestInfo, namer ScopeNamer) error {
	// TODO: SelfLink generation should return a full URL?
	uri, err := namer.GenerateLink(requestInfo, obj)
	if err != nil {
		return nil
	}

	return namer.SetSelfLink(obj, uri)
}

func hasUID(obj runtime.Object) (bool, error) {
	if obj == nil {
		return false, nil
	}
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return false, errors.NewInternalError(err)
	}
	if len(accessor.GetUID()) == 0 {
		return false, nil
	}
	return true, nil
}

// checkName checks the provided name against the request
func checkName(obj runtime.Object, name, namespace string, namer ScopeNamer) error {
	objNamespace, objName, err := namer.ObjectName(obj)
	if err != nil {
		return errors.NewBadRequest(fmt.Sprintf(
			"the name of the object (%s based on URL) was undeterminable: %v", name, err))
	}
	if objName != name {
		return errors.NewBadRequest(fmt.Sprintf(
			"the name of the object (%s) does not match the name on the URL (%s)", objName, name))
	}
	if len(namespace) > 0 {
		if len(objNamespace) > 0 && objNamespace != namespace {
			return errors.NewBadRequest(fmt.Sprintf(
				"the namespace of the object (%s) does not match the namespace on the request (%s)", objNamespace, namespace))
		}
	}

	return nil
}

// setListSelfLink sets the self link of a list to the base URL, then sets the self links
// on all child objects returned. Returns the number of items in the list.
func setListSelfLink(obj runtime.Object, ctx context.Context, req *http.Request, namer ScopeNamer) (int, error) {
	if !meta.IsListType(obj) {
		return 0, nil
	}

	uri, err := namer.GenerateListLink(req)
	if err != nil {
		return 0, err
	}
	if err := namer.SetSelfLink(obj, uri); err != nil {
		glog.V(4).Infof("Unable to set self link on object: %v", err)
	}
	requestInfo, ok := request.RequestInfoFrom(ctx)
	if !ok {
		return 0, fmt.Errorf("missing requestInfo")
	}

	count := 0
	err = meta.EachListItem(obj, func(obj runtime.Object) error {
		count++
		return setSelfLink(obj, requestInfo, namer)
	})
	return count, err
}

func summarizeData(data []byte, maxLength int) string {
	switch {
	case len(data) == 0:
		return "<empty>"
	case data[0] == '{':
		if len(data) > maxLength {
			return string(data[:maxLength]) + " ..."
		}
		return string(data)
	default:
		if len(data) > maxLength {
			return hex.EncodeToString(data[:maxLength]) + " ..."
		}
		return hex.EncodeToString(data)
	}
}

func readBody(req *http.Request) ([]byte, error) {
	defer req.Body.Close()
	return ioutil.ReadAll(req.Body)
}

func parseTimeout(str string) time.Duration {
	if str != "" {
		timeout, err := time.ParseDuration(str)
		if err == nil {
			return timeout
		}
		glog.Errorf("Failed to parse %q: %v", str, err)
	}
	return 30 * time.Second
}

func isDryRun(url *url.URL) bool {
	return len(url.Query()["dryRun"]) != 0
}
