/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"net/http"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

	"github.com/emicklei/go-restful"
)

// ContextFunc returns a Context given a request - a context must be returned
type ContextFunc func(req *restful.Request) api.Context

// ResourceNameFunc returns a name (and optional namespace) given a request - if no name is present
// an error must be returned.
type ResourceNameFunc func(req *restful.Request) (namespace, name string, err error)

// ObjectNameFunc  returns the name (and optional namespace) of an object
type ObjectNameFunc func(obj runtime.Object) (namespace, name string, err error)

// ResourceNamespaceFunc returns the namespace associated with the given request - if no namespace
// is present an error must be returned.
type ResourceNamespaceFunc func(req *restful.Request) (namespace string, err error)

// LinkResourceFunc updates the provided object with a SelfLink that is appropriate for the current
// request.
type LinkResourceFunc func(req *restful.Request, obj runtime.Object) error

// GetResource returns a function that handles retrieving a single resource from a RESTStorage object.
func GetResource(r RESTGetter, ctxFn ContextFunc, nameFn ResourceNameFunc, linkFn LinkResourceFunc, codec runtime.Codec) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter
		namespace, name, err := nameFn(req)
		if err != nil {
			notFound(w, req.Request)
			return
		}
		ctx := ctxFn(req)
		if len(namespace) > 0 {
			ctx = api.WithNamespace(ctx, namespace)
		}
		item, err := r.Get(ctx, name)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		if err := linkFn(req, item); err != nil {
			errorJSON(err, codec, w)
			return
		}
		writeJSON(http.StatusOK, codec, item, w)
	}
}

// ListResource returns a function that handles retrieving a list of resources from a RESTStorage object.
func ListResource(r RESTLister, ctxFn ContextFunc, namespaceFn ResourceNamespaceFunc, linkFn LinkResourceFunc, codec runtime.Codec) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		namespace, err := namespaceFn(req)
		if err != nil {
			notFound(w, req.Request)
			return
		}
		ctx := ctxFn(req)
		if len(namespace) > 0 {
			ctx = api.WithNamespace(ctx, namespace)
		}
		label, err := labels.ParseSelector(req.Request.URL.Query().Get("labels"))
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		field, err := labels.ParseSelector(req.Request.URL.Query().Get("fields"))
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		item, err := r.List(ctx, label, field)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		if err := linkFn(req, item); err != nil {
			errorJSON(err, codec, w)
			return
		}
		writeJSON(http.StatusOK, codec, item, w)
	}
}

// CreateResource returns a function that will handle a resource creation.
func CreateResource(r RESTCreater, ctxFn ContextFunc, namespaceFn ResourceNamespaceFunc, linkFn LinkResourceFunc, codec runtime.Codec, resource string, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, err := namespaceFn(req)
		if err != nil {
			notFound(w, req.Request)
			return
		}
		ctx := ctxFn(req)
		if len(namespace) > 0 {
			ctx = api.WithNamespace(ctx, namespace)
		}

		body, err := readBody(req.Request)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		obj := r.New()
		if err := codec.DecodeInto(body, obj); err != nil {
			errorJSON(err, codec, w)
			return
		}

		err = admit.Admit(admission.NewAttributesRecord(obj, namespace, resource, "CREATE"))
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			out, err := r.Create(ctx, obj)
			if status, ok := out.(*api.Status); ok && err == nil && status.Code == 0 {
				status.Code = http.StatusCreated
			}
			return out, err
		})
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		if err := linkFn(req, result); err != nil {
			errorJSON(err, codec, w)
			return
		}

		writeJSON(http.StatusCreated, codec, result, w)
	}
}

// UpdateResource returns a function that will handle a resource update
func UpdateResource(r RESTUpdater, ctxFn ContextFunc, nameFn ResourceNameFunc, objNameFunc ObjectNameFunc, linkFn LinkResourceFunc, codec runtime.Codec, resource string, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, name, err := nameFn(req)
		if err != nil {
			notFound(w, req.Request)
			return
		}
		ctx := ctxFn(req)
		if len(namespace) > 0 {
			ctx = api.WithNamespace(ctx, namespace)
		}

		body, err := readBody(req.Request)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		obj := r.New()
		if err := codec.DecodeInto(body, obj); err != nil {
			errorJSON(err, codec, w)
			return
		}

		objNamespace, objName, err := objNameFunc(obj)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		if objName != name {
			errorJSON(errors.NewBadRequest("the name of the object does not match the name on the URL"), codec, w)
			return
		}
		if len(namespace) > 0 {
			if len(objNamespace) > 0 && objNamespace != namespace {
				errorJSON(errors.NewBadRequest("the namespace of the object does not match the namespace on the request"), codec, w)
				return
			}
		}

		err = admit.Admit(admission.NewAttributesRecord(obj, namespace, resource, "UPDATE"))
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		wasCreated := false
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			obj, created, err := r.Update(ctx, obj)
			wasCreated = created
			return obj, err
		})
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		if err := linkFn(req, result); err != nil {
			errorJSON(err, codec, w)
			return
		}

		status := http.StatusOK
		if wasCreated {
			status = http.StatusCreated
		}
		writeJSON(status, codec, result, w)
	}
}

// DeleteResource returns a function that will handle a resource deletion
func DeleteResource(r RESTDeleter, ctxFn ContextFunc, nameFn ResourceNameFunc, linkFn LinkResourceFunc, codec runtime.Codec, resource, kind string, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, name, err := nameFn(req)
		if err != nil {
			notFound(w, req.Request)
			return
		}
		ctx := ctxFn(req)
		if len(namespace) > 0 {
			ctx = api.WithNamespace(ctx, namespace)
		}

		err = admit.Admit(admission.NewAttributesRecord(nil, namespace, resource, "DELETE"))
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			return r.Delete(ctx, name)
		})
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		// if the RESTDeleter returns a nil object, fill out a status. Callers may return a valid
		// object with the response.
		if result == nil {
			result = &api.Status{
				Status: api.StatusSuccess,
				Code:   http.StatusOK,
				Details: &api.StatusDetails{
					ID:   name,
					Kind: kind,
				},
			}
		} else {
			// when a non-status response is returned, set the self link
			if _, ok := result.(*api.Status); !ok {
				if err := linkFn(req, result); err != nil {
					errorJSON(err, codec, w)
					return
				}
			}
		}
		writeJSON(http.StatusOK, codec, result, w)
	}
}

// resultFunc is a function that returns a rest result and can be run in a goroutine
type resultFunc func() (runtime.Object, error)

// finishRequest makes a given resultFunc asynchronous and handles errors returned by the response.
// Any api.Status object returned is considered an "error", which interrupts the normal response flow.
func finishRequest(timeout time.Duration, fn resultFunc) (result runtime.Object, err error) {
	ch := make(chan runtime.Object)
	errCh := make(chan error)
	go func() {
		if result, err := fn(); err != nil {
			errCh <- err
		} else {
			ch <- result
		}
	}()

	select {
	case result = <-ch:
		if status, ok := result.(*api.Status); ok {
			return nil, errors.FromObject(status)
		}
		return result, nil
	case err = <-errCh:
		return nil, err
	case <-time.After(timeout):
		return nil, errors.NewTimeoutError("request did not complete within allowed duration")
	}
}

type linkFunc func(namespace, name string) (path string, query string)

// setSelfLink sets the self link of an object (or the child items in a list) to the base URL of the request
// plus the path and query generated by the provided linkFunc
func setSelfLink(obj runtime.Object, req *http.Request, linker runtime.SelfLinker, fn linkFunc) error {
	namespace, err := linker.Namespace(obj)
	if err != nil {
		return err
	}
	name, err := linker.Name(obj)
	if err != nil {
		return err
	}
	path, query := fn(namespace, name)

	newURL := *req.URL
	newURL.Path = path
	newURL.RawQuery = query
	newURL.Fragment = ""

	if err := linker.SetSelfLink(obj, newURL.String()); err != nil {
		return err
	}
	if !runtime.IsListType(obj) {
		return nil
	}

	// Set self-link of objects in the list.
	items, err := runtime.ExtractList(obj)
	if err != nil {
		return err
	}
	for i := range items {
		if err := setSelfLink(items[i], req, linker, fn); err != nil {
			return err
		}
	}
	return runtime.SetList(obj, items)
}
