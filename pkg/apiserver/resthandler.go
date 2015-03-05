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
	"fmt"
	"net/http"
	"net/url"
	gpath "path"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

	"github.com/emicklei/go-restful"
	"github.com/evanphx/json-patch"
	"github.com/golang/glog"
)

// ContextFunc returns a Context given a request - a context must be returned
type ContextFunc func(req *restful.Request) api.Context

// ScopeNamer handles accessing names from requests and objects
type ScopeNamer interface {
	// Namespace returns the appropriate namespace value from the request (may be empty) or an
	// error.
	Namespace(req *restful.Request) (namespace string, err error)
	// Name returns the name from the request, and an optional namespace value if this is a namespace
	// scoped call. An error is returned if the name is not available.
	Name(req *restful.Request) (namespace, name string, err error)
	// ObjectName returns the namespace and name from an object if they exist, or an error if the object
	// does not support names.
	ObjectName(obj runtime.Object) (namespace, name string, err error)
	// SetSelfLink sets the provided URL onto the object. The method should return nil if the object
	// does not support selfLinks.
	SetSelfLink(obj runtime.Object, url string) error
	// GenerateLink creates a path and query for a given runtime object that represents the canonical path.
	GenerateLink(req *restful.Request, obj runtime.Object) (path, query string, err error)
	// GenerateLink creates a path and query for a list that represents the canonical path.
	GenerateListLink(req *restful.Request) (path, query string, err error)
}

// GetResource returns a function that handles retrieving a single resource from a RESTStorage object.
func GetResource(r RESTGetter, ctxFn ContextFunc, namer ScopeNamer, codec runtime.Codec) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter
		namespace, name, err := namer.Name(req)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		ctx := ctxFn(req)
		ctx = api.WithNamespace(ctx, namespace)

		result, err := r.Get(ctx, name)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		if err := setSelfLink(result, req, namer); err != nil {
			errorJSON(err, codec, w)
			return
		}
		writeJSON(http.StatusOK, codec, result, w)
	}
}

func parseSelectorQueryParams(query url.Values, version, apiResource string) (label labels.Selector, field fields.Selector, err error) {
	labelString := query.Get(api.LabelSelectorQueryParam(version))
	label, err = labels.Parse(labelString)
	if err != nil {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("The 'labels' selector parameter (%s) could not be parsed: %v", labelString, err))
	}

	convertToInternalVersionFunc := func(label, value string) (newLabel, newValue string, err error) {
		return api.Scheme.ConvertFieldLabel(version, apiResource, label, value)
	}
	fieldString := query.Get(api.FieldSelectorQueryParam(version))
	field, err = fields.ParseAndTransformSelector(fieldString, convertToInternalVersionFunc)
	if err != nil {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("The 'fields' selector parameter (%s) could not be parsed: %v", fieldString, err))
	}
	return label, field, nil
}

// ListResource returns a function that handles retrieving a list of resources from a RESTStorage object.
func ListResource(r RESTLister, ctxFn ContextFunc, namer ScopeNamer, codec runtime.Codec, version, apiResource string) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		namespace, err := namer.Namespace(req)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		ctx := ctxFn(req)
		ctx = api.WithNamespace(ctx, namespace)

		label, field, err := parseSelectorQueryParams(req.Request.URL.Query(), version, apiResource)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		result, err := r.List(ctx, label, field)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		if err := setListSelfLink(result, req, namer); err != nil {
			errorJSON(err, codec, w)
			return
		}
		writeJSON(http.StatusOK, codec, result, w)
	}
}

// CreateResource returns a function that will handle a resource creation.
func CreateResource(r RESTCreater, ctxFn ContextFunc, namer ScopeNamer, codec runtime.Codec, typer runtime.ObjectTyper, resource string, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, err := namer.Namespace(req)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		ctx := ctxFn(req)
		ctx = api.WithNamespace(ctx, namespace)

		body, err := readBody(req.Request)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		obj := r.New()
		if err := codec.DecodeInto(body, obj); err != nil {
			err = transformDecodeError(typer, err, obj, body)
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

		if err := setSelfLink(result, req, namer); err != nil {
			errorJSON(err, codec, w)
			return
		}

		writeJSON(http.StatusCreated, codec, result, w)
	}
}

// PatchResource returns a function that will handle a resource patch
// TODO: Eventually PatchResource should just use AtomicUpdate and this routine should be a bit cleaner
func PatchResource(r RESTPatcher, ctxFn ContextFunc, namer ScopeNamer, codec runtime.Codec, typer runtime.ObjectTyper, resource string, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, name, err := namer.Name(req)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		obj := r.New()
		// PATCH requires same permission as UPDATE
		err = admit.Admit(admission.NewAttributesRecord(obj, namespace, resource, "UPDATE"))
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		ctx := ctxFn(req)
		ctx = api.WithNamespace(ctx, namespace)

		original, err := r.Get(ctx, name)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		originalObjJs, err := codec.Encode(original)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		patchJs, err := readBody(req.Request)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		patchedObjJs, err := jsonpatch.MergePatch(originalObjJs, patchJs)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		if err := codec.DecodeInto(patchedObjJs, obj); err != nil {
			errorJSON(err, codec, w)
			return
		}
		if err := checkName(obj, name, namespace, namer); err != nil {
			errorJSON(err, codec, w)
			return
		}

		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			// update should never create as previous get would fail
			obj, _, err := r.Update(ctx, obj)
			return obj, err
		})
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		if err := setSelfLink(result, req, namer); err != nil {
			errorJSON(err, codec, w)
			return
		}

		writeJSON(http.StatusOK, codec, result, w)
	}
}

// UpdateResource returns a function that will handle a resource update
func UpdateResource(r RESTUpdater, ctxFn ContextFunc, namer ScopeNamer, codec runtime.Codec, typer runtime.ObjectTyper, resource string, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, name, err := namer.Name(req)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		ctx := ctxFn(req)
		ctx = api.WithNamespace(ctx, namespace)

		body, err := readBody(req.Request)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		obj := r.New()
		if err := codec.DecodeInto(body, obj); err != nil {
			err = transformDecodeError(typer, err, obj, body)
			errorJSON(err, codec, w)
			return
		}

		if err := checkName(obj, name, namespace, namer); err != nil {
			errorJSON(err, codec, w)
			return
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

		if err := setSelfLink(result, req, namer); err != nil {
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
func DeleteResource(r RESTGracefulDeleter, checkBody bool, ctxFn ContextFunc, namer ScopeNamer, codec runtime.Codec, resource, kind string, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, name, err := namer.Name(req)
		if err != nil {
			errorJSON(err, codec, w)
			return
		}
		ctx := ctxFn(req)
		if len(namespace) > 0 {
			ctx = api.WithNamespace(ctx, namespace)
		}

		options := &api.DeleteOptions{}
		if checkBody {
			body, err := readBody(req.Request)
			if err != nil {
				errorJSON(err, codec, w)
				return
			}
			if len(body) > 0 {
				if err := codec.DecodeInto(body, options); err != nil {
					errorJSON(err, codec, w)
					return
				}
			}
		}

		err = admit.Admit(admission.NewAttributesRecord(nil, namespace, resource, "DELETE"))
		if err != nil {
			errorJSON(err, codec, w)
			return
		}

		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			return r.Delete(ctx, name, options)
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
				if err := setSelfLink(result, req, namer); err != nil {
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
	// these channels need to be buffered to prevent the goroutine below from hanging indefinitely
	// when the select statement reads something other than the one the goroutine sends on.
	ch := make(chan runtime.Object, 1)
	errCh := make(chan error, 1)
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

// transformDecodeError adds additional information when a decode fails.
func transformDecodeError(typer runtime.ObjectTyper, baseErr error, into runtime.Object, body []byte) error {
	_, kind, err := typer.ObjectVersionAndKind(into)
	if err != nil {
		return err
	}
	if version, dataKind, err := typer.DataVersionAndKind(body); err == nil && len(dataKind) > 0 {
		return errors.NewBadRequest(fmt.Sprintf("%s in version %s cannot be handled as a %s: %v", dataKind, version, kind, baseErr))
	}
	return errors.NewBadRequest(fmt.Sprintf("the object provided is unrecognized (must be of type %s): %v", kind, baseErr))
}

// setSelfLink sets the self link of an object (or the child items in a list) to the base URL of the request
// plus the path and query generated by the provided linkFunc
func setSelfLink(obj runtime.Object, req *restful.Request, namer ScopeNamer) error {
	// TODO: SelfLink generation should return a full URL?
	path, query, err := namer.GenerateLink(req, obj)
	if err == errEmptyName {
		return nil
	}
	if err != nil {
		return err
	}

	newURL := *req.Request.URL
	// use only canonical paths
	newURL.Path = gpath.Clean(path)
	newURL.RawQuery = query
	newURL.Fragment = ""

	return namer.SetSelfLink(obj, newURL.String())
}

// checkName checks the provided name against the request
func checkName(obj runtime.Object, name, namespace string, namer ScopeNamer) error {
	if objNamespace, objName, err := namer.ObjectName(obj); err == nil {
		if objName != name {
			return errors.NewBadRequest("the name of the object does not match the name on the URL")
		}
		if len(namespace) > 0 {
			if len(objNamespace) > 0 && objNamespace != namespace {
				return errors.NewBadRequest("the namespace of the object does not match the namespace on the request")
			}
		}
	}
	return nil
}

// setListSelfLink sets the self link of a list to the base URL, then sets the self links
// on all child objects returned.
func setListSelfLink(obj runtime.Object, req *restful.Request, namer ScopeNamer) error {
	if !runtime.IsListType(obj) {
		return nil
	}

	// TODO: List SelfLink generation should return a full URL?
	path, query, err := namer.GenerateListLink(req)
	if err != nil {
		return err
	}
	newURL := *req.Request.URL
	newURL.Path = path
	newURL.RawQuery = query
	// use the path that got us here
	newURL.Fragment = ""
	if err := namer.SetSelfLink(obj, newURL.String()); err != nil {
		glog.V(4).Infof("Unable to set self link on object: %v", err)
	}

	// Set self-link of objects in the list.
	items, err := runtime.ExtractList(obj)
	if err != nil {
		return err
	}
	for i := range items {
		if err := setSelfLink(items[i], req, namer); err != nil {
			return err
		}
	}
	return runtime.SetList(obj, items)

}
