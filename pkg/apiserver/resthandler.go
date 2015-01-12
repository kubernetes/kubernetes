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
	"path"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

	"github.com/golang/glog"
)

// RESTHandler implements HTTP verbs on a set of RESTful resources identified by name.
type RESTHandler struct {
	storage          map[string]RESTStorage
	codec            runtime.Codec
	canonicalPrefix  string
	selfLinker       runtime.SelfLinker
	ops              *Operations
	asyncOpWait      time.Duration
	admissionControl admission.Interface
}

// ServeHTTP handles requests to all RESTStorage objects.
func (h *RESTHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	namespace, kind, parts, err := KindAndNamespace(req)
	if err != nil {
		notFound(w, req)
		return
	}
	storage, ok := h.storage[kind]
	if !ok {
		notFound(w, req)
		return
	}

	h.handleRESTStorage(parts, req, w, storage, namespace, kind)
}

// Sets the SelfLink field of the object.
func (h *RESTHandler) setSelfLink(obj runtime.Object, req *http.Request) error {
	newURL := *req.URL
	newURL.Path = path.Join(h.canonicalPrefix, req.URL.Path)
	newURL.RawQuery = ""
	newURL.Fragment = ""
	namespace, err := h.selfLinker.Namespace(obj)
	if err != nil {
		return err
	}

	// we need to add namespace as a query param, if its not in the resource path
	if len(namespace) > 0 {
		parts := splitPath(req.URL.Path)
		if parts[0] != "ns" {
			query := newURL.Query()
			query.Set("namespace", namespace)
			newURL.RawQuery = query.Encode()
		}
	}

	err = h.selfLinker.SetSelfLink(obj, newURL.String())
	if err != nil {
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
		if err := h.setSelfLinkAddName(items[i], req); err != nil {
			return err
		}
	}
	return runtime.SetList(obj, items)
}

// Like setSelfLink, but appends the object's name.
func (h *RESTHandler) setSelfLinkAddName(obj runtime.Object, req *http.Request) error {
	name, err := h.selfLinker.Name(obj)
	if err != nil {
		return err
	}
	namespace, err := h.selfLinker.Namespace(obj)
	if err != nil {
		return err
	}
	newURL := *req.URL
	newURL.Path = path.Join(h.canonicalPrefix, req.URL.Path, name)
	newURL.RawQuery = ""
	newURL.Fragment = ""
	// we need to add namespace as a query param, if its not in the resource path
	if len(namespace) > 0 {
		parts := splitPath(req.URL.Path)
		if parts[0] != "ns" {
			query := newURL.Query()
			query.Set("namespace", namespace)
			newURL.RawQuery = query.Encode()
		}
	}
	return h.selfLinker.SetSelfLink(obj, newURL.String())
}

// curry adapts either of the self link setting functions into a function appropriate for operation's hook.
func curry(f func(runtime.Object, *http.Request) error, req *http.Request) func(RESTResult) {
	return func(obj RESTResult) {
		if err := f(obj.Object, req); err != nil {
			glog.Errorf("unable to set self link for %#v: %v", obj, err)
		}
	}
}

// handleRESTStorage is the main dispatcher for a storage object.  It switches on the HTTP method, and then
// on path length, according to the following table:
//   Method     Path          Action
//   GET        /foo          list
//   GET        /foo/bar      get 'bar'
//   POST       /foo          create
//   PUT        /foo/bar      update 'bar'
//   DELETE     /foo/bar      delete 'bar'
// Returns 404 if the method/pattern doesn't match one of these entries
// The s accepts several query parameters:
//    sync=[false|true] Synchronous request (only applies to create, update, delete operations)
//    timeout=<duration> Timeout for synchronous requests, only applies if sync=true
//    labels=<label-selector> Used for filtering list operations
func (h *RESTHandler) handleRESTStorage(parts []string, req *http.Request, w http.ResponseWriter, storage RESTStorage, namespace, kind string) {
	ctx := api.WithNamespace(api.NewContext(), namespace)
	sync := req.URL.Query().Get("sync") == "true"
	timeout := parseTimeout(req.URL.Query().Get("timeout"))
	switch req.Method {
	case "GET":
		switch len(parts) {
		case 1:
			label, err := labels.ParseSelector(req.URL.Query().Get("labels"))
			if err != nil {
				errorJSON(err, h.codec, w)
				return
			}
			field, err := labels.ParseSelector(req.URL.Query().Get("fields"))
			if err != nil {
				errorJSON(err, h.codec, w)
				return
			}
			lister, ok := storage.(RESTLister)
			if !ok {
				errorJSON(errors.NewMethodNotSupported(kind, "list"), h.codec, w)
				return
			}
			list, err := lister.List(ctx, label, field)
			if err != nil {
				errorJSON(err, h.codec, w)
				return
			}
			if err := h.setSelfLink(list, req); err != nil {
				errorJSON(err, h.codec, w)
				return
			}
			writeJSON(http.StatusOK, h.codec, list, w)
		case 2:
			getter, ok := storage.(RESTGetter)
			if !ok {
				errorJSON(errors.NewMethodNotSupported(kind, "get"), h.codec, w)
				return
			}
			item, err := getter.Get(ctx, parts[1])
			if err != nil {
				errorJSON(err, h.codec, w)
				return
			}
			if err := h.setSelfLink(item, req); err != nil {
				errorJSON(err, h.codec, w)
				return
			}
			writeJSON(http.StatusOK, h.codec, item, w)
		default:
			notFound(w, req)
		}

	case "POST":
		if len(parts) != 1 {
			notFound(w, req)
			return
		}
		creater, ok := storage.(RESTCreater)
		if !ok {
			errorJSON(errors.NewMethodNotSupported(kind, "create"), h.codec, w)
			return
		}

		body, err := readBody(req)
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}
		obj := storage.New()
		err = h.codec.DecodeInto(body, obj)
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}

		// invoke admission control
		err = h.admissionControl.Admit(admission.NewAttributesRecord(obj, namespace, parts[0], "CREATE"))
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}

		out, err := creater.Create(ctx, obj)
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}
		op := h.createOperation(out, sync, timeout, curry(h.setSelfLinkAddName, req))
		h.finishReq(op, req, w)

	case "DELETE":
		if len(parts) != 2 {
			notFound(w, req)
			return
		}
		deleter, ok := storage.(RESTDeleter)
		if !ok {
			errorJSON(errors.NewMethodNotSupported(kind, "delete"), h.codec, w)
			return
		}

		// invoke admission control
		err := h.admissionControl.Admit(admission.NewAttributesRecord(nil, namespace, parts[0], "DELETE"))
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}

		out, err := deleter.Delete(ctx, parts[1])
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}
		op := h.createOperation(out, sync, timeout, nil)
		h.finishReq(op, req, w)

	case "PUT":
		if len(parts) != 2 {
			notFound(w, req)
			return
		}
		updater, ok := storage.(RESTUpdater)
		if !ok {
			errorJSON(errors.NewMethodNotSupported(kind, "create"), h.codec, w)
			return
		}

		body, err := readBody(req)
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}
		obj := storage.New()
		err = h.codec.DecodeInto(body, obj)
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}

		// invoke admission control
		err = h.admissionControl.Admit(admission.NewAttributesRecord(obj, namespace, parts[0], "UPDATE"))
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}

		out, err := updater.Update(ctx, obj)
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}
		op := h.createOperation(out, sync, timeout, curry(h.setSelfLink, req))
		h.finishReq(op, req, w)

	default:
		notFound(w, req)
	}
}

// createOperation creates an operation to process a channel response.
func (h *RESTHandler) createOperation(out <-chan RESTResult, sync bool, timeout time.Duration, onReceive func(RESTResult)) *Operation {
	op := h.ops.NewOperation(out, onReceive)
	if sync {
		op.WaitFor(timeout)
	} else if h.asyncOpWait != 0 {
		op.WaitFor(h.asyncOpWait)
	}
	return op
}

// finishReq finishes up a request, waiting until the operation finishes or, after a timeout, creating an
// Operation to receive the result and returning its ID down the writer.
func (h *RESTHandler) finishReq(op *Operation, req *http.Request, w http.ResponseWriter) {
	result, complete := op.StatusOrResult()
	obj := result.Object
	if complete {
		status := http.StatusOK
		if result.Created {
			status = http.StatusCreated
		}
		switch stat := obj.(type) {
		case *api.Status:
			if stat.Code != 0 {
				status = stat.Code
			}
		}
		writeJSON(status, h.codec, obj, w)
	} else {
		writeJSON(http.StatusAccepted, h.codec, obj, w)
	}
}
