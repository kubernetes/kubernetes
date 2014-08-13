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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/httplog"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

type RESTHandler struct {
	storage     map[string]RESTStorage
	codec       Codec
	ops         *Operations
	asyncOpWait time.Duration
}

// ServeHTTP handles requests to all RESTStorage objects.
func (h *RESTHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	parts := splitPath(req.URL.Path)
	if len(parts) < 1 {
		notFound(w, req)
		return
	}
	storage := h.storage[parts[0]]
	if storage == nil {
		httplog.LogOf(w).Addf("'%v' has no storage object", parts[0])
		notFound(w, req)
		return
	}

	h.handleRESTStorage(parts, req, w, storage)
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
func (h *RESTHandler) handleRESTStorage(parts []string, req *http.Request, w http.ResponseWriter, storage RESTStorage) {
	sync := req.URL.Query().Get("sync") == "true"
	timeout := parseTimeout(req.URL.Query().Get("timeout"))
	switch req.Method {
	case "GET":
		switch len(parts) {
		case 1:
			selector, err := labels.ParseSelector(req.URL.Query().Get("labels"))
			if err != nil {
				errorJSON(err, h.codec, w)
				return
			}
			list, err := storage.List(selector)
			if err != nil {
				errorJSON(err, h.codec, w)
				return
			}
			writeJSON(http.StatusOK, h.codec, list, w)
		case 2:
			item, err := storage.Get(parts[1])
			if err != nil {
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
		out, err := storage.Create(obj)
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}
		op := h.createOperation(out, sync, timeout)
		h.finishReq(op, w)

	case "DELETE":
		if len(parts) != 2 {
			notFound(w, req)
			return
		}
		out, err := storage.Delete(parts[1])
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}
		op := h.createOperation(out, sync, timeout)
		h.finishReq(op, w)

	case "PUT":
		if len(parts) != 2 {
			notFound(w, req)
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
		out, err := storage.Update(obj)
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}
		op := h.createOperation(out, sync, timeout)
		h.finishReq(op, w)

	default:
		notFound(w, req)
	}
}

// createOperation creates an operation to process a channel response
func (h *RESTHandler) createOperation(out <-chan interface{}, sync bool, timeout time.Duration) *Operation {
	op := h.ops.NewOperation(out)
	if sync {
		op.WaitFor(timeout)
	} else if h.asyncOpWait != 0 {
		op.WaitFor(h.asyncOpWait)
	}
	return op
}

// finishReq finishes up a request, waiting until the operation finishes or, after a timeout, creating an
// Operation to receive the result and returning its ID down the writer.
func (h *RESTHandler) finishReq(op *Operation, w http.ResponseWriter) {
	obj, complete := op.StatusOrResult()
	if complete {
		status := http.StatusOK
		switch stat := obj.(type) {
		case api.Status:
			httplog.LogOf(w).Addf("programmer error: use *api.Status as a result, not api.Status.")
			if stat.Code != 0 {
				status = stat.Code
			}
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
