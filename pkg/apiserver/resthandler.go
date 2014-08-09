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
	storage RESTStorage
	codec   Codec
	// Optional, will track completion status of long running jobs
	ops         *Operations
	asyncOpWait time.Duration
	// Optional, used when ops is nil to limit how long requests can run
	maxWait time.Duration
}

// NewRESTHandler creates a handler capable of serving RESTful actions on a storage object
func NewRESTHandler(storage RESTStorage, codec Codec, ops *Operations) http.Handler {
	return &RESTHandler{
		storage: storage,
		codec:   codec,
		ops:     ops,
		// Delay just long enough to handle most simple write operations
		asyncOpWait: time.Millisecond * 25,
		maxWait:     time.Second * 10,
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
// The h accepts several query parameters:
//    sync=[false|true] Synchronous request (only applies to create, update, delete operations)
//    timeout=<duration> Timeout for synchronous requests, only applies if sync=true
//    labels=<label-selector> Used for filtering list operations
func (h *RESTHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	parts := splitPath(req.URL.Path)
	storage := h.storage
	sync := req.URL.Query().Get("sync") == "true"
	timeout := parseTimeout(req.URL.Query().Get("timeout"))
	switch req.Method {
	case "GET":
		switch len(parts) {
		case 0:
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
		case 1:
			item, err := storage.Get(parts[0])
			if err != nil {
				errorJSON(err, h.codec, w)
				return
			}
			writeJSON(http.StatusOK, h.codec, item, w)
		default:
			notFound(w, req)
		}

	case "POST":
		if len(parts) != 0 {
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
		obj, complete := h.createOperation(out, sync, timeout)
		h.finishReq(obj, complete, w)

	case "DELETE":
		if len(parts) != 1 {
			notFound(w, req)
			return
		}
		out, err := storage.Delete(parts[0])
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}
		obj, complete := h.createOperation(out, sync, timeout)
		h.finishReq(obj, complete, w)

	case "PUT":
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
		out, err := storage.Update(obj)
		if err != nil {
			errorJSON(err, h.codec, w)
			return
		}
		obj, complete := h.createOperation(out, sync, timeout)
		h.finishReq(obj, complete, w)

	default:
		notFound(w, req)
	}
}

// createOperation creates an operation to process a channel response
func (h *RESTHandler) createOperation(out <-chan interface{}, sync bool, timeout time.Duration) (interface{}, bool) {
	if h.ops == nil {
		if !sync && h.asyncOpWait != 0 {
			timeout = h.asyncOpWait
		}
		if timeout == 0 {
			timeout = h.maxWait
		}
		select {
		case obj := <-out:
			return obj, true
		case <-time.After(timeout):
			return &api.Status{
				Status:  api.StatusWorking,
				Reason:  api.ReasonTypeWorking,
				Message: "Operation is still in progress",
			}, false
		}
	}
	op := h.ops.NewOperation(out)
	if sync {
		op.WaitFor(timeout)
	} else if h.asyncOpWait != 0 {
		op.WaitFor(h.asyncOpWait)
	}
	return op.StatusOrResult()
}

// finishReq finishes up a request, waiting until the operation finishes or, after a timeout, creating an
// Operation to receive the result and returning its ID down the writer.
func (h *RESTHandler) finishReq(obj interface{}, complete bool, w http.ResponseWriter) {
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
