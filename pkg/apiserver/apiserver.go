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
	"io/ioutil"
	"net/http"
	"net/url"
	"runtime/debug"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// RESTStorage is a generic interface for RESTful storage services
type RESTStorage interface {
	List(labels.Selector) (interface{}, error)
	Get(id string) (interface{}, error)
	Delete(id string) (<-chan interface{}, error)
	Extract(body []byte) (interface{}, error)
	Create(interface{}) (<-chan interface{}, error)
	Update(interface{}) (<-chan interface{}, error)
}

// WorkFunc is used to perform any time consuming work for an api call, after
// the input has been validated. Pass one of these to MakeAsync to create an
// appropriate return value for the Update, Delete, and Create methods.
type WorkFunc func() (result interface{}, err error)

// MakeAsync takes a function and executes it, delivering the result in the way required
// by RESTStorage's Update, Delete, and Create methods.
func MakeAsync(fn WorkFunc) <-chan interface{} {
	channel := make(chan interface{})
	go func() {
		defer util.HandleCrash()
		obj, err := fn()
		if err != nil {
			channel <- &api.Status{
				Status:  api.StatusFailure,
				Details: err.Error(),
			}
		} else {
			channel <- obj
		}
		// 'close' is used to signal that no further values will
		// be written to the channel. Not strictly necessary, but
		// also won't hurt.
		close(channel)
	}()
	return channel
}

// ApiServer is an HTTPHandler that delegates to RESTStorage objects.
// It handles URLs of the form:
// ${prefix}/${storage_key}[/${object_name}]
// Where 'prefix' is an arbitrary string, and 'storage_key' points to a RESTStorage object stored in storage.
//
// TODO: consider migrating this to go-restful which is a more full-featured version of the same thing.
type ApiServer struct {
	prefix    string
	storage   map[string]RESTStorage
	ops       *Operations
	logserver http.Handler
}

// New creates a new ApiServer object.
// 'storage' contains a map of handlers.
// 'prefix' is the hosting path prefix.
func New(storage map[string]RESTStorage, prefix string) *ApiServer {
	return &ApiServer{
		storage:   storage,
		prefix:    prefix,
		ops:       NewOperations(),
		logserver: http.StripPrefix("/logs/", http.FileServer(http.Dir("/var/log/"))),
	}
}

func (server *ApiServer) handleIndex(w http.ResponseWriter) {
	w.WriteHeader(http.StatusOK)
	// TODO: serve this out of a file?
	data := "<html><body>Welcome to Kubernetes</body></html>"
	fmt.Fprint(w, data)
}

// HTTP Handler interface
func (server *ApiServer) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	defer func() {
		if x := recover(); x != nil {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprint(w, "apiserver panic. Look in log for details.")
			glog.Infof("ApiServer panic'd on %v %v: %#v\n%s\n", req.Method, req.RequestURI, x, debug.Stack())
		}
	}()
	logger := MakeLogged(req, w)
	w = logger
	defer logger.Log()
	url, err := url.ParseRequestURI(req.RequestURI)
	if err != nil {
		server.error(err, w)
		return
	}
	if url.Path == "/index.html" || url.Path == "/" || url.Path == "" {
		server.handleIndex(w)
		return
	}
	if strings.HasPrefix(url.Path, "/logs/") {
		server.logserver.ServeHTTP(w, req)
		return
	}
	if !strings.HasPrefix(url.Path, server.prefix) {
		server.notFound(req, w)
		return
	}
	requestParts := strings.Split(url.Path[len(server.prefix):], "/")[1:]
	if len(requestParts) < 1 {
		server.notFound(req, w)
		return
	}
	if requestParts[0] == "operations" {
		server.handleOperationRequest(requestParts[1:], w, req)
		return
	}
	storage := server.storage[requestParts[0]]
	if storage == nil {
		logger.Addf("'%v' has no storage object", requestParts[0])
		server.notFound(req, w)
		return
	} else {
		server.handleREST(requestParts, url, req, w, storage)
	}
}

func (server *ApiServer) notFound(req *http.Request, w http.ResponseWriter) {
	w.WriteHeader(http.StatusNotFound)
	fmt.Fprintf(w, "Not Found: %#v", req)
}

func (server *ApiServer) write(statusCode int, object interface{}, w http.ResponseWriter) {
	w.WriteHeader(statusCode)
	output, err := api.Encode(object)
	if err != nil {
		server.error(err, w)
		return
	}
	w.Write(output)
}

func (server *ApiServer) error(err error, w http.ResponseWriter) {
	w.WriteHeader(500)
	fmt.Fprintf(w, "Internal Error: %#v", err)
}

func (server *ApiServer) readBody(req *http.Request) ([]byte, error) {
	defer req.Body.Close()
	body, err := ioutil.ReadAll(req.Body)
	return body, err
}

// finishReq finishes up a request, waiting until the operation finishes or, after a timeout, creating an
// Operation to recieve the result and returning its ID down the writer.
func (server *ApiServer) finishReq(out <-chan interface{}, sync bool, timeout time.Duration, w http.ResponseWriter) {
	op := server.ops.NewOperation(out)
	if sync {
		op.WaitFor(timeout)
	}
	obj, complete := op.StatusOrResult()
	if complete {
		server.write(http.StatusOK, obj, w)
	} else {
		server.write(http.StatusAccepted, obj, w)
	}
}

func parseTimeout(str string) time.Duration {
	if str != "" {
		timeout, err := time.ParseDuration(str)
		if err == nil {
			return timeout
		}
		glog.Errorf("Failed to parse: %#v '%s'", err, str)
	}
	return 30 * time.Second
}

// handleREST is the main dispatcher for the server.  It switches on the HTTP method, and then
// on path length, according to the following table:
//   Method     Path          Action
//   GET        /foo          list
//   GET        /foo/bar      get 'bar'
//   POST       /foo          create
//   PUT        /foo/bar      update 'bar'
//   DELETE     /foo/bar      delete 'bar'
// Returns 404 if the method/pattern doesn't match one of these entries
// The server accepts several query parameters:
//    sync=[false|true] Synchronous request (only applies to create, update, delete operations)
//    timeout=<duration> Timeout for synchronous requests, only applies if sync=true
//    labels=<label-selector> Used for filtering list operations
func (server *ApiServer) handleREST(parts []string, requestUrl *url.URL, req *http.Request, w http.ResponseWriter, storage RESTStorage) {
	sync := requestUrl.Query().Get("sync") == "true"
	timeout := parseTimeout(requestUrl.Query().Get("timeout"))
	switch req.Method {
	case "GET":
		switch len(parts) {
		case 1:
			selector, err := labels.ParseSelector(requestUrl.Query().Get("labels"))
			if err != nil {
				server.error(err, w)
				return
			}
			list, err := storage.List(selector)
			if err != nil {
				server.error(err, w)
				return
			}
			server.write(http.StatusOK, list, w)
		case 2:
			item, err := storage.Get(parts[1])
			if err != nil {
				server.error(err, w)
				return
			}
			if item == nil {
				server.notFound(req, w)
				return
			}
			server.write(http.StatusOK, item, w)
		default:
			server.notFound(req, w)
		}
	case "POST":
		if len(parts) != 1 {
			server.notFound(req, w)
			return
		}
		body, err := server.readBody(req)
		if err != nil {
			server.error(err, w)
			return
		}
		obj, err := storage.Extract(body)
		if err != nil {
			server.error(err, w)
			return
		}
		out, err := storage.Create(obj)
		if err != nil {
			server.error(err, w)
			return
		}
		server.finishReq(out, sync, timeout, w)
	case "DELETE":
		if len(parts) != 2 {
			server.notFound(req, w)
			return
		}
		out, err := storage.Delete(parts[1])
		if err != nil {
			server.error(err, w)
			return
		}
		server.finishReq(out, sync, timeout, w)
	case "PUT":
		if len(parts) != 2 {
			server.notFound(req, w)
			return
		}
		body, err := server.readBody(req)
		if err != nil {
			server.error(err, w)
		}
		obj, err := storage.Extract(body)
		if err != nil {
			server.error(err, w)
			return
		}
		out, err := storage.Update(obj)
		if err != nil {
			server.error(err, w)
			return
		}
		server.finishReq(out, sync, timeout, w)
	default:
		server.notFound(req, w)
	}
}

func (server *ApiServer) handleOperationRequest(parts []string, w http.ResponseWriter, req *http.Request) {
	if req.Method != "GET" {
		server.notFound(req, w)
	}
	if len(parts) == 0 {
		// List outstanding operations.
		list := server.ops.List()
		server.write(http.StatusOK, list, w)
		return
	}

	op := server.ops.Get(parts[0])
	if op == nil {
		server.notFound(req, w)
	}

	obj, complete := op.StatusOrResult()
	if complete {
		server.write(http.StatusOK, obj, w)
	} else {
		server.write(http.StatusAccepted, obj, w)
	}
}
