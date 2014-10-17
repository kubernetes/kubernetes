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
	"encoding/json"
	"io/ioutil"
	"net/http"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/healthz"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/golang/glog"
)

// mux is an object that can register http handlers.
type mux interface {
	Handle(pattern string, handler http.Handler)
	HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request))
}

// defaultAPIServer exposes nested objects for testability.
type defaultAPIServer struct {
	http.Handler
	group *APIGroup
}

const (
	StatusUnprocessableEntity = 422
)

// Handle returns a Handler function that exposes the provided storage interfaces
// as RESTful resources at prefix, serialized by codec, and also includes the support
// http resources.
func Handle(storage map[string]RESTStorage, codec runtime.Codec, prefix string, selfLinker runtime.SelfLinker) http.Handler {
	group := NewAPIGroup(storage, codec, prefix, selfLinker)

	mux := http.NewServeMux()
	group.InstallREST(mux, prefix)
	InstallSupport(mux)
	return &defaultAPIServer{mux, group}
}

// APIGroup is a http.Handler that exposes multiple RESTStorage objects
// It handles URLs of the form:
// /${storage_key}[/${object_name}]
// Where 'storage_key' points to a RESTStorage object stored in storage.
//
// TODO: consider migrating this to go-restful which is a more full-featured version of the same thing.
type APIGroup struct {
	handler RESTHandler
}

// NewAPIGroup returns an object that will serve a set of REST resources and their
// associated operations.  The provided codec controls serialization and deserialization.
// This is a helper method for registering multiple sets of REST handlers under different
// prefixes onto a server.
// TODO: add multitype codec serialization
func NewAPIGroup(storage map[string]RESTStorage, codec runtime.Codec, canonicalPrefix string, selfLinker runtime.SelfLinker) *APIGroup {
	return &APIGroup{RESTHandler{
		storage:         storage,
		codec:           codec,
		canonicalPrefix: canonicalPrefix,
		selfLinker:      selfLinker,
		ops:             NewOperations(),
		// Delay just long enough to handle most simple write operations
		asyncOpWait: time.Millisecond * 25,
	}}
}

// InstallREST registers the REST handlers (storage, watch, and operations) into a mux.
// It is expected that the provided prefix will serve all operations. Path MUST NOT end
// in a slash.
func (g *APIGroup) InstallREST(mux mux, paths ...string) {
	restHandler := &g.handler
	watchHandler := &WatchHandler{g.handler.storage, g.handler.codec}
	redirectHandler := &RedirectHandler{g.handler.storage, g.handler.codec}
	opHandler := &OperationHandler{g.handler.ops, g.handler.codec}

	servers := map[string]string{
		"controller-manager": "127.0.0.1:10252",
		"scheduler":          "127.0.0.1:10251",
		// TODO: Add minion health checks here too.
	}
	validator, err := NewValidator(servers)
	if err != nil {
		glog.Errorf("failed to set up validator: %v", err)
		validator = nil
	}
	for _, prefix := range paths {
		prefix = strings.TrimRight(prefix, "/")
		proxyHandler := &ProxyHandler{prefix + "/proxy/", g.handler.storage, g.handler.codec}
		mux.Handle(prefix+"/", http.StripPrefix(prefix, restHandler))
		mux.Handle(prefix+"/watch/", http.StripPrefix(prefix+"/watch/", watchHandler))
		mux.Handle(prefix+"/proxy/", http.StripPrefix(prefix+"/proxy/", proxyHandler))
		mux.Handle(prefix+"/redirect/", http.StripPrefix(prefix+"/redirect/", redirectHandler))
		mux.Handle(prefix+"/operations", http.StripPrefix(prefix+"/operations", opHandler))
		mux.Handle(prefix+"/operations/", http.StripPrefix(prefix+"/operations/", opHandler))
		if validator != nil {
			mux.Handle(prefix+"/validate", validator)
		}
	}
}

// InstallSupport registers the APIServer support functions into a mux.
func InstallSupport(mux mux) {
	healthz.InstallHandler(mux)
	mux.HandleFunc("/version", handleVersion)
	mux.HandleFunc("/", handleIndex)
}

// InstallLogsSupport registers the APIServer log support function into a mux.
func InstallLogsSupport(mux mux) {
	mux.Handle("/logs/", http.StripPrefix("/logs/", http.FileServer(http.Dir("/var/log/"))))
}

// handleVersion writes the server's version information.
func handleVersion(w http.ResponseWriter, req *http.Request) {
	writeRawJSON(http.StatusOK, version.Get(), w)
}

// writeJSON renders an object as JSON to the response.
func writeJSON(statusCode int, codec runtime.Codec, object runtime.Object, w http.ResponseWriter) {
	output, err := codec.Encode(object)
	if err != nil {
		errorJSON(err, codec, w)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(output)
}

// errorJSON renders an error to the response.
func errorJSON(err error, codec runtime.Codec, w http.ResponseWriter) {
	status := errToAPIStatus(err)
	writeJSON(status.Code, codec, status, w)
}

// writeRawJSON writes a non-API object in JSON.
func writeRawJSON(statusCode int, object interface{}, w http.ResponseWriter) {
	output, err := json.Marshal(object)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(output)
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

func readBody(req *http.Request) ([]byte, error) {
	defer req.Body.Close()
	return ioutil.ReadAll(req.Body)
}

// splitPath returns the segments for a URL path.
func splitPath(path string) []string {
	path = strings.Trim(path, "/")
	if path == "" {
		return []string{}
	}
	return strings.Split(path, "/")
}
