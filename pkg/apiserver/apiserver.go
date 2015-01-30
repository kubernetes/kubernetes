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
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"path"
	"reflect"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/admission"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/healthz"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"

	"github.com/emicklei/go-restful"
	"github.com/golang/glog"
)

// mux is an object that can register http handlers.
type Mux interface {
	Handle(pattern string, handler http.Handler)
	HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request))
}

// defaultAPIServer exposes nested objects for testability.
type defaultAPIServer struct {
	http.Handler
	group *APIGroupVersion
}

// Handle returns a Handler function that exposes the provided storage interfaces
// as RESTful resources at prefix, serialized by codec, and also includes the support
// http resources.
// Note: This method is used only in tests.
func Handle(storage map[string]RESTStorage, codec runtime.Codec, root string, version string, selfLinker runtime.SelfLinker, admissionControl admission.Interface, mapper meta.RESTMapper) http.Handler {
	prefix := root + "/" + version
	group := NewAPIGroupVersion(storage, codec, prefix, selfLinker, admissionControl, mapper)
	container := restful.NewContainer()
	mux := container.ServeMux
	group.InstallREST(container, mux, root, version)
	ws := new(restful.WebService)
	InstallSupport(mux, ws)
	container.Add(ws)
	return &defaultAPIServer{mux, group}
}

// TODO: This is a whole API version right now. Maybe should rename it.
// APIGroupVersion is a http.Handler that exposes multiple RESTStorage objects
// It handles URLs of the form:
// /${storage_key}[/${object_name}]
// Where 'storage_key' points to a RESTStorage object stored in storage.
//
// TODO: consider migrating this to go-restful which is a more full-featured version of the same thing.
type APIGroupVersion struct {
	handler RESTHandler
	mapper  meta.RESTMapper
}

// NewAPIGroupVersion returns an object that will serve a set of REST resources and their
// associated operations.  The provided codec controls serialization and deserialization.
// This is a helper method for registering multiple sets of REST handlers under different
// prefixes onto a server.
// TODO: add multitype codec serialization
func NewAPIGroupVersion(storage map[string]RESTStorage, codec runtime.Codec, canonicalPrefix string, selfLinker runtime.SelfLinker, admissionControl admission.Interface, mapper meta.RESTMapper) *APIGroupVersion {
	return &APIGroupVersion{
		handler: RESTHandler{
			storage:          storage,
			codec:            codec,
			canonicalPrefix:  canonicalPrefix,
			selfLinker:       selfLinker,
			ops:              NewOperations(),
			admissionControl: admissionControl,
		},
		mapper: mapper,
	}
}

// This magic incantation returns *ptrToObject for an arbitrary pointer
func indirectArbitraryPointer(ptrToObject interface{}) interface{} {
	return reflect.Indirect(reflect.ValueOf(ptrToObject)).Interface()
}

func registerResourceHandlers(ws *restful.WebService, version string, path string, storage RESTStorage, h restful.RouteFunction, mapper meta.RESTMapper) error {
	object := storage.New()
	_, kind, err := api.Scheme.ObjectVersionAndKind(object)
	if err != nil {
		return err
	}
	versionedPtr, err := api.Scheme.New(version, kind)
	if err != nil {
		return err
	}
	versionedObject := indirectArbitraryPointer(versionedPtr)

	var versionedList interface{}
	if lister, ok := storage.(RESTLister); ok {
		list := lister.NewList()
		_, listKind, err := api.Scheme.ObjectVersionAndKind(list)
		versionedListPtr, err := api.Scheme.New(version, listKind)
		if err != nil {
			return err
		}
		versionedList = indirectArbitraryPointer(versionedListPtr)
	}

	mapping, err := mapper.RESTMapping(kind, version)
	if err != nil {
		glog.V(1).Infof("OH NOES kind %s version %s err: %v", kind, version, err)
		return err
	}

	// names are always in path
	nameParam := ws.PathParameter("name", "name of the "+kind).DataType("string")

	// what verbs are supported by the storage, used to know what verbs we support per path
	storageVerbs := map[string]bool{}
	if _, ok := storage.(RESTCreater); ok {
		storageVerbs["RESTCreater"] = true
	}
	if _, ok := storage.(RESTLister); ok {
		storageVerbs["RESTLister"] = true
	}
	if _, ok := storage.(RESTGetter); ok {
		storageVerbs["RESTGetter"] = true
	}
	if _, ok := storage.(RESTDeleter); ok {
		storageVerbs["RESTDeleter"] = true
	}
	if _, ok := storage.(RESTUpdater); ok {
		storageVerbs["RESTUpdater"] = true
	}

	// path to verbs takes a path, and set of http verbs we should claim to support on this path
	// given current url formats, this is scope specific
	// for example, in v1beta3 url styles we would support the following:
	// /ns/{namespace}/{resource} = []{POST, GET}  // list resources in this namespace scope
	// /ns/{namespace}/{resource}/{name} = []{GET, PUT, DELETE} // handle an individual resource
	// /{resource} = []{GET} // list resources across all namespaces
	pathToVerbs := map[string][]string{}
	// similar to the above, it maps the ordered set of parameters that need to be documented
	pathToParam := map[string][]*restful.Parameter{}
	// pathToStorageVerb lets us distinguish between RESTLister and RESTGetter on verb=GET
	pathToStorageVerb := map[string]string{}
	scope := mapping.Scope
	if scope.Name() != meta.RESTScopeNameNamespace {
		// non-namespaced resources support a smaller set of resource paths
		resourcesPath := path
		resourcesPathVerbs := []string{}
		resourcesPathVerbs = appendStringIf(resourcesPathVerbs, "POST", storageVerbs["RESTCreater"])
		resourcesPathVerbs = appendStringIf(resourcesPathVerbs, "GET", storageVerbs["RESTLister"])
		pathToVerbs[resourcesPath] = resourcesPathVerbs
		pathToParam[resourcesPath] = []*restful.Parameter{}
		pathToStorageVerb[resourcesPath] = "RESTLister"

		itemPath := resourcesPath + "/{name}"
		itemPathVerbs := []string{}
		itemPathVerbs = appendStringIf(itemPathVerbs, "GET", storageVerbs["RESTGetter"])
		itemPathVerbs = appendStringIf(itemPathVerbs, "PUT", storageVerbs["RESTUpdater"])
		itemPathVerbs = appendStringIf(itemPathVerbs, "DELETE", storageVerbs["RESTDeleter"])
		pathToVerbs[itemPath] = itemPathVerbs
		pathToParam[itemPath] = []*restful.Parameter{nameParam}
		pathToStorageVerb[itemPath] = "RESTGetter"

	} else {
		// v1beta3 format with namespace in path
		if scope.ParamPath() {
			namespaceParam := ws.PathParameter(scope.ParamName(), scope.ParamDescription()).DataType("string")

			resourceByNamespace := scope.ParamName() + "/{" + scope.ParamName() + "}/" + path
			resourceByNamespaceVerbs := []string{}
			resourceByNamespaceVerbs = appendStringIf(resourceByNamespaceVerbs, "POST", storageVerbs["RESTCreater"])
			resourceByNamespaceVerbs = appendStringIf(resourceByNamespaceVerbs, "GET", storageVerbs["RESTLister"])
			pathToVerbs[resourceByNamespace] = resourceByNamespaceVerbs
			pathToParam[resourceByNamespace] = []*restful.Parameter{namespaceParam}
			pathToStorageVerb[resourceByNamespace] = "RESTLister"

			itemPath := resourceByNamespace + "/{name}"
			itemPathVerbs := []string{}
			itemPathVerbs = appendStringIf(itemPathVerbs, "GET", storageVerbs["RESTGetter"])
			itemPathVerbs = appendStringIf(itemPathVerbs, "PUT", storageVerbs["RESTUpdater"])
			itemPathVerbs = appendStringIf(itemPathVerbs, "DELETE", storageVerbs["RESTDeleter"])
			pathToVerbs[itemPath] = itemPathVerbs
			pathToParam[itemPath] = []*restful.Parameter{namespaceParam, nameParam}
			pathToStorageVerb[itemPath] = "RESTGetter"

			listAcrossNamespace := path
			listAcrossNamespaceVerbs := []string{}
			listAcrossNamespaceVerbs = appendStringIf(listAcrossNamespaceVerbs, "GET", storageVerbs["RESTLister"])
			pathToVerbs[listAcrossNamespace] = listAcrossNamespaceVerbs
			pathToParam[listAcrossNamespace] = []*restful.Parameter{}
			pathToStorageVerb[listAcrossNamespace] = "RESTLister"

		} else {
			// v1beta1/v1beta2 format where namespace was a query parameter
			namespaceParam := ws.QueryParameter(scope.ParamName(), scope.ParamDescription()).DataType("string")

			resourcesPath := path
			resourcesPathVerbs := []string{}
			resourcesPathVerbs = appendStringIf(resourcesPathVerbs, "POST", storageVerbs["RESTCreater"])
			resourcesPathVerbs = appendStringIf(resourcesPathVerbs, "GET", storageVerbs["RESTLister"])
			pathToVerbs[resourcesPath] = resourcesPathVerbs
			pathToParam[resourcesPath] = []*restful.Parameter{namespaceParam}
			pathToStorageVerb[resourcesPath] = "RESTLister"

			itemPath := resourcesPath + "/{name}"
			itemPathVerbs := []string{}
			itemPathVerbs = appendStringIf(itemPathVerbs, "GET", storageVerbs["RESTGetter"])
			itemPathVerbs = appendStringIf(itemPathVerbs, "PUT", storageVerbs["RESTUpdater"])
			itemPathVerbs = appendStringIf(itemPathVerbs, "DELETE", storageVerbs["RESTDeleter"])
			pathToVerbs[itemPath] = itemPathVerbs
			pathToParam[itemPath] = []*restful.Parameter{namespaceParam, nameParam}
			pathToStorageVerb[itemPath] = "RESTGetter"
		}
	}

	// See github.com/emicklei/go-restful/blob/master/jsr311.go for routing logic
	// and status-code behavior
	for path, verbs := range pathToVerbs {
		glog.V(5).Infof("Installing version=/%s, kind=/%s, path=/%s", version, kind, path)

		params := pathToParam[path]
		for _, verb := range verbs {
			var route *restful.RouteBuilder
			switch verb {
			case "POST":
				route = ws.POST(path).To(h).
					Doc("create a " + kind).
					Operation("create" + kind).Reads(versionedObject)
			case "PUT":
				route = ws.PUT(path).To(h).
					Doc("update the specified " + kind).
					Operation("update" + kind).Reads(versionedObject)
			case "DELETE":
				route = ws.DELETE(path).To(h).
					Doc("delete the specified " + kind).
					Operation("delete" + kind)
			case "GET":
				doc := "read the specified " + kind
				op := "read" + kind
				if pathToStorageVerb[path] == "RESTLister" {
					doc = "list objects of kind " + kind
					op = "list" + kind
				}
				route = ws.GET(path).To(h).
					Doc(doc).
					Operation(op)
				if pathToStorageVerb[path] == "RESTLister" {
					route = route.Returns(http.StatusOK, "OK", versionedList)
				} else {
					route = route.Writes(versionedObject)
				}
			}
			for paramIndex := range params {
				route.Param(params[paramIndex])
			}
			ws.Route(route)
		}
	}
	return nil
}

// appendStringIf appends the value to the slice if shouldAdd is true
func appendStringIf(slice []string, value string, shouldAdd bool) []string {
	if shouldAdd {
		return append(slice, value)
	}
	return slice
}

// InstallREST registers the REST handlers (storage, watch, proxy and redirect) into a restful Container.
// It is expected that the provided path root prefix will serve all operations. Root MUST NOT end
// in a slash. A restful WebService is created for the group and version.
func (g *APIGroupVersion) InstallREST(container *restful.Container, mux Mux, root string, version string) error {
	prefix := path.Join(root, version)
	restHandler := &g.handler
	strippedHandler := http.StripPrefix(prefix, restHandler)
	watchHandler := &WatchHandler{
		storage:         g.handler.storage,
		codec:           g.handler.codec,
		canonicalPrefix: g.handler.canonicalPrefix,
		selfLinker:      g.handler.selfLinker,
	}
	proxyHandler := &ProxyHandler{prefix + "/proxy/", g.handler.storage, g.handler.codec}
	redirectHandler := &RedirectHandler{g.handler.storage, g.handler.codec}

	// Create a new WebService for this APIGroupVersion at the specified path prefix
	// TODO: Pass in more descriptive documentation
	ws := new(restful.WebService)
	ws.Path(prefix)
	ws.Doc("API at " + root + ", version " + version)
	// TODO: change to restful.MIME_JSON when we convert YAML->JSON and set content type in client
	ws.Consumes("*/*")
	ws.Produces(restful.MIME_JSON)
	// TODO: require json on input
	//ws.Consumes(restful.MIME_JSON)
	ws.ApiVersion(version)

	// TODO: add scheme to APIGroupVersion rather than using api.Scheme

	// TODO: #2057: Return API resources on "/".

	// TODO: Add status documentation using Returns()
	// Errors (see api/errors/errors.go as well as go-restful router):
	// http.StatusNotFound, http.StatusMethodNotAllowed,
	// http.StatusUnsupportedMediaType, http.StatusNotAcceptable,
	// http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden,
	// http.StatusRequestTimeout, http.StatusConflict, http.StatusPreconditionFailed,
	// 422 (StatusUnprocessableEntity), http.StatusInternalServerError,
	// http.StatusServiceUnavailable
	// and api error codes
	// Note that if we specify a versioned Status object here, we may need to
	// create one for the tests, also
	// Success:
	// http.StatusOK, http.StatusCreated, http.StatusAccepted, http.StatusNoContent
	//
	// test/integration/auth_test.go is currently the most comprehensive status code test

	// TODO: eliminate all the restful wrappers
	// TODO: create a separate handler per verb
	h := func(req *restful.Request, resp *restful.Response) {
		strippedHandler.ServeHTTP(resp.ResponseWriter, req.Request)
	}

	registrationErrors := make([]error, 0)

	for path, storage := range g.handler.storage {
		if err := registerResourceHandlers(ws, version, path, storage, h, g.mapper); err != nil {
			registrationErrors = append(registrationErrors, err)
		}
	}

	// TODO: port the rest of these. Sadly, if we don't, we'll have inconsistent
	// API behavior, as well as lack of documentation
	// Note: update GetAttribs() when adding a handler.
	mux.Handle(prefix+"/watch/", http.StripPrefix(prefix+"/watch/", watchHandler))
	mux.Handle(prefix+"/proxy/", http.StripPrefix(prefix+"/proxy/", proxyHandler))
	mux.Handle(prefix+"/redirect/", http.StripPrefix(prefix+"/redirect/", redirectHandler))

	container.Add(ws)

	return errors.NewAggregate(registrationErrors)
}

// TODO: Convert to go-restful
func InstallValidator(mux Mux, servers func() map[string]Server) {
	validator, err := NewValidator(servers)
	if err != nil {
		glog.Errorf("failed to set up validator: %v", err)
		return
	}
	if validator != nil {
		mux.Handle("/validate", validator)
	}
}

// TODO: document all handlers
// InstallSupport registers the APIServer support functions
func InstallSupport(mux Mux, ws *restful.WebService) {
	// TODO: convert healthz to restful and remove container arg
	healthz.InstallHandler(mux)

	// Set up a service to return the git code version.
	ws.Path("/version")
	ws.Doc("git code version from which this is built")
	ws.Route(
		ws.GET("/").To(handleVersion).
			Doc("get the code version").
			Operation("getCodeVersion").
			Produces(restful.MIME_JSON).
			Consumes(restful.MIME_JSON))
}

// InstallLogsSupport registers the APIServer log support function into a mux.
func InstallLogsSupport(mux Mux) {
	// TODO: use restful: ws.Route(ws.GET("/logs/{logpath:*}").To(fileHandler))
	// See github.com/emicklei/go-restful/blob/master/examples/restful-serve-static.go
	mux.Handle("/logs/", http.StripPrefix("/logs/", http.FileServer(http.Dir("/var/log/"))))
}

// Adds a service to return the supported api versions.
func AddApiWebService(container *restful.Container, apiPrefix string, versions []string) {
	// TODO: InstallREST should register each version automatically

	versionHandler := APIVersionHandler(versions[:]...)
	getApiVersionsWebService := new(restful.WebService)
	getApiVersionsWebService.Path(apiPrefix)
	getApiVersionsWebService.Doc("get available api versions")
	getApiVersionsWebService.Route(getApiVersionsWebService.GET("/").To(versionHandler).
		Doc("get available api versions").
		Operation("getApiVersions").
		Produces(restful.MIME_JSON).
		Consumes(restful.MIME_JSON))
	container.Add(getApiVersionsWebService)
}

// handleVersion writes the server's version information.
func handleVersion(req *restful.Request, resp *restful.Response) {
	// TODO: use restful's Response methods
	writeRawJSON(http.StatusOK, version.Get(), resp.ResponseWriter)
}

// APIVersionHandler returns a handler which will list the provided versions as available.
func APIVersionHandler(versions ...string) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		// TODO: use restful's Response methods
		writeRawJSON(http.StatusOK, api.APIVersions{Versions: versions}, resp.ResponseWriter)
	}
}

// writeJSON renders an object as JSON to the response.
func writeJSON(statusCode int, codec runtime.Codec, object runtime.Object, w http.ResponseWriter) {
	output, err := codec.Encode(object)
	if err != nil {
		errorJSONFatal(err, codec, w)
		return
	}
	// PR #2243: Pretty-print JSON by default.
	formatted := &bytes.Buffer{}
	err = json.Indent(formatted, output, "", "  ")
	if err != nil {
		errorJSONFatal(err, codec, w)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(formatted.Bytes())
}

// errorJSON renders an error to the response.
func errorJSON(err error, codec runtime.Codec, w http.ResponseWriter) {
	status := errToAPIStatus(err)
	writeJSON(status.Code, codec, status, w)
}

// errorJSONFatal renders an error to the response, and if codec fails will render plaintext
func errorJSONFatal(err error, codec runtime.Codec, w http.ResponseWriter) {
	util.HandleError(fmt.Errorf("apiserver was unable to write a JSON response: %v", err))
	status := errToAPIStatus(err)
	output, err := codec.Encode(status)
	if err != nil {
		w.WriteHeader(status.Code)
		fmt.Fprintf(w, "%s: %s", status.Reason, status.Message)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status.Code)
	w.Write(output)
}

// writeRawJSON writes a non-API object in JSON.
func writeRawJSON(statusCode int, object interface{}, w http.ResponseWriter) {
	output, err := json.MarshalIndent(object, "", "  ")
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
		glog.Errorf("Failed to parse %q: %v", str, err)
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
