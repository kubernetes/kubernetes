/*
Copyright 2015 Google Inc. All rights reserved.

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
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"

	"github.com/emicklei/go-restful"
)

type APIInstaller struct {
	prefix      string // Path prefix where API resources are to be registered.
	version     string // The API version being installed.
	restHandler *RESTHandler
	mapper      meta.RESTMapper
}

// Struct capturing information about an action ("GET", "POST", "WATCH", PROXY", etc).
type action struct {
	Verb   string               // Verb identifying the action ("GET", "POST", "WATCH", PROXY", etc).
	Path   string               // The path of the action
	Params []*restful.Parameter // List of parameters associated with the action.
}

// Installs handlers for API resources.
func (a *APIInstaller) Install() (ws *restful.WebService, errors []error) {
	errors = make([]error, 0)

	// Create the WebService.
	ws = a.newWebService()

	// Initialize the custom handlers.
	watchHandler := (&WatchHandler{
		storage:                a.restHandler.storage,
		codec:                  a.restHandler.codec,
		canonicalPrefix:        a.restHandler.canonicalPrefix,
		selfLinker:             a.restHandler.selfLinker,
		apiRequestInfoResolver: a.restHandler.apiRequestInfoResolver,
	})
	redirectHandler := (&RedirectHandler{a.restHandler.storage, a.restHandler.codec, a.restHandler.apiRequestInfoResolver})
	proxyHandler := (&ProxyHandler{a.prefix + "/proxy/", a.restHandler.storage, a.restHandler.codec, a.restHandler.apiRequestInfoResolver})

	for path, storage := range a.restHandler.storage {
		if err := a.registerResourceHandlers(path, storage, ws, watchHandler, redirectHandler, proxyHandler); err != nil {
			errors = append(errors, err)
		}
	}
	return ws, errors
}

func (a *APIInstaller) newWebService() *restful.WebService {
	ws := new(restful.WebService)
	ws.Path(a.prefix)
	ws.Doc("API at " + a.prefix + " version " + a.version)
	// TODO: change to restful.MIME_JSON when we set content type in client
	ws.Consumes("*/*")
	ws.Produces(restful.MIME_JSON)
	ws.ApiVersion(a.version)
	return ws
}

func (a *APIInstaller) registerResourceHandlers(path string, storage RESTStorage, ws *restful.WebService, watchHandler http.Handler, redirectHandler http.Handler, proxyHandler http.Handler) error {

	// Handler for standard REST verbs (GET, PUT, POST and DELETE).
	restVerbHandler := restfulStripPrefix(a.prefix, a.restHandler)
	object := storage.New()
	// TODO: add scheme to APIInstaller rather than using api.Scheme
	_, kind, err := api.Scheme.ObjectVersionAndKind(object)
	if err != nil {
		return err
	}
	versionedPtr, err := api.Scheme.New(a.version, kind)
	if err != nil {
		return err
	}
	versionedObject := indirectArbitraryPointer(versionedPtr)

	var versionedList interface{}
	if lister, ok := storage.(RESTLister); ok {
		list := lister.NewList()
		_, listKind, err := api.Scheme.ObjectVersionAndKind(list)
		versionedListPtr, err := api.Scheme.New(a.version, listKind)
		if err != nil {
			return err
		}
		versionedList = indirectArbitraryPointer(versionedListPtr)
	}

	mapping, err := a.mapper.RESTMapping(kind, a.version)
	if err != nil {
		return err
	}

	// what verbs are supported by the storage, used to know what verbs we support per path
	storageVerbs := map[string]bool{}
	if _, ok := storage.(RESTCreater); ok {
		// Handler for standard REST verbs (GET, PUT, POST and DELETE).
		storageVerbs["RESTCreater"] = true
	}
	if _, ok := storage.(RESTLister); ok {
		// Handler for standard REST verbs (GET, PUT, POST and DELETE).
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
	if _, ok := storage.(ResourceWatcher); ok {
		storageVerbs["ResourceWatcher"] = true
	}
	if _, ok := storage.(Redirector); ok {
		storageVerbs["Redirector"] = true
	}

	allowWatchList := storageVerbs["ResourceWatcher"] && storageVerbs["RESTLister"] // watching on lists is allowed only for kinds that support both watch and list.
	scope := mapping.Scope
	nameParam := ws.PathParameter("name", "name of the "+kind).DataType("string")
	params := []*restful.Parameter{}
	actions := []action{}
	// Get the list of actions for the given scope.
	if scope.Name() != meta.RESTScopeNameNamespace {
		// Handler for standard REST verbs (GET, PUT, POST and DELETE).
		actions = appendIf(actions, action{"LIST", path, params}, storageVerbs["RESTLister"])
		actions = appendIf(actions, action{"POST", path, params}, storageVerbs["RESTCreater"])
		actions = appendIf(actions, action{"WATCHLIST", "/watch/" + path, params}, allowWatchList)

		itemPath := path + "/{name}"
		nameParams := append(params, nameParam)
		actions = appendIf(actions, action{"GET", itemPath, nameParams}, storageVerbs["RESTGetter"])
		actions = appendIf(actions, action{"PUT", itemPath, nameParams}, storageVerbs["RESTUpdater"])
		actions = appendIf(actions, action{"DELETE", itemPath, nameParams}, storageVerbs["RESTDeleter"])
		actions = appendIf(actions, action{"WATCH", "/watch/" + itemPath, nameParams}, storageVerbs["ResourceWatcher"])
		actions = appendIf(actions, action{"REDIRECT", "/redirect/" + itemPath, nameParams}, storageVerbs["Redirector"])
		actions = appendIf(actions, action{"PROXY", "/proxy/" + itemPath + "/{path:*}", nameParams}, storageVerbs["Redirector"])
		actions = appendIf(actions, action{"PROXY", "/proxy/" + itemPath, nameParams}, storageVerbs["Redirector"])
	} else {
		// v1beta3 format with namespace in path
		if scope.ParamPath() {
			// Handler for standard REST verbs (GET, PUT, POST and DELETE).
			namespaceParam := ws.PathParameter(scope.ParamName(), scope.ParamDescription()).DataType("string")
			namespacedPath := scope.ParamName() + "/{" + scope.ParamName() + "}/" + path
			namespaceParams := []*restful.Parameter{namespaceParam}
			actions = appendIf(actions, action{"LIST", namespacedPath, namespaceParams}, storageVerbs["RESTLister"])
			actions = appendIf(actions, action{"POST", namespacedPath, namespaceParams}, storageVerbs["RESTCreater"])
			actions = appendIf(actions, action{"WATCHLIST", "/watch/" + namespacedPath, namespaceParams}, allowWatchList)

			itemPath := namespacedPath + "/{name}"
			nameParams := append(namespaceParams, nameParam)
			actions = appendIf(actions, action{"GET", itemPath, nameParams}, storageVerbs["RESTGetter"])
			actions = appendIf(actions, action{"PUT", itemPath, nameParams}, storageVerbs["RESTUpdater"])
			actions = appendIf(actions, action{"DELETE", itemPath, nameParams}, storageVerbs["RESTDeleter"])
			actions = appendIf(actions, action{"WATCH", "/watch/" + itemPath, nameParams}, storageVerbs["ResourceWatcher"])
			actions = appendIf(actions, action{"REDIRECT", "/redirect/" + itemPath, nameParams}, storageVerbs["Redirector"])
			actions = appendIf(actions, action{"PROXY", "/proxy/" + itemPath + "/{path:*}", nameParams}, storageVerbs["Redirector"])
			actions = appendIf(actions, action{"PROXY", "/proxy/" + itemPath, nameParams}, storageVerbs["Redirector"])

			// list across namespace.
			actions = appendIf(actions, action{"LIST", path, params}, storageVerbs["RESTLister"])
			actions = appendIf(actions, action{"WATCHLIST", "/watch/" + path, params}, allowWatchList)
		} else {
			// Handler for standard REST verbs (GET, PUT, POST and DELETE).
			// v1beta1/v1beta2 format where namespace was a query parameter
			namespaceParam := ws.QueryParameter(scope.ParamName(), scope.ParamDescription()).DataType("string")
			namespaceParams := []*restful.Parameter{namespaceParam}
			actions = appendIf(actions, action{"LIST", path, namespaceParams}, storageVerbs["RESTLister"])
			actions = appendIf(actions, action{"POST", path, namespaceParams}, storageVerbs["RESTCreater"])
			actions = appendIf(actions, action{"WATCHLIST", "/watch/" + path, namespaceParams}, allowWatchList)

			itemPath := path + "/{name}"
			nameParams := append(namespaceParams, nameParam)
			actions = appendIf(actions, action{"GET", itemPath, nameParams}, storageVerbs["RESTGetter"])
			actions = appendIf(actions, action{"PUT", itemPath, nameParams}, storageVerbs["RESTUpdater"])
			actions = appendIf(actions, action{"DELETE", itemPath, nameParams}, storageVerbs["RESTDeleter"])
			actions = appendIf(actions, action{"WATCH", "/watch/" + itemPath, nameParams}, storageVerbs["ResourceWatcher"])
			actions = appendIf(actions, action{"REDIRECT", "/redirect/" + itemPath, nameParams}, storageVerbs["Redirector"])
			actions = appendIf(actions, action{"PROXY", "/proxy/" + itemPath + "/{path:*}", nameParams}, storageVerbs["Redirector"])
			actions = appendIf(actions, action{"PROXY", "/proxy/" + itemPath, nameParams}, storageVerbs["Redirector"])
		}
	}

	// Create Routes for the actions.
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

	for _, action := range actions {
		switch action.Verb {
		case "GET": // Get a resource.
			route := ws.GET(action.Path).To(restVerbHandler).
				Doc("read the specified " + kind).
				Operation("read" + kind).
				Writes(versionedObject)
			addParams(route, action.Params)
			ws.Route(route)
		case "LIST": // List all resources of a kind.
			route := ws.GET(action.Path).To(restVerbHandler).
				Doc("list objects of kind " + kind).
				Operation("list" + kind).
				Writes(versionedList)
			addParams(route, action.Params)
			ws.Route(route)
		case "PUT": // Update a resource.
			route := ws.PUT(action.Path).To(restVerbHandler).
				Doc("update the specified " + kind).
				Operation("update" + kind).
				Reads(versionedObject)
			addParams(route, action.Params)
			ws.Route(route)
		case "POST": // Create a resource.
			route := ws.POST(action.Path).To(restVerbHandler).
				Doc("create a " + kind).
				Operation("create" + kind).
				Reads(versionedObject)
			addParams(route, action.Params)
			ws.Route(route)
		case "DELETE": // Delete a resource.
			route := ws.DELETE(action.Path).To(restVerbHandler).
				Doc("delete a " + kind).
				Operation("delete" + kind)
			addParams(route, action.Params)
			ws.Route(route)
		case "WATCH": // Watch a resource.
			route := ws.GET(action.Path).To(restfulStripPrefix(a.prefix+"/watch", watchHandler)).
				Doc("watch a particular " + kind).
				Operation("watch" + kind).
				Writes(versionedObject)
			addParams(route, action.Params)
			ws.Route(route)
		case "WATCHLIST": // Watch all resources of a kind.
			route := ws.GET(action.Path).To(restfulStripPrefix(a.prefix+"/watch", watchHandler)).
				Doc("watch a list of " + kind).
				Operation("watch" + kind + "list").
				Writes(versionedList)
			addParams(route, action.Params)
			ws.Route(route)
		case "REDIRECT": // Get the redirect URL for a resource.
			route := ws.GET(action.Path).To(restfulStripPrefix(a.prefix+"/redirect", redirectHandler)).
				Doc("redirect GET request to " + kind).
				Operation("redirect" + kind).
				Produces("*/*").
				Consumes("*/*")
			addParams(route, action.Params)
			ws.Route(route)
		case "PROXY": // Proxy requests to a resource.
			// Accept all methods as per https://github.com/GoogleCloudPlatform/kubernetes/issues/3996
			addProxyRoute(ws, "GET", a.prefix, action.Path, proxyHandler, kind, action.Params)
			addProxyRoute(ws, "PUT", a.prefix, action.Path, proxyHandler, kind, action.Params)
			addProxyRoute(ws, "POST", a.prefix, action.Path, proxyHandler, kind, action.Params)
			addProxyRoute(ws, "DELETE", a.prefix, action.Path, proxyHandler, kind, action.Params)
		}
		// Note: update GetAttribs() when adding a custom handler.
	}
	return nil
}

// This magic incantation returns *ptrToObject for an arbitrary pointer
func indirectArbitraryPointer(ptrToObject interface{}) interface{} {
	return reflect.Indirect(reflect.ValueOf(ptrToObject)).Interface()
}

func appendIf(actions []action, a action, shouldAppend bool) []action {
	if shouldAppend {
		actions = append(actions, a)
	}
	return actions
}

// Returns a restful RouteFunction that calls the given handler after stripping prefix from the request path.
func restfulStripPrefix(prefix string, handler http.Handler) restful.RouteFunction {
	return func(restReq *restful.Request, restResp *restful.Response) {
		http.StripPrefix(prefix, handler).ServeHTTP(restResp.ResponseWriter, restReq.Request)
	}
}

func addProxyRoute(ws *restful.WebService, method string, prefix string, path string, proxyHandler http.Handler, kind string, params []*restful.Parameter) {
	proxyRoute := ws.Method(method).Path(path).To(restfulStripPrefix(prefix+"/proxy", proxyHandler)).
		Doc("proxy " + method + " requests to " + kind).
		Operation("proxy" + method + kind).
		Produces("*/*").
		Consumes("*/*")
	addParams(proxyRoute, params)
	ws.Route(proxyRoute)
}

func addParams(route *restful.RouteBuilder, params []*restful.Parameter) {
	for _, param := range params {
		route.Param(param)
	}
}
