/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	gpath "path"
	"reflect"
	"sort"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	watchjson "k8s.io/kubernetes/pkg/watch/json"

	"github.com/emicklei/go-restful"
)

type APIInstaller struct {
	group             *APIGroupVersion
	info              *APIRequestInfoResolver
	prefix            string // Path prefix where API resources are to be registered.
	minRequestTimeout time.Duration
	proxyDialerFn     ProxyDialerFunc
}

// Struct capturing information about an action ("GET", "POST", "WATCH", PROXY", etc).
type action struct {
	Verb   string               // Verb identifying the action ("GET", "POST", "WATCH", PROXY", etc).
	Path   string               // The path of the action
	Params []*restful.Parameter // List of parameters associated with the action.
	Namer  ScopeNamer
}

// An interface to see if an object supports swagger documentation as a method
type documentable interface {
	SwaggerDoc() map[string]string
}

// errEmptyName is returned when API requests do not fill the name section of the path.
var errEmptyName = errors.NewBadRequest("name must be provided")

// Installs handlers for API resources.
func (a *APIInstaller) Install() (ws *restful.WebService, errors []error) {
	errors = make([]error, 0)

	// Create the WebService.
	ws = a.newWebService()

	proxyHandler := (&ProxyHandler{a.prefix + "/proxy/", a.group.Storage, a.group.Codec, a.group.Context, a.info, a.proxyDialerFn})

	// Register the paths in a deterministic (sorted) order to get a deterministic swagger spec.
	paths := make([]string, len(a.group.Storage))
	var i int = 0
	for path := range a.group.Storage {
		paths[i] = path
		i++
	}
	sort.Strings(paths)
	for _, path := range paths {
		if err := a.registerResourceHandlers(path, a.group.Storage[path], ws, proxyHandler); err != nil {
			errors = append(errors, err)
		}
	}
	return ws, errors
}

func (a *APIInstaller) newWebService() *restful.WebService {
	ws := new(restful.WebService)
	ws.Path(a.prefix)
	ws.Doc("API at " + a.prefix + " version " + a.group.Version)
	// TODO: change to restful.MIME_JSON when we set content type in client
	ws.Consumes("*/*")
	ws.Produces(restful.MIME_JSON)
	ws.ApiVersion(a.group.Version)

	return ws
}

func (a *APIInstaller) registerResourceHandlers(path string, storage rest.Storage, ws *restful.WebService, proxyHandler http.Handler) error {
	admit := a.group.Admit
	context := a.group.Context

	serverVersion := a.group.ServerVersion
	if len(serverVersion) == 0 {
		serverVersion = a.group.Version
	}

	var resource, subresource string
	switch parts := strings.Split(path, "/"); len(parts) {
	case 2:
		resource, subresource = parts[0], parts[1]
	case 1:
		resource = parts[0]
	default:
		// TODO: support deeper paths
		return fmt.Errorf("api_installer allows only one or two segment paths (resource or resource/subresource)")
	}
	hasSubresource := len(subresource) > 0

	object := storage.New()
	_, kind, err := a.group.Typer.ObjectVersionAndKind(object)
	if err != nil {
		return err
	}
	versionedPtr, err := a.group.Creater.New(a.group.Version, kind)
	if err != nil {
		return err
	}
	versionedObject := indirectArbitraryPointer(versionedPtr)

	mapping, err := a.group.Mapper.RESTMapping(kind, a.group.Version)
	if err != nil {
		return err
	}

	// subresources must have parent resources, and follow the namespacing rules of their parent
	if hasSubresource {
		parentStorage, ok := a.group.Storage[resource]
		if !ok {
			return fmt.Errorf("subresources can only be declared when the parent is also registered: %s needs %s", path, resource)
		}
		parentObject := parentStorage.New()
		_, parentKind, err := a.group.Typer.ObjectVersionAndKind(parentObject)
		if err != nil {
			return err
		}
		parentMapping, err := a.group.Mapper.RESTMapping(parentKind, a.group.Version)
		if err != nil {
			return err
		}
		mapping.Scope = parentMapping.Scope
	}

	// what verbs are supported by the storage, used to know what verbs we support per path
	creater, isCreater := storage.(rest.Creater)
	namedCreater, isNamedCreater := storage.(rest.NamedCreater)
	lister, isLister := storage.(rest.Lister)
	getter, isGetter := storage.(rest.Getter)
	getterWithOptions, isGetterWithOptions := storage.(rest.GetterWithOptions)
	deleter, isDeleter := storage.(rest.Deleter)
	gracefulDeleter, isGracefulDeleter := storage.(rest.GracefulDeleter)
	updater, isUpdater := storage.(rest.Updater)
	patcher, isPatcher := storage.(rest.Patcher)
	watcher, isWatcher := storage.(rest.Watcher)
	_, isRedirector := storage.(rest.Redirector)
	connecter, isConnecter := storage.(rest.Connecter)
	storageMeta, isMetadata := storage.(rest.StorageMetadata)
	if !isMetadata {
		storageMeta = defaultStorageMetadata{}
	}

	if isNamedCreater {
		isCreater = true
	}

	var versionedList interface{}
	if isLister {
		list := lister.NewList()
		_, listKind, err := a.group.Typer.ObjectVersionAndKind(list)
		versionedListPtr, err := a.group.Creater.New(a.group.Version, listKind)
		if err != nil {
			return err
		}
		versionedList = indirectArbitraryPointer(versionedListPtr)
	}

	versionedListOptions, err := a.group.Creater.New(serverVersion, "ListOptions")
	if err != nil {
		return err
	}

	var versionedDeleterObject interface{}
	switch {
	case isGracefulDeleter:
		objectPtr, err := a.group.Creater.New(serverVersion, "DeleteOptions")
		if err != nil {
			return err
		}
		versionedDeleterObject = indirectArbitraryPointer(objectPtr)
		isDeleter = true
	case isDeleter:
		gracefulDeleter = rest.GracefulDeleteAdapter{Deleter: deleter}
	}

	versionedStatusPtr, err := a.group.Creater.New(serverVersion, "Status")
	if err != nil {
		return err
	}
	versionedStatus := indirectArbitraryPointer(versionedStatusPtr)
	var (
		getOptions          runtime.Object
		versionedGetOptions runtime.Object
		getOptionsKind      string
		getSubpath          bool
		getSubpathKey       string
	)
	if isGetterWithOptions {
		getOptions, getSubpath, getSubpathKey = getterWithOptions.NewGetOptions()
		_, getOptionsKind, err = a.group.Typer.ObjectVersionAndKind(getOptions)
		if err != nil {
			return err
		}
		versionedGetOptions, err = a.group.Creater.New(serverVersion, getOptionsKind)
		if err != nil {
			return err
		}
		isGetter = true
	}

	var (
		connectOptions          runtime.Object
		versionedConnectOptions runtime.Object
		connectOptionsKind      string
		connectSubpath          bool
		connectSubpathKey       string
	)
	if isConnecter {
		connectOptions, connectSubpath, connectSubpathKey = connecter.NewConnectOptions()
		if connectOptions != nil {
			_, connectOptionsKind, err = a.group.Typer.ObjectVersionAndKind(connectOptions)
			if err != nil {
				return err
			}
			versionedConnectOptions, err = a.group.Creater.New(serverVersion, connectOptionsKind)
		}
	}

	var ctxFn ContextFunc
	ctxFn = func(req *restful.Request) api.Context {
		if context == nil {
			return api.NewContext()
		}
		if ctx, ok := context.Get(req.Request); ok {
			return ctx
		}
		return api.NewContext()
	}

	allowWatchList := isWatcher && isLister // watching on lists is allowed only for kinds that support both watch and list.
	scope := mapping.Scope
	nameParam := ws.PathParameter("name", "name of the "+kind).DataType("string")
	pathParam := ws.PathParameter("path:*", "path to the resource").DataType("string")
	params := []*restful.Parameter{}
	actions := []action{}

	// Get the list of actions for the given scope.
	switch scope.Name() {
	case meta.RESTScopeNameRoot:
		// Handle non-namespace scoped resources like nodes.
		resourcePath := resource
		resourceParams := params
		itemPath := resourcePath + "/{name}"
		nameParams := append(params, nameParam)
		proxyParams := append(nameParams, pathParam)
		if hasSubresource {
			itemPath = itemPath + "/" + subresource
			resourcePath = itemPath
			resourceParams = nameParams
		}
		namer := rootScopeNaming{scope, a.group.Linker, gpath.Join(a.prefix, itemPath)}

		// Handler for standard REST verbs (GET, PUT, POST and DELETE).
		// Add actions at the resource path: /api/apiVersion/resource
		actions = appendIf(actions, action{"LIST", resourcePath, resourceParams, namer}, isLister)
		actions = appendIf(actions, action{"POST", resourcePath, resourceParams, namer}, isCreater)
		actions = appendIf(actions, action{"WATCHLIST", "watch/" + resourcePath, resourceParams, namer}, allowWatchList)

		// Add actions at the item path: /api/apiVersion/resource/{name}
		actions = appendIf(actions, action{"GET", itemPath, nameParams, namer}, isGetter)
		if getSubpath {
			actions = appendIf(actions, action{"GET", itemPath + "/{path:*}", proxyParams, namer}, isGetter)
		}
		actions = appendIf(actions, action{"PUT", itemPath, nameParams, namer}, isUpdater)
		actions = appendIf(actions, action{"PATCH", itemPath, nameParams, namer}, isPatcher)
		actions = appendIf(actions, action{"DELETE", itemPath, nameParams, namer}, isDeleter)
		actions = appendIf(actions, action{"WATCH", "watch/" + itemPath, nameParams, namer}, isWatcher)
		actions = appendIf(actions, action{"PROXY", "proxy/" + itemPath + "/{path:*}", proxyParams, namer}, isRedirector)
		actions = appendIf(actions, action{"PROXY", "proxy/" + itemPath, nameParams, namer}, isRedirector)
		actions = appendIf(actions, action{"CONNECT", itemPath, nameParams, namer}, isConnecter)
		actions = appendIf(actions, action{"CONNECT", itemPath + "/{path:*}", proxyParams, namer}, isConnecter && connectSubpath)
		break
	case meta.RESTScopeNameNamespace:
		// Handler for standard REST verbs (GET, PUT, POST and DELETE).
		namespaceParam := ws.PathParameter(scope.ArgumentName(), scope.ParamDescription()).DataType("string")
		namespacedPath := scope.ParamName() + "/{" + scope.ArgumentName() + "}/" + resource
		namespaceParams := []*restful.Parameter{namespaceParam}

		resourcePath := namespacedPath
		resourceParams := namespaceParams
		itemPath := namespacedPath + "/{name}"
		nameParams := append(namespaceParams, nameParam)
		proxyParams := append(nameParams, pathParam)
		if hasSubresource {
			itemPath = itemPath + "/" + subresource
			resourcePath = itemPath
			resourceParams = nameParams
		}
		namer := scopeNaming{scope, a.group.Linker, gpath.Join(a.prefix, itemPath), false}

		actions = appendIf(actions, action{"LIST", resourcePath, resourceParams, namer}, isLister)
		actions = appendIf(actions, action{"POST", resourcePath, resourceParams, namer}, isCreater)
		// DEPRECATED
		actions = appendIf(actions, action{"WATCHLIST", "watch/" + resourcePath, resourceParams, namer}, allowWatchList)

		actions = appendIf(actions, action{"GET", itemPath, nameParams, namer}, isGetter)
		if getSubpath {
			actions = appendIf(actions, action{"GET", itemPath + "/{path:*}", proxyParams, namer}, isGetter)
		}
		actions = appendIf(actions, action{"PUT", itemPath, nameParams, namer}, isUpdater)
		actions = appendIf(actions, action{"PATCH", itemPath, nameParams, namer}, isPatcher)
		actions = appendIf(actions, action{"DELETE", itemPath, nameParams, namer}, isDeleter)
		actions = appendIf(actions, action{"WATCH", "watch/" + itemPath, nameParams, namer}, isWatcher)
		actions = appendIf(actions, action{"PROXY", "proxy/" + itemPath + "/{path:*}", proxyParams, namer}, isRedirector)
		actions = appendIf(actions, action{"PROXY", "proxy/" + itemPath, nameParams, namer}, isRedirector)
		actions = appendIf(actions, action{"CONNECT", itemPath, nameParams, namer}, isConnecter)
		actions = appendIf(actions, action{"CONNECT", itemPath + "/{path:*}", proxyParams, namer}, isConnecter && connectSubpath)

		// list or post across namespace.
		// For ex: LIST all pods in all namespaces by sending a LIST request at /api/apiVersion/pods.
		// TODO: more strongly type whether a resource allows these actions on "all namespaces" (bulk delete)
		if !hasSubresource {
			namer = scopeNaming{scope, a.group.Linker, gpath.Join(a.prefix, itemPath), true}
			actions = appendIf(actions, action{"LIST", resource, params, namer}, isLister)
			actions = appendIf(actions, action{"WATCHLIST", "watch/" + resource, params, namer}, allowWatchList)
		}
		break
	default:
		return fmt.Errorf("unsupported restscope: %s", scope.Name())
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

	reqScope := RequestScope{
		ContextFunc:      ctxFn,
		Creater:          a.group.Creater,
		Convertor:        a.group.Convertor,
		Codec:            mapping.Codec,
		APIVersion:       a.group.Version,
		ServerAPIVersion: serverVersion,
		Resource:         resource,
		Subresource:      subresource,
		Kind:             kind,
	}
	for _, action := range actions {
		reqScope.Namer = action.Namer
		m := monitorFilter(action.Verb, resource)
		namespaced := ""
		if strings.Contains(action.Path, scope.ArgumentName()) {
			namespaced = "Namespaced"
		}
		switch action.Verb {
		case "GET": // Get a resource.
			var handler restful.RouteFunction
			if isGetterWithOptions {
				handler = GetResourceWithOptions(getterWithOptions, reqScope, getOptionsKind, getSubpath, getSubpathKey)
			} else {
				handler = GetResource(getter, reqScope)
			}
			doc := "read the specified " + kind
			if hasSubresource {
				doc = "read " + subresource + " of the specified " + kind
			}
			route := ws.GET(action.Path).To(handler).
				Filter(m).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("read"+namespaced+kind+strings.Title(subresource)).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), "application/json")...).
				Returns(http.StatusOK, "OK", versionedObject).
				Writes(versionedObject)
			if isGetterWithOptions {
				if err := addObjectParams(ws, route, versionedGetOptions); err != nil {
					return err
				}
			}
			addParams(route, action.Params)
			ws.Route(route)
		case "LIST": // List all resources of a kind.
			doc := "list objects of kind " + kind
			if hasSubresource {
				doc = "list " + subresource + " of objects of kind " + kind
			}
			route := ws.GET(action.Path).To(ListResource(lister, watcher, reqScope, false, a.minRequestTimeout)).
				Filter(m).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("list"+namespaced+kind+strings.Title(subresource)).
				Produces("application/json").
				Returns(http.StatusOK, "OK", versionedList).
				Writes(versionedList)
			if err := addObjectParams(ws, route, versionedListOptions); err != nil {
				return err
			}
			switch {
			case isLister && isWatcher:
				doc := "list or watch objects of kind " + kind
				if hasSubresource {
					doc = "list or watch " + subresource + " of objects of kind " + kind
				}
				route.Doc(doc)
			case isWatcher:
				doc := "watch objects of kind " + kind
				if hasSubresource {
					doc = "watch " + subresource + "of objects of kind " + kind
				}
				route.Doc(doc)
			}
			addParams(route, action.Params)
			ws.Route(route)
		case "PUT": // Update a resource.
			doc := "replace the specified " + kind
			if hasSubresource {
				doc = "replace " + subresource + " of the specified " + kind
			}
			route := ws.PUT(action.Path).To(UpdateResource(updater, reqScope, a.group.Typer, admit)).
				Filter(m).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("replace"+namespaced+kind+strings.Title(subresource)).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), "application/json")...).
				Returns(http.StatusOK, "OK", versionedObject).
				Reads(versionedObject).
				Writes(versionedObject)
			addParams(route, action.Params)
			ws.Route(route)
		case "PATCH": // Partially update a resource
			doc := "partially update the specified " + kind
			if hasSubresource {
				doc = "partially update " + subresource + " of the specified " + kind
			}
			route := ws.PATCH(action.Path).To(PatchResource(patcher, reqScope, a.group.Typer, admit, mapping.ObjectConvertor)).
				Filter(m).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Consumes(string(api.JSONPatchType), string(api.MergePatchType), string(api.StrategicMergePatchType)).
				Operation("patch"+namespaced+kind+strings.Title(subresource)).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), "application/json")...).
				Returns(http.StatusOK, "OK", versionedObject).
				Reads(api.Patch{}).
				Writes(versionedObject)
			addParams(route, action.Params)
			ws.Route(route)
		case "POST": // Create a resource.
			var handler restful.RouteFunction
			if isNamedCreater {
				handler = CreateNamedResource(namedCreater, reqScope, a.group.Typer, admit)
			} else {
				handler = CreateResource(creater, reqScope, a.group.Typer, admit)
			}
			doc := "create a " + kind
			if hasSubresource {
				doc = "create " + subresource + " of a " + kind
			}
			route := ws.POST(action.Path).To(handler).
				Filter(m).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("create"+namespaced+kind+strings.Title(subresource)).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), "application/json")...).
				Returns(http.StatusOK, "OK", versionedObject).
				Reads(versionedObject).
				Writes(versionedObject)
			addParams(route, action.Params)
			ws.Route(route)
		case "DELETE": // Delete a resource.
			doc := "delete a " + kind
			if hasSubresource {
				doc = "delete " + subresource + " of a " + kind
			}
			route := ws.DELETE(action.Path).To(DeleteResource(gracefulDeleter, isGracefulDeleter, reqScope, admit)).
				Filter(m).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("delete"+namespaced+kind+strings.Title(subresource)).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), "application/json")...).
				Writes(versionedStatus).
				Returns(http.StatusOK, "OK", versionedStatus)
			if isGracefulDeleter {
				route.Reads(versionedDeleterObject)
			}
			addParams(route, action.Params)
			ws.Route(route)
		// TODO: deprecated
		case "WATCH": // Watch a resource.
			doc := "watch changes to an object of kind " + kind
			if hasSubresource {
				doc = "watch changes to " + subresource + " of an object of kind " + kind
			}
			route := ws.GET(action.Path).To(ListResource(lister, watcher, reqScope, true, a.minRequestTimeout)).
				Filter(m).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("watch"+namespaced+kind+strings.Title(subresource)).
				Produces("application/json").
				Returns(http.StatusOK, "OK", watchjson.WatchEvent{}).
				Writes(watchjson.WatchEvent{})
			if err := addObjectParams(ws, route, versionedListOptions); err != nil {
				return err
			}
			addParams(route, action.Params)
			ws.Route(route)
		// TODO: deprecated
		case "WATCHLIST": // Watch all resources of a kind.
			doc := "watch individual changes to a list of " + kind
			if hasSubresource {
				doc = "watch individual changes to a list of " + subresource + " of " + kind
			}
			route := ws.GET(action.Path).To(ListResource(lister, watcher, reqScope, true, a.minRequestTimeout)).
				Filter(m).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("watch"+namespaced+kind+strings.Title(subresource)+"List").
				Produces("application/json").
				Returns(http.StatusOK, "OK", watchjson.WatchEvent{}).
				Writes(watchjson.WatchEvent{})
			if err := addObjectParams(ws, route, versionedListOptions); err != nil {
				return err
			}
			addParams(route, action.Params)
			ws.Route(route)
		case "PROXY": // Proxy requests to a resource.
			// Accept all methods as per http://issue.k8s.io/3996
			addProxyRoute(ws, "GET", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params)
			addProxyRoute(ws, "PUT", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params)
			addProxyRoute(ws, "POST", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params)
			addProxyRoute(ws, "DELETE", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params)
			addProxyRoute(ws, "HEAD", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params)
			addProxyRoute(ws, "OPTIONS", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params)
		case "CONNECT":
			for _, method := range connecter.ConnectMethods() {
				doc := "connect " + method + " requests to " + kind
				if hasSubresource {
					doc = "connect " + method + " requests to " + subresource + " of " + kind
				}
				route := ws.Method(method).Path(action.Path).
					To(ConnectResource(connecter, reqScope, admit, connectOptionsKind, path, connectSubpath, connectSubpathKey)).
					Filter(m).
					Doc(doc).
					Operation("connect" + strings.Title(strings.ToLower(method)) + namespaced + kind + strings.Title(subresource)).
					Produces("*/*").
					Consumes("*/*").
					Writes("string")
				if versionedConnectOptions != nil {
					if err := addObjectParams(ws, route, versionedConnectOptions); err != nil {
						return err
					}
				}
				addParams(route, action.Params)
				ws.Route(route)
			}
		default:
			return fmt.Errorf("unrecognized action verb: %s", action.Verb)
		}
		// Note: update GetAttribs() when adding a custom handler.
	}
	return nil
}

// rootScopeNaming reads only names from a request and ignores namespaces. It implements ScopeNamer
// for root scoped resources.
type rootScopeNaming struct {
	scope meta.RESTScope
	runtime.SelfLinker
	itemPath string
}

// rootScopeNaming implements ScopeNamer
var _ ScopeNamer = rootScopeNaming{}

// Namespace returns an empty string because root scoped objects have no namespace.
func (n rootScopeNaming) Namespace(req *restful.Request) (namespace string, err error) {
	return "", nil
}

// Name returns the name from the path and an empty string for namespace, or an error if the
// name is empty.
func (n rootScopeNaming) Name(req *restful.Request) (namespace, name string, err error) {
	name = req.PathParameter("name")
	if len(name) == 0 {
		return "", "", errEmptyName
	}
	return "", name, nil
}

// GenerateLink returns the appropriate path and query to locate an object by its canonical path.
func (n rootScopeNaming) GenerateLink(req *restful.Request, obj runtime.Object) (path, query string, err error) {
	_, name, err := n.ObjectName(obj)
	if err != nil {
		return "", "", err
	}
	if len(name) == 0 {
		_, name, err = n.Name(req)
		if err != nil {
			return "", "", err
		}
	}
	path = strings.Replace(n.itemPath, "{name}", name, 1)
	return path, "", nil
}

// GenerateListLink returns the appropriate path and query to locate a list by its canonical path.
func (n rootScopeNaming) GenerateListLink(req *restful.Request) (path, query string, err error) {
	path = req.Request.URL.Path
	return path, "", nil
}

// ObjectName returns the name set on the object, or an error if the
// name cannot be returned. Namespace is empty
// TODO: distinguish between objects with name/namespace and without via a specific error.
func (n rootScopeNaming) ObjectName(obj runtime.Object) (namespace, name string, err error) {
	name, err = n.SelfLinker.Name(obj)
	if err != nil {
		return "", "", err
	}
	if len(name) == 0 {
		return "", "", errEmptyName
	}
	return "", name, nil
}

// scopeNaming returns naming information from a request. It implements ScopeNamer for
// namespace scoped resources.
type scopeNaming struct {
	scope meta.RESTScope
	runtime.SelfLinker
	itemPath      string
	allNamespaces bool
}

// scopeNaming implements ScopeNamer
var _ ScopeNamer = scopeNaming{}

// Namespace returns the namespace from the path or the default.
func (n scopeNaming) Namespace(req *restful.Request) (namespace string, err error) {
	if n.allNamespaces {
		return "", nil
	}
	namespace = req.PathParameter(n.scope.ArgumentName())
	if len(namespace) == 0 {
		// a URL was constructed without the namespace, or this method was invoked
		// on an object without a namespace path parameter.
		return "", fmt.Errorf("no namespace parameter found on request")
	}
	return namespace, nil
}

// Name returns the name from the path, the namespace (or default), or an error if the
// name is empty.
func (n scopeNaming) Name(req *restful.Request) (namespace, name string, err error) {
	namespace, _ = n.Namespace(req)
	name = req.PathParameter("name")
	if len(name) == 0 {
		return "", "", errEmptyName
	}
	return
}

// GenerateLink returns the appropriate path and query to locate an object by its canonical path.
func (n scopeNaming) GenerateLink(req *restful.Request, obj runtime.Object) (path, query string, err error) {
	namespace, name, err := n.ObjectName(obj)
	if err != nil {
		return "", "", err
	}
	if len(namespace) == 0 && len(name) == 0 {
		namespace, name, err = n.Name(req)
		if err != nil {
			return "", "", err
		}
	}
	if len(name) == 0 {
		return "", "", errEmptyName
	}
	path = strings.Replace(n.itemPath, "{name}", name, 1)
	path = strings.Replace(path, "{"+n.scope.ArgumentName()+"}", namespace, 1)
	return path, "", nil
}

// GenerateListLink returns the appropriate path and query to locate a list by its canonical path.
func (n scopeNaming) GenerateListLink(req *restful.Request) (path, query string, err error) {
	path = req.Request.URL.Path
	return path, "", nil
}

// ObjectName returns the name and namespace set on the object, or an error if the
// name cannot be returned.
// TODO: distinguish between objects with name/namespace and without via a specific error.
func (n scopeNaming) ObjectName(obj runtime.Object) (namespace, name string, err error) {
	name, err = n.SelfLinker.Name(obj)
	if err != nil {
		return "", "", err
	}
	namespace, err = n.SelfLinker.Namespace(obj)
	if err != nil {
		return "", "", err
	}
	return namespace, name, err
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

// Wraps a http.Handler function inside a restful.RouteFunction
func routeFunction(handler http.Handler) restful.RouteFunction {
	return func(restReq *restful.Request, restResp *restful.Response) {
		handler.ServeHTTP(restResp.ResponseWriter, restReq.Request)
	}
}

func addProxyRoute(ws *restful.WebService, method string, prefix string, path string, proxyHandler http.Handler, namespaced, kind, resource, subresource string, hasSubresource bool, params []*restful.Parameter) {
	doc := "proxy " + method + " requests to " + kind
	if hasSubresource {
		doc = "proxy " + method + " requests to " + subresource + " of " + kind
	}
	proxyRoute := ws.Method(method).Path(path).To(routeFunction(proxyHandler)).
		Filter(monitorFilter("PROXY", resource)).
		Doc(doc).
		Operation("proxy" + strings.Title(method) + namespaced + kind + strings.Title(subresource)).
		Produces("*/*").
		Consumes("*/*").
		Writes("string")
	addParams(proxyRoute, params)
	ws.Route(proxyRoute)
}

func addParams(route *restful.RouteBuilder, params []*restful.Parameter) {
	for _, param := range params {
		route.Param(param)
	}
}

// addObjectParams converts a runtime.Object into a set of go-restful Param() definitions on the route.
// The object must be a pointer to a struct; only fields at the top level of the struct that are not
// themselves interfaces or structs are used; only fields with a json tag that is non empty (the standard
// Go JSON behavior for omitting a field) become query parameters. The name of the query parameter is
// the JSON field name. If a description struct tag is set on the field, that description is used on the
// query parameter. In essence, it converts a standard JSON top level object into a query param schema.
func addObjectParams(ws *restful.WebService, route *restful.RouteBuilder, obj interface{}) error {
	sv, err := conversion.EnforcePtr(obj)
	if err != nil {
		return err
	}
	st := sv.Type()
	switch st.Kind() {
	case reflect.Struct:
		for i := 0; i < st.NumField(); i++ {
			name := st.Field(i).Name
			sf, ok := st.FieldByName(name)
			if !ok {
				continue
			}
			switch sf.Type.Kind() {
			case reflect.Interface, reflect.Struct:
			default:
				jsonTag := sf.Tag.Get("json")
				if len(jsonTag) == 0 {
					continue
				}
				jsonName := strings.SplitN(jsonTag, ",", 2)[0]
				if len(jsonName) == 0 {
					continue
				}

				var desc string
				if docable, ok := obj.(documentable); ok {
					desc = docable.SwaggerDoc()[jsonName]
				}
				route.Param(ws.QueryParameter(jsonName, desc).DataType(typeToJSON(sf.Type.Name())))
			}
		}
	}
	return nil
}

// TODO: this is incomplete, expand as needed.
// Convert the name of a golang type to the name of a JSON type
func typeToJSON(typeName string) string {
	switch typeName {
	case "bool":
		return "boolean"
	case "uint8", "int", "int32", "int64", "uint32", "uint64":
		return "integer"
	case "byte":
		return "string"
	case "float64", "float32":
		return "number"
	case "util.Time":
		return "string"
	default:
		return typeName
	}
}

// defaultStorageMetadata provides default answers to rest.StorageMetadata.
type defaultStorageMetadata struct{}

// defaultStorageMetadata implements rest.StorageMetadata
var _ rest.StorageMetadata = defaultStorageMetadata{}

func (defaultStorageMetadata) ProducesMIMETypes(verb string) []string {
	return nil
}
