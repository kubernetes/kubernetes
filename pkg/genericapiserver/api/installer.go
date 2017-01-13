/*
Copyright 2015 The Kubernetes Authors.

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

package api

import (
	"bytes"
	"fmt"
	"net/http"
	"net/url"
	gpath "path"
	"reflect"
	"sort"
	"strings"
	"time"
	"unicode"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/handlers/negotiation"
	"k8s.io/apiserver/pkg/metrics"
	"k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/genericapiserver/api/handlers"

	"github.com/emicklei/go-restful"
)

type APIInstaller struct {
	group             *APIGroupVersion
	prefix            string // Path prefix where API resources are to be registered.
	minRequestTimeout time.Duration
}

// Struct capturing information about an action ("GET", "POST", "WATCH", PROXY", etc).
type action struct {
	Verb          string               // Verb identifying the action ("GET", "POST", "WATCH", PROXY", etc).
	Path          string               // The path of the action
	Params        []*restful.Parameter // List of parameters associated with the action.
	Namer         handlers.ScopeNamer
	AllNamespaces bool // true iff the action is namespaced but works on aggregate result for all namespaces
}

// An interface to see if an object supports swagger documentation as a method
type documentable interface {
	SwaggerDoc() map[string]string
}

// toDiscoveryKubeVerb maps an action.Verb to the logical kube verb, used for discovery
var toDiscoveryKubeVerb = map[string]string{
	"CONNECT":          "", // do not list in discovery.
	"DELETE":           "delete",
	"DELETECOLLECTION": "deletecollection",
	"GET":              "get",
	"LIST":             "list",
	"PATCH":            "patch",
	"POST":             "create",
	"PROXY":            "proxy",
	"PUT":              "update",
	"WATCH":            "watch",
	"WATCHLIST":        "watch",
}

// errEmptyName is returned when API requests do not fill the name section of the path.
var errEmptyName = errors.NewBadRequest("name must be provided")

// Installs handlers for API resources.
func (a *APIInstaller) Install(ws *restful.WebService) (apiResources []metav1.APIResource, errors []error) {
	errors = make([]error, 0)

	proxyHandler := (&handlers.ProxyHandler{
		Prefix:     a.prefix + "/proxy/",
		Storage:    a.group.Storage,
		Serializer: a.group.Serializer,
		Mapper:     a.group.Context,
	})

	// Register the paths in a deterministic (sorted) order to get a deterministic swagger spec.
	paths := make([]string, len(a.group.Storage))
	var i int = 0
	for path := range a.group.Storage {
		paths[i] = path
		i++
	}
	sort.Strings(paths)
	for _, path := range paths {
		apiResource, err := a.registerResourceHandlers(path, a.group.Storage[path], ws, proxyHandler)
		if err != nil {
			errors = append(errors, fmt.Errorf("error in registering resource: %s, %v", path, err))
		}
		if apiResource != nil {
			apiResources = append(apiResources, *apiResource)
		}
	}
	return apiResources, errors
}

// NewWebService creates a new restful webservice with the api installer's prefix and version.
func (a *APIInstaller) NewWebService() *restful.WebService {
	ws := new(restful.WebService)
	ws.Path(a.prefix)
	// a.prefix contains "prefix/group/version"
	ws.Doc("API at " + a.prefix)
	// Backwards compatibility, we accepted objects with empty content-type at V1.
	// If we stop using go-restful, we can default empty content-type to application/json on an
	// endpoint by endpoint basis
	ws.Consumes("*/*")
	mediaTypes, streamMediaTypes := negotiation.MediaTypesForSerializer(a.group.Serializer)
	ws.Produces(append(mediaTypes, streamMediaTypes...)...)
	ws.ApiVersion(a.group.GroupVersion.String())

	return ws
}

// getResourceKind returns the external group version kind registered for the given storage
// object. If the storage object is a subresource and has an override supplied for it, it returns
// the group version kind supplied in the override.
func (a *APIInstaller) getResourceKind(path string, storage rest.Storage) (schema.GroupVersionKind, error) {
	if fqKindToRegister, ok := a.group.SubresourceGroupVersionKind[path]; ok {
		return fqKindToRegister, nil
	}

	object := storage.New()
	fqKinds, _, err := a.group.Typer.ObjectKinds(object)
	if err != nil {
		return schema.GroupVersionKind{}, err
	}

	// a given go type can have multiple potential fully qualified kinds.  Find the one that corresponds with the group
	// we're trying to register here
	fqKindToRegister := schema.GroupVersionKind{}
	for _, fqKind := range fqKinds {
		if fqKind.Group == a.group.GroupVersion.Group {
			fqKindToRegister = a.group.GroupVersion.WithKind(fqKind.Kind)
			break
		}

		// TODO This keeps it doing what it was doing before, but it doesn't feel right.
		if fqKind.Group == extensions.GroupName && fqKind.Kind == "ThirdPartyResourceData" {
			fqKindToRegister = a.group.GroupVersion.WithKind(fqKind.Kind)
		}
	}
	if fqKindToRegister.Empty() {
		return schema.GroupVersionKind{}, fmt.Errorf("unable to locate fully qualified kind for %v: found %v when registering for %v", reflect.TypeOf(object), fqKinds, a.group.GroupVersion)
	}
	return fqKindToRegister, nil
}

// restMapping returns rest mapper for the resource.
// Example REST paths that this mapper maps.
// 1. Resource only, no subresource:
//      Resource Type:    batch/v1.Job (input args: resource = "jobs")
//      REST path:        /apis/batch/v1/namespaces/{namespace}/job/{name}
// 2. Subresource and its parent belong to different API groups and/or versions:
//      Resource Type:    extensions/v1beta1.ReplicaSet (input args: resource = "replicasets")
//      Subresource Type: autoscaling/v1.Scale
//      REST path:        /apis/extensions/v1beta1/namespaces/{namespace}/replicaset/{name}/scale
func (a *APIInstaller) restMapping(resource string) (*meta.RESTMapping, error) {
	// subresources must have parent resources, and follow the namespacing rules of their parent.
	// So get the storage of the resource (which is the parent resource in case of subresources)
	storage, ok := a.group.Storage[resource]
	if !ok {
		return nil, fmt.Errorf("unable to locate the storage object for resource: %s", resource)
	}
	fqKindToRegister, err := a.getResourceKind(resource, storage)
	if err != nil {
		return nil, fmt.Errorf("unable to locate fully qualified kind for mapper resource %s: %v", resource, err)
	}
	return a.group.Mapper.RESTMapping(fqKindToRegister.GroupKind(), fqKindToRegister.Version)
}

func (a *APIInstaller) registerResourceHandlers(path string, storage rest.Storage, ws *restful.WebService, proxyHandler http.Handler) (*metav1.APIResource, error) {
	admit := a.group.Admit
	context := a.group.Context

	optionsExternalVersion := a.group.GroupVersion
	if a.group.OptionsExternalVersion != nil {
		optionsExternalVersion = *a.group.OptionsExternalVersion
	}

	resource, subresource, err := splitSubresource(path)
	if err != nil {
		return nil, err
	}

	mapping, err := a.restMapping(resource)
	if err != nil {
		return nil, err
	}

	fqKindToRegister, err := a.getResourceKind(path, storage)
	if err != nil {
		return nil, err
	}

	versionedPtr, err := a.group.Creater.New(fqKindToRegister)
	if err != nil {
		return nil, err
	}
	defaultVersionedObject := indirectArbitraryPointer(versionedPtr)
	kind := fqKindToRegister.Kind
	hasSubresource := len(subresource) > 0

	// what verbs are supported by the storage, used to know what verbs we support per path
	creater, isCreater := storage.(rest.Creater)
	namedCreater, isNamedCreater := storage.(rest.NamedCreater)
	lister, isLister := storage.(rest.Lister)
	getter, isGetter := storage.(rest.Getter)
	getterWithOptions, isGetterWithOptions := storage.(rest.GetterWithOptions)
	deleter, isDeleter := storage.(rest.Deleter)
	gracefulDeleter, isGracefulDeleter := storage.(rest.GracefulDeleter)
	collectionDeleter, isCollectionDeleter := storage.(rest.CollectionDeleter)
	updater, isUpdater := storage.(rest.Updater)
	patcher, isPatcher := storage.(rest.Patcher)
	watcher, isWatcher := storage.(rest.Watcher)
	_, isRedirector := storage.(rest.Redirector)
	connecter, isConnecter := storage.(rest.Connecter)
	storageMeta, isMetadata := storage.(rest.StorageMetadata)
	if !isMetadata {
		storageMeta = defaultStorageMetadata{}
	}
	exporter, isExporter := storage.(rest.Exporter)
	if !isExporter {
		exporter = nil
	}

	versionedExportOptions, err := a.group.Creater.New(optionsExternalVersion.WithKind("ExportOptions"))
	if err != nil {
		return nil, err
	}

	if isNamedCreater {
		isCreater = true
	}

	var versionedList interface{}
	if isLister {
		list := lister.NewList()
		listGVKs, _, err := a.group.Typer.ObjectKinds(list)
		if err != nil {
			return nil, err
		}
		versionedListPtr, err := a.group.Creater.New(a.group.GroupVersion.WithKind(listGVKs[0].Kind))
		if err != nil {
			return nil, err
		}
		versionedList = indirectArbitraryPointer(versionedListPtr)
	}

	versionedListOptions, err := a.group.Creater.New(optionsExternalVersion.WithKind("ListOptions"))
	if err != nil {
		return nil, err
	}

	var versionedDeleteOptions runtime.Object
	var versionedDeleterObject interface{}
	switch {
	case isGracefulDeleter:
		versionedDeleteOptions, err = a.group.Creater.New(optionsExternalVersion.WithKind("DeleteOptions"))
		if err != nil {
			return nil, err
		}
		versionedDeleterObject = indirectArbitraryPointer(versionedDeleteOptions)
		isDeleter = true
	case isDeleter:
		gracefulDeleter = rest.GracefulDeleteAdapter{Deleter: deleter}
	}

	versionedStatusPtr, err := a.group.Creater.New(optionsExternalVersion.WithKind("Status"))
	if err != nil {
		return nil, err
	}
	versionedStatus := indirectArbitraryPointer(versionedStatusPtr)
	var (
		getOptions             runtime.Object
		versionedGetOptions    runtime.Object
		getOptionsInternalKind schema.GroupVersionKind
		getSubpath             bool
	)
	if isGetterWithOptions {
		getOptions, getSubpath, _ = getterWithOptions.NewGetOptions()
		getOptionsInternalKinds, _, err := a.group.Typer.ObjectKinds(getOptions)
		if err != nil {
			return nil, err
		}
		getOptionsInternalKind = getOptionsInternalKinds[0]
		versionedGetOptions, err = a.group.Creater.New(optionsExternalVersion.WithKind(getOptionsInternalKind.Kind))
		if err != nil {
			return nil, err
		}
		isGetter = true
	}

	var versionedWatchEvent interface{}
	if isWatcher {
		versionedWatchEventPtr, err := a.group.Creater.New(a.group.GroupVersion.WithKind("WatchEvent"))
		if err != nil {
			return nil, err
		}
		versionedWatchEvent = indirectArbitraryPointer(versionedWatchEventPtr)
	}

	var (
		connectOptions             runtime.Object
		versionedConnectOptions    runtime.Object
		connectOptionsInternalKind schema.GroupVersionKind
		connectSubpath             bool
	)
	if isConnecter {
		connectOptions, connectSubpath, _ = connecter.NewConnectOptions()
		if connectOptions != nil {
			connectOptionsInternalKinds, _, err := a.group.Typer.ObjectKinds(connectOptions)
			if err != nil {
				return nil, err
			}

			connectOptionsInternalKind = connectOptionsInternalKinds[0]
			versionedConnectOptions, err = a.group.Creater.New(optionsExternalVersion.WithKind(connectOptionsInternalKind.Kind))
			if err != nil {
				return nil, err
			}
		}
	}

	var ctxFn handlers.ContextFunc
	ctxFn = func(req *restful.Request) request.Context {
		if context == nil {
			return request.WithUserAgent(request.NewContext(), req.HeaderParameter("User-Agent"))
		}
		if ctx, ok := context.Get(req.Request); ok {
			return request.WithUserAgent(ctx, req.HeaderParameter("User-Agent"))
		}
		return request.WithUserAgent(request.NewContext(), req.HeaderParameter("User-Agent"))
	}

	allowWatchList := isWatcher && isLister // watching on lists is allowed only for kinds that support both watch and list.
	scope := mapping.Scope
	nameParam := ws.PathParameter("name", "name of the "+kind).DataType("string")
	pathParam := ws.PathParameter("path", "path to the resource").DataType("string")

	params := []*restful.Parameter{}
	actions := []action{}

	var resourceKind string
	kindProvider, ok := storage.(rest.KindProvider)
	if ok {
		resourceKind = kindProvider.Kind()
	} else {
		resourceKind = kind
	}

	var apiResource metav1.APIResource
	// Get the list of actions for the given scope.
	switch scope.Name() {
	case meta.RESTScopeNameRoot:
		// Handle non-namespace scoped resources like nodes.
		resourcePath := resource
		resourceParams := params
		itemPath := resourcePath + "/{name}"
		nameParams := append(params, nameParam)
		proxyParams := append(nameParams, pathParam)
		suffix := ""
		if hasSubresource {
			suffix = "/" + subresource
			itemPath = itemPath + suffix
			resourcePath = itemPath
			resourceParams = nameParams
		}
		apiResource.Name = path
		apiResource.Namespaced = false
		apiResource.Kind = resourceKind
		namer := rootScopeNaming{scope, a.group.Linker, gpath.Join(a.prefix, resourcePath, "/"), suffix}

		// Handler for standard REST verbs (GET, PUT, POST and DELETE).
		// Add actions at the resource path: /api/apiVersion/resource
		actions = appendIf(actions, action{"LIST", resourcePath, resourceParams, namer, false}, isLister)
		actions = appendIf(actions, action{"POST", resourcePath, resourceParams, namer, false}, isCreater)
		actions = appendIf(actions, action{"DELETECOLLECTION", resourcePath, resourceParams, namer, false}, isCollectionDeleter)
		// DEPRECATED
		actions = appendIf(actions, action{"WATCHLIST", "watch/" + resourcePath, resourceParams, namer, false}, allowWatchList)

		// Add actions at the item path: /api/apiVersion/resource/{name}
		actions = appendIf(actions, action{"GET", itemPath, nameParams, namer, false}, isGetter)
		if getSubpath {
			actions = appendIf(actions, action{"GET", itemPath + "/{path:*}", proxyParams, namer, false}, isGetter)
		}
		actions = appendIf(actions, action{"PUT", itemPath, nameParams, namer, false}, isUpdater)
		actions = appendIf(actions, action{"PATCH", itemPath, nameParams, namer, false}, isPatcher)
		actions = appendIf(actions, action{"DELETE", itemPath, nameParams, namer, false}, isDeleter)
		actions = appendIf(actions, action{"WATCH", "watch/" + itemPath, nameParams, namer, false}, isWatcher)
		// We add "proxy" subresource to remove the need for the generic top level prefix proxy.
		// The generic top level prefix proxy is deprecated in v1.2, and will be removed in 1.3, or 1.4 at the latest.
		// TODO: DEPRECATED in v1.2.
		actions = appendIf(actions, action{"PROXY", "proxy/" + itemPath + "/{path:*}", proxyParams, namer, false}, isRedirector)
		// TODO: DEPRECATED in v1.2.
		actions = appendIf(actions, action{"PROXY", "proxy/" + itemPath, nameParams, namer, false}, isRedirector)
		actions = appendIf(actions, action{"CONNECT", itemPath, nameParams, namer, false}, isConnecter)
		actions = appendIf(actions, action{"CONNECT", itemPath + "/{path:*}", proxyParams, namer, false}, isConnecter && connectSubpath)
		break
	case meta.RESTScopeNameNamespace:
		// Handler for standard REST verbs (GET, PUT, POST and DELETE).
		namespaceParam := ws.PathParameter(scope.ArgumentName(), scope.ParamDescription()).DataType("string")
		namespacedPath := scope.ParamName() + "/{" + scope.ArgumentName() + "}/" + resource
		namespaceParams := []*restful.Parameter{namespaceParam}

		resourcePath := namespacedPath
		resourceParams := namespaceParams
		itemPathPrefix := gpath.Join(a.prefix, scope.ParamName()) + "/"
		itemPath := namespacedPath + "/{name}"
		itemPathMiddle := "/" + resource + "/"
		nameParams := append(namespaceParams, nameParam)
		proxyParams := append(nameParams, pathParam)
		itemPathSuffix := ""
		if hasSubresource {
			itemPathSuffix = "/" + subresource
			itemPath = itemPath + itemPathSuffix
			resourcePath = itemPath
			resourceParams = nameParams
		}
		apiResource.Name = path
		apiResource.Namespaced = true
		apiResource.Kind = resourceKind

		itemPathFn := func(name, namespace string) bytes.Buffer {
			var buf bytes.Buffer
			buf.WriteString(itemPathPrefix)
			buf.WriteString(url.QueryEscape(namespace))
			buf.WriteString(itemPathMiddle)
			buf.WriteString(url.QueryEscape(name))
			buf.WriteString(itemPathSuffix)
			return buf
		}
		namer := scopeNaming{scope, a.group.Linker, itemPathFn, false}

		actions = appendIf(actions, action{"LIST", resourcePath, resourceParams, namer, false}, isLister)
		actions = appendIf(actions, action{"POST", resourcePath, resourceParams, namer, false}, isCreater)
		actions = appendIf(actions, action{"DELETECOLLECTION", resourcePath, resourceParams, namer, false}, isCollectionDeleter)
		// DEPRECATED
		actions = appendIf(actions, action{"WATCHLIST", "watch/" + resourcePath, resourceParams, namer, false}, allowWatchList)

		actions = appendIf(actions, action{"GET", itemPath, nameParams, namer, false}, isGetter)
		if getSubpath {
			actions = appendIf(actions, action{"GET", itemPath + "/{path:*}", proxyParams, namer, false}, isGetter)
		}
		actions = appendIf(actions, action{"PUT", itemPath, nameParams, namer, false}, isUpdater)
		actions = appendIf(actions, action{"PATCH", itemPath, nameParams, namer, false}, isPatcher)
		actions = appendIf(actions, action{"DELETE", itemPath, nameParams, namer, false}, isDeleter)
		actions = appendIf(actions, action{"WATCH", "watch/" + itemPath, nameParams, namer, false}, isWatcher)
		// We add "proxy" subresource to remove the need for the generic top level prefix proxy.
		// The generic top level prefix proxy is deprecated in v1.2, and will be removed in 1.3, or 1.4 at the latest.
		// TODO: DEPRECATED in v1.2.
		actions = appendIf(actions, action{"PROXY", "proxy/" + itemPath + "/{path:*}", proxyParams, namer, false}, isRedirector)
		// TODO: DEPRECATED in v1.2.
		actions = appendIf(actions, action{"PROXY", "proxy/" + itemPath, nameParams, namer, false}, isRedirector)
		actions = appendIf(actions, action{"CONNECT", itemPath, nameParams, namer, false}, isConnecter)
		actions = appendIf(actions, action{"CONNECT", itemPath + "/{path:*}", proxyParams, namer, false}, isConnecter && connectSubpath)

		// list or post across namespace.
		// For ex: LIST all pods in all namespaces by sending a LIST request at /api/apiVersion/pods.
		// TODO: more strongly type whether a resource allows these actions on "all namespaces" (bulk delete)
		if !hasSubresource {
			namer = scopeNaming{scope, a.group.Linker, itemPathFn, true}
			actions = appendIf(actions, action{"LIST", resource, params, namer, true}, isLister)
			actions = appendIf(actions, action{"WATCHLIST", "watch/" + resource, params, namer, true}, allowWatchList)
		}
		break
	default:
		return nil, fmt.Errorf("unsupported restscope: %s", scope.Name())
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

	mediaTypes, streamMediaTypes := negotiation.MediaTypesForSerializer(a.group.Serializer)
	allMediaTypes := append(mediaTypes, streamMediaTypes...)
	ws.Produces(allMediaTypes...)

	kubeVerbs := map[string]struct{}{}
	reqScope := handlers.RequestScope{
		ContextFunc:    ctxFn,
		Serializer:     a.group.Serializer,
		ParameterCodec: a.group.ParameterCodec,
		Creater:        a.group.Creater,
		Convertor:      a.group.Convertor,
		Copier:         a.group.Copier,

		// TODO: This seems wrong for cross-group subresources. It makes an assumption that a subresource and its parent are in the same group version. Revisit this.
		Resource:    a.group.GroupVersion.WithResource(resource),
		Subresource: subresource,
		Kind:        fqKindToRegister,
	}
	for _, action := range actions {
		versionedObject := storageMeta.ProducesObject(action.Verb)
		if versionedObject == nil {
			versionedObject = defaultVersionedObject
		}
		reqScope.Namer = action.Namer
		namespaced := ""
		if apiResource.Namespaced {
			namespaced = "Namespaced"
		}
		operationSuffix := ""
		if strings.HasSuffix(action.Path, "/{path:*}") {
			operationSuffix = operationSuffix + "WithPath"
		}
		if action.AllNamespaces {
			operationSuffix = operationSuffix + "ForAllNamespaces"
			namespaced = ""
		}

		if kubeVerb, found := toDiscoveryKubeVerb[action.Verb]; found {
			if len(kubeVerb) != 0 {
				kubeVerbs[kubeVerb] = struct{}{}
			}
		} else {
			return nil, fmt.Errorf("unknown action verb for discovery: %s", action.Verb)
		}

		switch action.Verb {
		case "GET": // Get a resource.
			var handler restful.RouteFunction
			if isGetterWithOptions {
				handler = handlers.GetResourceWithOptions(getterWithOptions, reqScope)
			} else {
				handler = handlers.GetResource(getter, exporter, reqScope)
			}
			handler = metrics.InstrumentRouteFunc(action.Verb, resource, handler)
			doc := "read the specified " + kind
			if hasSubresource {
				doc = "read " + subresource + " of the specified " + kind
			}
			route := ws.GET(action.Path).To(handler).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("read"+namespaced+kind+strings.Title(subresource)+operationSuffix).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), mediaTypes...)...).
				Returns(http.StatusOK, "OK", versionedObject).
				Writes(versionedObject)
			if isGetterWithOptions {
				if err := addObjectParams(ws, route, versionedGetOptions); err != nil {
					return nil, err
				}
			}
			if isExporter {
				if err := addObjectParams(ws, route, versionedExportOptions); err != nil {
					return nil, err
				}
			}
			addParams(route, action.Params)
			ws.Route(route)
		case "LIST": // List all resources of a kind.
			doc := "list objects of kind " + kind
			if hasSubresource {
				doc = "list " + subresource + " of objects of kind " + kind
			}
			handler := metrics.InstrumentRouteFunc(action.Verb, resource, handlers.ListResource(lister, watcher, reqScope, false, a.minRequestTimeout))
			route := ws.GET(action.Path).To(handler).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("list"+namespaced+kind+strings.Title(subresource)+operationSuffix).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), allMediaTypes...)...).
				Returns(http.StatusOK, "OK", versionedList).
				Writes(versionedList)
			if err := addObjectParams(ws, route, versionedListOptions); err != nil {
				return nil, err
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
			handler := metrics.InstrumentRouteFunc(action.Verb, resource, handlers.UpdateResource(updater, reqScope, a.group.Typer, admit))
			route := ws.PUT(action.Path).To(handler).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("replace"+namespaced+kind+strings.Title(subresource)+operationSuffix).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), mediaTypes...)...).
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
			handler := metrics.InstrumentRouteFunc(action.Verb, resource, handlers.PatchResource(patcher, reqScope, admit, mapping.ObjectConvertor))
			route := ws.PATCH(action.Path).To(handler).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Consumes(string(api.JSONPatchType), string(api.MergePatchType), string(api.StrategicMergePatchType)).
				Operation("patch"+namespaced+kind+strings.Title(subresource)+operationSuffix).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), mediaTypes...)...).
				Returns(http.StatusOK, "OK", versionedObject).
				Reads(metav1.Patch{}).
				Writes(versionedObject)
			addParams(route, action.Params)
			ws.Route(route)
		case "POST": // Create a resource.
			var handler restful.RouteFunction
			if isNamedCreater {
				handler = handlers.CreateNamedResource(namedCreater, reqScope, a.group.Typer, admit)
			} else {
				handler = handlers.CreateResource(creater, reqScope, a.group.Typer, admit)
			}
			handler = metrics.InstrumentRouteFunc(action.Verb, resource, handler)
			article := getArticleForNoun(kind, " ")
			doc := "create" + article + kind
			if hasSubresource {
				doc = "create " + subresource + " of" + article + kind
			}
			route := ws.POST(action.Path).To(handler).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("create"+namespaced+kind+strings.Title(subresource)+operationSuffix).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), mediaTypes...)...).
				Returns(http.StatusOK, "OK", versionedObject).
				Reads(versionedObject).
				Writes(versionedObject)
			addParams(route, action.Params)
			ws.Route(route)
		case "DELETE": // Delete a resource.
			article := getArticleForNoun(kind, " ")
			doc := "delete" + article + kind
			if hasSubresource {
				doc = "delete " + subresource + " of" + article + kind
			}
			handler := metrics.InstrumentRouteFunc(action.Verb, resource, handlers.DeleteResource(gracefulDeleter, isGracefulDeleter, reqScope, admit))
			route := ws.DELETE(action.Path).To(handler).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("delete"+namespaced+kind+strings.Title(subresource)+operationSuffix).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), mediaTypes...)...).
				Writes(versionedStatus).
				Returns(http.StatusOK, "OK", versionedStatus)
			if isGracefulDeleter {
				route.Reads(versionedDeleterObject)
				if err := addObjectParams(ws, route, versionedDeleteOptions); err != nil {
					return nil, err
				}
			}
			addParams(route, action.Params)
			ws.Route(route)
		case "DELETECOLLECTION":
			doc := "delete collection of " + kind
			if hasSubresource {
				doc = "delete collection of " + subresource + " of a " + kind
			}
			handler := metrics.InstrumentRouteFunc(action.Verb, resource, handlers.DeleteCollection(collectionDeleter, isCollectionDeleter, reqScope, admit))
			route := ws.DELETE(action.Path).To(handler).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("deletecollection"+namespaced+kind+strings.Title(subresource)+operationSuffix).
				Produces(append(storageMeta.ProducesMIMETypes(action.Verb), mediaTypes...)...).
				Writes(versionedStatus).
				Returns(http.StatusOK, "OK", versionedStatus)
			if err := addObjectParams(ws, route, versionedListOptions); err != nil {
				return nil, err
			}
			addParams(route, action.Params)
			ws.Route(route)
		// TODO: deprecated
		case "WATCH": // Watch a resource.
			doc := "watch changes to an object of kind " + kind
			if hasSubresource {
				doc = "watch changes to " + subresource + " of an object of kind " + kind
			}
			handler := metrics.InstrumentRouteFunc(action.Verb, resource, handlers.ListResource(lister, watcher, reqScope, true, a.minRequestTimeout))
			route := ws.GET(action.Path).To(handler).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("watch"+namespaced+kind+strings.Title(subresource)+operationSuffix).
				Produces(allMediaTypes...).
				Returns(http.StatusOK, "OK", versionedWatchEvent).
				Writes(versionedWatchEvent)
			if err := addObjectParams(ws, route, versionedListOptions); err != nil {
				return nil, err
			}
			addParams(route, action.Params)
			ws.Route(route)
		// TODO: deprecated
		case "WATCHLIST": // Watch all resources of a kind.
			doc := "watch individual changes to a list of " + kind
			if hasSubresource {
				doc = "watch individual changes to a list of " + subresource + " of " + kind
			}
			handler := metrics.InstrumentRouteFunc(action.Verb, resource, handlers.ListResource(lister, watcher, reqScope, true, a.minRequestTimeout))
			route := ws.GET(action.Path).To(handler).
				Doc(doc).
				Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
				Operation("watch"+namespaced+kind+strings.Title(subresource)+"List"+operationSuffix).
				Produces(allMediaTypes...).
				Returns(http.StatusOK, "OK", versionedWatchEvent).
				Writes(versionedWatchEvent)
			if err := addObjectParams(ws, route, versionedListOptions); err != nil {
				return nil, err
			}
			addParams(route, action.Params)
			ws.Route(route)
		// We add "proxy" subresource to remove the need for the generic top level prefix proxy.
		// The generic top level prefix proxy is deprecated in v1.2, and will be removed in 1.3, or 1.4 at the latest.
		// TODO: DEPRECATED in v1.2.
		case "PROXY": // Proxy requests to a resource.
			// Accept all methods as per http://issue.k8s.io/3996
			addProxyRoute(ws, "GET", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params, operationSuffix)
			addProxyRoute(ws, "PUT", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params, operationSuffix)
			addProxyRoute(ws, "POST", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params, operationSuffix)
			addProxyRoute(ws, "DELETE", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params, operationSuffix)
			addProxyRoute(ws, "HEAD", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params, operationSuffix)
			addProxyRoute(ws, "OPTIONS", a.prefix, action.Path, proxyHandler, namespaced, kind, resource, subresource, hasSubresource, action.Params, operationSuffix)
		case "CONNECT":
			for _, method := range connecter.ConnectMethods() {
				doc := "connect " + method + " requests to " + kind
				if hasSubresource {
					doc = "connect " + method + " requests to " + subresource + " of " + kind
				}
				handler := metrics.InstrumentRouteFunc(action.Verb, resource, handlers.ConnectResource(connecter, reqScope, admit, path))
				route := ws.Method(method).Path(action.Path).
					To(handler).
					Doc(doc).
					Operation("connect" + strings.Title(strings.ToLower(method)) + namespaced + kind + strings.Title(subresource) + operationSuffix).
					Produces("*/*").
					Consumes("*/*").
					Writes("string")
				if versionedConnectOptions != nil {
					if err := addObjectParams(ws, route, versionedConnectOptions); err != nil {
						return nil, err
					}
				}
				addParams(route, action.Params)
				ws.Route(route)
			}
		default:
			return nil, fmt.Errorf("unrecognized action verb: %s", action.Verb)
		}
		// Note: update GetAuthorizerAttributes() when adding a custom handler.
	}

	apiResource.Verbs = make([]string, 0, len(kubeVerbs))
	for kubeVerb := range kubeVerbs {
		apiResource.Verbs = append(apiResource.Verbs, kubeVerb)
	}
	sort.Strings(apiResource.Verbs)

	return &apiResource, nil
}

// rootScopeNaming reads only names from a request and ignores namespaces. It implements ScopeNamer
// for root scoped resources.
type rootScopeNaming struct {
	scope meta.RESTScope
	runtime.SelfLinker
	pathPrefix string
	pathSuffix string
}

// rootScopeNaming implements ScopeNamer
var _ handlers.ScopeNamer = rootScopeNaming{}

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
func (n rootScopeNaming) GenerateLink(req *restful.Request, obj runtime.Object) (uri string, err error) {
	_, name, err := n.ObjectName(obj)
	if err != nil {
		return "", err
	}
	if len(name) == 0 {
		_, name, err = n.Name(req)
		if err != nil {
			return "", err
		}
	}
	return n.pathPrefix + url.QueryEscape(name) + n.pathSuffix, nil
}

// GenerateListLink returns the appropriate path and query to locate a list by its canonical path.
func (n rootScopeNaming) GenerateListLink(req *restful.Request) (uri string, err error) {
	if len(req.Request.URL.RawPath) > 0 {
		return req.Request.URL.RawPath, nil
	}
	return req.Request.URL.EscapedPath(), nil
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
	itemPathFn    func(name, namespace string) bytes.Buffer
	allNamespaces bool
}

// scopeNaming implements ScopeNamer
var _ handlers.ScopeNamer = scopeNaming{}

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
func (n scopeNaming) GenerateLink(req *restful.Request, obj runtime.Object) (uri string, err error) {
	namespace, name, err := n.ObjectName(obj)
	if err != nil {
		return "", err
	}
	if len(namespace) == 0 && len(name) == 0 {
		namespace, name, err = n.Name(req)
		if err != nil {
			return "", err
		}
	}
	if len(name) == 0 {
		return "", errEmptyName
	}
	result := n.itemPathFn(name, namespace)
	return result.String(), nil
}

// GenerateListLink returns the appropriate path and query to locate a list by its canonical path.
func (n scopeNaming) GenerateListLink(req *restful.Request) (uri string, err error) {
	if len(req.Request.URL.RawPath) > 0 {
		return req.Request.URL.RawPath, nil
	}
	return req.Request.URL.EscapedPath(), nil
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

func addProxyRoute(ws *restful.WebService, method string, prefix string, path string, proxyHandler http.Handler, namespaced, kind, resource, subresource string, hasSubresource bool, params []*restful.Parameter, operationSuffix string) {
	doc := "proxy " + method + " requests to " + kind
	if hasSubresource {
		doc = "proxy " + method + " requests to " + subresource + " of " + kind
	}
	handler := metrics.InstrumentRouteFunc("PROXY", resource, routeFunction(proxyHandler))
	proxyRoute := ws.Method(method).Path(path).To(handler).
		Doc(doc).
		Operation("proxy" + strings.Title(method) + namespaced + kind + strings.Title(subresource) + operationSuffix).
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
			case reflect.Ptr:
				// TODO: This is a hack to let metav1.Time through. This needs to be fixed in a more generic way eventually. bug #36191
				if (sf.Type.Elem().Kind() == reflect.Interface || sf.Type.Elem().Kind() == reflect.Struct) && strings.TrimPrefix(sf.Type.String(), "*") != "metav1.Time" {
					continue
				}
				fallthrough
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
				route.Param(ws.QueryParameter(jsonName, desc).DataType(typeToJSON(sf.Type.String())))
			}
		}
	}
	return nil
}

// TODO: this is incomplete, expand as needed.
// Convert the name of a golang type to the name of a JSON type
func typeToJSON(typeName string) string {
	switch typeName {
	case "bool", "*bool":
		return "boolean"
	case "uint8", "*uint8", "int", "*int", "int32", "*int32", "int64", "*int64", "uint32", "*uint32", "uint64", "*uint64":
		return "integer"
	case "float64", "*float64", "float32", "*float32":
		return "number"
	case "metav1.Time", "*metav1.Time":
		return "string"
	case "byte", "*byte":
		return "string"
	case "[]string", "[]*string":
		// TODO: Fix this when go-restful supports a way to specify an array query param:
		// https://github.com/emicklei/go-restful/issues/225
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

func (defaultStorageMetadata) ProducesObject(verb string) interface{} {
	return nil
}

// splitSubresource checks if the given storage path is the path of a subresource and returns
// the resource and subresource components.
func splitSubresource(path string) (string, string, error) {
	var resource, subresource string
	switch parts := strings.Split(path, "/"); len(parts) {
	case 2:
		resource, subresource = parts[0], parts[1]
	case 1:
		resource = parts[0]
	default:
		// TODO: support deeper paths
		return "", "", fmt.Errorf("api_installer allows only one or two segment paths (resource or resource/subresource)")
	}
	return resource, subresource, nil
}

// getArticleForNoun returns the article needed for the given noun.
func getArticleForNoun(noun string, padding string) string {
	if noun[len(noun)-2:] != "ss" && noun[len(noun)-1:] == "s" {
		// Plurals don't have an article.
		// Don't catch words like class
		return fmt.Sprintf("%v", padding)
	}

	article := "a"
	if isVowel(rune(noun[0])) {
		article = "an"
	}

	return fmt.Sprintf("%s%s%s", padding, article, padding)
}

// isVowel returns true if the rune is a vowel (case insensitive).
func isVowel(c rune) bool {
	vowels := []rune{'a', 'e', 'i', 'o', 'u'}
	for _, value := range vowels {
		if value == unicode.ToLower(c) {
			return true
		}
	}
	return false
}
