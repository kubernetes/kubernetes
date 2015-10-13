/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"path"
	rt "runtime"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apiserver/metrics"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/flushwriter"
	"k8s.io/kubernetes/pkg/util/wsstream"
	"k8s.io/kubernetes/pkg/version"

	"github.com/emicklei/go-restful"
	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"
)

func init() {
	metrics.Register()
}

// monitorFilter creates a filter that reports the metrics for a given resource and action.
func monitorFilter(action, resource string) restful.FilterFunction {
	return func(req *restful.Request, res *restful.Response, chain *restful.FilterChain) {
		reqStart := time.Now()
		chain.ProcessFilter(req, res)
		httpCode := res.StatusCode()
		metrics.Monitor(&action, &resource, util.GetClient(req.Request), &httpCode, reqStart)
	}
}

// mux is an object that can register http handlers.
type Mux interface {
	Handle(pattern string, handler http.Handler)
	HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request))
}

// APIGroupVersion is a helper for exposing rest.Storage objects as http.Handlers via go-restful
// It handles URLs of the form:
// /${storage_key}[/${object_name}]
// Where 'storage_key' points to a rest.Storage object stored in storage.
// This object should contain all parameterization necessary for running a particular API version
type APIGroupVersion struct {
	Storage map[string]rest.Storage

	Root string
	// TODO: caesarxuchao: Version actually contains "group/version", refactor it to avoid confusion.
	Version string

	// APIRequestInfoResolver is used to parse URLs for the legacy proxy handler.  Don't use this for anything else
	// TODO: refactor proxy handler to use sub resources
	APIRequestInfoResolver *APIRequestInfoResolver

	// ServerVersion controls the Kubernetes APIVersion used for common objects in the apiserver
	// schema like api.Status, api.DeleteOptions, and api.ListOptions. Other implementors may
	// define a version "v1beta1" but want to use the Kubernetes "v1" internal objects. If
	// empty, defaults to Version.
	// TODO: caesarxuchao: ServerVersion actually contains "group/version",
	// refactor it to avoid confusion.
	ServerVersion string

	Mapper meta.RESTMapper

	Codec     runtime.Codec
	Typer     runtime.ObjectTyper
	Creater   runtime.ObjectCreater
	Convertor runtime.ObjectConvertor
	Linker    runtime.SelfLinker

	Admit   admission.Interface
	Context api.RequestContextMapper

	MinRequestTimeout time.Duration
}

type ProxyDialerFunc func(network, addr string) (net.Conn, error)

// TODO: Pipe these in through the apiserver cmd line
const (
	// Minimum duration before timing out read/write requests
	MinTimeoutSecs = 300
	// Maximum duration before timing out read/write requests
	MaxTimeoutSecs = 600
)

// InstallREST registers the REST handlers (storage, watch, proxy and redirect) into a restful Container.
// It is expected that the provided path root prefix will serve all operations. Root MUST NOT end
// in a slash.
func (g *APIGroupVersion) InstallREST(container *restful.Container) error {
	installer := g.newInstaller()
	ws := installer.NewWebService()
	apiResources, registrationErrors := installer.Install(ws)
	// TODO: g.Version only contains "version" now, it will contain "group/version" in the near future.
	AddSupportedResourcesWebService(ws, g.Version, apiResources)
	container.Add(ws)
	return errors.NewAggregate(registrationErrors)
}

// UpdateREST registers the REST handlers for this APIGroupVersion to an existing web service
// in the restful Container.  It will use the prefix (root/version) to find the existing
// web service.  If a web service does not exist within the container to support the prefix
// this method will return an error.
func (g *APIGroupVersion) UpdateREST(container *restful.Container) error {
	installer := g.newInstaller()
	var ws *restful.WebService = nil

	for i, s := range container.RegisteredWebServices() {
		if s.RootPath() == installer.prefix {
			ws = container.RegisteredWebServices()[i]
			break
		}
	}

	if ws == nil {
		return apierrors.NewInternalError(fmt.Errorf("unable to find an existing webservice for prefix %s", installer.prefix))
	}
	apiResources, registrationErrors := installer.Install(ws)
	// TODO: g.Version only contains "version" now, it will contain "group/version" in the near future.
	AddSupportedResourcesWebService(ws, g.Version, apiResources)
	return errors.NewAggregate(registrationErrors)
}

// newInstaller is a helper to create the installer.  Used by InstallREST and UpdateREST.
func (g *APIGroupVersion) newInstaller() *APIInstaller {
	prefix := path.Join(g.Root, g.Version)
	installer := &APIInstaller{
		group:             g,
		info:              g.APIRequestInfoResolver,
		prefix:            prefix,
		minRequestTimeout: g.MinRequestTimeout,
	}
	return installer
}

// TODO: document all handlers
// InstallSupport registers the APIServer support functions
func InstallSupport(mux Mux, ws *restful.WebService, enableResettingMetrics bool, checks ...healthz.HealthzChecker) {
	// TODO: convert healthz and metrics to restful and remove container arg
	healthz.InstallHandler(mux, checks...)
	mux.Handle("/metrics", prometheus.Handler())
	if enableResettingMetrics {
		mux.HandleFunc("/resetMetrics", metrics.Reset)
	}

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

func InstallRecoverHandler(container *restful.Container) {
	container.RecoverHandler(logStackOnRecover)
}

//TODO: Unify with RecoverPanics?
func logStackOnRecover(panicReason interface{}, httpWriter http.ResponseWriter) {
	var buffer bytes.Buffer
	buffer.WriteString(fmt.Sprintf("recover from panic situation: - %v\r\n", panicReason))
	for i := 2; ; i += 1 {
		_, file, line, ok := rt.Caller(i)
		if !ok {
			break
		}
		buffer.WriteString(fmt.Sprintf("    %s:%d\r\n", file, line))
	}
	glog.Errorln(buffer.String())

	// TODO: make status unversioned or plumb enough of the request to deduce the requested API version
	errorJSON(apierrors.NewGenericServerResponse(http.StatusInternalServerError, "", "", "", "", 0, false), latest.GroupOrDie("").Codec, httpWriter)
}

func InstallServiceErrorHandler(container *restful.Container, requestResolver *APIRequestInfoResolver, apiVersions []string) {
	container.ServiceErrorHandler(func(serviceErr restful.ServiceError, request *restful.Request, response *restful.Response) {
		serviceErrorHandler(requestResolver, apiVersions, serviceErr, request, response)
	})
}

func serviceErrorHandler(requestResolver *APIRequestInfoResolver, apiVersions []string, serviceErr restful.ServiceError, request *restful.Request, response *restful.Response) {
	requestInfo, err := requestResolver.GetAPIRequestInfo(request.Request)
	codec := latest.GroupOrDie("").Codec
	if err == nil && requestInfo.APIVersion != "" {
		// check if the api version is valid.
		for _, version := range apiVersions {
			if requestInfo.APIVersion == version {
				// valid api version.
				codec = runtime.CodecFor(api.Scheme, requestInfo.APIVersion)
				break
			}
		}
	}

	errorJSON(apierrors.NewGenericServerResponse(serviceErr.Code, "", "", "", "", 0, false), codec, response.ResponseWriter)
}

// Adds a service to return the supported api versions at the legacy /api.
func AddApiWebService(container *restful.Container, apiPrefix string, versions []string) {
	// TODO: InstallREST should register each version automatically

	versionHandler := APIVersionHandler(versions[:]...)
	ws := new(restful.WebService)
	ws.Path(apiPrefix)
	ws.Doc("get available API versions")
	ws.Route(ws.GET("/").To(versionHandler).
		Doc("get available API versions").
		Operation("getAPIVersions").
		Produces(restful.MIME_JSON).
		Consumes(restful.MIME_JSON))
	container.Add(ws)
}

// Adds a service to return the supported api versions at /apis.
func AddApisWebService(container *restful.Container, apiPrefix string, groups []unversioned.APIGroup) {
	rootAPIHandler := RootAPIHandler(groups)
	ws := new(restful.WebService)
	ws.Path(apiPrefix)
	ws.Doc("get available API versions")
	ws.Route(ws.GET("/").To(rootAPIHandler).
		Doc("get available API versions").
		Operation("getAPIVersions").
		Produces(restful.MIME_JSON).
		Consumes(restful.MIME_JSON))
	container.Add(ws)
}

// Adds a service to return the supported versions, preferred version, and name
// of a group. E.g., a such web service will be registered at /apis/extensions.
func AddGroupWebService(container *restful.Container, path string, group unversioned.APIGroup) {
	groupHandler := GroupHandler(group)
	ws := new(restful.WebService)
	ws.Path(path)
	ws.Doc("get information of a group")
	ws.Route(ws.GET("/").To(groupHandler).
		Doc("get information of a group").
		Operation("getAPIGroup").
		Produces(restful.MIME_JSON).
		Consumes(restful.MIME_JSON))
	container.Add(ws)
}

// Adds a service to return the supported resources, E.g., a such web service
// will be registered at /apis/extensions/v1.
func AddSupportedResourcesWebService(ws *restful.WebService, groupVersion string, apiResources []unversioned.APIResource) {
	resourceHandler := SupportedResourcesHandler(groupVersion, apiResources)
	ws.Route(ws.GET("/").To(resourceHandler).
		Doc("get available resources").
		Operation("getAPIResources").
		Produces(restful.MIME_JSON).
		Consumes(restful.MIME_JSON))
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
		writeRawJSON(http.StatusOK, unversioned.APIVersions{Versions: versions}, resp.ResponseWriter)
	}
}

// RootAPIHandler returns a handler which will list the provided groups and versions as available.
func RootAPIHandler(groups []unversioned.APIGroup) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		// TODO: use restful's Response methods
		writeRawJSON(http.StatusOK, unversioned.APIGroupList{Groups: groups}, resp.ResponseWriter)
	}
}

// GroupHandler returns a handler which will return the api.GroupAndVersion of
// the group.
func GroupHandler(group unversioned.APIGroup) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		// TODO: use restful's Response methods
		writeRawJSON(http.StatusOK, group, resp.ResponseWriter)
	}
}

// SupportedResourcesHandler returns a handler which will list the provided resources as available.
func SupportedResourcesHandler(groupVersion string, apiResources []unversioned.APIResource) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		// TODO: use restful's Response methods
		writeRawJSON(http.StatusOK, unversioned.APIResourceList{GroupVersion: groupVersion, APIResources: apiResources}, resp.ResponseWriter)
	}
}

// write renders a returned runtime.Object to the response as a stream or an encoded object. If the object
// returned by the response implements rest.ResourceStreamer that interface will be used to render the
// response. The Accept header and current API version will be passed in, and the output will be copied
// directly to the response body. If content type is returned it is used, otherwise the content type will
// be "application/octet-stream". All other objects are sent to standard JSON serialization.
func write(statusCode int, apiVersion string, codec runtime.Codec, object runtime.Object, w http.ResponseWriter, req *http.Request) {
	if stream, ok := object.(rest.ResourceStreamer); ok {
		out, flush, contentType, err := stream.InputStream(apiVersion, req.Header.Get("Accept"))
		if err != nil {
			errorJSONFatal(err, codec, w)
			return
		}
		if out == nil {
			// No output provided - return StatusNoContent
			w.WriteHeader(http.StatusNoContent)
			return
		}
		defer out.Close()

		if wsstream.IsWebSocketRequest(req) {
			r := wsstream.NewReader(out, true)
			if err := r.Copy(w, req); err != nil {
				util.HandleError(fmt.Errorf("error encountered while streaming results via websocket: %v", err))
			}
			return
		}

		if len(contentType) == 0 {
			contentType = "application/octet-stream"
		}
		w.Header().Set("Content-Type", contentType)
		w.WriteHeader(statusCode)
		writer := w.(io.Writer)
		if flush {
			writer = flushwriter.Wrap(w)
		}
		io.Copy(writer, out)
		return
	}
	writeJSON(statusCode, codec, object, w, isPrettyPrint(req))
}

func isPrettyPrint(req *http.Request) bool {
	pp := req.URL.Query().Get("pretty")
	if len(pp) > 0 {
		pretty, _ := strconv.ParseBool(pp)
		return pretty
	}
	userAgent := req.UserAgent()
	// This covers basic all browers and cli http tools
	if strings.HasPrefix(userAgent, "curl") || strings.HasPrefix(userAgent, "Wget") || strings.HasPrefix(userAgent, "Mozilla/5.0") {
		return true
	}
	return false
}

// writeJSON renders an object as JSON to the response.
func writeJSON(statusCode int, codec runtime.Codec, object runtime.Object, w http.ResponseWriter, pretty bool) {
	w.Header().Set("Content-Type", "application/json")
	// We send the status code before we encode the object, so if we error, the status code stays but there will
	// still be an error object.  This seems ok, the alternative is to validate the object before
	// encoding, but this really should never happen, so it's wasted compute for every API request.
	w.WriteHeader(statusCode)
	if pretty {
		prettyJSON(codec, object, w)
		return
	}
	err := codec.EncodeToStream(object, w)
	if err != nil {
		errorJSONFatal(err, codec, w)
	}
}

func prettyJSON(codec runtime.Codec, object runtime.Object, w http.ResponseWriter) {
	formatted := &bytes.Buffer{}
	output, err := codec.Encode(object)
	if err != nil {
		errorJSONFatal(err, codec, w)
	}
	if err := json.Indent(formatted, output, "", "  "); err != nil {
		errorJSONFatal(err, codec, w)
		return
	}
	w.Write(formatted.Bytes())
}

// errorJSON renders an error to the response. Returns the HTTP status code of the error.
func errorJSON(err error, codec runtime.Codec, w http.ResponseWriter) int {
	status := errToAPIStatus(err)
	writeJSON(status.Code, codec, status, w, true)
	return status.Code
}

// errorJSONFatal renders an error to the response, and if codec fails will render plaintext.
// Returns the HTTP status code of the error.
func errorJSONFatal(err error, codec runtime.Codec, w http.ResponseWriter) int {
	util.HandleError(fmt.Errorf("apiserver was unable to write a JSON response: %v", err))
	status := errToAPIStatus(err)
	output, err := codec.Encode(status)
	if err != nil {
		w.WriteHeader(status.Code)
		fmt.Fprintf(w, "%s: %s", status.Reason, status.Message)
		return status.Code
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status.Code)
	w.Write(output)
	return status.Code
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
