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
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/latest"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/apiserver/metrics"
	"k8s.io/kubernetes/pkg/healthz"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/flushwriter"
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

	Root    string
	Version string

	// ServerVersion controls the Kubernetes APIVersion used for common objects in the apiserver
	// schema like api.Status, api.DeleteOptions, and api.ListOptions. Other implementors may
	// define a version "v1beta1" but want to use the Kubernetes "v1" internal objects. If
	// empty, defaults to Version.
	ServerVersion string

	Mapper meta.RESTMapper

	Codec     runtime.Codec
	Typer     runtime.ObjectTyper
	Creater   runtime.ObjectCreater
	Convertor runtime.ObjectConvertor
	Linker    runtime.SelfLinker

	Admit   admission.Interface
	Context api.RequestContextMapper

	ProxyDialerFn     ProxyDialerFunc
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
// in a slash. A restful WebService is created for the group and version.
func (g *APIGroupVersion) InstallREST(container *restful.Container) error {
	info := &APIRequestInfoResolver{util.NewStringSet(strings.TrimPrefix(g.Root, "/")), g.Mapper}

	prefix := path.Join(g.Root, g.Version)
	installer := &APIInstaller{
		group:             g,
		info:              info,
		prefix:            prefix,
		minRequestTimeout: g.MinRequestTimeout,
		proxyDialerFn:     g.ProxyDialerFn,
	}
	ws, registrationErrors := installer.Install()
	container.Add(ws)
	return errors.NewAggregate(registrationErrors)
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

func InstallServiceErrorHandler(container *restful.Container, requestResolver *APIRequestInfoResolver, apiVersions []string) {
	container.ServiceErrorHandler(func(serviceErr restful.ServiceError, request *restful.Request, response *restful.Response) {
		serviceErrorHandler(requestResolver, apiVersions, serviceErr, request, response)
	})
}

func serviceErrorHandler(requestResolver *APIRequestInfoResolver, apiVersions []string, serviceErr restful.ServiceError, request *restful.Request, response *restful.Response) {
	requestInfo, err := requestResolver.GetAPIRequestInfo(request.Request)
	codec := latest.Codec
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

// Adds a service to return the supported api versions.
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
	output, err := codec.Encode(object)
	if err != nil {
		errorJSONFatal(err, codec, w)
		return
	}
	if pretty {
		// PR #2243: Pretty-print JSON by default.
		formatted := &bytes.Buffer{}
		err = json.Indent(formatted, output, "", "  ")
		if err != nil {
			errorJSONFatal(err, codec, w)
			return
		}
		output = formatted.Bytes()
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	w.Write(output)
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
	// TODO: change back to 30s once #5180 is fixed
	return 2 * time.Minute
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
