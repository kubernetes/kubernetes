/*
Copyright 2014 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/rest"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apiserver/metrics"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/flushwriter"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wsstream"
	"k8s.io/kubernetes/pkg/version"

	"github.com/emicklei/go-restful"
	"github.com/golang/glog"
)

func init() {
	metrics.Register()
}

type APIResourceLister interface {
	ListAPIResources() []metav1.APIResource
}

// APIGroupVersion is a helper for exposing rest.Storage objects as http.Handlers via go-restful
// It handles URLs of the form:
// /${storage_key}[/${object_name}]
// Where 'storage_key' points to a rest.Storage object stored in storage.
// This object should contain all parameterization necessary for running a particular API version
type APIGroupVersion struct {
	Storage map[string]rest.Storage

	Root string

	// GroupVersion is the external group version
	GroupVersion schema.GroupVersion

	// OptionsExternalVersion controls the Kubernetes APIVersion used for common objects in the apiserver
	// schema like api.Status, api.DeleteOptions, and api.ListOptions. Other implementors may
	// define a version "v1beta1" but want to use the Kubernetes "v1" internal objects. If
	// empty, defaults to GroupVersion.
	OptionsExternalVersion *schema.GroupVersion

	Mapper meta.RESTMapper

	// Serializer is used to determine how to convert responses from API methods into bytes to send over
	// the wire.
	Serializer     runtime.NegotiatedSerializer
	ParameterCodec runtime.ParameterCodec

	Typer     runtime.ObjectTyper
	Creater   runtime.ObjectCreater
	Convertor runtime.ObjectConvertor
	Copier    runtime.ObjectCopier
	Linker    runtime.SelfLinker

	Admit   admission.Interface
	Context api.RequestContextMapper

	MinRequestTimeout time.Duration

	// SubresourceGroupVersionKind contains the GroupVersionKind overrides for each subresource that is
	// accessible from this API group version. The GroupVersionKind is that of the external version of
	// the subresource. The key of this map should be the path of the subresource. The keys here should
	// match the keys in the Storage map above for subresources.
	SubresourceGroupVersionKind map[string]schema.GroupVersionKind

	// ResourceLister is an interface that knows how to list resources
	// for this API Group.
	ResourceLister APIResourceLister
}

type ProxyDialerFunc func(network, addr string) (net.Conn, error)

// TODO: Pipe these in through the apiserver cmd line
const (
	// Minimum duration before timing out read/write requests
	MinTimeoutSecs = 300
	// Maximum duration before timing out read/write requests
	MaxTimeoutSecs = 600
)

// staticLister implements the APIResourceLister interface
type staticLister struct {
	list []metav1.APIResource
}

func (s staticLister) ListAPIResources() []metav1.APIResource {
	return s.list
}

var _ APIResourceLister = &staticLister{}

// InstallREST registers the REST handlers (storage, watch, proxy and redirect) into a restful Container.
// It is expected that the provided path root prefix will serve all operations. Root MUST NOT end
// in a slash.
func (g *APIGroupVersion) InstallREST(container *restful.Container) error {
	installer := g.newInstaller()
	ws := installer.NewWebService()
	apiResources, registrationErrors := installer.Install(ws)
	lister := g.ResourceLister
	if lister == nil {
		lister = staticLister{apiResources}
	}
	AddSupportedResourcesWebService(g.Serializer, ws, g.GroupVersion, lister)
	container.Add(ws)
	return utilerrors.NewAggregate(registrationErrors)
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
	lister := g.ResourceLister
	if lister == nil {
		lister = staticLister{apiResources}
	}
	AddSupportedResourcesWebService(g.Serializer, ws, g.GroupVersion, lister)
	return utilerrors.NewAggregate(registrationErrors)
}

// newInstaller is a helper to create the installer.  Used by InstallREST and UpdateREST.
func (g *APIGroupVersion) newInstaller() *APIInstaller {
	prefix := path.Join(g.Root, g.GroupVersion.Group, g.GroupVersion.Version)
	installer := &APIInstaller{
		group:             g,
		prefix:            prefix,
		minRequestTimeout: g.MinRequestTimeout,
	}
	return installer
}

// TODO: needs to perform response type negotiation, this is probably the wrong way to recover panics
func InstallRecoverHandler(s runtime.NegotiatedSerializer, container *restful.Container) {
	container.RecoverHandler(func(panicReason interface{}, httpWriter http.ResponseWriter) {
		logStackOnRecover(s, panicReason, httpWriter)
	})
}

//TODO: Unify with RecoverPanics?
func logStackOnRecover(s runtime.NegotiatedSerializer, panicReason interface{}, w http.ResponseWriter) {
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

	headers := http.Header{}
	if ct := w.Header().Get("Content-Type"); len(ct) > 0 {
		headers.Set("Accept", ct)
	}
	errorNegotiated(apierrors.NewGenericServerResponse(http.StatusInternalServerError, "", api.Resource(""), "", "", 0, false), s, schema.GroupVersion{}, w, &http.Request{Header: headers})
}

// Adds a service to return the supported api versions at the legacy /api.
func AddApiWebService(s runtime.NegotiatedSerializer, container *restful.Container, apiPrefix string, getAPIVersionsFunc func(req *restful.Request) *metav1.APIVersions) {
	// TODO: InstallREST should register each version automatically

	// Because in release 1.1, /api returns response with empty APIVersion, we
	// use StripVersionNegotiatedSerializer to keep the response backwards
	// compatible.
	mediaTypes, _ := mediaTypesForSerializer(s)
	ss := StripVersionNegotiatedSerializer{s}
	versionHandler := APIVersionHandler(ss, getAPIVersionsFunc)
	ws := new(restful.WebService)
	ws.Path(apiPrefix)
	ws.Doc("get available API versions")
	ws.Route(ws.GET("/").To(versionHandler).
		Doc("get available API versions").
		Operation("getAPIVersions").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIVersions{}))
	container.Add(ws)
}

// stripVersionEncoder strips APIVersion field from the encoding output. It's
// used to keep the responses at the discovery endpoints backward compatible
// with release-1.1, when the responses have empty APIVersion.
type stripVersionEncoder struct {
	encoder    runtime.Encoder
	serializer runtime.Serializer
}

func (c stripVersionEncoder) Encode(obj runtime.Object, w io.Writer) error {
	buf := bytes.NewBuffer([]byte{})
	err := c.encoder.Encode(obj, buf)
	if err != nil {
		return err
	}
	roundTrippedObj, gvk, err := c.serializer.Decode(buf.Bytes(), nil, nil)
	if err != nil {
		return err
	}
	gvk.Group = ""
	gvk.Version = ""
	roundTrippedObj.GetObjectKind().SetGroupVersionKind(*gvk)
	return c.serializer.Encode(roundTrippedObj, w)
}

// StripVersionNegotiatedSerializer will return stripVersionEncoder when
// EncoderForVersion is called. See comments for stripVersionEncoder.
type StripVersionNegotiatedSerializer struct {
	runtime.NegotiatedSerializer
}

func (n StripVersionNegotiatedSerializer) EncoderForVersion(encoder runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	serializer, ok := encoder.(runtime.Serializer)
	if !ok {
		// The stripVersionEncoder needs both an encoder and decoder, but is called from a context that doesn't have access to the
		// decoder. We do a best effort cast here (since this code path is only for backwards compatibility) to get access to the caller's
		// decoder.
		panic(fmt.Sprintf("Unable to extract serializer from %#v", encoder))
	}
	versioned := n.NegotiatedSerializer.EncoderForVersion(encoder, gv)
	return stripVersionEncoder{versioned, serializer}
}

func keepUnversioned(group string) bool {
	return group == "" || group == "extensions"
}

// NewApisWebService returns a webservice serving the available api version under /apis.
func NewApisWebService(s runtime.NegotiatedSerializer, apiPrefix string, f func(req *restful.Request) []metav1.APIGroup) *restful.WebService {
	// Because in release 1.1, /apis returns response with empty APIVersion, we
	// use StripVersionNegotiatedSerializer to keep the response backwards
	// compatible.
	ss := StripVersionNegotiatedSerializer{s}
	mediaTypes, _ := mediaTypesForSerializer(s)
	rootAPIHandler := RootAPIHandler(ss, f)
	ws := new(restful.WebService)
	ws.Path(apiPrefix)
	ws.Doc("get available API versions")
	ws.Route(ws.GET("/").To(rootAPIHandler).
		Doc("get available API versions").
		Operation("getAPIVersions").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIGroupList{}))
	return ws
}

// NewGroupWebService returns a webservice serving the supported versions, preferred version, and name
// of a group. E.g., such a web service will be registered at /apis/extensions.
func NewGroupWebService(s runtime.NegotiatedSerializer, path string, group metav1.APIGroup) *restful.WebService {
	ss := s
	if keepUnversioned(group.Name) {
		// Because in release 1.1, /apis/extensions returns response with empty
		// APIVersion, we use StripVersionNegotiatedSerializer to keep the
		// response backwards compatible.
		ss = StripVersionNegotiatedSerializer{s}
	}
	mediaTypes, _ := mediaTypesForSerializer(s)
	groupHandler := GroupHandler(ss, group)
	ws := new(restful.WebService)
	ws.Path(path)
	ws.Doc("get information of a group")
	ws.Route(ws.GET("/").To(groupHandler).
		Doc("get information of a group").
		Operation("getAPIGroup").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIGroup{}))
	return ws
}

// Adds a service to return the supported resources, E.g., a such web service
// will be registered at /apis/extensions/v1.
func AddSupportedResourcesWebService(s runtime.NegotiatedSerializer, ws *restful.WebService, groupVersion schema.GroupVersion, lister APIResourceLister) {
	ss := s
	if keepUnversioned(groupVersion.Group) {
		// Because in release 1.1, /apis/extensions/v1beta1 returns response
		// with empty APIVersion, we use StripVersionNegotiatedSerializer to
		// keep the response backwards compatible.
		ss = StripVersionNegotiatedSerializer{s}
	}
	mediaTypes, _ := mediaTypesForSerializer(s)
	resourceHandler := SupportedResourcesHandler(ss, groupVersion, lister)
	ws.Route(ws.GET("/").To(resourceHandler).
		Doc("get available resources").
		Operation("getAPIResources").
		Produces(mediaTypes...).
		Consumes(mediaTypes...).
		Writes(metav1.APIResourceList{}))
}

// APIVersionHandler returns a handler which will list the provided versions as available.
func APIVersionHandler(s runtime.NegotiatedSerializer, getAPIVersionsFunc func(req *restful.Request) *metav1.APIVersions) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		writeNegotiated(s, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, getAPIVersionsFunc(req))
	}
}

// TODO: Remove in 1.6. Returns if kubectl is older than v1.5.0
func isOldKubectl(userAgent string) bool {
	// example userAgent string: kubectl-1.3/v1.3.8 (linux/amd64) kubernetes/e328d5b
	if !strings.Contains(userAgent, "kubectl") {
		return false
	}
	userAgent = strings.Split(userAgent, " ")[0]
	subs := strings.Split(userAgent, "/")
	if len(subs) != 2 {
		return false
	}
	kubectlVersion, versionErr := version.Parse(subs[1])
	if versionErr != nil {
		return false
	}
	return kubectlVersion.LT(version.MustParse("v1.5.0"))
}

// TODO: Remove in 1.6. This is for backward compatibility with 1.4 kubectl.
// See https://github.com/kubernetes/kubernetes/issues/35791
var groupsWithNewVersionsIn1_5 = sets.NewString("apps", "policy")

// TODO: Remove in 1.6.
func filterAPIGroups(req *restful.Request, groups []metav1.APIGroup) []metav1.APIGroup {
	if !isOldKubectl(req.HeaderParameter("User-Agent")) {
		return groups
	}
	// hide API group that has new versions added in 1.5.
	var ret []metav1.APIGroup
	for _, group := range groups {
		if groupsWithNewVersionsIn1_5.Has(group.Name) {
			continue
		}
		ret = append(ret, group)
	}
	return ret
}

// RootAPIHandler returns a handler which will list the provided groups and versions as available.
func RootAPIHandler(s runtime.NegotiatedSerializer, f func(req *restful.Request) []metav1.APIGroup) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		writeNegotiated(s, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, &metav1.APIGroupList{Groups: filterAPIGroups(req, f(req))})
	}
}

// GroupHandler returns a handler which will return the api.GroupAndVersion of
// the group.
func GroupHandler(s runtime.NegotiatedSerializer, group metav1.APIGroup) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		writeNegotiated(s, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, &group)
	}
}

// SupportedResourcesHandler returns a handler which will list the provided resources as available.
func SupportedResourcesHandler(s runtime.NegotiatedSerializer, groupVersion schema.GroupVersion, lister APIResourceLister) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		writeNegotiated(s, schema.GroupVersion{}, resp.ResponseWriter, req.Request, http.StatusOK, &metav1.APIResourceList{GroupVersion: groupVersion.String(), APIResources: lister.ListAPIResources()})
	}
}

// write renders a returned runtime.Object to the response as a stream or an encoded object. If the object
// returned by the response implements rest.ResourceStreamer that interface will be used to render the
// response. The Accept header and current API version will be passed in, and the output will be copied
// directly to the response body. If content type is returned it is used, otherwise the content type will
// be "application/octet-stream". All other objects are sent to standard JSON serialization.
func write(statusCode int, gv schema.GroupVersion, s runtime.NegotiatedSerializer, object runtime.Object, w http.ResponseWriter, req *http.Request) {
	stream, ok := object.(rest.ResourceStreamer)
	if !ok {
		writeNegotiated(s, gv, w, req, statusCode, object)
		return
	}

	out, flush, contentType, err := stream.InputStream(gv.String(), req.Header.Get("Accept"))
	if err != nil {
		errorNegotiated(err, s, gv, w, req)
		return
	}
	if out == nil {
		// No output provided - return StatusNoContent
		w.WriteHeader(http.StatusNoContent)
		return
	}
	defer out.Close()

	if wsstream.IsWebSocketRequest(req) {
		r := wsstream.NewReader(out, true, wsstream.NewDefaultReaderProtocols())
		if err := r.Copy(w, req); err != nil {
			utilruntime.HandleError(fmt.Errorf("error encountered while streaming results via websocket: %v", err))
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
}

// writeNegotiated renders an object in the content type negotiated by the client
func writeNegotiated(s runtime.NegotiatedSerializer, gv schema.GroupVersion, w http.ResponseWriter, req *http.Request, statusCode int, object runtime.Object) {
	serializer, err := negotiateOutputSerializer(req, s)
	if err != nil {
		status := errToAPIStatus(err)
		WriteRawJSON(int(status.Code), status, w)
		return
	}

	w.Header().Set("Content-Type", serializer.MediaType)
	w.WriteHeader(statusCode)

	encoder := s.EncoderForVersion(serializer.Serializer, gv)
	if err := encoder.Encode(object, w); err != nil {
		errorJSONFatal(err, encoder, w)
	}
}

// errorNegotiated renders an error to the response. Returns the HTTP status code of the error.
func errorNegotiated(err error, s runtime.NegotiatedSerializer, gv schema.GroupVersion, w http.ResponseWriter, req *http.Request) int {
	status := errToAPIStatus(err)
	code := int(status.Code)
	// when writing an error, check to see if the status indicates a retry after period
	if status.Details != nil && status.Details.RetryAfterSeconds > 0 {
		delay := strconv.Itoa(int(status.Details.RetryAfterSeconds))
		w.Header().Set("Retry-After", delay)
	}
	writeNegotiated(s, gv, w, req, code, status)
	return code
}

// errorJSONFatal renders an error to the response, and if codec fails will render plaintext.
// Returns the HTTP status code of the error.
func errorJSONFatal(err error, codec runtime.Encoder, w http.ResponseWriter) int {
	utilruntime.HandleError(fmt.Errorf("apiserver was unable to write a JSON response: %v", err))
	status := errToAPIStatus(err)
	code := int(status.Code)
	output, err := runtime.Encode(codec, status)
	if err != nil {
		w.WriteHeader(code)
		fmt.Fprintf(w, "%s: %s", status.Reason, status.Message)
		return code
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	w.Write(output)
	return code
}

// WriteRawJSON writes a non-API object in JSON.
func WriteRawJSON(statusCode int, object interface{}, w http.ResponseWriter) {
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
