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

package unversioned

import (
	"bytes"
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"mime"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/metrics"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/watch"
	watchjson "k8s.io/kubernetes/pkg/watch/json"
)

// specialParams lists parameters that are handled specially and which users of Request
// are therefore not allowed to set manually.
var specialParams = sets.NewString("timeout")

// HTTPClient is an interface for testing a request object.
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// ResponseWrapper is an interface for getting a response.
// The response may be either accessed as a raw data (the whole output is put into memory) or as a stream.
type ResponseWrapper interface {
	DoRaw() ([]byte, error)
	Stream() (io.ReadCloser, error)
}

// RequestConstructionError is returned when there's an error assembling a request.
type RequestConstructionError struct {
	Err error
}

// Error returns a textual description of 'r'.
func (r *RequestConstructionError) Error() string {
	return fmt.Sprintf("request construction error: '%v'", r.Err)
}

// Request allows for building up a request to a server in a chained fashion.
// Any errors are stored until the end of your call, so you only have to
// check once.
type Request struct {
	// required
	client  HTTPClient
	verb    string
	baseURL *url.URL
	codec   runtime.Codec

	// generic components accessible via method setters
	path    string
	subpath string
	params  url.Values
	headers http.Header

	// structural elements of the request that are part of the Kubernetes API conventions
	namespace    string
	namespaceSet bool
	resource     string
	resourceName string
	subresource  string
	selector     labels.Selector
	timeout      time.Duration

	apiVersion string

	// output
	err  error
	body io.Reader

	// The constructed request and the response
	req  *http.Request
	resp *http.Response
}

// NewRequest creates a new request helper object for accessing runtime.Objects on a server.
func NewRequest(client HTTPClient, verb string, baseURL *url.URL, apiVersion string,
	codec runtime.Codec) *Request {
	metrics.Register()
	return &Request{
		client:     client,
		verb:       verb,
		baseURL:    baseURL,
		path:       baseURL.Path,
		apiVersion: apiVersion,
		codec:      codec,
	}
}

// Prefix adds segments to the relative beginning to the request path. These
// items will be placed before the optional Namespace, Resource, or Name sections.
// Setting AbsPath will clear any previously set Prefix segments
func (r *Request) Prefix(segments ...string) *Request {
	if r.err != nil {
		return r
	}
	r.path = path.Join(r.path, path.Join(segments...))
	return r
}

// Suffix appends segments to the end of the path. These items will be placed after the prefix and optional
// Namespace, Resource, or Name sections.
func (r *Request) Suffix(segments ...string) *Request {
	if r.err != nil {
		return r
	}
	r.subpath = path.Join(r.subpath, path.Join(segments...))
	return r
}

// Resource sets the resource to access (<resource>/[ns/<namespace>/]<name>)
func (r *Request) Resource(resource string) *Request {
	if r.err != nil {
		return r
	}
	if len(r.resource) != 0 {
		r.err = fmt.Errorf("resource already set to %q, cannot change to %q", r.resource, resource)
		return r
	}
	r.resource = resource
	return r
}

// SubResource sets a sub-resource path which can be multiple segments segment after the resource
// name but before the suffix.
func (r *Request) SubResource(subresources ...string) *Request {
	if r.err != nil {
		return r
	}
	subresource := path.Join(subresources...)
	if len(r.subresource) != 0 {
		r.err = fmt.Errorf("subresource already set to %q, cannot change to %q", r.resource, subresource)
		return r
	}
	r.subresource = subresource
	return r
}

// Name sets the name of a resource to access (<resource>/[ns/<namespace>/]<name>)
func (r *Request) Name(resourceName string) *Request {
	if r.err != nil {
		return r
	}
	if len(resourceName) == 0 {
		r.err = fmt.Errorf("resource name may not be empty")
		return r
	}
	if len(r.resourceName) != 0 {
		r.err = fmt.Errorf("resource name already set to %q, cannot change to %q", r.resourceName, resourceName)
		return r
	}
	r.resourceName = resourceName
	return r
}

// Namespace applies the namespace scope to a request (<resource>/[ns/<namespace>/]<name>)
func (r *Request) Namespace(namespace string) *Request {
	if r.err != nil {
		return r
	}
	if r.namespaceSet {
		r.err = fmt.Errorf("namespace already set to %q, cannot change to %q", r.namespace, namespace)
		return r
	}
	r.namespaceSet = true
	r.namespace = namespace
	return r
}

// NamespaceIfScoped is a convenience function to set a namespace if scoped is true
func (r *Request) NamespaceIfScoped(namespace string, scoped bool) *Request {
	if scoped {
		return r.Namespace(namespace)
	}
	return r
}

// UnversionedPath strips the apiVersion from the baseURL before appending segments.
func (r *Request) UnversionedPath(segments ...string) *Request {
	if r.err != nil {
		return r
	}
	upath := path.Clean(r.baseURL.Path)
	//TODO(jdef) this is a pretty hackish version test
	if strings.HasPrefix(path.Base(upath), "v") {
		upath = path.Dir(upath)
		if upath == "." {
			upath = "/"
		}
	}
	r.path = path.Join(append([]string{upath}, segments...)...)
	return r
}

// AbsPath overwrites an existing path with the segments provided. Trailing slashes are preserved
// when a single segment is passed.
func (r *Request) AbsPath(segments ...string) *Request {
	if r.err != nil {
		return r
	}
	if len(segments) == 1 {
		// preserve any trailing slashes for legacy behavior
		r.path = segments[0]
	} else {
		r.path = path.Join(segments...)
	}
	return r
}

// RequestURI overwrites existing path and parameters with the value of the provided server relative
// URI. Some parameters (those in specialParameters) cannot be overwritten.
func (r *Request) RequestURI(uri string) *Request {
	if r.err != nil {
		return r
	}
	locator, err := url.Parse(uri)
	if err != nil {
		r.err = err
		return r
	}
	r.path = locator.Path
	if len(locator.Query()) > 0 {
		if r.params == nil {
			r.params = make(url.Values)
		}
		for k, v := range locator.Query() {
			r.params[k] = v
		}
	}
	return r
}

const (
	// A constant that clients can use to refer in a field selector to the object name field.
	// Will be automatically emitted as the correct name for the API version.
	NodeUnschedulable = "spec.unschedulable"
	ObjectNameField   = "metadata.name"
	PodHost           = "spec.nodeName"
	PodStatus         = "status.phase"
	SecretType        = "type"

	EventReason                  = "reason"
	EventSource                  = "source"
	EventInvolvedKind            = "involvedObject.kind"
	EventInvolvedNamespace       = "involvedObject.namespace"
	EventInvolvedName            = "involvedObject.name"
	EventInvolvedUID             = "involvedObject.uid"
	EventInvolvedAPIVersion      = "involvedObject.apiVersion"
	EventInvolvedResourceVersion = "involvedObject.resourceVersion"
	EventInvolvedFieldPath       = "involvedObject.fieldPath"
)

type clientFieldNameToAPIVersionFieldName map[string]string

func (c clientFieldNameToAPIVersionFieldName) filterField(field, value string) (newField, newValue string, err error) {
	newFieldName, ok := c[field]
	if !ok {
		return "", "", fmt.Errorf("%v - %v - no field mapping defined", field, value)
	}
	return newFieldName, value, nil
}

type resourceTypeToFieldMapping map[string]clientFieldNameToAPIVersionFieldName

func (r resourceTypeToFieldMapping) filterField(resourceType, field, value string) (newField, newValue string, err error) {
	fMapping, ok := r[resourceType]
	if !ok {
		return "", "", fmt.Errorf("%v - %v - %v - no field mapping defined", resourceType, field, value)
	}
	return fMapping.filterField(field, value)
}

type versionToResourceToFieldMapping map[string]resourceTypeToFieldMapping

func (v versionToResourceToFieldMapping) filterField(apiVersion, resourceType, field, value string) (newField, newValue string, err error) {
	rMapping, ok := v[apiVersion]
	if !ok {
		glog.Warningf("Field selector: %v - %v - %v - %v: need to check if this is versioned correctly.", apiVersion, resourceType, field, value)
		return field, value, nil
	}
	newField, newValue, err = rMapping.filterField(resourceType, field, value)
	if err != nil {
		// This is only a warning until we find and fix all of the client's usages.
		glog.Warningf("Field selector: %v - %v - %v - %v: need to check if this is versioned correctly.", apiVersion, resourceType, field, value)
		return field, value, nil
	}
	return newField, newValue, nil
}

var fieldMappings = versionToResourceToFieldMapping{
	"v1": resourceTypeToFieldMapping{
		"nodes": clientFieldNameToAPIVersionFieldName{
			ObjectNameField:   "metadata.name",
			NodeUnschedulable: "spec.unschedulable",
		},
		"pods": clientFieldNameToAPIVersionFieldName{
			PodHost:   "spec.nodeName",
			PodStatus: "status.phase",
		},
		"secrets": clientFieldNameToAPIVersionFieldName{
			SecretType: "type",
		},
		"serviceAccounts": clientFieldNameToAPIVersionFieldName{
			ObjectNameField: "metadata.name",
		},
		"endpoints": clientFieldNameToAPIVersionFieldName{
			ObjectNameField: "metadata.name",
		},
		"events": clientFieldNameToAPIVersionFieldName{
			ObjectNameField:              "metadata.name",
			EventReason:                  "reason",
			EventSource:                  "source",
			EventInvolvedKind:            "involvedObject.kind",
			EventInvolvedNamespace:       "involvedObject.namespace",
			EventInvolvedName:            "involvedObject.name",
			EventInvolvedUID:             "involvedObject.uid",
			EventInvolvedAPIVersion:      "involvedObject.apiVersion",
			EventInvolvedResourceVersion: "involvedObject.resourceVersion",
			EventInvolvedFieldPath:       "involvedObject.fieldPath",
		},
	},
}

// FieldsSelectorParam adds the given selector as a query parameter with the name paramName.
func (r *Request) FieldsSelectorParam(s fields.Selector) *Request {
	if r.err != nil {
		return r
	}
	if s == nil {
		return r
	}
	if s.Empty() {
		return r
	}
	s2, err := s.Transform(func(field, value string) (newField, newValue string, err error) {
		return fieldMappings.filterField(r.apiVersion, r.resource, field, value)
	})
	if err != nil {
		r.err = err
		return r
	}
	return r.setParam(api.FieldSelectorQueryParam(r.apiVersion), s2.String())
}

// LabelsSelectorParam adds the given selector as a query parameter
func (r *Request) LabelsSelectorParam(s labels.Selector) *Request {
	if r.err != nil {
		return r
	}
	if s == nil {
		return r
	}
	if s.Empty() {
		return r
	}
	return r.setParam(api.LabelSelectorQueryParam(r.apiVersion), s.String())
}

// UintParam creates a query parameter with the given value.
func (r *Request) UintParam(paramName string, u uint64) *Request {
	if r.err != nil {
		return r
	}
	return r.setParam(paramName, strconv.FormatUint(u, 10))
}

// Param creates a query parameter with the given string value.
func (r *Request) Param(paramName, s string) *Request {
	if r.err != nil {
		return r
	}
	return r.setParam(paramName, s)
}

func (r *Request) setParam(paramName, value string) *Request {
	if specialParams.Has(paramName) {
		r.err = fmt.Errorf("must set %v through the corresponding function, not directly.", paramName)
		return r
	}
	if r.params == nil {
		r.params = make(url.Values)
	}
	r.params[paramName] = append(r.params[paramName], value)
	return r
}

func (r *Request) SetHeader(key, value string) *Request {
	if r.headers == nil {
		r.headers = http.Header{}
	}
	r.headers.Set(key, value)
	return r
}

// Timeout makes the request use the given duration as a timeout. Sets the "timeout"
// parameter.
func (r *Request) Timeout(d time.Duration) *Request {
	if r.err != nil {
		return r
	}
	r.timeout = d
	return r
}

// Body makes the request use obj as the body. Optional.
// If obj is a string, try to read a file of that name.
// If obj is a []byte, send it directly.
// If obj is an io.Reader, use it directly.
// If obj is a runtime.Object, marshal it correctly.
// Otherwise, set an error.
func (r *Request) Body(obj interface{}) *Request {
	if r.err != nil {
		return r
	}
	switch t := obj.(type) {
	case string:
		data, err := ioutil.ReadFile(t)
		if err != nil {
			r.err = err
			return r
		}
		glog.V(8).Infof("Request Body: %s", string(data))
		r.body = bytes.NewBuffer(data)
	case []byte:
		glog.V(8).Infof("Request Body: %s", string(t))
		r.body = bytes.NewBuffer(t)
	case io.Reader:
		r.body = t
	case runtime.Object:
		data, err := r.codec.Encode(t)
		if err != nil {
			r.err = err
			return r
		}
		glog.V(8).Infof("Request Body: %s", string(data))
		r.body = bytes.NewBuffer(data)
	default:
		r.err = fmt.Errorf("unknown type used for body: %+v", obj)
	}
	return r
}

// URL returns the current working URL.
func (r *Request) URL() *url.URL {
	p := r.path
	if r.namespaceSet && len(r.namespace) > 0 {
		p = path.Join(p, "namespaces", r.namespace)
	}
	if len(r.resource) != 0 {
		p = path.Join(p, strings.ToLower(r.resource))
	}
	// Join trims trailing slashes, so preserve r.path's trailing slash for backwards compat if nothing was changed
	if len(r.resourceName) != 0 || len(r.subpath) != 0 || len(r.subresource) != 0 {
		p = path.Join(p, r.resourceName, r.subresource, r.subpath)
	}

	finalURL := &url.URL{}
	if r.baseURL != nil {
		*finalURL = *r.baseURL
	}
	finalURL.Path = p

	query := url.Values{}
	for key, values := range r.params {
		for _, value := range values {
			query.Add(key, value)
		}
	}

	// timeout is handled specially here.
	if r.timeout != 0 {
		query.Set("timeout", r.timeout.String())
	}
	finalURL.RawQuery = query.Encode()
	return finalURL
}

// finalURLTemplate is similar to URL(), but will make all specific parameter values equal
// - instead of name or namespace, "{name}" and "{namespace}" will be used, and all query
// parameters will be reset. This creates a copy of the request so as not to change the
// underyling object.  This means some useful request info (like the types of field
// selectors in use) will be lost.
// TODO: preserve field selector keys
func (r Request) finalURLTemplate() string {
	if len(r.resourceName) != 0 {
		r.resourceName = "{name}"
	}
	if r.namespaceSet && len(r.namespace) != 0 {
		r.namespace = "{namespace}"
	}
	newParams := url.Values{}
	v := []string{"{value}"}
	for k := range r.params {
		newParams[k] = v
	}
	r.params = newParams
	return r.URL().String()
}

// Watch attempts to begin watching the requested location.
// Returns a watch.Interface, or an error.
func (r *Request) Watch() (watch.Interface, error) {
	if r.err != nil {
		return nil, r.err
	}
	url := r.URL().String()
	req, err := http.NewRequest(r.verb, url, r.body)
	if err != nil {
		return nil, err
	}
	client := r.client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		// The watch stream mechanism handles many common partial data errors, so closed
		// connections can be retried in many cases.
		if util.IsProbableEOF(err) {
			return watch.NewEmptyWatch(), nil
		}
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		if result := r.transformResponse(resp, req); result.err != nil {
			return nil, result.err
		}
		return nil, fmt.Errorf("for request '%+v', got status: %v", url, resp.StatusCode)
	}
	return watch.NewStreamWatcher(watchjson.NewDecoder(resp.Body, r.codec)), nil
}

// Stream formats and executes the request, and offers streaming of the response.
// Returns io.ReadCloser which could be used for streaming of the response, or an error
// Any non-2xx http status code causes an error.  If we get a non-2xx code, we try to convert the body into an APIStatus object.
// If we can, we return that as an error.  Otherwise, we create an error that lists the http status and the content of the response.
func (r *Request) Stream() (io.ReadCloser, error) {
	if r.err != nil {
		return nil, r.err
	}
	url := r.URL().String()
	req, err := http.NewRequest(r.verb, url, nil)
	if err != nil {
		return nil, err
	}
	client := r.client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}

	switch {
	case (resp.StatusCode >= 200) && (resp.StatusCode < 300):
		return resp.Body, nil

	default:
		// ensure we close the body before returning the error
		defer resp.Body.Close()

		// we have a decent shot at taking the object returned, parsing it as a status object and returning a more normal error
		bodyBytes, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("%v while accessing %v", resp.Status, url)
		}

		if runtimeObject, err := r.codec.Decode(bodyBytes); err == nil {
			statusError := errors.FromObject(runtimeObject)

			if _, ok := statusError.(APIStatus); ok {
				return nil, statusError
			}
		}

		bodyText := string(bodyBytes)
		return nil, fmt.Errorf("%s while accessing %v: %s", resp.Status, url, bodyText)
	}
}

// Upgrade upgrades the request so that it supports multiplexed bidirectional
// streams. The current implementation uses SPDY, but this could be replaced
// with HTTP/2 once it's available, or something else.
func (r *Request) Upgrade(config *Config, newRoundTripperFunc func(*tls.Config) httpstream.UpgradeRoundTripper) (httpstream.Connection, error) {
	if r.err != nil {
		return nil, r.err
	}

	tlsConfig, err := TLSConfigFor(config)
	if err != nil {
		return nil, err
	}

	upgradeRoundTripper := newRoundTripperFunc(tlsConfig)
	wrapper, err := HTTPWrappersForConfig(config, upgradeRoundTripper)
	if err != nil {
		return nil, err
	}

	r.client = &http.Client{Transport: wrapper}

	req, err := http.NewRequest(r.verb, r.URL().String(), nil)
	if err != nil {
		return nil, fmt.Errorf("Error creating request: %s", err)
	}

	resp, err := r.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Error sending request: %s", err)
	}
	defer resp.Body.Close()

	return upgradeRoundTripper.NewConnection(resp)
}

// request connects to the server and invokes the provided function when a server response is
// received. It handles retry behavior and up front validation of requests. It wil invoke
// fn at most once. It will return an error if a problem occurred prior to connecting to the
// server - the provided function is responsible for handling server errors.
func (r *Request) request(fn func(*http.Request, *http.Response)) error {
	if r.err != nil {
		return r.err
	}

	// TODO: added to catch programmer errors (invoking operations with an object with an empty namespace)
	if (r.verb == "GET" || r.verb == "PUT" || r.verb == "DELETE") && r.namespaceSet && len(r.resourceName) > 0 && len(r.namespace) == 0 {
		return fmt.Errorf("an empty namespace may not be set when a resource name is provided")
	}
	if (r.verb == "POST") && r.namespaceSet && len(r.namespace) == 0 {
		return fmt.Errorf("an empty namespace may not be set during creation")
	}

	client := r.client
	if client == nil {
		client = http.DefaultClient
	}

	// Right now we make about ten retry attempts if we get a Retry-After response.
	// TODO: Change to a timeout based approach.
	maxRetries := 10
	retries := 0
	for {
		url := r.URL().String()
		req, err := http.NewRequest(r.verb, url, r.body)
		if err != nil {
			return err
		}
		req.Header = r.headers

		resp, err := client.Do(req)
		if err != nil {
			return err
		}

		done := func() bool {
			// ensure the response body is closed before we reconnect, so that we reuse the same
			// TCP connection
			defer resp.Body.Close()

			retries++
			if seconds, wait := checkWait(resp); wait && retries < maxRetries {
				glog.V(4).Infof("Got a Retry-After %s response for attempt %d to %v", seconds, retries, url)
				time.Sleep(time.Duration(seconds) * time.Second)
				return false
			}
			fn(req, resp)
			return true
		}()
		if done {
			return nil
		}
	}
}

// Do formats and executes the request. Returns a Result object for easy response
// processing.
//
// Error type:
//  * If the request can't be constructed, or an error happened earlier while building its
//    arguments: *RequestConstructionError
//  * If the server responds with a status: *errors.StatusError or *errors.UnexpectedObjectError
//  * http.Client.Do errors are returned directly.
func (r *Request) Do() Result {
	start := time.Now()
	defer func() {
		metrics.RequestLatency.WithLabelValues(r.verb, r.finalURLTemplate()).Observe(metrics.SinceInMicroseconds(start))
	}()
	var result Result
	err := r.request(func(req *http.Request, resp *http.Response) {
		result = r.transformResponse(resp, req)
	})
	if err != nil {
		return Result{err: err}
	}
	return result
}

// DoRaw executes the request but does not process the response body.
func (r *Request) DoRaw() ([]byte, error) {
	start := time.Now()
	defer func() {
		metrics.RequestLatency.WithLabelValues(r.verb, r.finalURLTemplate()).Observe(metrics.SinceInMicroseconds(start))
	}()
	var result Result
	err := r.request(func(req *http.Request, resp *http.Response) {
		result.body, result.err = ioutil.ReadAll(resp.Body)
	})
	if err != nil {
		return nil, err
	}
	return result.body, result.err
}

// transformResponse converts an API response into a structured API object
func (r *Request) transformResponse(resp *http.Response, req *http.Request) Result {
	var body []byte
	if resp.Body != nil {
		if data, err := ioutil.ReadAll(resp.Body); err == nil {
			body = data
		}
	}
	glog.V(8).Infof("Response Body: %s", string(body))

	// Did the server give us a status response?
	isStatusResponse := false
	var status unversioned.Status
	if err := r.codec.DecodeInto(body, &status); err == nil && status.Status != "" {
		isStatusResponse = true
	}

	switch {
	case resp.StatusCode == http.StatusSwitchingProtocols:
		// no-op, we've been upgraded
	case resp.StatusCode < http.StatusOK || resp.StatusCode > http.StatusPartialContent:
		if !isStatusResponse {
			return Result{err: r.transformUnstructuredResponseError(resp, req, body)}
		}
		return Result{err: errors.FromObject(&status)}
	}

	// If the server gave us a status back, look at what it was.
	success := resp.StatusCode >= http.StatusOK && resp.StatusCode <= http.StatusPartialContent
	if isStatusResponse && (status.Status != unversioned.StatusSuccess && !success) {
		// "Failed" requests are clearly just an error and it makes sense to return them as such.
		return Result{err: errors.FromObject(&status)}
	}

	return Result{
		body:       body,
		statusCode: resp.StatusCode,
		codec:      r.codec,
	}
}

// transformUnstructuredResponseError handles an error from the server that is not in a structured form.
// It is expected to transform any response that is not recognizable as a clear server sent error from the
// K8S API using the information provided with the request. In practice, HTTP proxies and client libraries
// introduce a level of uncertainty to the responses returned by servers that in common use result in
// unexpected responses. The rough structure is:
//
// 1. Assume the server sends you something sane - JSON + well defined error objects + proper codes
//    - this is the happy path
//    - when you get this output, trust what the server sends
// 2. Guard against empty fields / bodies in received JSON and attempt to cull sufficient info from them to
//    generate a reasonable facsimile of the original failure.
//    - Be sure to use a distinct error type or flag that allows a client to distinguish between this and error 1 above
// 3. Handle true disconnect failures / completely malformed data by moving up to a more generic client error
// 4. Distinguish between various connection failures like SSL certificates, timeouts, proxy errors, unexpected
//    initial contact, the presence of mismatched body contents from posted content types
//    - Give these a separate distinct error type and capture as much as possible of the original message
//
// TODO: introduce transformation of generic http.Client.Do() errors that separates 4.
func (r *Request) transformUnstructuredResponseError(resp *http.Response, req *http.Request, body []byte) error {
	if body == nil && resp.Body != nil {
		if data, err := ioutil.ReadAll(resp.Body); err == nil {
			body = data
		}
	}
	glog.V(8).Infof("Response Body: %s", string(body))

	message := "unknown"
	if isTextResponse(resp) {
		message = strings.TrimSpace(string(body))
	}
	retryAfter, _ := retryAfterSeconds(resp)
	return errors.NewGenericServerResponse(resp.StatusCode, req.Method, r.resource, r.resourceName, message, retryAfter, true)
}

// isTextResponse returns true if the response appears to be a textual media type.
func isTextResponse(resp *http.Response) bool {
	contentType := resp.Header.Get("Content-Type")
	if len(contentType) == 0 {
		return true
	}
	media, _, err := mime.ParseMediaType(contentType)
	if err != nil {
		return false
	}
	return strings.HasPrefix(media, "text/")
}

// checkWait returns true along with a number of seconds if the server instructed us to wait
// before retrying.
func checkWait(resp *http.Response) (int, bool) {
	if resp.StatusCode != errors.StatusTooManyRequests {
		return 0, false
	}
	i, ok := retryAfterSeconds(resp)
	return i, ok
}

// retryAfterSeconds returns the value of the Retry-After header and true, or 0 and false if
// the header was missing or not a valid number.
func retryAfterSeconds(resp *http.Response) (int, bool) {
	if h := resp.Header.Get("Retry-After"); len(h) > 0 {
		if i, err := strconv.Atoi(h); err == nil {
			return i, true
		}
	}
	return 0, false
}

// Result contains the result of calling Request.Do().
type Result struct {
	body       []byte
	err        error
	statusCode int

	codec runtime.Codec
}

// Raw returns the raw result.
func (r Result) Raw() ([]byte, error) {
	return r.body, r.err
}

// Get returns the result as an object.
func (r Result) Get() (runtime.Object, error) {
	if r.err != nil {
		return nil, r.err
	}
	return r.codec.Decode(r.body)
}

// StatusCode returns the HTTP status code of the request. (Only valid if no
// error was returned.)
func (r Result) StatusCode(statusCode *int) Result {
	*statusCode = r.statusCode
	return r
}

// Into stores the result into obj, if possible.
func (r Result) Into(obj runtime.Object) error {
	if r.err != nil {
		return r.err
	}
	return r.codec.DecodeInto(r.body, obj)
}

// WasCreated updates the provided bool pointer to whether the server returned
// 201 created or a different response.
func (r Result) WasCreated(wasCreated *bool) Result {
	*wasCreated = r.statusCode == http.StatusCreated
	return r
}

// Error returns the error executing the request, nil if no error occurred.
// See the Request.Do() comment for what errors you might get.
func (r Result) Error() error {
	return r.err
}
