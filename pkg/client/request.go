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

package client

import (
	"bytes"
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/httpstream"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	watchjson "github.com/GoogleCloudPlatform/kubernetes/pkg/watch/json"
	"github.com/golang/glog"
)

// specialParams lists parameters that are handled specially and which users of Request
// are therefore not allowed to set manually.
var specialParams = util.NewStringSet("timeout")

// HTTPClient is an interface for testing a request object.
type HTTPClient interface {
	Do(req *http.Request) (*http.Response, error)
}

// UnexpectedStatusError is returned as an error if a response's body and HTTP code don't
// make sense together.
type UnexpectedStatusError struct {
	Request  *http.Request
	Response *http.Response
	Body     string
}

// Error returns a textual description of 'u'.
func (u *UnexpectedStatusError) Error() string {
	return fmt.Sprintf("request [%+v] failed (%d) %s: %s", u.Request, u.Response.StatusCode, u.Response.Status, u.Body)
}

// IsUnexpectedStatusError determines if err is due to an unexpected status from the server.
func IsUnexpectedStatusError(err error) bool {
	_, ok := err.(*UnexpectedStatusError)
	return ok
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

	// If true, add "?namespace=<namespace>" as a query parameter, if false put ns/<namespace> in path
	// Query parameter is considered legacy behavior
	namespaceInQuery bool
	// If true, lowercase resource prior to inserting into a path, if false, leave it as is. Preserving
	// case is considered legacy behavior.
	preserveResourceCase bool

	// generic components accessible via method setters
	path    string
	subpath string
	params  url.Values

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
	codec runtime.Codec, namespaceInQuery bool, preserveResourceCase bool) *Request {
	return &Request{
		client:  client,
		verb:    verb,
		baseURL: baseURL,
		path:    baseURL.Path,

		codec:                codec,
		namespaceInQuery:     namespaceInQuery,
		preserveResourceCase: preserveResourceCase,
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

// ParseSelectorParam parses the given string as a resource selector.
// This is a convenience function so you don't have to first check that it's a
// validly formatted selector.
func (r *Request) ParseSelectorParam(paramName, item string) *Request {
	if r.err != nil {
		return r
	}
	var selector string
	var err error
	switch paramName {
	case "labels":
		var lsel labels.Selector
		if lsel, err = labels.Parse(item); err == nil {
			selector = lsel.String()
		}
	case "fields":
		var fsel fields.Selector
		if fsel, err = fields.ParseSelector(item); err == nil {
			selector = fsel.String()
		}
	default:
		err = fmt.Errorf("unknown parameter name '%s'", paramName)
	}
	if err != nil {
		r.err = err
		return r
	}
	return r.setParam(paramName, selector)
}

// SelectorParam adds the given selector as a query parameter with the name paramName.
func (r *Request) SelectorParam(paramName string, s labels.Selector) *Request {
	if r.err != nil {
		return r
	}
	if s.Empty() {
		return r
	}
	return r.setParam(paramName, s.String())
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
		r.body = bytes.NewBuffer(data)
	case []byte:
		r.body = bytes.NewBuffer(t)
	case io.Reader:
		r.body = t
	case runtime.Object:
		data, err := r.codec.Encode(t)
		if err != nil {
			r.err = err
			return r
		}
		r.body = bytes.NewBuffer(data)
	default:
		r.err = fmt.Errorf("unknown type used for body: %+v", obj)
	}
	return r
}

func (r *Request) finalURL() string {
	p := r.path
	if r.namespaceSet && !r.namespaceInQuery && len(r.namespace) > 0 {
		p = path.Join(p, "namespaces", r.namespace)
	}
	if len(r.resource) != 0 {
		resource := r.resource
		if !r.preserveResourceCase {
			resource = strings.ToLower(resource)
		}
		p = path.Join(p, resource)
	}
	// Join trims trailing slashes, so preserve r.path's trailing slash for backwards compat if nothing was changed
	if len(r.resourceName) != 0 || len(r.subpath) != 0 || len(r.subresource) != 0 {
		p = path.Join(p, r.resourceName, r.subresource, r.subpath)
	}

	finalURL := *r.baseURL
	finalURL.Path = p

	query := url.Values{}
	for key, values := range r.params {
		for _, value := range values {
			query.Add(key, value)
		}
	}

	if r.namespaceSet && r.namespaceInQuery {
		query.Set("namespace", r.namespace)
	}

	// timeout is handled specially here.
	if r.timeout != 0 {
		query.Set("timeout", r.timeout.String())
	}
	finalURL.RawQuery = query.Encode()
	return finalURL.String()
}

// Watch attempts to begin watching the requested location.
// Returns a watch.Interface, or an error.
func (r *Request) Watch() (watch.Interface, error) {
	if r.err != nil {
		return nil, r.err
	}
	req, err := http.NewRequest(r.verb, r.finalURL(), r.body)
	if err != nil {
		return nil, err
	}
	client := r.client
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(req)
	if err != nil {
		if isProbableEOF(err) {
			return watch.NewEmptyWatch(), nil
		}
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		var body []byte
		if resp.Body != nil {
			body, _ = ioutil.ReadAll(resp.Body)
		}
		return nil, fmt.Errorf("for request '%+v', got status: %v\nbody: %v", req.URL, resp.StatusCode, string(body))
	}
	return watch.NewStreamWatcher(watchjson.NewDecoder(resp.Body, r.codec)), nil
}

// isProbableEOF returns true if the given error resembles a connection termination
// scenario that would justify assuming that the watch is empty. The watch stream
// mechanism handles many common partial data errors, so closed connections can be
// retried in many cases.
func isProbableEOF(err error) bool {
	if uerr, ok := err.(*url.Error); ok {
		err = uerr.Err
	}
	switch {
	case err == io.EOF:
		return true
	case err.Error() == "http: can't write HTTP request on broken connection":
		return true
	case strings.Contains(err.Error(), "connection reset by peer"):
		return true
	}
	return false
}

// Stream formats and executes the request, and offers streaming of the response.
// Returns io.ReadCloser which could be used for streaming of the response, or an error
func (r *Request) Stream() (io.ReadCloser, error) {
	if r.err != nil {
		return nil, r.err
	}
	req, err := http.NewRequest(r.verb, r.finalURL(), nil)
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
	return resp.Body, nil
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

	req, err := http.NewRequest(r.verb, r.finalURL(), nil)
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

// DoRaw executes a raw request which is not subject to interpretation as an API response.
func (r *Request) DoRaw() ([]byte, error) {
	client := r.client
	if client == nil {
		client = http.DefaultClient
	}

	// Right now we make about ten retry attempts if we get a Retry-After response.
	// TODO: Change to a timeout based approach.
	retries := 0

	for {
		if r.err != nil {
			return nil, r.err
		}

		// TODO: added to catch programmer errors (invoking operations with an object with an empty namespace)
		if (r.verb == "GET" || r.verb == "PUT" || r.verb == "DELETE") && r.namespaceSet && len(r.resourceName) > 0 && len(r.namespace) == 0 {
			return nil, fmt.Errorf("an empty namespace may not be set when a resource name is provided")
		}
		if (r.verb == "POST") && r.namespaceSet && len(r.namespace) == 0 {
			return nil, fmt.Errorf("an empty namespace may not be set during creation")
		}

		var err error
		r.req, err = http.NewRequest(r.verb, r.finalURL(), r.body)
		if err != nil {
			return nil, err
		}
		r.resp, err = client.Do(r.req)
		if err != nil {
			return nil, err
		}
		defer r.resp.Body.Close()

		// Check to see if we got a 429 Too Many Requests response code.
		if r.resp.StatusCode == errors.StatusTooManyRequests {
			if retries < 10 {
				retries++
				if waitFor := r.resp.Header.Get("Retry-After"); waitFor != "" {
					delay, err := strconv.Atoi(waitFor)
					if err == nil {
						glog.V(4).Infof("Got a Retry-After %s response for attempt %d to %v", waitFor, retries, r.finalURL())
						time.Sleep(time.Duration(delay) * time.Second)
						continue
					}
				}
			}
		}
		body, err := ioutil.ReadAll(r.resp.Body)
		if err != nil {
			return nil, err
		}
		return body, err
	}
}

// Do formats and executes the request. Returns a Result object for easy response
// processing.
//
// Error type:
//  * If the request can't be constructed, or an error happened earlier while building its
//    arguments: *RequestConstructionError
//  * If the server responds with a status: *errors.StatusError or *errors.UnexpectedObjectError
//  * If the status code and body don't make sense together: *UnexpectedStatusError
//  * http.Client.Do errors are returned directly.
func (r *Request) Do() Result {
	body, err := r.DoRaw()
	if err != nil {
		return Result{err: err}
	}
	respBody, created, err := r.transformResponse(body, r.resp, r.req)
	return Result{respBody, created, err, r.codec}
}

// transformResponse converts an API response into a structured API object.
func (r *Request) transformResponse(body []byte, resp *http.Response, req *http.Request) ([]byte, bool, error) {
	// Did the server give us a status response?
	isStatusResponse := false
	var status api.Status
	if err := r.codec.DecodeInto(body, &status); err == nil && status.Status != "" {
		isStatusResponse = true
	}

	switch {
	case resp.StatusCode == http.StatusSwitchingProtocols:
		// no-op, we've been upgraded
	case resp.StatusCode < http.StatusOK || resp.StatusCode > http.StatusPartialContent:
		if !isStatusResponse {
			var err error
			err = &UnexpectedStatusError{
				Request:  req,
				Response: resp,
				Body:     string(body),
			}
			// TODO: handle other error classes we know about
			switch resp.StatusCode {
			case http.StatusConflict:
				if req.Method == "POST" {
					err = errors.NewAlreadyExists(r.resource, r.resourceName)
				} else {
					err = errors.NewConflict(r.resource, r.resourceName, err)
				}
			case http.StatusNotFound:
				err = errors.NewNotFound(r.resource, r.resourceName)
			case http.StatusBadRequest:
				err = errors.NewBadRequest(err.Error())
			}
			return nil, false, err
		}
		return nil, false, errors.FromObject(&status)
	}

	// If the server gave us a status back, look at what it was.
	success := resp.StatusCode >= http.StatusOK && resp.StatusCode <= http.StatusPartialContent
	if isStatusResponse && (status.Status != api.StatusSuccess && !success) {
		// "Working" requests need to be handled specially.
		// "Failed" requests are clearly just an error and it makes sense to return them as such.
		return nil, false, errors.FromObject(&status)
	}

	created := resp.StatusCode == http.StatusCreated
	return body, created, nil
}

// Result contains the result of calling Request.Do().
type Result struct {
	body    []byte
	created bool
	err     error

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
	*wasCreated = r.created
	return r
}

// Error returns the error executing the request, nil if no error occurred.
// See the Request.Do() comment for what errors you might get.
func (r Result) Error() error {
	return r.err
}
