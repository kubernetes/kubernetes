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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	watchjson "github.com/GoogleCloudPlatform/kubernetes/pkg/watch/json"
	"github.com/golang/glog"
)

// specialParams lists parameters that are handled specially and which users of Request
// are therefore not allowed to set manually.
var specialParams = util.NewStringSet("sync", "timeout")

// PollFunc is called when a server operation returns 202 accepted. The name of the
// operation is extracted from the response and passed to this function. Return a
// request to retrieve the result of the operation, or false for the second argument
// if polling should end.
type PollFunc func(name string) (*Request, bool)

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

	// optional, will be invoked if the server returns a 202 to decide
	// whether to poll.
	poller PollFunc

	// If true, add "?namespace=<namespace>" as a query parameter, if false put ns/<namespace> in path
	// Query parameter is considered legacy behavior
	namespaceInQuery bool
	// If true, lowercase resource prior to inserting into a path, if false, leave it as is. Preserving
	// case is considered legacy behavior.
	preserveResourceCase bool

	// generic components accessible via method setters
	path    string
	subpath string
	params  map[string]string

	// structural elements of the request that are part of the Kubernetes API conventions
	namespace    string
	namespaceSet bool
	resource     string
	resourceName string
	selector     labels.Selector
	sync         bool
	timeout      time.Duration

	// output
	err  error
	body io.Reader
}

// NewRequest creates a new request helper object for accessing runtime.Objects on a server.
func NewRequest(client HTTPClient, verb string, baseURL *url.URL,
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

// Sync sets sync/async call status by setting the "sync" parameter to "true"/"false".
func (r *Request) Sync(sync bool) *Request {
	if r.err != nil {
		return r
	}
	r.sync = sync
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

// ParseSelectorParam parses the given string as a resource label selector.
// This is a convenience function so you don't have to first check that it's a
// validly formatted selector.
func (r *Request) ParseSelectorParam(paramName, item string) *Request {
	if r.err != nil {
		return r
	}
	sel, err := labels.ParseSelector(item)
	if err != nil {
		r.err = err
		return r
	}
	return r.setParam(paramName, sel.String())
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
		r.params = make(map[string]string)
	}
	r.params[paramName] = value
	return r
}

// Timeout makes the request use the given duration as a timeout. Sets the "timeout"
// parameter. Ignored if sync=false.
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

// NoPoll indicates a server "working" response should be returned as an error
func (r *Request) NoPoll() *Request {
	return r.Poller(nil)
}

// Poller indicates this request should use the specified poll function to determine whether
// a server "working" response should be retried. The poller is responsible for waiting or
// outputting messages to the client.
func (r *Request) Poller(poller PollFunc) *Request {
	if r.err != nil {
		return r
	}
	r.poller = poller
	return r
}

func (r *Request) finalURL() string {
	p := r.path
	if r.namespaceSet && !r.namespaceInQuery && len(r.namespace) > 0 {
		p = path.Join(p, "ns", r.namespace)
	}
	if len(r.resource) != 0 {
		resource := r.resource
		if !r.preserveResourceCase {
			resource = strings.ToLower(resource)
		}
		p = path.Join(p, resource)
	}
	// Join trims trailing slashes, so preserve r.path's trailing slash for backwards compat if nothing was changed
	if len(r.resourceName) != 0 || len(r.subpath) != 0 {
		p = path.Join(p, r.resourceName, r.subpath)
	}

	finalURL := *r.baseURL
	finalURL.Path = p

	query := url.Values{}
	for key, value := range r.params {
		query.Add(key, value)
	}

	if r.namespaceSet && r.namespaceInQuery && len(r.namespace) > 0 {
		query.Add("namespace", r.namespace)
	}

	// sync and timeout are handled specially here, to allow setting them
	// in any order.
	if r.sync {
		query.Add("sync", "true")
		if r.timeout != 0 {
			query.Add("timeout", r.timeout.String())
		}
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

// Do formats and executes the request. Returns a Result object for easy response
// processing. Handles polling the server in the event a continuation was sent.
//
// Error type:
//  * If the request can't be constructed, or an error happened earlier while building its
//    arguments: *RequestConstructionError
//  * If the server responds with a status: *errors.StatusError or *errors.UnexpectedObjectError
//  * If the status code and body don't make sense together: *UnexpectedStatusError
//  * http.Client.Do errors are returned directly.
func (r *Request) Do() Result {
	client := r.client
	if client == nil {
		client = http.DefaultClient
	}

	// Right now we make about ten retry attempts if we get a Retry-After response.
	// TODO: Change to a timeout based approach.
	retries := 0

	for {
		if r.err != nil {
			return Result{err: &RequestConstructionError{r.err}}
		}

		req, err := http.NewRequest(r.verb, r.finalURL(), r.body)
		if err != nil {
			return Result{err: &RequestConstructionError{err}}
		}

		resp, err := client.Do(req)
		if err != nil {
			return Result{err: err}
		}

		respBody, created, err := r.transformResponse(resp, req)
		if poll, ok := r.shouldPoll(err); ok {
			r = poll
			continue
		}

		if resp.StatusCode == http.StatusServiceUnavailable {
			if retries < 10 {
				retries++
				if waitFor := resp.Header.Get("Retry-After"); waitFor != "" {
					delay, err := strconv.Atoi(waitFor)
					if err == nil {
						glog.V(4).Infof("Got a Retry-After %s response for attempt %d to %v", waitFor, retries, r.finalURL())
						time.Sleep(time.Duration(delay) * time.Second)
						continue
					}
				}
			}
		}
		return Result{respBody, created, err, r.codec}
	}
}

// shouldPoll checks the server error for an incomplete operation
// and if found returns a request that would check the response.
// If no polling is necessary or possible, it will return false.
func (r *Request) shouldPoll(err error) (*Request, bool) {
	if err == nil || r.poller == nil {
		return nil, false
	}
	apistatus, ok := err.(APIStatus)
	if !ok {
		return nil, false
	}
	status := apistatus.Status()
	if status.Status != api.StatusWorking {
		return nil, false
	}
	if status.Details == nil || len(status.Details.ID) == 0 {
		return nil, false
	}
	return r.poller(status.Details.ID)
}

// transformResponse converts an API response into a structured API object.
func (r *Request) transformResponse(resp *http.Response, req *http.Request) ([]byte, bool, error) {
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, false, err
	}

	// Did the server give us a status response?
	isStatusResponse := false
	var status api.Status
	if err := r.codec.DecodeInto(body, &status); err == nil && status.Status != "" {
		isStatusResponse = true
	}

	switch {
	case resp.StatusCode < http.StatusOK || resp.StatusCode > http.StatusPartialContent:
		if !isStatusResponse {
			var err error = &UnexpectedStatusError{
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
	if isStatusResponse && status.Status != api.StatusSuccess {
		// "Working" requests need to be handled specially.
		// "Failed" requests are clearly just an error and it makes sense to return them as such.
		return nil, false, errors.FromObject(&status)
	}

	created := resp.StatusCode == http.StatusCreated
	return body, created, err
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
