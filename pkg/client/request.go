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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	watchjson "github.com/GoogleCloudPlatform/kubernetes/pkg/watch/json"
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

	// accessible via method setters
	path     string
	params   map[string]string
	selector labels.Selector
	sync     bool
	timeout  time.Duration

	// output
	err  error
	body io.Reader
}

// NewRequest creates a new request with the core attributes.
func NewRequest(client HTTPClient, verb string, baseURL *url.URL, codec runtime.Codec) *Request {
	return &Request{
		client:  client,
		verb:    verb,
		baseURL: baseURL,
		codec:   codec,

		path: baseURL.Path,
	}
}

// Path appends an item to the request path. You must call Path at least once.
func (r *Request) Path(item string) *Request {
	if r.err != nil {
		return r
	}
	r.path = path.Join(r.path, item)
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

// Namespace applies the namespace scope to a request
func (r *Request) Namespace(namespace string) *Request {
	if r.err != nil {
		return r
	}
	if len(namespace) > 0 {
		return r.setParam("namespace", namespace)
	}
	return r
}

// AbsPath overwrites an existing path with the path parameter.
func (r *Request) AbsPath(path string) *Request {
	if r.err != nil {
		return r
	}
	r.path = path
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
		r.err = fmt.Errorf("Unknown type used for body: %#v", obj)
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
	finalURL := *r.baseURL
	finalURL.Path = r.path
	query := url.Values{}
	for key, value := range r.params {
		query.Add(key, value)
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
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Got status: %v", resp.StatusCode)
	}
	return watch.NewStreamWatcher(watchjson.NewDecoder(resp.Body, r.codec)), nil
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
func (r *Request) Do() Result {
	client := r.client
	if client == nil {
		client = http.DefaultClient
	}

	for {
		if r.err != nil {
			return Result{err: r.err}
		}

		req, err := http.NewRequest(r.verb, r.finalURL(), r.body)
		if err != nil {
			return Result{err: err}
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
			return nil, false, fmt.Errorf("request [%#v] failed (%d) %s: %s", req, resp.StatusCode, resp.Status, string(body))
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
func (r Result) Error() error {
	return r.err
}
