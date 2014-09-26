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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	"github.com/golang/glog"
)

// specialParams lists parameters that are handled specially and which users of Request
// are therefore not allowed to set manually.
var specialParams = util.NewStringSet("sync", "timeout")

// Verb begins a request with a verb (GET, POST, PUT, DELETE).
//
// Example usage of Client's request building interface:
// auth, err := LoadAuth(filename)
// c := New(url, auth)
// resp, err := c.Verb("GET").
//	Path("pods").
//	SelectorParam("labels", "area=staging").
//	Timeout(10*time.Second).
//	Do()
// if err != nil { ... }
// list, ok := resp.(*api.PodList)
//
func (c *RESTClient) Verb(verb string) *Request {
	return &Request{
		verb:       verb,
		c:          c,
		path:       c.prefix,
		sync:       c.Sync,
		timeout:    c.Timeout,
		params:     map[string]string{},
		pollPeriod: c.PollPeriod,
	}
}

// Post begins a POST request. Short for c.Verb("POST").
func (c *RESTClient) Post() *Request {
	return c.Verb("POST")
}

// Put begins a PUT request. Short for c.Verb("PUT").
func (c *RESTClient) Put() *Request {
	return c.Verb("PUT")
}

// Get begins a GET request. Short for c.Verb("GET").
func (c *RESTClient) Get() *Request {
	return c.Verb("GET")
}

// Delete begins a DELETE request. Short for c.Verb("DELETE").
func (c *RESTClient) Delete() *Request {
	return c.Verb("DELETE")
}

// PollFor makes a request to do a single poll of the completion of the given operation.
func (c *RESTClient) PollFor(operationID string) *Request {
	return c.Get().Path("operations").Path(operationID).Sync(false).PollPeriod(0)
}

// Request allows for building up a request to a server in a chained fashion.
// Any errors are stored until the end of your call, so you only have to
// check once.
type Request struct {
	c          *RESTClient
	err        error
	verb       string
	path       string
	body       io.Reader
	params     map[string]string
	selector   labels.Selector
	timeout    time.Duration
	sync       bool
	pollPeriod time.Duration
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

func (r *Request) setParam(paramName, value string) *Request {
	if specialParams.Has(paramName) {
		r.err = fmt.Errorf("must set %v through the corresponding function, not directly.", paramName)
		return r
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
// Otherwise, assume obj is an api type and marshall it correctly.
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
	default:
		data, err := runtime.Encode(obj)
		if err != nil {
			r.err = err
			return r
		}
		r.body = bytes.NewBuffer(data)
	}
	return r
}

// PollPeriod sets the poll period.
// If the server sends back a "working" status message, then repeatedly poll the server
// to see if the operation has completed yet, waiting 'd' between each poll.
// If you want to handle the "working" status yourself (it'll be delivered as StatusErr),
// set d to 0 to turn off this behavior.
func (r *Request) PollPeriod(d time.Duration) *Request {
	if r.err != nil {
		return r
	}
	r.pollPeriod = d
	return r
}

func (r *Request) finalURL() string {
	finalURL := r.c.host + r.path
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
	finalURL += "?" + query.Encode()
	return finalURL
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
	if r.c.auth != nil {
		req.SetBasicAuth(r.c.auth.User, r.c.auth.Password)
	}
	response, err := r.c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Got status: %v", response.StatusCode)
	}
	return watch.NewStreamWatcher(tools.NewAPIEventDecoder(response.Body)), nil
}

// Stream formats and executes the request.
// Returns io.ReadCloser which could be used for streaming of the response, or an error
func (r *Request) Stream() (io.ReadCloser, error) {
	if r.err != nil {
		return nil, r.err
	}
	req, err := http.NewRequest(r.verb, r.finalURL(), nil)
	if err != nil {
		return nil, err
	}
	if r.c.auth != nil {
		req.SetBasicAuth(r.c.auth.User, r.c.auth.Password)
	}
	response, err := r.c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	return response.Body, nil
}

// Do formats and executes the request. Returns the API object received, or an error.
func (r *Request) Do() Result {
	for {
		if r.err != nil {
			return Result{err: r.err}
		}
		req, err := http.NewRequest(r.verb, r.finalURL(), r.body)
		if err != nil {
			return Result{err: err}
		}
		respBody, err := r.c.doRequest(req)
		if err != nil {
			if statusErr, ok := err.(*StatusErr); ok {
				if statusErr.Status.Status == api.StatusWorking && r.pollPeriod != 0 {
					if statusErr.Status.Details != nil {
						id := statusErr.Status.Details.ID
						if len(id) > 0 {
							glog.Infof("Waiting for completion of /operations/%s", id)
							time.Sleep(r.pollPeriod)
							// Make a poll request
							pollOp := r.c.PollFor(id).PollPeriod(r.pollPeriod)
							// Could also say "return r.Do()" but this way doesn't grow the callstack.
							r = pollOp
							continue
						}
					}
				}
			}
		}
		return Result{respBody, err}
	}
}

// Result contains the result of calling Request.Do().
type Result struct {
	body []byte
	err  error
}

// Raw returns the raw result.
func (r Result) Raw() ([]byte, error) {
	return r.body, r.err
}

// Get returns the result as an object.
func (r Result) Get() (interface{}, error) {
	if r.err != nil {
		return nil, r.err
	}
	return runtime.Decode(r.body)
}

// Into stores the result into obj, if possible.
func (r Result) Into(obj interface{}) error {
	if r.err != nil {
		return r.err
	}
	return runtime.DecodeInto(r.body, obj)
}

// Error returns the error executing the request, nil if no error occurred.
func (r Result) Error() error {
	return r.err
}
