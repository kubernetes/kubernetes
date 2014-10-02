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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
	watchjson "github.com/GoogleCloudPlatform/kubernetes/pkg/watch/json"
	"github.com/golang/glog"
)

// specialParams lists parameters that are handled specially and which users of Request
// are therefore not allowed to set manually.
var specialParams = util.NewStringSet("sync", "timeout")

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
		data, err := r.c.Codec.Encode(t)
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
	finalURL := *r.c.baseURL
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
	client := r.c.Client
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Got status: %v", response.StatusCode)
	}
	return watch.NewStreamWatcher(watchjson.NewDecoder(response.Body, r.c.Codec)), nil
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
	client := r.c.Client
	if client == nil {
		client = http.DefaultClient
	}
	response, err := client.Do(req)
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
			if s, ok := err.(APIStatus); ok {
				status := s.Status()
				if status.Status == api.StatusWorking && r.pollPeriod != 0 {
					if status.Details != nil {
						id := status.Details.ID
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
		return Result{respBody, err, r.c.Codec}
	}
}

// Result contains the result of calling Request.Do().
type Result struct {
	body  []byte
	err   error
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

// Error returns the error executing the request, nil if no error occurred.
func (r Result) Error() error {
	return r.err
}
