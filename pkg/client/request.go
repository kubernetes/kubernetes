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
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"path"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

// Server contains info locating a kubernetes api server.
// Example usage:
// auth, err := LoadAuth(filename)
// c := New(url, auth)
// resp, err := c.Verb("GET").
//	Path("pods").
//	Selector("area=staging").
//	Timeout(10*time.Second).
//	Do()
// list, ok := resp.(api.PodList)

// Begin a request with a verb (GET, POST, PUT, DELETE)
func (c *Client) Verb(verb string) *Request {
	return &Request{
		verb:       verb,
		c:          c,
		path:       "/api/v1beta1",
		sync:       false,
		timeout:    20 * time.Second,
		pollPeriod: 20 * time.Second,
	}
}

// Begin a POST request.
func (c *Client) Post() *Request {
	return c.Verb("POST")
}

// Begin a PUT request.
func (c *Client) Put() *Request {
	return c.Verb("PUT")
}

// Begin a GET request.
func (c *Client) Get() *Request {
	return c.Verb("GET")
}

// Begin a DELETE request.
func (c *Client) Delete() *Request {
	return c.Verb("DELETE")
}

// Make a request to do a single poll of the completion of the given operation.
func (c *Client) PollFor(operationId string) *Request {
	return c.Get().Path("operations").Path(operationId).Sync(false).PollPeriod(0)
}

// Request allows for building up a request to a server in a chained fashion.
// Any errors are stored until the end of your call, so you only have to
// check once.
type Request struct {
	c          *Client
	err        error
	verb       string
	path       string
	body       io.Reader
	selector   labels.Selector
	timeout    time.Duration
	sync       bool
	pollPeriod time.Duration
}

// Append an item to the request path. You must call Path at least once.
func (r *Request) Path(item string) *Request {
	if r.err != nil {
		return r
	}
	r.path = path.Join(r.path, item)
	return r
}

// Set sync/async call status.
func (r *Request) Sync(sync bool) *Request {
	if r.err != nil {
		return r
	}
	r.sync = sync
	return r
}

// Overwrite an existing path with the path parameter.
func (r *Request) AbsPath(path string) *Request {
	if r.err != nil {
		return r
	}
	r.path = path
	return r
}

// Parse the given string as a resource label selector. Optional.
func (r *Request) ParseSelector(item string) *Request {
	if r.err != nil {
		return r
	}
	r.selector, r.err = labels.ParseSelector(item)
	return r
}

// Use the given selector.
func (r *Request) Selector(s labels.Selector) *Request {
	if r.err != nil {
		return r
	}
	r.selector = s
	return r
}

// Use the given duration as a timeout. Optional.
func (r *Request) Timeout(d time.Duration) *Request {
	if r.err != nil {
		return r
	}
	r.timeout = d
	return r
}

// Use obj as the body of the request. Optional.
// If obj is a string, try to read a file of that name.
// If obj is a []byte, send it directly.
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
		r.body = obj.(io.Reader)
	default:
		data, err := api.Encode(obj)
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

// Format and execute the request. Returns the API object received, or an error.
func (r *Request) Do() Result {
	for {
		if r.err != nil {
			return Result{err: r.err}
		}
		finalUrl := r.c.host + r.path
		query := url.Values{}
		if r.selector != nil {
			query.Add("labels", r.selector.String())
		}
		if r.sync {
			query.Add("sync", "true")
			if r.timeout != 0 {
				query.Add("timeout", r.timeout.String())
			}
		}
		finalUrl += "?" + query.Encode()
		req, err := http.NewRequest(r.verb, finalUrl, r.body)
		if err != nil {
			return Result{err: err}
		}
		respBody, err := r.c.doRequest(req)
		if err != nil {
			if statusErr, ok := err.(*StatusErr); ok {
				if statusErr.Status.Status == api.StatusWorking && r.pollPeriod != 0 {
					time.Sleep(r.pollPeriod)
					// Make a poll request
					pollOp := r.c.PollFor(statusErr.Status.Details).PollPeriod(r.pollPeriod)
					// Could also say "return r.Do()" but this way doesn't grow the callstack.
					r = pollOp
					continue
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
	return api.Decode(r.body)
}

// Into stores the result into obj, if possible..
func (r Result) Into(obj interface{}) error {
	if r.err != nil {
		return r.err
	}
	return api.DecodeInto(r.body, obj)
}

// Returns the error executing the request, nil if no error occurred.
func (r Result) Error() error {
	return r.err
}
