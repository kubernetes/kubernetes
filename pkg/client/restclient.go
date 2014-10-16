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
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// RESTClient imposes common Kubernetes API conventions on a set of resource paths.
// The baseURL is expected to point to an HTTP or HTTPS path that is the parent
// of one or more resources.  The server should return a decodable API resource
// object, or an api.Status object which contains information about the reason for
// any failure.
//
// Most consumers should use client.New() to get a Kubernetes API client.
type RESTClient struct {
	baseURL *url.URL

	// Codec is the encoding and decoding scheme that applies to a particular set of
	// REST resources.
	Codec runtime.Codec

	// Set specific behavior of the client.  If not set http.DefaultClient will be
	// used.
	Client *http.Client

	Sync       bool
	PollPeriod time.Duration
	Timeout    time.Duration
}

// NewRESTClient creates a new RESTClient. This client performs generic REST functions
// such as Get, Put, Post, and Delete on specified paths.  Codec controls encoding and
// decoding of responses from the server.
func NewRESTClient(baseURL *url.URL, c runtime.Codec) *RESTClient {
	base := *baseURL
	if !strings.HasSuffix(base.Path, "/") {
		base.Path += "/"
	}
	base.RawQuery = ""
	base.Fragment = ""

	return &RESTClient{
		baseURL: &base,
		Codec:   c,

		// Make asynchronous requests by default
		// TODO: flip me to the default
		Sync: false,
		// Poll frequently when asynchronous requests are provided
		PollPeriod: time.Second * 2,
	}
}

// doRequest executes a request against a server
func (c *RESTClient) doRequest(request *http.Request) ([]byte, error) {
	client := c.Client
	if client == nil {
		client = http.DefaultClient
	}

	response, err := client.Do(request)
	if err != nil {
		return nil, err
	}
	defer response.Body.Close()
	body, err := ioutil.ReadAll(response.Body)
	if err != nil {
		return body, err
	}

	// Did the server give us a status response?
	isStatusResponse := false
	var status api.Status
	if err := c.Codec.DecodeInto(body, &status); err == nil && status.Status != "" {
		isStatusResponse = true
	}

	switch {
	case response.StatusCode < http.StatusOK || response.StatusCode > http.StatusPartialContent:
		if !isStatusResponse {
			return nil, fmt.Errorf("request [%#v] failed (%d) %s: %s", request, response.StatusCode, response.Status, string(body))
		}
		return nil, errors.FromObject(&status)
	}

	// If the server gave us a status back, look at what it was.
	if isStatusResponse && status.Status != api.StatusSuccess {
		// "Working" requests need to be handled specially.
		// "Failed" requests are clearly just an error and it makes sense to return them as such.
		return nil, errors.FromObject(&status)
	}

	return body, err
}

// Verb begins a request with a verb (GET, POST, PUT, DELETE).
//
// Example usage of RESTClient's request building interface:
// c := NewRESTClient(url, codec)
// resp, err := c.Verb("GET").
//  Path("pods").
//  SelectorParam("labels", "area=staging").
//  Timeout(10*time.Second).
//  Do()
// if err != nil { ... }
// list, ok := resp.(*api.PodList)
//
func (c *RESTClient) Verb(verb string) *Request {
	// TODO: uncomment when Go 1.2 support is dropped
	//var timeout time.Duration = 0
	// if c.Client != nil {
	// 	timeout = c.Client.Timeout
	// }
	return &Request{
		verb:       verb,
		c:          c,
		path:       c.baseURL.Path,
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
