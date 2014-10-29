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
	"net/url"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

	"github.com/golang/glog"
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
	Client HTTPClient

	// Set the poll behavior of this client. If not set the DefaultPoll method will
	// be called.
	Poller PollFunc

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
		Sync: false,

		// Poll frequently when asynchronous requests are provided
		PollPeriod: time.Second * 2,
	}
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
	poller := c.Poller
	if poller == nil {
		poller = c.DefaultPoll
	}
	return NewRequest(c.Client, verb, c.baseURL, c.Codec).Poller(poller).Sync(c.Sync).Timeout(c.Timeout)
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
func (c *RESTClient) Operation(name string) *Request {
	return c.Get().Path("operations").Path(name).Sync(false).NoPoll()
}

func (c *RESTClient) DefaultPoll(name string) (*Request, bool) {
	if c.PollPeriod == 0 {
		return nil, false
	}
	glog.Infof("Waiting for completion of operation %s", name)
	time.Sleep(c.PollPeriod)
	// Make a poll request
	return c.Operation(name).Poller(c.DefaultPoll), true
}
