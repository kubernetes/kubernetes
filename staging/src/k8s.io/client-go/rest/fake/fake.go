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

// This is made a separate package and should only be imported by tests, because
// it imports testapi
package fake

import (
	"net/http"
	"net/url"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/flowcontrol"
)

// CreateHTTPClient creates an http.Client that will invoke the provided roundTripper func
// when a request is made.
func CreateHTTPClient(roundTripper func(*http.Request) (*http.Response, error)) *http.Client {
	return &http.Client{
		Transport: roundTripperFunc(roundTripper),
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

// RESTClient provides a fake RESTClient interface. It is used to mock network
// interactions via a rest.Request, or to make them via the provided Client to
// a specific server.
type RESTClient struct {
	NegotiatedSerializer runtime.NegotiatedSerializer
	GroupVersion         schema.GroupVersion
	VersionedAPIPath     string

	// Err is returned when any request would be made to the server. If Err is set,
	// Req will not be recorded, Resp will not be returned, and Client will not be
	// invoked.
	Err error
	// Req is set to the last request that was executed (had the methods Do/DoRaw) invoked.
	Req *http.Request
	// If Client is specified, the client will be invoked instead of returning Resp if
	// Err is not set.
	Client *http.Client
	// Resp is returned to the caller after Req is recorded, unless Err or Client are set.
	Resp *http.Response
}

func (c *RESTClient) Get() *restclient.Request {
	return c.Verb("GET")
}

func (c *RESTClient) Put() *restclient.Request {
	return c.Verb("PUT")
}

func (c *RESTClient) Patch(pt types.PatchType) *restclient.Request {
	return c.Verb("PATCH").SetHeader("Content-Type", string(pt))
}

func (c *RESTClient) Post() *restclient.Request {
	return c.Verb("POST")
}

func (c *RESTClient) Delete() *restclient.Request {
	return c.Verb("DELETE")
}

func (c *RESTClient) Verb(verb string) *restclient.Request {
	return c.Request().Verb(verb)
}

func (c *RESTClient) APIVersion() schema.GroupVersion {
	return c.GroupVersion
}

func (c *RESTClient) GetRateLimiter() flowcontrol.RateLimiter {
	return nil
}

func (c *RESTClient) Request() *restclient.Request {
	config := restclient.ClientContentConfig{
		ContentType:  runtime.ContentTypeJSON,
		GroupVersion: c.GroupVersion,
		Negotiator:   runtime.NewClientNegotiator(c.NegotiatedSerializer, c.GroupVersion),
	}
	return restclient.NewRequestWithClient(&url.URL{Scheme: "https", Host: "localhost"}, c.VersionedAPIPath, config, CreateHTTPClient(c.do))
}

// do is invoked when a Request() created by this client is executed.
func (c *RESTClient) do(req *http.Request) (*http.Response, error) {
	if c.Err != nil {
		return nil, c.Err
	}
	c.Req = req
	if c.Client != nil {
		return c.Client.Do(req)
	}
	return c.Resp, nil
}
