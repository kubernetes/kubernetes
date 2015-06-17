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

package client

import (
	"net/http"
	"net/url"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

type HTTPClientFunc func(*http.Request) (*http.Response, error)

func (f HTTPClientFunc) Do(req *http.Request) (*http.Response, error) {
	return f(req)
}

// FakeRESTClient provides a fake RESTClient interface.
type FakeRESTClient struct {
	Client HTTPClient
	Codec  runtime.Codec
	Req    *http.Request
	Resp   *http.Response
	Err    error
}

func (c *FakeRESTClient) Get() *Request {
	return NewRequest(c, "GET", &url.URL{Host: "localhost"}, testapi.Version(), c.Codec)
}

func (c *FakeRESTClient) Put() *Request {
	return NewRequest(c, "PUT", &url.URL{Host: "localhost"}, testapi.Version(), c.Codec)
}

func (c *FakeRESTClient) Patch(_ api.PatchType) *Request {
	return NewRequest(c, "PATCH", &url.URL{Host: "localhost"}, testapi.Version(), c.Codec)
}

func (c *FakeRESTClient) Post() *Request {
	return NewRequest(c, "POST", &url.URL{Host: "localhost"}, testapi.Version(), c.Codec)
}

func (c *FakeRESTClient) Delete() *Request {
	return NewRequest(c, "DELETE", &url.URL{Host: "localhost"}, testapi.Version(), c.Codec)
}

func (c *FakeRESTClient) Do(req *http.Request) (*http.Response, error) {
	c.Req = req
	if c.Client != HTTPClient(nil) {
		return c.Client.Do(req)
	}
	return c.Resp, c.Err
}
