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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/runtime"
)

func CreateHTTPClient(roundTripper func(*http.Request) (*http.Response, error)) *http.Client {
	return &http.Client{
		Transport: roundTripperFunc(roundTripper),
	}
}

type roundTripperFunc func(*http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(req *http.Request) (*http.Response, error) {
	return f(req)
}

// RESTClient provides a fake RESTClient interface.
type RESTClient struct {
	Client               *http.Client
	NegotiatedSerializer runtime.NegotiatedSerializer

	Req  *http.Request
	Resp *http.Response
	Err  error
}

func (c *RESTClient) Get() *restclient.Request {
	return c.request("GET")
}

func (c *RESTClient) Put() *restclient.Request {
	return c.request("PUT")
}

func (c *RESTClient) Patch(_ api.PatchType) *restclient.Request {
	return c.request("PATCH")
}

func (c *RESTClient) Post() *restclient.Request {
	return c.request("POST")
}

func (c *RESTClient) Delete() *restclient.Request {
	return c.request("DELETE")
}

func (c *RESTClient) request(verb string) *restclient.Request {
	config := restclient.ContentConfig{
		ContentType:          runtime.ContentTypeJSON,
		GroupVersion:         testapi.Default.GroupVersion(),
		NegotiatedSerializer: c.NegotiatedSerializer,
	}
	ns := c.NegotiatedSerializer
	serializer, _ := ns.SerializerForMediaType(runtime.ContentTypeJSON, nil)
	streamingSerializer, _ := ns.StreamingSerializerForMediaType(runtime.ContentTypeJSON, nil)
	internalVersion := unversioned.GroupVersion{
		Group:   testapi.Default.GroupVersion().Group,
		Version: runtime.APIVersionInternal,
	}
	internalVersion.Version = runtime.APIVersionInternal
	serializers := restclient.Serializers{
		Encoder:             ns.EncoderForVersion(serializer, *testapi.Default.GroupVersion()),
		Decoder:             ns.DecoderToVersion(serializer, internalVersion),
		StreamingSerializer: streamingSerializer,
		Framer:              streamingSerializer.Framer,
	}
	return restclient.NewRequest(c, verb, &url.URL{Host: "localhost"}, "", config, serializers, nil, nil)
}

func (c *RESTClient) Do(req *http.Request) (*http.Response, error) {
	if c.Err != nil {
		return nil, c.Err
	}
	c.Req = req
	if c.Client != nil {
		return c.Client.Do(req)
	}
	return c.Resp, nil
}
