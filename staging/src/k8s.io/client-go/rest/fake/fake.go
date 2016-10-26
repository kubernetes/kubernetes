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

	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/testapi"
	"k8s.io/client-go/pkg/api/unversioned"
	"k8s.io/client-go/pkg/apimachinery/registered"
	"k8s.io/client-go/pkg/runtime"
	"k8s.io/client-go/pkg/util/flowcontrol"
	"k8s.io/client-go/rest"
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
	GroupName            string

	Req  *http.Request
	Resp *http.Response
	Err  error
}

func (c *RESTClient) Get() *rest.Request {
	return c.request("GET")
}

func (c *RESTClient) Put() *rest.Request {
	return c.request("PUT")
}

func (c *RESTClient) Patch(_ api.PatchType) *rest.Request {
	return c.request("PATCH")
}

func (c *RESTClient) Post() *rest.Request {
	return c.request("POST")
}

func (c *RESTClient) Delete() *rest.Request {
	return c.request("DELETE")
}

func (c *RESTClient) Verb(verb string) *rest.Request {
	return c.request(verb)
}

func (c *RESTClient) APIVersion() unversioned.GroupVersion {
	return *(testapi.Default.GroupVersion())
}

func (c *RESTClient) GetRateLimiter() flowcontrol.RateLimiter {
	return nil
}

func (c *RESTClient) request(verb string) *rest.Request {
	config := rest.ContentConfig{
		ContentType:          runtime.ContentTypeJSON,
		GroupVersion:         &registered.GroupOrDie(api.GroupName).GroupVersion,
		NegotiatedSerializer: c.NegotiatedSerializer,
	}

	groupName := api.GroupName
	if c.GroupName != "" {
		groupName = c.GroupName
	}
	ns := c.NegotiatedSerializer
	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	internalVersion := unversioned.GroupVersion{
		Group:   registered.GroupOrDie(groupName).GroupVersion.Group,
		Version: runtime.APIVersionInternal,
	}
	internalVersion.Version = runtime.APIVersionInternal
	serializers := rest.Serializers{
		Encoder: ns.EncoderForVersion(info.Serializer, registered.GroupOrDie(api.GroupName).GroupVersion),
		Decoder: ns.DecoderToVersion(info.Serializer, internalVersion),
	}
	if info.StreamSerializer != nil {
		serializers.StreamingSerializer = info.StreamSerializer.Serializer
		serializers.Framer = info.StreamSerializer.Framer
	}
	return rest.NewRequest(c, verb, &url.URL{Host: "localhost"}, "", config, serializers, nil, nil)
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
