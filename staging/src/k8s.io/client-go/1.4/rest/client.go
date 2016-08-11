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

package rest

import (
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"k8s.io/client-go/1.4/pkg/api"
	"k8s.io/client-go/1.4/pkg/api/unversioned"
	"k8s.io/client-go/1.4/pkg/runtime"
	"k8s.io/client-go/1.4/pkg/util/flowcontrol"
)

const (
	// Environment variables: Note that the duration should be long enough that the backoff
	// persists for some reasonable time (i.e. 120 seconds).  The typical base might be "1".
	envBackoffBase     = "KUBE_CLIENT_BACKOFF_BASE"
	envBackoffDuration = "KUBE_CLIENT_BACKOFF_DURATION"
)

// RESTClient imposes common Kubernetes API conventions on a set of resource paths.
// The baseURL is expected to point to an HTTP or HTTPS path that is the parent
// of one or more resources.  The server should return a decodable API resource
// object, or an api.Status object which contains information about the reason for
// any failure.
//
// Most consumers should use client.New() to get a Kubernetes API client.
type RESTClient struct {
	// base is the root URL for all invocations of the client
	base *url.URL
	// versionedAPIPath is a path segment connecting the base URL to the resource root
	versionedAPIPath string

	// contentConfig is the information used to communicate with the server.
	contentConfig ContentConfig

	// serializers contain all serializers for undelying content type.
	serializers Serializers

	// creates BackoffManager that is passed to requests.
	createBackoffMgr func() BackoffManager

	// TODO extract this into a wrapper interface via the RESTClient interface in kubectl.
	Throttle flowcontrol.RateLimiter

	// Set specific behavior of the client.  If not set http.DefaultClient will be used.
	Client *http.Client
}

type Serializers struct {
	Encoder             runtime.Encoder
	Decoder             runtime.Decoder
	StreamingSerializer runtime.Serializer
	Framer              runtime.Framer
	RenegotiatedDecoder func(contentType string, params map[string]string) (runtime.Decoder, error)
}

// NewRESTClient creates a new RESTClient. This client performs generic REST functions
// such as Get, Put, Post, and Delete on specified paths.  Codec controls encoding and
// decoding of responses from the server.
func NewRESTClient(baseURL *url.URL, versionedAPIPath string, config ContentConfig, maxQPS float32, maxBurst int, rateLimiter flowcontrol.RateLimiter, client *http.Client) (*RESTClient, error) {
	base := *baseURL
	if !strings.HasSuffix(base.Path, "/") {
		base.Path += "/"
	}
	base.RawQuery = ""
	base.Fragment = ""

	if config.GroupVersion == nil {
		config.GroupVersion = &unversioned.GroupVersion{}
	}
	if len(config.ContentType) == 0 {
		config.ContentType = "application/json"
	}
	serializers, err := createSerializers(config)
	if err != nil {
		return nil, err
	}

	var throttle flowcontrol.RateLimiter
	if maxQPS > 0 && rateLimiter == nil {
		throttle = flowcontrol.NewTokenBucketRateLimiter(maxQPS, maxBurst)
	} else if rateLimiter != nil {
		throttle = rateLimiter
	}
	return &RESTClient{
		base:             &base,
		versionedAPIPath: versionedAPIPath,
		contentConfig:    config,
		serializers:      *serializers,
		createBackoffMgr: readExpBackoffConfig,
		Throttle:         throttle,
		Client:           client,
	}, nil
}

// GetRateLimiter returns rate limier for a given client, or nil if it's called on a nil client
func (c *RESTClient) GetRateLimiter() flowcontrol.RateLimiter {
	if c == nil {
		return nil
	}
	return c.Throttle
}

// readExpBackoffConfig handles the internal logic of determining what the
// backoff policy is.  By default if no information is available, NoBackoff.
// TODO Generalize this see #17727 .
func readExpBackoffConfig() BackoffManager {
	backoffBase := os.Getenv(envBackoffBase)
	backoffDuration := os.Getenv(envBackoffDuration)

	backoffBaseInt, errBase := strconv.ParseInt(backoffBase, 10, 64)
	backoffDurationInt, errDuration := strconv.ParseInt(backoffDuration, 10, 64)
	if errBase != nil || errDuration != nil {
		return &NoBackoff{}
	}
	return &URLBackoff{
		Backoff: flowcontrol.NewBackOff(
			time.Duration(backoffBaseInt)*time.Second,
			time.Duration(backoffDurationInt)*time.Second)}
}

// createSerializers creates all necessary serializers for given contentType.
func createSerializers(config ContentConfig) (*Serializers, error) {
	negotiated := config.NegotiatedSerializer
	contentType := config.ContentType
	info, ok := negotiated.SerializerForMediaType(contentType, nil)
	if !ok {
		return nil, fmt.Errorf("serializer for %s not registered", contentType)
	}
	streamInfo, ok := negotiated.StreamingSerializerForMediaType(contentType, nil)
	if !ok {
		return nil, fmt.Errorf("streaming serializer for %s not registered", contentType)
	}
	internalGV := unversioned.GroupVersion{
		Group:   config.GroupVersion.Group,
		Version: runtime.APIVersionInternal,
	}
	return &Serializers{
		Encoder:             negotiated.EncoderForVersion(info.Serializer, *config.GroupVersion),
		Decoder:             negotiated.DecoderToVersion(info.Serializer, internalGV),
		StreamingSerializer: streamInfo.Serializer,
		Framer:              streamInfo.Framer,
		RenegotiatedDecoder: func(contentType string, params map[string]string) (runtime.Decoder, error) {
			renegotiated, ok := negotiated.SerializerForMediaType(contentType, params)
			if !ok {
				return nil, fmt.Errorf("serializer for %s not registered", contentType)
			}
			return negotiated.DecoderToVersion(renegotiated.Serializer, internalGV), nil
		},
	}, nil
}

// Verb begins a request with a verb (GET, POST, PUT, DELETE).
//
// Example usage of RESTClient's request building interface:
// c, err := NewRESTClient(...)
// if err != nil { ... }
// resp, err := c.Verb("GET").
//  Path("pods").
//  SelectorParam("labels", "area=staging").
//  Timeout(10*time.Second).
//  Do()
// if err != nil { ... }
// list, ok := resp.(*api.PodList)
//
func (c *RESTClient) Verb(verb string) *Request {
	backoff := c.createBackoffMgr()

	if c.Client == nil {
		return NewRequest(nil, verb, c.base, c.versionedAPIPath, c.contentConfig, c.serializers, backoff, c.Throttle)
	}
	return NewRequest(c.Client, verb, c.base, c.versionedAPIPath, c.contentConfig, c.serializers, backoff, c.Throttle)
}

// Post begins a POST request. Short for c.Verb("POST").
func (c *RESTClient) Post() *Request {
	return c.Verb("POST")
}

// Put begins a PUT request. Short for c.Verb("PUT").
func (c *RESTClient) Put() *Request {
	return c.Verb("PUT")
}

// Patch begins a PATCH request. Short for c.Verb("Patch").
func (c *RESTClient) Patch(pt api.PatchType) *Request {
	return c.Verb("PATCH").SetHeader("Content-Type", string(pt))
}

// Get begins a GET request. Short for c.Verb("GET").
func (c *RESTClient) Get() *Request {
	return c.Verb("GET")
}

// Delete begins a DELETE request. Short for c.Verb("DELETE").
func (c *RESTClient) Delete() *Request {
	return c.Verb("DELETE")
}

// APIVersion returns the APIVersion this RESTClient is expected to use.
func (c *RESTClient) APIVersion() unversioned.GroupVersion {
	return *c.contentConfig.GroupVersion
}
