/*
Copyright 2021 The Kubernetes Authors.

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
	"net/http"
	"net/url"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/util/flowcontrol"
)

type RESTClientFactory struct {
	// base is the root URL for all invocations of the client
	base *url.URL
	// rateLimiter is shared among all requests created by this client unless specifically
	// overridden.
	rateLimiter flowcontrol.RateLimiter
	// warningHandler is shared among all requests created by this client.
	// If not set, defaultWarningHandler is used.
	warningHandler WarningHandler
	// creates BackoffManager that is passed to requests.
	createBackoffMgr func() BackoffManager
	// Set specific behavior of the client.  If not set http.DefaultClient will be used.
	Client *http.Client
}

func RESTClientFactoryFor(config *Config) (*RESTClientFactory, error) {
	// Base URL
	base, _, err := defaultServerUrlFor(config)
	if err != nil {
		return nil, err
	}

	if !strings.HasSuffix(base.Path, "/") {
		base.Path += "/"
	}
	base.RawQuery = ""
	base.Fragment = ""

	// Transport
	if config.UserAgent == "" {
		config.UserAgent = DefaultKubernetesUserAgent()
	}

	transport, err := TransportFor(config)
	if err != nil {
		return nil, err
	}

	var httpClient *http.Client
	if transport != http.DefaultTransport {
		httpClient = &http.Client{Transport: transport}
		if config.Timeout > 0 {
			httpClient.Timeout = config.Timeout
		}
	}

	// RateLimiter
	rateLimiter := config.RateLimiter
	if rateLimiter == nil {
		qps := config.QPS
		if config.QPS == 0.0 {
			qps = DefaultQPS
		}
		burst := config.Burst
		if config.Burst == 0 {
			burst = DefaultBurst
		}
		if qps > 0 {
			rateLimiter = flowcontrol.NewTokenBucketRateLimiter(qps, burst)
		}
	}

	return &RESTClientFactory{
		base:             base,
		rateLimiter:      rateLimiter,
		warningHandler:   config.WarningHandler,
		createBackoffMgr: readExpBackoffConfig,
		Client:           httpClient,
	}, err
}

func RESTClientFactoryFromClient(c RESTClient) *RESTClientFactory {
	return &RESTClientFactory{
		base:             c.base,
		rateLimiter:      c.GetRateLimiter(),
		warningHandler:   c.warningHandler,
		createBackoffMgr: readExpBackoffConfig,
		Client:           c.Client,
	}
}

func (r RESTClientFactory) NewFor(apiPath string, config ClientContentConfig) *RESTClient {
	// shallow copy
	c := config
	if len(c.ContentType) == 0 {
		c.ContentType = "application/json"
	}

	if c.Negotiator == nil {
		c.Negotiator = runtime.NewClientNegotiator(scheme.Codecs.WithoutConversion(), c.GroupVersion)
	}

	return &RESTClient{
		// shared
		base:           r.base,
		rateLimiter:    r.rateLimiter,
		warningHandler: r.warningHandler,
		Client:         r.Client,
		// per client values
		versionedAPIPath: DefaultVersionedAPIPath(apiPath, c.GroupVersion),
		content:          c,
	}
}
