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
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/util/flowcontrol"
)

// RESTClientFactory allows to create RESTClients sharing the same Transport.
type RESTClientFactory struct {
	// base is the root URL for all invocations of the client
	base *url.URL
	// Set specific behavior of the client.  If not set http.DefaultClient will be used.
	Client *http.Client
}

// RESTClientFactoryFor create a new RESTClientFactory for the configuration.
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

	return &RESTClientFactory{
		base:   base,
		Client: httpClient,
	}, err
}

func RESTClientFactoryFromClient(c RESTClient) *RESTClientFactory {
	return &RESTClientFactory{
		base:   c.base,
		Client: c.Client,
	}
}

func (r RESTClientFactory) NewClientWithOptions(apiPath string, config ClientContentConfig, options ...RESTClientOption) (*RESTClient, error) {
	if config.GroupVersion == (schema.GroupVersion{}) {
		return nil, fmt.Errorf("GroupVersion is required when initializing a RESTClient")
	}
	if config.Negotiator == nil {
		return nil, fmt.Errorf("NegotiatedSerializer is required when initializing a RESTClient")
	}

	if len(config.ContentType) == 0 {
		config.ContentType = "application/json"
	}

	client := &RESTClient{
		// shared
		base:   r.base,
		Client: r.Client,
		// per client values
		versionedAPIPath: DefaultVersionedAPIPath(apiPath, config.GroupVersion),
		content:          config,
		// defaults
		rateLimiter:      flowcontrol.NewTokenBucketRateLimiter(DefaultQPS, DefaultBurst),
		createBackoffMgr: readExpBackoffConfig,
	}
	// Apply all options
	for _, opt := range options {
		client = opt(client)
	}

	return client, nil
}
