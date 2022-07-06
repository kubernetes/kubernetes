/*
Copyright 2016 The Kubernetes Authors.

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
	"net/url"
	"path"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// DefaultServerURL converts a host, host:port, or URL string to the default base server API path
// to use with a Client at a given API version following the standard conventions for a
// Kubernetes API.
func DefaultServerURL(host, apiPath string, groupVersion schema.GroupVersion, defaultTLS bool) (*url.URL, string, error) {
	if host == "" {
		return nil, "", fmt.Errorf("host must be a URL or a host:port pair")
	}
	base := host
	hostURL, err := url.Parse(base)
	if err != nil || hostURL.Scheme == "" || hostURL.Host == "" {
		scheme := "http://"
		if defaultTLS {
			scheme = "https://"
		}
		hostURL, err = url.Parse(scheme + base)
		if err != nil {
			return nil, "", err
		}
		if hostURL.Path != "" && hostURL.Path != "/" {
			return nil, "", fmt.Errorf("host must be a URL or a host:port pair: %q", base)
		}
	}

	// hostURL.Path is optional; a non-empty Path is treated as a prefix that is to be applied to
	// all URIs used to access the host. this is useful when there's a proxy in front of the
	// apiserver that has relocated the apiserver endpoints, forwarding all requests from, for
	// example, /a/b/c to the apiserver. in this case the Path should be /a/b/c.
	//
	// if running without a frontend proxy (that changes the location of the apiserver), then
	// hostURL.Path should be blank.
	//
	// versionedAPIPath, a path relative to baseURL.Path, points to a versioned API base
	versionedAPIPath := DefaultVersionedAPIPath(apiPath, groupVersion)

	return hostURL, versionedAPIPath, nil
}

// DefaultVersionedAPIPathFor constructs the default path for the given group version, assuming the given
// API path, following the standard conventions of the Kubernetes API.
func DefaultVersionedAPIPath(apiPath string, groupVersion schema.GroupVersion) string {
	versionedAPIPath := path.Join("/", apiPath)

	// Add the version to the end of the path
	if len(groupVersion.Group) > 0 {
		versionedAPIPath = path.Join(versionedAPIPath, groupVersion.Group, groupVersion.Version)

	} else {
		versionedAPIPath = path.Join(versionedAPIPath, groupVersion.Version)
	}

	return versionedAPIPath
}

// defaultServerUrlFor is shared between IsConfigTransportTLS and RESTClientFor. It
// requires Host and Version to be set prior to being called.
func defaultServerUrlFor(config *Config) (*url.URL, string, error) {
	// TODO: move the default to secure when the apiserver supports TLS by default
	// config.Insecure is taken to mean "I want HTTPS but don't bother checking the certs against a CA."
	hasCA := len(config.CAFile) != 0 || len(config.CAData) != 0
	hasCert := len(config.CertFile) != 0 || len(config.CertData) != 0
	defaultTLS := hasCA || hasCert || config.Insecure
	host := config.Host
	if host == "" {
		host = "localhost"
	}

	if config.GroupVersion != nil {
		return DefaultServerURL(host, config.APIPath, *config.GroupVersion, defaultTLS)
	}
	return DefaultServerURL(host, config.APIPath, schema.GroupVersion{}, defaultTLS)
}
