/*
Copyright The Kubernetes Authors.

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

package h2c

import (
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"

	v1 "k8s.io/api/core/v1"
	"k8s.io/component-base/version"
)

// NewRequestForH2CGetAction builds a GET request to http://<podIP>:<port><path> for an h2c probe.
func NewRequestForH2CGetAction(a *v1.H2CGetAction, podIP string, userAgentFragment string) (*http.Request, error) {
	if a == nil {
		return nil, fmt.Errorf("h2cGet action is nil")
	}
	path := a.Path
	if path == "" {
		path = "/"
	}
	u := formatURL("http", podIP, int(a.Port), path)
	headers := v1HeaderToHTTPHeader(a.HTTPHeaders)
	return newProbeRequest(u, headers, userAgentFragment)
}

func newProbeRequest(u *url.URL, headers http.Header, userAgentFragment string) (*http.Request, error) {
	req, err := http.NewRequest(http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, err
	}
	if headers == nil {
		headers = make(http.Header)
	}
	if _, ok := headers["User-Agent"]; !ok {
		headers.Set("User-Agent", userAgent(userAgentFragment))
	}
	if _, ok := headers["Accept"]; !ok {
		headers.Set("Accept", "*/*")
	} else if headers.Get("Accept") == "" {
		headers.Del("Accept")
	}
	req.Header = headers
	req.Host = headers.Get("Host")
	return req, nil
}

func userAgent(purpose string) string {
	v := version.Get()
	return fmt.Sprintf("kube-%s/%s.%s", purpose, v.Major, v.Minor)
}

func formatURL(scheme string, host string, port int, path string) *url.URL {
	u, err := url.Parse(path)
	if err != nil {
		u = &url.URL{Path: path}
	}
	u.Scheme = scheme
	u.Host = net.JoinHostPort(host, strconv.Itoa(port))
	return u
}

func v1HeaderToHTTPHeader(headerList []v1.HTTPHeader) http.Header {
	headers := make(http.Header)
	for _, header := range headerList {
		headers.Add(header.Name, header.Value)
	}
	return headers
}