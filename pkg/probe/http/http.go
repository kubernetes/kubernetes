/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package http

import (
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"

	"github.com/golang/glog"
)

func New() HTTPProber {
	transport := &http.Transport{}
	return httpProber{transport}
}

type HTTPProber interface {
	Probe(host string, port int, path string, timeout time.Duration) (probe.Result, string, error)
}

type httpProber struct {
	transport *http.Transport
}

// Probe returns a ProbeRunner capable of running an http check.
func (pr httpProber) Probe(host string, port int, path string, timeout time.Duration) (probe.Result, string, error) {
	return DoHTTPProbe(formatURL(host, port, path), &http.Client{Timeout: timeout, Transport: pr.transport})
}

type HTTPGetInterface interface {
	Get(u string) (*http.Response, error)
}

// DoHTTPProbe checks if a GET request to the url succeeds.
// If the HTTP response code is successful (i.e. 400 > code >= 200), it returns Success.
// If the HTTP response code is unsuccessful or HTTP communication fails, it returns Failure.
// This is exported because some other packages may want to do direct HTTP probes.
func DoHTTPProbe(url string, client HTTPGetInterface) (probe.Result, string, error) {
	res, err := client.Get(url)
	if err != nil {
		// Convert errors into failures to catch timeouts.
		return probe.Failure, err.Error(), nil
	}
	defer res.Body.Close()
	b, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return probe.Failure, "", err
	}
	body := string(b)
	if res.StatusCode >= http.StatusOK && res.StatusCode < http.StatusBadRequest {
		return probe.Success, body, nil
	}
	glog.V(4).Infof("Probe failed for %s, Response: %v", url, *res)
	return probe.Failure, body, nil
}

// formatURL formats a URL from args.  For testability.
func formatURL(host string, port int, path string) string {
	u := url.URL{
		Scheme: "http",
		Host:   net.JoinHostPort(host, strconv.Itoa(port)),
		Path:   path,
	}
	return u.String()
}
