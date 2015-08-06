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
	"net/http"
	"net/url"
	"time"

	"k8s.io/kubernetes/pkg/probe"

	"github.com/golang/glog"
)

type HTTPProber interface {
	Probe(url *url.URL, timeout time.Duration) (probe.Result, string, error)
}

func New() HTTPProber {
	return httpProber{}
}

type httpProber struct{}

// Probe returns a ProbeRunner capable of running an http check.
func (pr httpProber) Probe(url *url.URL, timeout time.Duration) (probe.Result, string, error) {
	return DoHTTPProbe(url, NewTimeoutClient(timeout))
}

type HTTPGetter interface {
	Get(u string) (*http.Response, error)
}

type BodyValidator interface {
	// Validate returns an error if the response body is unhealthy
	Validate(body []byte) error
}

// DoHTTPProbe checks if a GET request to the url succeeds.
// If the HTTP communication fails or times out, it returns Failure.
// If the HTTP response code is < 200 or >= 400, it returns Failure.
// If the HTTP response body causes a validation error, it returns Failure.
// Otherwise the probe returns Success
// This is exported because some other packages may want to do direct HTTP probes.
func DoHTTPProbe(url *url.URL, client HTTPGetter, validators ...BodyValidator) (probe.Result, string, error) {
	res, err := client.Get(url.String())
	if err != nil {
		// Convert errors into failures to catch timeouts.
		return probe.Failure, err.Error(), nil
	}
	defer res.Body.Close()
	bodyBytes, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return probe.Failure, "", err
	}
	body := string(bodyBytes)
	if res.StatusCode < http.StatusOK || res.StatusCode >= http.StatusBadRequest {
		glog.V(4).Infof("Probe failed for %s, Response: %v, Invalid Status Code", url.String(), *res)
		return probe.Failure, body, nil
	}
	for _, validator := range validators {
		if err = validator.Validate(bodyBytes); err != nil {
			glog.V(4).Infof("Probe failed for %s, Response: %v: Invalid Body: %v", url.String(), *res, err)
			return probe.Failure, body, nil
		}
	}
	glog.V(4).Infof("Probe succeeded for %s, Response: %v", url.String(), *res)
	return probe.Success, body, nil
}
