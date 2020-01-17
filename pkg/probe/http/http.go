/*
Copyright 2015 The Kubernetes Authors.

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
	"crypto/tls"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"time"

	"path/filepath"

	utilfeature "k8s.io/apiserver/pkg/util/feature"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/component-base/version"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/probe"

	"k8s.io/klog"
	utilio "k8s.io/utils/io"
)

const (
	maxRespBodyLength = 10 * 1 << 10 // 10KB
)

// New creates Prober that will skip TLS verification while probing.
// followNonLocalRedirects configures whether the prober should follow redirects to a different hostname.
//   If disabled, redirects to other hosts will trigger a warning result.
func New(followNonLocalRedirects bool) Prober {
	tlsConfig := &tls.Config{InsecureSkipVerify: true}
	return NewWithTLSConfig(tlsConfig, followNonLocalRedirects)
}

// NewWithTLSConfig takes tls config as parameter.
// followNonLocalRedirects configures whether the prober should follow redirects to a different hostname.
//   If disabled, redirects to other hosts will trigger a warning result.
func NewWithTLSConfig(config *tls.Config, followNonLocalRedirects bool) Prober {
	// We do not want the probe use node's local proxy set.
	transport := utilnet.SetTransportDefaults(
		&http.Transport{
			TLSClientConfig:   config,
			DisableKeepAlives: true,
			Proxy:             http.ProxyURL(nil),
		})
	return httpProber{transport, followNonLocalRedirects}
}

// Prober is an interface that defines the Probe function for doing HTTP readiness/liveness checks.
type Prober interface {
	Probe(url *url.URL, headers http.Header, expectHTTPCodes []int, expectHTTPContent string, timeout time.Duration) (probe.Result, string, error)
}

type httpProber struct {
	transport               *http.Transport
	followNonLocalRedirects bool
}

// Probe returns a ProbeRunner capable of running an HTTP check.
func (pr httpProber) Probe(url *url.URL, headers http.Header, expectHTTPCodes []int, expectHTTPContent string, timeout time.Duration) (probe.Result, string, error) {
	client := &http.Client{
		Timeout:       timeout,
		Transport:     pr.transport,
		CheckRedirect: redirectChecker(pr.followNonLocalRedirects),
	}
	return DoHTTPProbe(url, headers, expectHTTPCodes, expectHTTPContent, client)
}

// GetHTTPInterface is an interface for making HTTP requests, that returns a response and error.
type GetHTTPInterface interface {
	Do(req *http.Request) (*http.Response, error)
}

// DoHTTPProbe checks if a GET request to the url succeeds.
// When HTTPProbePlus not enabled, If the HTTP response code is successful (i.e. 400 > code >= 200), it returns Success.
// If the HTTP response code is unsuccessful or HTTP communication fails(200 > code >= 400), it returns Failure.
// When HTTPProbePlus is enabled, if both response code in expectHTTPCodes and response content match expectHTTPContent, it return Success.
// This is exported because some other packages may want to do direct HTTP probes.
func DoHTTPProbe(url *url.URL, headers http.Header, expectHTTPCodes []int, expectHTTPContent string, client GetHTTPInterface) (probe.Result, string, error) {
	req, err := http.NewRequest("GET", url.String(), nil)
	if err != nil {
		// Convert errors into failures to catch timeouts.
		return probe.Failure, err.Error(), nil
	}
	if _, ok := headers["User-Agent"]; !ok {
		if headers == nil {
			headers = http.Header{}
		}
		// explicitly set User-Agent so it's not set to default Go value
		v := version.Get()
		headers.Set("User-Agent", fmt.Sprintf("kube-probe/%s.%s", v.Major, v.Minor))
	}
	req.Header = headers
	if headers.Get("Host") != "" {
		req.Host = headers.Get("Host")
	}
	res, err := client.Do(req)
	if err != nil {
		// Convert errors into failures to catch timeouts.
		return probe.Failure, err.Error(), nil
	}
	defer res.Body.Close()
	b, err := utilio.ReadAtMost(res.Body, maxRespBodyLength)
	if err != nil {
		if err == utilio.ErrLimitReached {
			klog.V(4).Infof("Non fatal body truncation for %s, Response: %v", url.String(), *res)
		} else {
			return probe.Failure, "", err
		}
	}
	body := string(b)
	result, err, reason := hTTPCheckerPlus(res, url, expectHTTPCodes, expectHTTPContent, body)
	if result == probe.Failure {
		klog.V(4).Infof("Probe failed for %s with request headers %v, response body: %v", url.String(), headers, body)
		body = reason
	}
	return result, body, err
}

// If  HTTPProbePlus not enabled run default statusCodechecker, else run contentChecker
// and statusCodeCheckerPlus
func hTTPCheckerPlus(res *http.Response, url *url.URL, expectHTTPCodes []int, expectHTTPContent, body string) (probe.Result, error, string) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.HTTPProbePlus) || (len(expectHTTPCodes) == 0 && expectHTTPContent == "") {
		return statusCodeChecker(res, url)
	}
	result, err, reason := contentChecker(res, url, expectHTTPContent, body)
	if result == probe.Failure || err != nil {
		return result, err, reason
	}
	return statusCodeCheckerPlus(res, url, expectHTTPCodes)
}

// If expectHTTPContent is not set. It returns Success
// If response body match expectHTTPContent pattern, it returns Success, else Failure
func contentChecker(res *http.Response, url *url.URL, expectHTTPContent, body string) (probe.Result, error, string) {
	// if expectHTTPContent is empty, return Success directly
	if expectHTTPContent == "" {
		klog.V(4).Infof("expectHTTPContent is empty. skip it.")
		return probe.Success, nil, ""
	}
	// check expectHTTPContent match body or not
	isMatch, err := filepath.Match(expectHTTPContent, body)
	if err != nil {
		return probe.Failure, err, ""
	}
	// match
	if isMatch {
		return probe.Success, nil, ""
	}
	// not match
	return probe.Failure, nil, fmt.Sprintf("HTTP probe failed with content: %s not match expectHTTPContent: %s", body, expectHTTPContent)
}

// If expectHTTPCodes is not set, it returns Success.
// if response code in  expectHTTPCodes, it returns Success, else Failure
func statusCodeCheckerPlus(res *http.Response, url *url.URL, expectHTTPCodes []int) (probe.Result, error, string) {
	if len(expectHTTPCodes) == 0 {
		//if expectHTTPCodes is empty return success directly
		klog.V(4).Infof("HTTPProbePlus is enabled, but expectHTTPCodes for url %s was empty, check status code treat as success", url.String())
		return probe.Success, nil, ""
	}
	for _, code := range expectHTTPCodes {
		if code == res.StatusCode {
			klog.V(4).Infof("Probe succeeded for %s, Response: %v", url.String(), *res)
			return probe.Success, nil, ""
		}
	}
	return probe.Failure, nil, fmt.Sprintf("HTTP probe failed with statuscode: %d, expectHTTPCodes: %v", res.StatusCode, expectHTTPCodes)
}

func statusCodeChecker(res *http.Response, url *url.URL) (probe.Result, error, string) {
	if res.StatusCode >= http.StatusOK && res.StatusCode < http.StatusBadRequest {
		if res.StatusCode >= http.StatusMultipleChoices { // Redirect
			klog.V(4).Infof("Probe terminated redirects for %s, Response: %v", url.String(), *res)
			return probe.Warning, nil, ""
		}
		klog.V(4).Infof("Probe succeeded for %s, Response: %v", url.String(), *res)
		return probe.Success, nil, ""
	}

	return probe.Failure, nil, fmt.Sprintf("HTTP probe failed with statuscode: %d, HTTPCodes: must >= %d and < %d", res.StatusCode, http.StatusOK, http.StatusBadRequest)
}

func redirectChecker(followNonLocalRedirects bool) func(*http.Request, []*http.Request) error {
	if followNonLocalRedirects {
		return nil // Use the default http client checker.
	}

	return func(req *http.Request, via []*http.Request) error {
		if req.URL.Hostname() != via[0].URL.Hostname() {
			return http.ErrUseLastResponse
		}
		// Default behavior: stop after 10 redirects.
		if len(via) >= 10 {
			return errors.New("stopped after 10 redirects")
		}
		return nil
	}
}
