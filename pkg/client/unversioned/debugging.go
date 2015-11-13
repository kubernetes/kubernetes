/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package unversioned

import (
	"fmt"
	"net/http"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
)

// RequestInfo keeps track of information about a request/response combination
type RequestInfo struct {
	RequestHeaders http.Header
	RequestVerb    string
	RequestURL     string

	ResponseStatus  string
	ResponseHeaders http.Header
	ResponseErr     error

	Duration time.Duration
}

// NewRequestInfo creates a new RequestInfo based on an http request
func NewRequestInfo(req *http.Request) *RequestInfo {
	reqInfo := &RequestInfo{}
	reqInfo.RequestURL = req.URL.String()
	reqInfo.RequestVerb = req.Method
	reqInfo.RequestHeaders = req.Header

	return reqInfo
}

// Complete adds information about the response to the RequestInfo
func (r *RequestInfo) Complete(response *http.Response, err error) {
	if err != nil {
		r.ResponseErr = err
		return
	}
	r.ResponseStatus = response.Status
	r.ResponseHeaders = response.Header
}

// ToCurl returns a string that can be run as a command in a terminal (minus the body)
func (r RequestInfo) ToCurl() string {
	headers := ""
	for key, values := range map[string][]string(r.RequestHeaders) {
		for _, value := range values {
			headers += fmt.Sprintf(` -H %q`, fmt.Sprintf("%s: %s", key, value))
		}
	}

	return fmt.Sprintf("curl -k -v -X%s %s %s", r.RequestVerb, headers, r.RequestURL)
}

// DebuggingRoundTripper will display information about the requests passing through it based on what is configured
type DebuggingRoundTripper struct {
	delegatedRoundTripper http.RoundTripper

	Levels sets.String
}

const (
	JustURL         string = "url"
	URLTiming       string = "urltiming"
	CurlCommand     string = "curlcommand"
	RequestHeaders  string = "requestheaders"
	ResponseStatus  string = "responsestatus"
	ResponseHeaders string = "responseheaders"
)

func NewDebuggingRoundTripper(rt http.RoundTripper, levels ...string) *DebuggingRoundTripper {
	return &DebuggingRoundTripper{rt, sets.NewString(levels...)}
}

func (rt *DebuggingRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	reqInfo := NewRequestInfo(req)

	if rt.Levels.Has(JustURL) {
		glog.Infof("%s %s", reqInfo.RequestVerb, reqInfo.RequestURL)
	}
	if rt.Levels.Has(CurlCommand) {
		glog.Infof("%s", reqInfo.ToCurl())

	}
	if rt.Levels.Has(RequestHeaders) {
		glog.Infof("Request Headers:")
		for key, values := range reqInfo.RequestHeaders {
			for _, value := range values {
				glog.Infof("    %s: %s", key, value)
			}
		}
	}

	startTime := time.Now()
	response, err := rt.delegatedRoundTripper.RoundTrip(req)
	reqInfo.Duration = time.Since(startTime)

	reqInfo.Complete(response, err)

	if rt.Levels.Has(URLTiming) {
		glog.Infof("%s %s %s in %d milliseconds", reqInfo.RequestVerb, reqInfo.RequestURL, reqInfo.ResponseStatus, reqInfo.Duration.Nanoseconds()/int64(time.Millisecond))
	}
	if rt.Levels.Has(ResponseStatus) {
		glog.Infof("Response Status: %s in %d milliseconds", reqInfo.ResponseStatus, reqInfo.Duration.Nanoseconds()/int64(time.Millisecond))
	}
	if rt.Levels.Has(ResponseHeaders) {
		glog.Infof("Response Headers:")
		for key, values := range reqInfo.ResponseHeaders {
			for _, value := range values {
				glog.Infof("    %s: %s", key, value)
			}
		}
	}

	return response, err
}

var _ = util.RoundTripperWrapper(&DebuggingRoundTripper{})

func (rt *DebuggingRoundTripper) WrappedRoundTripper() http.RoundTripper {
	return rt.delegatedRoundTripper
}
