package logger

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"regexp"
	"strings"
	"testing"
)

func TestNilLogger(t *testing.T) {
	// verify no crash with no logging
	Instance.WriteRequest(nil, Filter{})
}

const (
	reqURL                = "https://fakething/dot/com"
	reqHeaderKey          = "x-header"
	reqHeaderVal          = "value"
	reqBody               = "the request body"
	respHeaderKey         = "response-header"
	respHeaderVal         = "something"
	respBody              = "the response body"
	logFileTimeStampRegex = `\(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{7}(-\d{2}:\d{2}|Z)\)`
)

func TestLogReqRespNoBody(t *testing.T) {
	err := os.Setenv("AZURE_GO_SDK_LOG_LEVEL", "info")
	if err != nil {
		t.Fatalf("failed to set log level: %v", err)
	}
	lf := path.Join(os.TempDir(), "testloggingbasic.log")
	err = os.Setenv("AZURE_GO_SDK_LOG_FILE", lf)
	if err != nil {
		t.Fatalf("failed to set log file: %v", err)
	}
	initDefaultLogger()
	if Level() != LogInfo {
		t.Fatalf("wrong log level: %d", Level())
	}
	// create mock request and response for logging
	req, err := http.NewRequest(http.MethodGet, reqURL, nil)
	if err != nil {
		t.Fatalf("failed to create mock request: %v", err)
	}
	req.Header.Add(reqHeaderKey, reqHeaderVal)
	Instance.WriteRequest(req, Filter{})
	resp := &http.Response{
		Status:     "200 OK",
		StatusCode: 200,
		Proto:      "HTTP/1.0",
		ProtoMajor: 1,
		ProtoMinor: 0,
		Request:    req,
		Header:     http.Header{},
	}
	resp.Header.Add(respHeaderKey, respHeaderVal)
	Instance.WriteResponse(resp, Filter{})
	if fl, ok := Instance.(fileLogger); ok {
		fl.logFile.Close()
	} else {
		t.Fatal("expected Instance to be fileLogger")
	}
	// parse log file to ensure contents match
	b, err := ioutil.ReadFile(lf)
	if err != nil {
		t.Fatalf("failed to read log file: %v", err)
	}
	parts := strings.Split(string(b), "\n")
	reqMatch := fmt.Sprintf("%s INFO: REQUEST: %s %s", logFileTimeStampRegex, req.Method, req.URL.String())
	respMatch := fmt.Sprintf("%s INFO: RESPONSE: %d %s", logFileTimeStampRegex, resp.StatusCode, resp.Request.URL.String())
	if !matchRegex(t, reqMatch, parts[0]) {
		t.Fatalf("request header doesn't match: %s", parts[0])
	}
	if !matchRegex(t, fmt.Sprintf("(?i)%s: %s", reqHeaderKey, reqHeaderVal), parts[1]) {
		t.Fatalf("request header entry doesn't match: %s", parts[1])
	}
	if !matchRegex(t, respMatch, parts[2]) {
		t.Fatalf("response header doesn't match: %s", parts[2])
	}
	if !matchRegex(t, fmt.Sprintf("(?i)%s: %s", respHeaderKey, respHeaderVal), parts[3]) {
		t.Fatalf("response header value doesn't match: %s", parts[3])
	}
	// disable logging
	err = os.Setenv("AZURE_GO_SDK_LOG_LEVEL", "")
	if err != nil {
		t.Fatalf("failed to clear log level: %v", err)
	}
}

func TestLogReqRespWithBody(t *testing.T) {
	err := os.Setenv("AZURE_GO_SDK_LOG_LEVEL", "debug")
	if err != nil {
		t.Fatalf("failed to set log level: %v", err)
	}
	lf := path.Join(os.TempDir(), "testloggingfull.log")
	err = os.Setenv("AZURE_GO_SDK_LOG_FILE", lf)
	if err != nil {
		t.Fatalf("failed to set log file: %v", err)
	}
	initDefaultLogger()
	if Level() != LogDebug {
		t.Fatalf("wrong log level: %d", Level())
	}
	// create mock request and response for logging
	req, err := http.NewRequest(http.MethodGet, reqURL, strings.NewReader(reqBody))
	if err != nil {
		t.Fatalf("failed to create mock request: %v", err)
	}
	req.Header.Add(reqHeaderKey, reqHeaderVal)
	Instance.WriteRequest(req, Filter{})
	resp := &http.Response{
		Status:     "200 OK",
		StatusCode: 200,
		Proto:      "HTTP/1.0",
		ProtoMajor: 1,
		ProtoMinor: 0,
		Request:    req,
		Header:     http.Header{},
		Body:       ioutil.NopCloser(strings.NewReader(respBody)),
	}
	resp.Header.Add(respHeaderKey, respHeaderVal)
	Instance.WriteResponse(resp, Filter{})
	if fl, ok := Instance.(fileLogger); ok {
		fl.logFile.Close()
	} else {
		t.Fatal("expected Instance to be fileLogger")
	}
	// parse log file to ensure contents match
	b, err := ioutil.ReadFile(lf)
	if err != nil {
		t.Fatalf("failed to read log file: %v", err)
	}
	parts := strings.Split(string(b), "\n")
	reqMatch := fmt.Sprintf("%s INFO: REQUEST: %s %s", logFileTimeStampRegex, req.Method, req.URL.String())
	respMatch := fmt.Sprintf("%s INFO: RESPONSE: %d %s", logFileTimeStampRegex, resp.StatusCode, resp.Request.URL.String())
	if !matchRegex(t, reqMatch, parts[0]) {
		t.Fatalf("request header doesn't match: %s", parts[0])
	}
	if !matchRegex(t, fmt.Sprintf("(?i)%s: %s", reqHeaderKey, reqHeaderVal), parts[1]) {
		t.Fatalf("request header value doesn't match: %s", parts[1])
	}
	if !matchRegex(t, reqBody, parts[2]) {
		t.Fatalf("request body doesn't match: %s", parts[2])
	}
	if !matchRegex(t, respMatch, parts[3]) {
		t.Fatalf("response header doesn't match: %s", parts[3])
	}
	if !matchRegex(t, fmt.Sprintf("(?i)%s: %s", respHeaderKey, respHeaderVal), parts[4]) {
		t.Fatalf("response header value doesn't match: %s", parts[4])
	}
	if !matchRegex(t, respBody, parts[5]) {
		t.Fatalf("response body doesn't match: %s", parts[5])
	}
	// disable logging
	err = os.Setenv("AZURE_GO_SDK_LOG_LEVEL", "")
	if err != nil {
		t.Fatalf("failed to clear log level: %v", err)
	}
}

func matchRegex(t *testing.T, pattern, s string) bool {
	match, err := regexp.MatchString(pattern, s)
	if err != nil {
		t.Fatalf("regexp failure: %v", err)
	}
	return match
}
