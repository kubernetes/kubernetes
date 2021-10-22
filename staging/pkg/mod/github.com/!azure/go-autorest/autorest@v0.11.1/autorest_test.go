package autorest

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
	"net/http"
	"testing"

	"github.com/Azure/go-autorest/autorest/mocks"
)

func TestResponseHasStatusCode(t *testing.T) {
	codes := []int{http.StatusOK, http.StatusAccepted}
	resp := &http.Response{StatusCode: http.StatusAccepted}
	if !ResponseHasStatusCode(resp, codes...) {
		t.Fatalf("autorest: ResponseHasStatusCode failed to find %v in %v", resp.StatusCode, codes)
	}
}

func TestResponseHasStatusCodeNotPresent(t *testing.T) {
	codes := []int{http.StatusOK, http.StatusAccepted}
	resp := &http.Response{StatusCode: http.StatusInternalServerError}
	if ResponseHasStatusCode(resp, codes...) {
		t.Fatalf("autorest: ResponseHasStatusCode unexpectedly found %v in %v", resp.StatusCode, codes)
	}
}

func TestNewPollingRequestDoesNotReturnARequestWhenLocationHeaderIsMissing(t *testing.T) {
	resp := mocks.NewResponseWithStatus("500 InternalServerError", http.StatusInternalServerError)

	req, _ := NewPollingRequest(resp, nil)
	if req != nil {
		t.Fatal("autorest: NewPollingRequest returned an http.Request when the Location header was missing")
	}
}

func TestNewPollingRequestReturnsAnErrorWhenPrepareFails(t *testing.T) {
	resp := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)
	mocks.SetAcceptedHeaders(resp)
	resp.Header.Set(http.CanonicalHeaderKey(HeaderLocation), mocks.TestBadURL)

	_, err := NewPollingRequest(resp, nil)
	if err == nil {
		t.Fatal("autorest: NewPollingRequest failed to return an error when Prepare fails")
	}
}

func TestNewPollingRequestDoesNotReturnARequestWhenPrepareFails(t *testing.T) {
	resp := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)
	mocks.SetAcceptedHeaders(resp)
	resp.Header.Set(http.CanonicalHeaderKey(HeaderLocation), mocks.TestBadURL)

	req, _ := NewPollingRequest(resp, nil)
	if req != nil {
		t.Fatal("autorest: NewPollingRequest returned an http.Request when Prepare failed")
	}
}

func TestNewPollingRequestReturnsAGetRequest(t *testing.T) {
	resp := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)
	mocks.SetAcceptedHeaders(resp)

	req, _ := NewPollingRequest(resp, nil)
	if req.Method != "GET" {
		t.Fatalf("autorest: NewPollingRequest did not create an HTTP GET request -- actual method %v", req.Method)
	}
}

func TestNewPollingRequestProvidesTheURL(t *testing.T) {
	resp := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)
	mocks.SetAcceptedHeaders(resp)

	req, _ := NewPollingRequest(resp, nil)
	if req.URL.String() != mocks.TestURL {
		t.Fatalf("autorest: NewPollingRequest did not create an HTTP with the expected URL -- received %v, expected %v", req.URL, mocks.TestURL)
	}
}

func TestGetLocation(t *testing.T) {
	resp := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)
	mocks.SetAcceptedHeaders(resp)

	l := GetLocation(resp)
	if len(l) == 0 {
		t.Fatalf("autorest: GetLocation failed to return Location header -- expected %v, received %v", mocks.TestURL, l)
	}
}

func TestGetLocationReturnsEmptyStringForMissingLocation(t *testing.T) {
	resp := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)

	l := GetLocation(resp)
	if len(l) != 0 {
		t.Fatalf("autorest: GetLocation return a value without a Location header -- received %v", l)
	}
}

func TestGetRetryAfter(t *testing.T) {
	resp := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)
	mocks.SetAcceptedHeaders(resp)

	d := GetRetryAfter(resp, DefaultPollingDelay)
	if d != mocks.TestDelay {
		t.Fatalf("autorest: GetRetryAfter failed to returned the expected delay -- expected %v, received %v", mocks.TestDelay, d)
	}
}

func TestGetRetryAfterReturnsDefaultDelayIfRetryHeaderIsMissing(t *testing.T) {
	resp := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)

	d := GetRetryAfter(resp, DefaultPollingDelay)
	if d != DefaultPollingDelay {
		t.Fatalf("autorest: GetRetryAfter failed to returned the default delay for a missing Retry-After header -- expected %v, received %v",
			DefaultPollingDelay, d)
	}
}

func TestGetRetryAfterReturnsDefaultDelayIfRetryHeaderIsMalformed(t *testing.T) {
	resp := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)
	mocks.SetAcceptedHeaders(resp)
	resp.Header.Set(http.CanonicalHeaderKey(HeaderRetryAfter), "a very bad non-integer value")

	d := GetRetryAfter(resp, DefaultPollingDelay)
	if d != DefaultPollingDelay {
		t.Fatalf("autorest: GetRetryAfter failed to returned the default delay for a malformed Retry-After header -- expected %v, received %v",
			DefaultPollingDelay, d)
	}
}
