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
)

func TestNewSharedKeyAuthorizer(t *testing.T) {
	auth, err := NewSharedKeyAuthorizer("golangrocksonazure", "YmFy", SharedKey)
	if err != nil {
		t.Fatalf("create shared key authorizer: %v", err)
	}
	req, err := http.NewRequest(http.MethodGet, "https://golangrocksonazure.blob.core.windows.net/some/blob.dat", nil)
	if err != nil {
		t.Fatalf("create HTTP request: %v", err)
	}
	req.Header.Add(headerAcceptCharset, "UTF-8")
	req.Header.Add(headerContentType, "application/json")
	req.Header.Add(headerXMSDate, "Wed, 23 Sep 2015 16:40:05 GMT")
	req.Header.Add(headerContentLength, "0")
	req.Header.Add(headerXMSVersion, "2015-02-21")
	req.Header.Add(headerAccept, "application/json;odata=nometadata")
	req, err = Prepare(req, auth.WithAuthorization())
	if err != nil {
		t.Fatalf("prepare HTTP request: %v", err)
	}
	const expected = "SharedKey golangrocksonazure:nYRqgbumDOTPs+Vv1FLH+hm0KPjwwt+Fmj/i16W+lO0="
	if auth := req.Header.Get(headerAuthorization); auth != expected {
		t.Fatalf("expected: %s, go %s", expected, auth)
	}
}

func TestNewSharedKeyAuthorizerWithRoot(t *testing.T) {
	auth, err := NewSharedKeyAuthorizer("golangrocksonazure", "YmFy", SharedKey)
	if err != nil {
		t.Fatalf("create shared key authorizer: %v", err)
	}
	req, err := http.NewRequest(http.MethodGet, "https://golangrocksonazure.blob.core.windows.net/?comp=properties&restype=service", nil)
	if err != nil {
		t.Fatalf("create HTTP request: %v", err)
	}
	req.Header.Add(headerAcceptCharset, "UTF-8")
	req.Header.Add(headerContentType, "application/json")
	req.Header.Add(headerXMSDate, "Tue, 10 Mar 2020 10:04:41 GMT")
	req.Header.Add(headerContentLength, "0")
	req.Header.Add(headerXMSVersion, "2018-11-09")
	req.Header.Add(headerAccept, "application/json;odata=nometadata")
	req, err = Prepare(req, auth.WithAuthorization())
	if err != nil {
		t.Fatalf("prepare HTTP request: %v", err)
	}
	const expected = "SharedKey golangrocksonazure:BfdIC0K5OwkRbZjewqRXgjQJ2PBMZDoaBCCL3qhrEIs="
	if auth := req.Header.Get(headerAuthorization); auth != expected {
		t.Fatalf("expected: %s, go %s", expected, auth)
	}
}

func TestNewSharedKeyAuthorizerWithoutRoot(t *testing.T) {
	auth, err := NewSharedKeyAuthorizer("golangrocksonazure", "YmFy", SharedKey)
	if err != nil {
		t.Fatalf("create shared key authorizer: %v", err)
	}
	req, err := http.NewRequest(http.MethodGet, "https://golangrocksonazure.blob.core.windows.net?comp=properties&restype=service", nil)
	if err != nil {
		t.Fatalf("create HTTP request: %v", err)
	}
	req.Header.Add(headerAcceptCharset, "UTF-8")
	req.Header.Add(headerContentType, "application/json")
	req.Header.Add(headerXMSDate, "Tue, 10 Mar 2020 10:04:41 GMT")
	req.Header.Add(headerContentLength, "0")
	req.Header.Add(headerXMSVersion, "2018-11-09")
	req.Header.Add(headerAccept, "application/json;odata=nometadata")
	req, err = Prepare(req, auth.WithAuthorization())
	if err != nil {
		t.Fatalf("prepare HTTP request: %v", err)
	}
	const expected = "SharedKey golangrocksonazure:BfdIC0K5OwkRbZjewqRXgjQJ2PBMZDoaBCCL3qhrEIs="
	if auth := req.Header.Get(headerAuthorization); auth != expected {
		t.Fatalf("expected: %s, go %s", expected, auth)
	}
}

func TestNewSharedKeyForTableAuthorizer(t *testing.T) {
	auth, err := NewSharedKeyAuthorizer("golangrocksonazure", "YmFy", SharedKeyForTable)
	if err != nil {
		t.Fatalf("create shared key authorizer: %v", err)
	}
	req, err := http.NewRequest(http.MethodGet, "https://golangrocksonazure.table.core.windows.net/tquery()", nil)
	if err != nil {
		t.Fatalf("create HTTP request: %v", err)
	}
	req.Header.Add(headerAcceptCharset, "UTF-8")
	req.Header.Add(headerContentType, "application/json")
	req.Header.Add(headerXMSDate, "Wed, 23 Sep 2015 16:40:05 GMT")
	req.Header.Add(headerContentLength, "0")
	req.Header.Add(headerXMSVersion, "2015-02-21")
	req.Header.Add(headerAccept, "application/json;odata=nometadata")
	req, err = Prepare(req, auth.WithAuthorization())
	if err != nil {
		t.Fatalf("prepare HTTP request: %v", err)
	}
	const expected = "SharedKey golangrocksonazure:73oeIBA2dulLhOBdAlM3U0+DKIWS0UW6InBWCHpOY50="
	if auth := req.Header.Get(headerAuthorization); auth != expected {
		t.Fatalf("expected: %s, go %s", expected, auth)
	}
}

func TestNewSharedKeyLiteAuthorizer(t *testing.T) {
	auth, err := NewSharedKeyAuthorizer("golangrocksonazure", "YmFy", SharedKeyLite)
	if err != nil {
		t.Fatalf("create shared key authorizer: %v", err)
	}

	req, err := http.NewRequest(http.MethodGet, "https://golangrocksonazure.file.core.windows.net/some/file.dat", nil)
	if err != nil {
		t.Fatalf("create HTTP request: %v", err)
	}
	req.Header.Add(headerAcceptCharset, "UTF-8")
	req.Header.Add(headerContentType, "application/json")
	req.Header.Add(headerXMSDate, "Wed, 23 Sep 2015 16:40:05 GMT")
	req.Header.Add(headerContentLength, "0")
	req.Header.Add(headerXMSVersion, "2015-02-21")
	req.Header.Add(headerAccept, "application/json;odata=nometadata")
	req, err = Prepare(req, auth.WithAuthorization())
	if err != nil {
		t.Fatalf("prepare HTTP request: %v", err)
	}
	const expected = "SharedKeyLite golangrocksonazure:0VODf/mHRDa7lMShzTKbow7lxptaIZ0qIAcVD0lG9PE="
	if auth := req.Header.Get(headerAuthorization); auth != expected {
		t.Fatalf("expected: %s, go %s", expected, auth)
	}
}

func TestNewSharedKeyLiteForTableAuthorizer(t *testing.T) {
	auth, err := NewSharedKeyAuthorizer("golangrocksonazure", "YmFy", SharedKeyLiteForTable)
	if err != nil {
		t.Fatalf("create shared key authorizer: %v", err)
	}

	req, err := http.NewRequest(http.MethodGet, "https://golangrocksonazure.table.core.windows.net/tquery()", nil)
	if err != nil {
		t.Fatalf("create HTTP request: %v", err)
	}
	req.Header.Add(headerAcceptCharset, "UTF-8")
	req.Header.Add(headerContentType, "application/json")
	req.Header.Add(headerXMSDate, "Wed, 23 Sep 2015 16:40:05 GMT")
	req.Header.Add(headerContentLength, "0")
	req.Header.Add(headerXMSVersion, "2015-02-21")
	req.Header.Add(headerAccept, "application/json;odata=nometadata")
	req, err = Prepare(req, auth.WithAuthorization())
	if err != nil {
		t.Fatalf("prepare HTTP request: %v", err)
	}
	const expected = "SharedKeyLite golangrocksonazure:NusXSFXAvHqr6EQNXnZZ50CvU1sX0iP/FFDHehnixLc="
	if auth := req.Header.Get(headerAuthorization); auth != expected {
		t.Fatalf("expected: %s, go %s", expected, auth)
	}
}
