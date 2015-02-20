/*
Copyright 2014 Google Inc. All rights reserved.

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

package osinserver

import (
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/RangelReale/osin"
	"golang.org/x/oauth2"

	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/osinserver/teststorage"
)

func TokenTokenExchange(t *testing.T) {
	storage := teststorage.New()
	storage.Clients["test"] = &osin.DefaultClient{
		Id:          "test",
		Secret:      "secret",
		RedirectUri: "http://localhost/redirect",
	}
	oauthServer := New(
		NewDefaultServerConfig(),
		storage,
		AuthorizeHandlerFunc(func(ar *osin.AuthorizeRequest, w http.ResponseWriter) (bool, error) {
			ar.Authorized = true
			return false, nil
		}),
		AccessHandlerFunc(func(ar *osin.AccessRequest, w http.ResponseWriter) error {
			ar.Authorized = true
			ar.GenerateRefresh = false
			return nil
		}),
		NewDefaultErrorHandler(),
	)
	mux := http.NewServeMux()
	oauthServer.Install(mux, "")
	server := httptest.NewServer(mux)

	config := &oauth2.Config{
		ClientID:     "test",
		ClientSecret: "secret",
		Endpoint: oauth2.Endpoint{
			AuthURL:  server.URL + "/authorize",
			TokenURL: server.URL + "/token",
		},
		RedirectURL: "http://localhost/redirect",
		Scopes:      []string{"a_scope"},
	}

	// Get auth url
	authurl := config.AuthCodeURL("state")
	req, err := http.NewRequest("GET", authurl, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	resp, err := http.DefaultTransport.RoundTrip(req)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Extract code
	loc := resp.Header.Get("Location")
	returl, err := url.Parse(loc)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	code := returl.Query().Get("code")
	if code == "" {
		t.Fatalf("expected code")
	}

	// Exchange for token
	tok, err := config.Exchange(oauth2.NoContext, code)
	if err != nil {
		log.Fatal(err)
	}
	if tok.AccessToken == "" {
		t.Fatalf("expected access token")
	}

	// Ensure backing data
	if storage.AccessData == nil {
		t.Fatalf("unexpected nil access data")
	}
}
