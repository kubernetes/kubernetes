// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"strings"
	"testing"
)

func TestAuthLegacyConfig(t *testing.T) {
	auth := base64.StdEncoding.EncodeToString([]byte("user:pass"))
	read := strings.NewReader(fmt.Sprintf(`{"docker.io":{"auth":"%s","email":"user@example.com"}}`, auth))
	ac, err := NewAuthConfigurations(read)
	if err != nil {
		t.Error(err)
	}
	c, ok := ac.Configs["docker.io"]
	if !ok {
		t.Error("NewAuthConfigurations: Expected Configs to contain docker.io")
	}
	if got, want := c.Email, "user@example.com"; got != want {
		t.Errorf(`AuthConfigurations.Configs["docker.io"].Email: wrong result. Want %q. Got %q`, want, got)
	}
	if got, want := c.Username, "user"; got != want {
		t.Errorf(`AuthConfigurations.Configs["docker.io"].Username: wrong result. Want %q. Got %q`, want, got)
	}
	if got, want := c.Password, "pass"; got != want {
		t.Errorf(`AuthConfigurations.Configs["docker.io"].Password: wrong result. Want %q. Got %q`, want, got)
	}
	if got, want := c.ServerAddress, "docker.io"; got != want {
		t.Errorf(`AuthConfigurations.Configs["docker.io"].ServerAddress: wrong result. Want %q. Got %q`, want, got)
	}
}

func TestAuthBadConfig(t *testing.T) {
	auth := base64.StdEncoding.EncodeToString([]byte("userpass"))
	read := strings.NewReader(fmt.Sprintf(`{"docker.io":{"auth":"%s","email":"user@example.com"}}`, auth))
	ac, err := NewAuthConfigurations(read)
	if err != AuthParseError {
		t.Errorf("Incorrect error returned %v\n", err)
	}
	if ac != nil {
		t.Errorf("Invalid auth configuration returned, should be nil %v\n", ac)
	}
}

func TestAuthConfig(t *testing.T) {
	auth := base64.StdEncoding.EncodeToString([]byte("user:pass"))
	read := strings.NewReader(fmt.Sprintf(`{"auths":{"docker.io":{"auth":"%s","email":"user@example.com"}}}`, auth))
	ac, err := NewAuthConfigurations(read)
	if err != nil {
		t.Error(err)
	}
	c, ok := ac.Configs["docker.io"]
	if !ok {
		t.Error("NewAuthConfigurations: Expected Configs to contain docker.io")
	}
	if got, want := c.Email, "user@example.com"; got != want {
		t.Errorf(`AuthConfigurations.Configs["docker.io"].Email: wrong result. Want %q. Got %q`, want, got)
	}
	if got, want := c.Username, "user"; got != want {
		t.Errorf(`AuthConfigurations.Configs["docker.io"].Username: wrong result. Want %q. Got %q`, want, got)
	}
	if got, want := c.Password, "pass"; got != want {
		t.Errorf(`AuthConfigurations.Configs["docker.io"].Password: wrong result. Want %q. Got %q`, want, got)
	}
	if got, want := c.ServerAddress, "docker.io"; got != want {
		t.Errorf(`AuthConfigurations.Configs["docker.io"].ServerAddress: wrong result. Want %q. Got %q`, want, got)
	}
}

func TestAuthCheck(t *testing.T) {
	fakeRT := &FakeRoundTripper{status: http.StatusOK}
	client := newTestClient(fakeRT)
	if err := client.AuthCheck(nil); err == nil {
		t.Fatalf("expected error on nil auth config")
	}
	// test good auth
	if err := client.AuthCheck(&AuthConfiguration{}); err != nil {
		t.Fatal(err)
	}
	*fakeRT = FakeRoundTripper{status: http.StatusUnauthorized}
	if err := client.AuthCheck(&AuthConfiguration{}); err == nil {
		t.Fatal("expected failure from unauthorized auth")
	}
}
