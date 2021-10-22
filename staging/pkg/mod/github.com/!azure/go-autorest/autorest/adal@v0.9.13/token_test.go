package adal

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
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/big"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest/date"
	"github.com/Azure/go-autorest/autorest/mocks"
	jwt "github.com/form3tech-oss/jwt-go"
)

const (
	defaultFormData       = "client_id=id&client_secret=secret&grant_type=client_credentials&resource=resource"
	defaultManualFormData = "client_id=id&grant_type=refresh_token&refresh_token=refreshtoken&resource=resource"
)

func init() {
	// fake that the IMDS endpoint is available
	msiAvailableHook = func(ctx context.Context, sender Sender) bool {
		return true
	}
}

func TestTokenExpires(t *testing.T) {
	tt := time.Now().Add(5 * time.Second)
	tk := newTokenExpiresAt(tt)

	if tk.Expires().Equal(tt) {
		t.Fatalf("adal: Token#Expires miscalculated expiration time -- received %v, expected %v", tk.Expires(), tt)
	}
}

func TestTokenIsExpired(t *testing.T) {
	tk := newTokenExpiresAt(time.Now().Add(-5 * time.Second))

	if !tk.IsExpired() {
		t.Fatalf("adal: Token#IsExpired failed to mark a stale token as expired -- now %v, token expires at %v",
			time.Now().UTC(), tk.Expires())
	}
}

func TestTokenIsExpiredUninitialized(t *testing.T) {
	tk := &Token{}

	if !tk.IsExpired() {
		t.Fatalf("adal: An uninitialized Token failed to mark itself as expired (expiration time %v)", tk.Expires())
	}
}

func TestTokenIsNoExpired(t *testing.T) {
	tk := newTokenExpiresAt(time.Now().Add(1000 * time.Second))

	if tk.IsExpired() {
		t.Fatalf("adal: Token marked a fresh token as expired -- now %v, token expires at %v", time.Now().UTC(), tk.Expires())
	}
}

func TestTokenWillExpireIn(t *testing.T) {
	d := 5 * time.Second
	tk := newTokenExpiresIn(d)

	if !tk.WillExpireIn(d) {
		t.Fatal("adal: Token#WillExpireIn mismeasured expiration time")
	}
}

func TestParseExpiresOn(t *testing.T) {
	// get current time, round to nearest second, and add one hour
	n := time.Now().UTC().Round(time.Second).Add(time.Hour)
	amPM := "AM"
	if n.Hour() >= 12 {
		amPM = "PM"
	}
	testcases := []struct {
		Name   string
		String string
		Value  int64
	}{
		{
			Name:   "integer",
			String: "3600",
			Value:  3600,
		},
		{
			Name:   "timestamp with AM/PM",
			String: fmt.Sprintf("%d/%d/%d %d:%02d:%02d %s +00:00", n.Month(), n.Day(), n.Year(), n.Hour(), n.Minute(), n.Second(), amPM),
			Value:  3600,
		},
		{
			Name:   "timestamp without AM/PM",
			String: fmt.Sprintf("%d/%d/%d %d:%02d:%02d +00:00", n.Month(), n.Day(), n.Year(), n.Hour(), n.Minute(), n.Second()),
			Value:  3600,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.Name, func(subT *testing.T) {
			jn, err := parseExpiresOn(tc.String)
			if err != nil {
				subT.Error(err)
			}
			i, err := jn.Int64()
			if err != nil {
				subT.Error(err)
			}
			if i != tc.Value {
				subT.Logf("expected %d, got %d", tc.Value, i)
				subT.Fail()
			}
		})
	}
}

func TestServicePrincipalTokenSetAutoRefresh(t *testing.T) {
	spt := newServicePrincipalToken()

	if !spt.inner.AutoRefresh {
		t.Fatal("adal: ServicePrincipalToken did not default to automatic token refreshing")
	}

	spt.SetAutoRefresh(false)
	if spt.inner.AutoRefresh {
		t.Fatal("adal: ServicePrincipalToken#SetAutoRefresh did not disable automatic token refreshing")
	}
}

func TestServicePrincipalTokenSetCustomRefreshFunc(t *testing.T) {
	spt := newServicePrincipalToken()

	var refreshFunc TokenRefresh = func(context context.Context, resource string) (*Token, error) {
		return nil, nil
	}

	if spt.customRefreshFunc != nil {
		t.Fatalf("adal: ServicePrincipalToken#SetCustomRefreshFunc had a default custom refresh func when it shouldn't")
	}

	spt.SetCustomRefreshFunc(refreshFunc)

	if spt.customRefreshFunc == nil {
		t.Fatalf("adal: ServicePrincipalToken#SetCustomRefreshFunc didn't have a refresh func")
	}
}

func TestServicePrincipalTokenSetRefreshWithin(t *testing.T) {
	spt := newServicePrincipalToken()

	if spt.inner.RefreshWithin != defaultRefresh {
		t.Fatal("adal: ServicePrincipalToken did not correctly set the default refresh interval")
	}

	spt.SetRefreshWithin(2 * defaultRefresh)
	if spt.inner.RefreshWithin != 2*defaultRefresh {
		t.Fatal("adal: ServicePrincipalToken#SetRefreshWithin did not set the refresh interval")
	}
}

func TestServicePrincipalTokenSetSender(t *testing.T) {
	spt := newServicePrincipalToken()

	c := &http.Client{}
	spt.SetSender(c)
	if !reflect.DeepEqual(c, spt.sender) {
		t.Fatal("adal: ServicePrincipalToken#SetSender did not set the sender")
	}
}

func TestServicePrincipalTokenRefreshUsesCustomRefreshFunc(t *testing.T) {
	spt := newServicePrincipalToken()

	called := false
	var refreshFunc TokenRefresh = func(context context.Context, resource string) (*Token, error) {
		called = true
		return &Token{}, nil
	}
	spt.SetCustomRefreshFunc(refreshFunc)
	if called {
		t.Fatalf("adal: ServicePrincipalToken#refreshInternal called the refresh function prior to refreshing")
	}

	spt.refreshInternal(context.Background(), "https://example.com")

	if !called {
		t.Fatalf("adal: ServicePrincipalToken#refreshInternal didn't call the refresh function")
	}
}

func TestServicePrincipalTokenRefreshUsesPOST(t *testing.T) {
	spt := newServicePrincipalToken()

	body := mocks.NewBody(newTokenJSON(`"3600"`, "12345", "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					if r.Method != "POST" {
						t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set HTTP method -- expected %v, received %v", "POST", r.Method)
					}
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err := spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}

	if body.IsOpen() {
		t.Fatalf("the response was not closed!")
	}
}

func TestNewServicePrincipalTokenFromManagedIdentity(t *testing.T) {
	spt, err := NewServicePrincipalTokenFromManagedIdentity("https://resource", nil)
	if err != nil {
		t.Fatalf("Failed to get MSI SPT: %v", err)
	}

	body := mocks.NewBody(newTokenJSON(`"3600"`, "12345", "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					if r.Method != "GET" {
						t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set HTTP method -- expected %v, received %v", "GET", r.Method)
					}
					if h := r.Header.Get("Metadata"); h != "true" {
						t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set Metadata header for MSI")
					}
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err = spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}

	if body.IsOpen() {
		t.Fatalf("the response was not closed!")
	}
}

func TestServicePrincipalTokenFromMSICloudshell(t *testing.T) {
	os.Setenv(msiEndpointEnv, "http://dummy")
	defer func() {
		os.Unsetenv(msiEndpointEnv)
	}()
	spt, err := NewServicePrincipalTokenFromMSI("", "https://resource")
	if err != nil {
		t.Fatalf("Failed to get MSI SPT: %v", err)
	}

	body := mocks.NewBody(newTokenJSON(`"3600"`, "12345", "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					if r.Method != http.MethodPost {
						t.Fatalf("adal: cloudshell did not correctly set HTTP method -- expected %v, received %v", "GET", r.Method)
					}
					if h := r.Header.Get("Metadata"); h != "true" {
						t.Fatalf("adal: cloudshell did not correctly set Metadata header")
					}
					if h := r.Header.Get("Content-Type"); h != "application/x-www-form-urlencoded" {
						t.Fatalf("adal: cloudshell did not correctly set Content-Type header")
					}
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err = spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}

	if body.IsOpen() {
		t.Fatalf("the response was not closed!")
	}
}

func TestServicePrincipalTokenFromMSIRefreshZeroRetry(t *testing.T) {
	resource := "https://resource"
	cb := func(token Token) error { return nil }

	endpoint, _ := GetMSIVMEndpoint()
	spt, err := NewServicePrincipalTokenFromMSI(endpoint, resource, cb)
	if err != nil {
		t.Fatalf("Failed to get MSI SPT: %v", err)
	}
	spt.MaxMSIRefreshAttempts = 1

	body := mocks.NewBody(newTokenJSON(`"3600"`, "12345", "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					// second invocation, perform MSI request validation
					if r.Method != "GET" {
						t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set HTTP method -- expected %v, received %v", "GET", r.Method)
					}
					if h := r.Header.Get("Metadata"); h != "true" {
						t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set Metadata header for MSI")
					}
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err = spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}

	if body.IsOpen() {
		t.Fatalf("the response was not closed!")
	}
}

func TestServicePrincipalTokenFromASE(t *testing.T) {
	os.Setenv("MSI_ENDPOINT", "http://localhost")
	os.Setenv("MSI_SECRET", "super")
	defer func() {
		os.Unsetenv("MSI_ENDPOINT")
		os.Unsetenv("MSI_SECRET")
	}()
	resource := "https://resource"
	spt, err := NewServicePrincipalTokenFromMSI("", resource)
	if err != nil {
		t.Fatalf("Failed to get MSI SPT: %v", err)
	}
	spt.MaxMSIRefreshAttempts = 1
	// expires_on is sent in UTC
	expiresOn := time.Now().UTC().Add(time.Hour)
	// use int format for expires_in
	body := mocks.NewBody(newTokenJSON("3600", expiresOn.Format(expiresOnDateFormat), "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					if r.Method != "GET" {
						t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set HTTP method -- expected %v, received %v", "GET", r.Method)
					}
					if h := r.Header.Get(metadataHeader); h != "" {
						t.Fatalf("adal: ServicePrincipalToken#Refresh incorrectly set Metadata header for ASE")
					}
					if s := r.Header.Get(secretHeader); s != "super" {
						t.Fatalf("adal: unexpected secret header value %s", s)
					}
					if r.URL.Host != "localhost" {
						t.Fatalf("adal: unexpected host %s", r.URL.Host)
					}
					qp := r.URL.Query()
					if api := qp.Get("api-version"); api != appServiceAPIVersion2017 {
						t.Fatalf("adal: unexpected api-version %s", api)
					}
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err = spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}
	v, err := spt.inner.Token.ExpiresOn.Int64()
	if err != nil {
		t.Fatalf("adal: failed to get ExpiresOn %v", err)
	}
	// depending on elapsed time it might be slightly less that one hour
	const hourInSeconds = int64(time.Hour / time.Second)
	if v > hourInSeconds || v < hourInSeconds-1 {
		t.Fatalf("adal: expected %v, got %v", int64(time.Hour/time.Second), v)
	}
	if body.IsOpen() {
		t.Fatalf("the response was not closed!")
	}
}

func TestServicePrincipalTokenFromADFS(t *testing.T) {
	os.Setenv("MSI_ENDPOINT", "http://localhost")
	os.Setenv("MSI_SECRET", "super")
	defer func() {
		os.Unsetenv("MSI_ENDPOINT")
		os.Unsetenv("MSI_SECRET")
	}()
	resource := "https://resource"
	endpoint, _ := GetMSIEndpoint()
	spt, err := NewServicePrincipalTokenFromMSI(endpoint, resource)
	if err != nil {
		t.Fatalf("Failed to get MSI SPT: %v", err)
	}
	spt.MaxMSIRefreshAttempts = 1
	const expiresIn = 3600
	body := mocks.NewBody(newADFSTokenJSON(expiresIn))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					if r.Method != "GET" {
						t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set HTTP method -- expected %v, received %v", "GET", r.Method)
					}
					if h := r.Header.Get(metadataHeader); h != "" {
						t.Fatalf("adal: ServicePrincipalToken#Refresh incorrectly set Metadata header for ASE")
					}
					if s := r.Header.Get(secretHeader); s != "super" {
						t.Fatalf("adal: unexpected secret header value %s", s)
					}
					if r.URL.Host != "localhost" {
						t.Fatalf("adal: unexpected host %s", r.URL.Host)
					}
					qp := r.URL.Query()
					if api := qp.Get("api-version"); api != appServiceAPIVersion2017 {
						t.Fatalf("adal: unexpected api-version %s", api)
					}
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err = spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}
	i, err := spt.inner.Token.ExpiresIn.Int64()
	if err != nil {
		t.Fatalf("unexpected parsing of expires_in: %v", err)
	}
	if i != expiresIn {
		t.Fatalf("unexpected expires_in %d", i)
	}
	if spt.inner.Token.ExpiresOn.String() != "" {
		t.Fatal("expected empty expires_on")
	}
	if body.IsOpen() {
		t.Fatalf("the response was not closed!")
	}
}

func TestServicePrincipalTokenFromMSIRefreshCancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	endpoint, _ := GetMSIVMEndpoint()

	spt, err := NewServicePrincipalTokenFromMSI(endpoint, "https://resource")
	if err != nil {
		t.Fatalf("Failed to get MSI SPT: %v", err)
	}

	c := mocks.NewSender()
	c.AppendAndRepeatResponse(mocks.NewResponseWithStatus("Internal server error", http.StatusInternalServerError), 5)

	var wg sync.WaitGroup
	wg.Add(1)
	start := time.Now()
	end := time.Now()

	go func() {
		spt.SetSender(c)
		err = spt.RefreshWithContext(ctx)
		end = time.Now()
		wg.Done()
	}()

	cancel()
	wg.Wait()
	time.Sleep(5 * time.Millisecond)

	if end.Sub(start) >= time.Second {
		t.Fatalf("TestServicePrincipalTokenFromMSIRefreshCancel failed to cancel")
	}
}

func TestServicePrincipalTokenRefreshSetsMimeType(t *testing.T) {
	spt := newServicePrincipalToken()

	body := mocks.NewBody(newTokenJSON(`"3600"`, "12345", "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					if r.Header.Get(http.CanonicalHeaderKey("Content-Type")) != "application/x-www-form-urlencoded" {
						t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set Content-Type -- expected %v, received %v",
							"application/x-form-urlencoded",
							r.Header.Get(http.CanonicalHeaderKey("Content-Type")))
					}
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err := spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}
}

func TestServicePrincipalTokenRefreshSetsURL(t *testing.T) {
	spt := newServicePrincipalToken()

	body := mocks.NewBody(newTokenJSON(`"3600"`, "12345", "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					if r.URL.String() != TestOAuthConfig.TokenEndpoint.String() {
						t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set the URL -- expected %v, received %v",
							TestOAuthConfig.TokenEndpoint, r.URL)
					}
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err := spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}
}

func testServicePrincipalTokenRefreshSetsBody(t *testing.T, spt *ServicePrincipalToken, f func(*testing.T, []byte)) {
	body := mocks.NewBody(newTokenJSON(`"3600"`, "12345", "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					b, err := ioutil.ReadAll(r.Body)
					if err != nil {
						t.Fatalf("adal: Failed to read body of Service Principal token request (%v)", err)
					}
					f(t, b)
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err := spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}
}

func TestServicePrincipalTokenManualRefreshSetsBody(t *testing.T) {
	sptManual := newServicePrincipalTokenManual()
	testServicePrincipalTokenRefreshSetsBody(t, sptManual, func(t *testing.T, b []byte) {
		if string(b) != defaultManualFormData {
			t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set the HTTP Request Body -- expected %v, received %v",
				defaultManualFormData, string(b))
		}
	})
}

func TestServicePrincipalTokenCertficateRefreshSetsBody(t *testing.T) {
	sptCert := newServicePrincipalTokenCertificate(t)
	testServicePrincipalTokenRefreshSetsBody(t, sptCert, func(t *testing.T, b []byte) {
		body := string(b)

		values, _ := url.ParseQuery(body)
		if values["client_assertion_type"][0] != "urn:ietf:params:oauth:client-assertion-type:jwt-bearer" ||
			values["client_id"][0] != "id" ||
			values["grant_type"][0] != "client_credentials" ||
			values["resource"][0] != "resource" {
			t.Fatalf("adal: ServicePrincipalTokenCertificate#Refresh did not correctly set the HTTP Request Body.")
		}

		tok, _ := jwt.Parse(values["client_assertion"][0], nil)
		if tok == nil {
			t.Fatalf("adal: ServicePrincipalTokenCertificate#Expected client_assertion to be a JWT")
		}
		if _, ok := tok.Header["x5t"]; !ok {
			t.Fatalf("adal: ServicePrincipalTokenCertificate#Expected client_assertion to have an x5t header")
		}
		if _, ok := tok.Header["x5c"]; !ok {
			t.Fatalf("adal: ServicePrincipalTokenCertificate#Expected client_assertion to have an x5c header")
		}
		claims, ok := tok.Claims.(jwt.MapClaims)
		if !ok {
			t.Fatalf("expected MapClaims, got %T", tok.Claims)
		}
		if err := claims.Valid(); err != nil {
			t.Fatalf("invalid claim: %v", err)
		}
		if aud := claims["aud"]; aud != "https://login.test.com/SomeTenantID/oauth2/token?api-version=1.0" {
			t.Fatalf("unexpected aud: %s", aud)
		}
		if iss := claims["iss"]; iss != "id" {
			t.Fatalf("unexpected iss: %s", iss)
		}
		if sub := claims["sub"]; sub != "id" {
			t.Fatalf("unexpected sub: %s", sub)
		}
	})
}

func TestServicePrincipalTokenUsernamePasswordRefreshSetsBody(t *testing.T) {
	spt := newServicePrincipalTokenUsernamePassword(t)
	testServicePrincipalTokenRefreshSetsBody(t, spt, func(t *testing.T, b []byte) {
		body := string(b)

		values, _ := url.ParseQuery(body)
		if values["client_id"][0] != "id" ||
			values["grant_type"][0] != "password" ||
			values["username"][0] != "username" ||
			values["password"][0] != "password" ||
			values["resource"][0] != "resource" {
			t.Fatalf("adal: ServicePrincipalTokenUsernamePassword#Refresh did not correctly set the HTTP Request Body.")
		}
	})
}

func TestServicePrincipalTokenAuthorizationCodeRefreshSetsBody(t *testing.T) {
	spt := newServicePrincipalTokenAuthorizationCode(t)
	testServicePrincipalTokenRefreshSetsBody(t, spt, func(t *testing.T, b []byte) {
		body := string(b)

		values, _ := url.ParseQuery(body)
		if values["client_id"][0] != "id" ||
			values["grant_type"][0] != OAuthGrantTypeAuthorizationCode ||
			values["code"][0] != "code" ||
			values["client_secret"][0] != "clientSecret" ||
			values["redirect_uri"][0] != "http://redirectUri/getToken" ||
			values["resource"][0] != "resource" {
			t.Fatalf("adal: ServicePrincipalTokenAuthorizationCode#Refresh did not correctly set the HTTP Request Body.")
		}
	})
	testServicePrincipalTokenRefreshSetsBody(t, spt, func(t *testing.T, b []byte) {
		body := string(b)

		values, _ := url.ParseQuery(body)
		if values["client_id"][0] != "id" ||
			values["grant_type"][0] != OAuthGrantTypeRefreshToken ||
			values["code"][0] != "code" ||
			values["client_secret"][0] != "clientSecret" ||
			values["redirect_uri"][0] != "http://redirectUri/getToken" ||
			values["resource"][0] != "resource" {
			t.Fatalf("adal: ServicePrincipalTokenAuthorizationCode#Refresh did not correctly set the HTTP Request Body.")
		}
	})
}

func TestServicePrincipalTokenSecretRefreshSetsBody(t *testing.T) {
	spt := newServicePrincipalToken()
	testServicePrincipalTokenRefreshSetsBody(t, spt, func(t *testing.T, b []byte) {
		if string(b) != defaultFormData {
			t.Fatalf("adal: ServicePrincipalToken#Refresh did not correctly set the HTTP Request Body -- expected %v, received %v",
				defaultFormData, string(b))
		}

	})
}

func TestServicePrincipalTokenRefreshClosesRequestBody(t *testing.T) {
	spt := newServicePrincipalToken()

	body := mocks.NewBody(newTokenJSON(`"3600"`, "12345", "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err := spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}
	if resp.Body.(*mocks.Body).IsOpen() {
		t.Fatal("adal: ServicePrincipalToken#Refresh failed to close the HTTP Response Body")
	}
}

func TestServicePrincipalTokenRefreshRejectsResponsesWithStatusNotOK(t *testing.T) {
	spt := newServicePrincipalToken()

	body := mocks.NewBody(newTokenJSON(`"3600"`, "12345", "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusUnauthorized, "Unauthorized")

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err := spt.Refresh()
	if err == nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh should reject a response with status != %d", http.StatusOK)
	}
}

func TestServicePrincipalTokenRefreshRejectsEmptyBody(t *testing.T) {
	spt := newServicePrincipalToken()

	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					return mocks.NewResponse(), nil
				})
			}
		})())
	spt.SetSender(s)
	err := spt.Refresh()
	if err == nil {
		t.Fatal("adal: ServicePrincipalToken#Refresh should reject an empty token")
	}
}

func TestServicePrincipalTokenRefreshPropagatesErrors(t *testing.T) {
	spt := newServicePrincipalToken()

	c := mocks.NewSender()
	c.SetError(fmt.Errorf("Faux Error"))
	spt.SetSender(c)

	err := spt.Refresh()
	if err == nil {
		t.Fatal("adal: Failed to propagate the request error")
	}
}

func TestServicePrincipalTokenRefreshReturnsErrorIfNotOk(t *testing.T) {
	spt := newServicePrincipalToken()

	c := mocks.NewSender()
	c.AppendResponse(mocks.NewResponseWithStatus("401 NotAuthorized", http.StatusUnauthorized))
	spt.SetSender(c)

	err := spt.Refresh()
	if err == nil {
		t.Fatalf("adal: Failed to return an when receiving a status code other than HTTP %d", http.StatusOK)
	}
}

func TestServicePrincipalTokenRefreshUnmarshals(t *testing.T) {
	spt := newServicePrincipalToken()

	expiresOn := strconv.Itoa(int(time.Now().Add(3600 * time.Second).Sub(date.UnixEpoch()).Seconds()))
	j := newTokenJSON(`"3600"`, expiresOn, "resource")
	resp := mocks.NewResponseWithContent(j)
	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)

	err := spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	} else if spt.inner.Token.AccessToken != "accessToken" ||
		spt.inner.Token.ExpiresIn != "3600" ||
		spt.inner.Token.ExpiresOn != json.Number(expiresOn) ||
		spt.inner.Token.NotBefore != json.Number(expiresOn) ||
		spt.inner.Token.Resource != "resource" ||
		spt.inner.Token.Type != "Bearer" {
		t.Fatalf("adal: ServicePrincipalToken#Refresh failed correctly unmarshal the JSON -- expected %v, received %v",
			j, *spt)
	}
}

func TestServicePrincipalTokenEnsureFreshRefreshes(t *testing.T) {
	spt := newServicePrincipalToken()
	expireToken(&spt.inner.Token)

	body := mocks.NewBody(newTokenJSON(`"3600"`, "12345", "test"))
	resp := mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK")

	f := false
	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					f = true
					return resp, nil
				})
			}
		})())
	spt.SetSender(s)
	err := spt.EnsureFresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#EnsureFresh returned an unexpected error (%v)", err)
	}
	if !f {
		t.Fatal("adal: ServicePrincipalToken#EnsureFresh failed to call Refresh for stale token")
	}
}

func TestServicePrincipalTokenEnsureFreshFails1(t *testing.T) {
	spt := newServicePrincipalToken()
	expireToken(&spt.inner.Token)

	c := mocks.NewSender()
	c.SetError(fmt.Errorf("some failure"))

	spt.SetSender(c)
	err := spt.EnsureFresh()
	if err == nil {
		t.Fatal("adal: ServicePrincipalToken#EnsureFresh didn't return an error")
	}
	if _, ok := err.(TokenRefreshError); ok {
		t.Fatal("adal: ServicePrincipalToken#EnsureFresh unexpected TokenRefreshError")
	}
}

func TestServicePrincipalTokenEnsureFreshFails2(t *testing.T) {
	spt := newServicePrincipalToken()
	expireToken(&spt.inner.Token)

	c := mocks.NewSender()
	c.AppendResponse(mocks.NewResponseWithStatus("bad request", http.StatusBadRequest))

	spt.SetSender(c)
	err := spt.EnsureFresh()
	if err == nil {
		t.Fatal("adal: ServicePrincipalToken#EnsureFresh didn't return an error")
	}
	if _, ok := err.(TokenRefreshError); !ok {
		t.Fatal("adal: ServicePrincipalToken#EnsureFresh didn't return a TokenRefreshError")
	}
}

func TestServicePrincipalTokenEnsureFreshSkipsIfFresh(t *testing.T) {
	spt := newServicePrincipalToken()
	setTokenToExpireIn(&spt.inner.Token, 1000*time.Second)

	f := false
	c := mocks.NewSender()
	s := DecorateSender(c,
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					f = true
					return mocks.NewResponse(), nil
				})
			}
		})())
	spt.SetSender(s)
	err := spt.EnsureFresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#EnsureFresh returned an unexpected error (%v)", err)
	}
	if f {
		t.Fatal("adal: ServicePrincipalToken#EnsureFresh invoked Refresh for fresh token")
	}
}

func TestRefreshCallback(t *testing.T) {
	callbackTriggered := false
	spt := newServicePrincipalToken(func(Token) error {
		callbackTriggered = true
		return nil
	})

	expiresOn := strconv.Itoa(int(time.Now().Add(3600 * time.Second).Sub(date.UnixEpoch()).Seconds()))

	sender := mocks.NewSender()
	j := newTokenJSON(`"3600"`, expiresOn, "resource")
	sender.AppendResponse(mocks.NewResponseWithContent(j))
	spt.SetSender(sender)
	err := spt.Refresh()
	if err != nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh returned an unexpected error (%v)", err)
	}
	if !callbackTriggered {
		t.Fatalf("adal: RefreshCallback failed to trigger call callback")
	}
}

func TestRefreshCallbackErrorPropagates(t *testing.T) {
	errorText := "this is an error text"
	spt := newServicePrincipalToken(func(Token) error {
		return fmt.Errorf(errorText)
	})

	expiresOn := strconv.Itoa(int(time.Now().Add(3600 * time.Second).Sub(date.UnixEpoch()).Seconds()))

	sender := mocks.NewSender()
	j := newTokenJSON(`"3600"`, expiresOn, "resource")
	sender.AppendResponse(mocks.NewResponseWithContent(j))
	spt.SetSender(sender)
	err := spt.Refresh()

	if err == nil || !strings.Contains(err.Error(), errorText) {
		t.Fatalf("adal: RefreshCallback failed to propagate error")
	}
}

// This demonstrates the danger of manual token without a refresh token
func TestServicePrincipalTokenManualRefreshFailsWithoutRefresh(t *testing.T) {
	spt := newServicePrincipalTokenManual()
	spt.inner.Token.RefreshToken = ""
	err := spt.Refresh()
	if err == nil {
		t.Fatalf("adal: ServicePrincipalToken#Refresh should have failed with a ManualTokenSecret without a refresh token")
	}
}

func TestNewServicePrincipalTokenFromMSI(t *testing.T) {
	const resource = "https://resource"
	cb := func(token Token) error { return nil }

	spt, err := NewServicePrincipalTokenFromMSI("http://msiendpoint/", resource, cb)
	if err != nil {
		t.Fatalf("Failed to get MSI SPT: %v", err)
	}

	// check some of the SPT fields
	if _, ok := spt.inner.Secret.(*ServicePrincipalMSISecret); !ok {
		t.Fatal("SPT secret was not of MSI type")
	}

	if spt.inner.Resource != resource {
		t.Fatal("SPT came back with incorrect resource")
	}

	if len(spt.refreshCallbacks) != 1 {
		t.Fatal("SPT had incorrect refresh callbacks.")
	}
}

func TestNewServicePrincipalTokenFromMSIWithUserAssignedID(t *testing.T) {
	const (
		resource = "https://resource"
		userID   = "abc123"
	)
	cb := func(token Token) error { return nil }

	spt, err := NewServicePrincipalTokenFromMSIWithUserAssignedID("http://msiendpoint/", resource, userID, cb)
	if err != nil {
		t.Fatalf("Failed to get MSI SPT: %v", err)
	}

	// check some of the SPT fields
	if _, ok := spt.inner.Secret.(*ServicePrincipalMSISecret); !ok {
		t.Fatal("SPT secret was not of MSI type")
	}

	if spt.inner.Resource != resource {
		t.Fatal("SPT came back with incorrect resource")
	}

	if len(spt.refreshCallbacks) != 1 {
		t.Fatal("SPT had incorrect refresh callbacks.")
	}

	if spt.inner.ClientID != userID {
		t.Fatal("SPT had incorrect client ID")
	}
}

func TestNewServicePrincipalTokenFromMSIWithIdentityResourceID(t *testing.T) {
	const (
		resource           = "https://resource"
		identityResourceID = "/subscriptions/testSub/resourceGroups/testGroup/providers/Microsoft.ManagedIdentity/userAssignedIdentities/test-identity"
	)
	cb := func(token Token) error { return nil }

	spt, err := NewServicePrincipalTokenFromMSIWithIdentityResourceID("http://msiendpoint/", resource, identityResourceID, cb)
	if err != nil {
		t.Fatalf("Failed to get MSI SPT: %v", err)
	}

	// check some of the SPT fields
	if _, ok := spt.inner.Secret.(*ServicePrincipalMSISecret); !ok {
		t.Fatal("SPT secret was not of MSI type")
	}

	if spt.inner.Resource != resource {
		t.Fatal("SPT came back with incorrect resource")
	}

	if len(spt.refreshCallbacks) != 1 {
		t.Fatal("SPT had incorrect refresh callbacks.")
	}

	urlPathParameter := url.Values{}
	urlPathParameter.Set("mi_res_id", identityResourceID)

	if !strings.Contains(spt.inner.OauthConfig.TokenEndpoint.RawQuery, urlPathParameter.Encode()) {
		t.Fatal("SPT tokenEndpoint should contains mi_res_id")
	}
}

func TestNewServicePrincipalTokenFromManualTokenSecret(t *testing.T) {
	token := newToken()
	secret := &ServicePrincipalAuthorizationCodeSecret{
		ClientSecret:      "clientSecret",
		AuthorizationCode: "code123",
		RedirectURI:       "redirect",
	}

	spt, err := NewServicePrincipalTokenFromManualTokenSecret(TestOAuthConfig, "id", "resource", token, secret, nil)
	if err != nil {
		t.Fatalf("Failed creating new SPT: %s", err)
	}

	if !reflect.DeepEqual(token, spt.inner.Token) {
		t.Fatalf("Tokens do not match: %s, %s", token, spt.inner.Token)
	}

	if !reflect.DeepEqual(secret, spt.inner.Secret) {
		t.Fatalf("Secrets do not match: %s, %s", secret, spt.inner.Secret)
	}

}

func TestGetVMEndpoint(t *testing.T) {
	endpoint, err := GetMSIVMEndpoint()
	if err != nil {
		t.Fatal("Coudn't get VM endpoint")
	}

	if endpoint != msiEndpoint {
		t.Fatal("Didn't get correct endpoint")
	}
}

func TestGetAppServiceEndpoint(t *testing.T) {
	const testEndpoint = "http://172.16.1.2:8081/msi/token"
	const aseSecret = "the_secret"
	if err := os.Setenv(msiEndpointEnv, testEndpoint); err != nil {
		t.Fatalf("os.Setenv: %v", err)
	}
	if err := os.Setenv(msiSecretEnv, aseSecret); err != nil {
		t.Fatalf("os.Setenv: %v", err)
	}
	defer func() {
		os.Unsetenv(msiEndpointEnv)
		os.Unsetenv(msiSecretEnv)
	}()

	endpoint, err := GetMSIAppServiceEndpoint()
	if err != nil {
		t.Fatal("Coudn't get App Service endpoint")
	}

	if endpoint != testEndpoint {
		t.Fatal("Didn't get correct endpoint")
	}
}

func TestGetMSIEndpoint(t *testing.T) {
	const (
		testEndpoint = "http://172.16.1.2:8081/msi/token"
		testSecret   = "DEADBEEF-BBBB-AAAA-DDDD-DDD000000DDD"
	)

	// Test VM well-known endpoint is returned
	if err := os.Unsetenv(msiEndpointEnv); err != nil {
		t.Fatalf("os.Unsetenv: %v", err)
	}

	if err := os.Unsetenv(msiSecretEnv); err != nil {
		t.Fatalf("os.Unsetenv: %v", err)
	}

	vmEndpoint, err := GetMSIEndpoint()
	if err != nil {
		t.Fatal("Coudn't get VM endpoint")
	}

	if vmEndpoint != msiEndpoint {
		t.Fatal("Didn't get correct endpoint")
	}

	// Test App Service endpoint is returned
	if err := os.Setenv(msiEndpointEnv, testEndpoint); err != nil {
		t.Fatalf("os.Setenv: %v", err)
	}

	if err := os.Setenv(msiSecretEnv, testSecret); err != nil {
		t.Fatalf("os.Setenv: %v", err)
	}

	asEndpoint, err := GetMSIEndpoint()
	if err != nil {
		t.Fatal("Coudn't get App Service endpoint")
	}

	if asEndpoint != testEndpoint {
		t.Fatal("Didn't get correct endpoint")
	}

	if err := os.Unsetenv(msiEndpointEnv); err != nil {
		t.Fatalf("os.Unsetenv: %v", err)
	}

	if err := os.Unsetenv(msiSecretEnv); err != nil {
		t.Fatalf("os.Unsetenv: %v", err)
	}
}

func TestClientSecretWithASESet(t *testing.T) {
	if err := os.Setenv(msiEndpointEnv, "http://172.16.1.2:8081/msi/token"); err != nil {
		t.Fatalf("os.Setenv: %v", err)
	}
	if err := os.Setenv(msiSecretEnv, "the_secret"); err != nil {
		t.Fatalf("os.Setenv: %v", err)
	}
	defer func() {
		os.Unsetenv(msiEndpointEnv)
		os.Unsetenv(msiSecretEnv)
	}()
	spt := newServicePrincipalToken()
	if _, ok := spt.inner.Secret.(*ServicePrincipalMSISecret); ok {
		t.Fatal("should not have MSI secret for client secret token even when ASE is enabled")
	}
}

func TestMarshalServicePrincipalNoSecret(t *testing.T) {
	spt := newServicePrincipalTokenManual()
	b, err := json.Marshal(spt)
	if err != nil {
		t.Fatalf("failed to marshal token: %+v", err)
	}
	var spt2 *ServicePrincipalToken
	err = json.Unmarshal(b, &spt2)
	if err != nil {
		t.Fatalf("failed to unmarshal token: %+v", err)
	}
	if !reflect.DeepEqual(spt, spt2) {
		t.Fatal("tokens don't match")
	}
}

func TestMarshalServicePrincipalTokenSecret(t *testing.T) {
	spt := newServicePrincipalToken()
	b, err := json.Marshal(spt)
	if err != nil {
		t.Fatalf("failed to marshal token: %+v", err)
	}
	var spt2 *ServicePrincipalToken
	err = json.Unmarshal(b, &spt2)
	if err != nil {
		t.Fatalf("failed to unmarshal token: %+v", err)
	}
	if !reflect.DeepEqual(spt, spt2) {
		t.Fatal("tokens don't match")
	}
}

func TestMarshalServicePrincipalCertificateSecret(t *testing.T) {
	spt := newServicePrincipalTokenCertificate(t)
	b, err := json.Marshal(spt)
	if err == nil {
		t.Fatal("expected error when marshalling certificate token")
	}
	var spt2 *ServicePrincipalToken
	err = json.Unmarshal(b, &spt2)
	if err == nil {
		t.Fatal("expected error when unmarshalling certificate token")
	}
}

func TestMarshalServicePrincipalMSISecret(t *testing.T) {
	spt, err := newServicePrincipalTokenFromMSI("http://msiendpoint/", "https://resource", "", "")
	if err != nil {
		t.Fatalf("failed to get MSI SPT: %+v", err)
	}
	b, err := json.Marshal(spt)
	if err == nil {
		t.Fatal("expected error when marshalling MSI token")
	}
	var spt2 *ServicePrincipalToken
	err = json.Unmarshal(b, &spt2)
	if err == nil {
		t.Fatal("expected error when unmarshalling MSI token")
	}
}

func TestMarshalServicePrincipalUsernamePasswordSecret(t *testing.T) {
	spt := newServicePrincipalTokenUsernamePassword(t)
	b, err := json.Marshal(spt)
	if err != nil {
		t.Fatalf("failed to marshal token: %+v", err)
	}
	var spt2 *ServicePrincipalToken
	err = json.Unmarshal(b, &spt2)
	if err != nil {
		t.Fatalf("failed to unmarshal token: %+v", err)
	}
	if !reflect.DeepEqual(spt, spt2) {
		t.Fatal("tokens don't match")
	}
}

func TestMarshalServicePrincipalAuthorizationCodeSecret(t *testing.T) {
	spt := newServicePrincipalTokenAuthorizationCode(t)
	b, err := json.Marshal(spt)
	if err != nil {
		t.Fatalf("failed to marshal token: %+v", err)
	}
	var spt2 *ServicePrincipalToken
	err = json.Unmarshal(b, &spt2)
	if err != nil {
		t.Fatalf("failed to unmarshal token: %+v", err)
	}
	if !reflect.DeepEqual(spt, spt2) {
		t.Fatal("tokens don't match")
	}
}

func TestMarshalInnerToken(t *testing.T) {
	spt := newServicePrincipalTokenManual()
	tokenJSON, err := spt.MarshalTokenJSON()
	if err != nil {
		t.Fatalf("failed to marshal token: %+v", err)
	}

	testToken := newToken()
	testToken.RefreshToken = "refreshtoken"

	testTokenJSON, err := json.Marshal(testToken)
	if err != nil {
		t.Fatalf("failed to marshal test token: %+v", err)
	}

	if !reflect.DeepEqual(tokenJSON, testTokenJSON) {
		t.Fatalf("tokens don't match: %s, %s", tokenJSON, testTokenJSON)
	}

	var t1 Token
	err = json.Unmarshal(tokenJSON, &t1)
	if err != nil {
		t.Fatalf("failed to unmarshal token: %+v", err)
	}

	if !reflect.DeepEqual(t1, testToken) {
		t.Fatalf("tokens don't match: %s, %s", t1, testToken)
	}
}

func TestNewMultiTenantServicePrincipalToken(t *testing.T) {
	cfg, err := NewMultiTenantOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID, TestAuxTenantIDs, OAuthOptions{})
	if err != nil {
		t.Fatalf("autorest/adal: unexpected error while creating multitenant config: %v", err)
	}
	mt, err := NewMultiTenantServicePrincipalToken(cfg, "clientID", "superSecret", "resource")
	if err != nil {
		t.Fatalf("autorest/adal: unexpected error while creating multitenant service principal token: %v", err)
	}
	if !strings.Contains(mt.PrimaryToken.inner.OauthConfig.AuthorizeEndpoint.String(), TestTenantID) {
		t.Fatal("didn't find primary tenant ID in primary SPT")
	}
	for i := range mt.AuxiliaryTokens {
		if ep := mt.AuxiliaryTokens[i].inner.OauthConfig.AuthorizeEndpoint.String(); !strings.Contains(ep, fmt.Sprintf("%s%d", TestAuxTenantPrefix, i)) {
			t.Fatalf("didn't find auxiliary tenant ID in token %s", ep)
		}
	}
}

func TestNewMultiTenantServicePrincipalTokenFromCertificate(t *testing.T) {
	cfg, err := NewMultiTenantOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID, TestAuxTenantIDs, OAuthOptions{})
	if err != nil {
		t.Fatalf("autorest/adal: unexpected error while creating multitenant config: %v", err)
	}
	cert, key := newTestCertificate(t)
	mt, err := NewMultiTenantServicePrincipalTokenFromCertificate(cfg, "clientID", cert, key, "resource")
	if err != nil {
		t.Fatalf("autorest/adal: unexpected error while creating multitenant service principal token: %v", err)
	}
	if !strings.Contains(mt.PrimaryToken.inner.OauthConfig.AuthorizeEndpoint.String(), TestTenantID) {
		t.Fatal("didn't find primary tenant ID in primary SPT")
	}
	for i := range mt.AuxiliaryTokens {
		if ep := mt.AuxiliaryTokens[i].inner.OauthConfig.AuthorizeEndpoint.String(); !strings.Contains(ep, fmt.Sprintf("%s%d", TestAuxTenantPrefix, i)) {
			t.Fatalf("didn't find auxiliary tenant ID in token %s", ep)
		}
	}
}

func TestMSIAvailableSuccess(t *testing.T) {
	c := mocks.NewSender()
	c.AppendResponse(mocks.NewResponse())
	if !MSIAvailable(context.Background(), c) {
		t.Fatal("unexpected false")
	}
}

func TestMSIAvailableSlow(t *testing.T) {
	c := mocks.NewSender()
	// introduce a long response delay to simulate the endpoint not being available
	c.AppendResponseWithDelay(mocks.NewResponse(), 5*time.Second)
	if MSIAvailable(context.Background(), c) {
		t.Fatal("unexpected true")
	}
}

func TestMSIAvailableFail(t *testing.T) {
	expectErr := "failed to make msi http request"
	c := mocks.NewSender()
	c.AppendAndRepeatError(fmt.Errorf(expectErr), 2)
	if MSIAvailable(context.Background(), c) {
		t.Fatal("unexpected true")
	}
	_, err := getMSIEndpoint(context.Background(), c)
	if !strings.Contains(err.Error(), "") {
		t.Fatalf("expected error: '%s', but got error '%s'", expectErr, err)
	}
}

func newTokenJSON(expiresIn, expiresOn, resource string) string {
	nb, err := parseExpiresOn(expiresOn)
	if err != nil {
		panic(err)
	}
	return fmt.Sprintf(`{
		"access_token" : "accessToken",
		"expires_in"   : %s,
		"expires_on"   : "%s",
		"not_before"   : "%s",
		"resource"     : "%s",
		"token_type"   : "Bearer",
		"refresh_token": "ABC123"
		}`,
		expiresIn, expiresOn, nb, resource)
}

func newADFSTokenJSON(expiresIn int) string {
	return fmt.Sprintf(`{
		"access_token" : "accessToken",
		"expires_in"   : %d,
		"token_type"   : "Bearer"
		}`,
		expiresIn)
}

func newTokenExpiresIn(expireIn time.Duration) *Token {
	t := newToken()
	return setTokenToExpireIn(&t, expireIn)
}

func newTokenExpiresAt(expireAt time.Time) *Token {
	t := newToken()
	return setTokenToExpireAt(&t, expireAt)
}

func expireToken(t *Token) *Token {
	return setTokenToExpireIn(t, 0)
}

func setTokenToExpireAt(t *Token, expireAt time.Time) *Token {
	t.ExpiresIn = "3600"
	t.ExpiresOn = json.Number(strconv.FormatInt(int64(expireAt.Sub(date.UnixEpoch())/time.Second), 10))
	t.NotBefore = t.ExpiresOn
	return t
}

func setTokenToExpireIn(t *Token, expireIn time.Duration) *Token {
	return setTokenToExpireAt(t, time.Now().Add(expireIn))
}

func newServicePrincipalToken(callbacks ...TokenRefreshCallback) *ServicePrincipalToken {
	spt, _ := NewServicePrincipalToken(TestOAuthConfig, "id", "secret", "resource", callbacks...)
	return spt
}

func newServicePrincipalTokenManual() *ServicePrincipalToken {
	token := newToken()
	token.RefreshToken = "refreshtoken"
	spt, _ := NewServicePrincipalTokenFromManualToken(TestOAuthConfig, "id", "resource", token)
	return spt
}

func newTestCertificate(t *testing.T) (*x509.Certificate, *rsa.PrivateKey) {
	template := x509.Certificate{
		SerialNumber:          big.NewInt(0),
		Subject:               pkix.Name{CommonName: "test"},
		BasicConstraintsValid: true,
	}
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	certificateBytes, err := x509.CreateCertificate(rand.Reader, &template, &template, &privateKey.PublicKey, privateKey)
	if err != nil {
		t.Fatal(err)
	}
	certificate, err := x509.ParseCertificate(certificateBytes)
	if err != nil {
		t.Fatal(err)
	}
	return certificate, privateKey
}

func newServicePrincipalTokenCertificate(t *testing.T) *ServicePrincipalToken {
	certificate, privateKey := newTestCertificate(t)

	spt, _ := NewServicePrincipalTokenFromCertificate(TestOAuthConfig, "id", certificate, privateKey, "resource")
	return spt
}

func newServicePrincipalTokenUsernamePassword(t *testing.T) *ServicePrincipalToken {
	spt, _ := NewServicePrincipalTokenFromUsernamePassword(TestOAuthConfig, "id", "username", "password", "resource")
	return spt
}

func newServicePrincipalTokenAuthorizationCode(t *testing.T) *ServicePrincipalToken {
	spt, _ := NewServicePrincipalTokenFromAuthorizationCode(TestOAuthConfig, "id", "clientSecret", "code", "http://redirectUri/getToken", "resource")
	return spt
}
