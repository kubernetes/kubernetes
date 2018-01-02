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
	"fmt"
	"net/http"
	"reflect"
	"testing"

	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/mocks"
)

const (
	TestTenantID                = "TestTenantID"
	TestActiveDirectoryEndpoint = "https://login/test.com/"
)

func TestWithAuthorizer(t *testing.T) {
	r1 := mocks.NewRequest()

	na := &NullAuthorizer{}
	r2, err := Prepare(r1,
		na.WithAuthorization())
	if err != nil {
		t.Fatalf("autorest: NullAuthorizer#WithAuthorization returned an unexpected error (%v)", err)
	} else if !reflect.DeepEqual(r1, r2) {
		t.Fatalf("autorest: NullAuthorizer#WithAuthorization modified the request -- received %v, expected %v", r2, r1)
	}
}

func TestTokenWithAuthorization(t *testing.T) {
	token := &adal.Token{
		AccessToken: "TestToken",
		Resource:    "https://azure.microsoft.com/",
		Type:        "Bearer",
	}

	ba := NewBearerAuthorizer(token)
	req, err := Prepare(&http.Request{}, ba.WithAuthorization())
	if err != nil {
		t.Fatalf("azure: BearerAuthorizer#WithAuthorization returned an error (%v)", err)
	} else if req.Header.Get(http.CanonicalHeaderKey("Authorization")) != fmt.Sprintf("Bearer %s", token.AccessToken) {
		t.Fatal("azure: BearerAuthorizer#WithAuthorization failed to set Authorization header")
	}
}

func TestServicePrincipalTokenWithAuthorizationNoRefresh(t *testing.T) {
	oauthConfig, err := adal.NewOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID)
	if err != nil {
		t.Fatalf("azure: BearerAuthorizer#WithAuthorization returned an error (%v)", err)
	}
	spt, err := adal.NewServicePrincipalToken(*oauthConfig, "id", "secret", "resource", nil)
	if err != nil {
		t.Fatalf("azure: BearerAuthorizer#WithAuthorization returned an error (%v)", err)
	}
	spt.SetAutoRefresh(false)
	s := mocks.NewSender()
	spt.SetSender(s)

	ba := NewBearerAuthorizer(spt)
	req, err := Prepare(mocks.NewRequest(), ba.WithAuthorization())
	if err != nil {
		t.Fatalf("azure: BearerAuthorizer#WithAuthorization returned an error (%v)", err)
	} else if req.Header.Get(http.CanonicalHeaderKey("Authorization")) != fmt.Sprintf("Bearer %s", spt.AccessToken) {
		t.Fatal("azure: BearerAuthorizer#WithAuthorization failed to set Authorization header")
	}
}

func TestServicePrincipalTokenWithAuthorizationRefresh(t *testing.T) {

	oauthConfig, err := adal.NewOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID)
	if err != nil {
		t.Fatalf("azure: BearerAuthorizer#WithAuthorization returned an error (%v)", err)
	}
	refreshed := false
	spt, err := adal.NewServicePrincipalToken(*oauthConfig, "id", "secret", "resource", func(t adal.Token) error {
		refreshed = true
		return nil
	})
	if err != nil {
		t.Fatalf("azure: BearerAuthorizer#WithAuthorization returned an error (%v)", err)
	}

	jwt := `{
		"access_token" : "accessToken",
		"expires_in"   : "3600",
		"expires_on"   : "test",
		"not_before"   : "test",
		"resource"     : "test",
		"token_type"   : "Bearer"
	}`
	body := mocks.NewBody(jwt)
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

	ba := NewBearerAuthorizer(spt)
	req, err := Prepare(mocks.NewRequest(), ba.WithAuthorization())
	if err != nil {
		t.Fatalf("azure: BearerAuthorizer#WithAuthorization returned an error (%v)", err)
	} else if req.Header.Get(http.CanonicalHeaderKey("Authorization")) != fmt.Sprintf("Bearer %s", spt.AccessToken) {
		t.Fatal("azure: BearerAuthorizer#WithAuthorization failed to set Authorization header")
	}

	if !refreshed {
		t.Fatal("azure: BearerAuthorizer#WithAuthorization must refresh the token")
	}
}

func TestServicePrincipalTokenWithAuthorizationReturnsErrorIfConnotRefresh(t *testing.T) {
	oauthConfig, err := adal.NewOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID)
	if err != nil {
		t.Fatalf("azure: BearerAuthorizer#WithAuthorization returned an error (%v)", err)
	}
	spt, err := adal.NewServicePrincipalToken(*oauthConfig, "id", "secret", "resource", nil)
	if err != nil {
		t.Fatalf("azure: BearerAuthorizer#WithAuthorization returned an error (%v)", err)
	}

	s := mocks.NewSender()
	s.AppendResponse(mocks.NewResponseWithStatus("400 Bad Request", http.StatusBadRequest))
	spt.SetSender(s)

	ba := NewBearerAuthorizer(spt)
	_, err = Prepare(mocks.NewRequest(), ba.WithAuthorization())
	if err == nil {
		t.Fatal("azure: BearerAuthorizer#WithAuthorization failed to return an error when refresh fails")
	}
}

func TestBearerAuthorizerCallback(t *testing.T) {
	tenantString := "123-tenantID-456"
	resourceString := "https://fake.resource.net"

	s := mocks.NewSender()
	resp := mocks.NewResponseWithStatus("401 Unauthorized", http.StatusUnauthorized)
	mocks.SetResponseHeader(resp, bearerChallengeHeader, bearer+" \"authorization\"=\"https://fake.net/"+tenantString+"\",\"resource\"=\""+resourceString+"\"")
	s.AppendResponse(resp)

	auth := NewBearerAuthorizerCallback(s, func(tenantID, resource string) (*BearerAuthorizer, error) {
		if tenantID != tenantString {
			t.Fatal("BearerAuthorizerCallback: bad tenant ID")
		}
		if resource != resourceString {
			t.Fatal("BearerAuthorizerCallback: bad resource")
		}

		oauthConfig, err := adal.NewOAuthConfig(TestActiveDirectoryEndpoint, tenantID)
		if err != nil {
			t.Fatalf("azure: NewOAuthConfig returned an error (%v)", err)
		}

		spt, err := adal.NewServicePrincipalToken(*oauthConfig, "id", "secret", resource)
		if err != nil {
			t.Fatalf("azure: NewServicePrincipalToken returned an error (%v)", err)
		}

		spt.SetSender(s)
		return NewBearerAuthorizer(spt), nil
	})

	_, err := Prepare(mocks.NewRequest(), auth.WithAuthorization())
	if err == nil {
		t.Fatal("azure: BearerAuthorizerCallback#WithAuthorization failed to return an error when refresh fails")
	}
}
