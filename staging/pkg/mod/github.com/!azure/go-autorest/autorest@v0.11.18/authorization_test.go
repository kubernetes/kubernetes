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
	"strings"
	"testing"

	"github.com/Azure/go-autorest/autorest/adal"
	"github.com/Azure/go-autorest/autorest/mocks"
)

const (
	TestTenantID                = "TestTenantID"
	TestAuxTenent1              = "aux1"
	TestAuxTenent2              = "aux2"
	TestAuxTenent3              = "aux3"
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
	} else if req.Header.Get(http.CanonicalHeaderKey("Authorization")) != fmt.Sprintf("Bearer %s", spt.OAuthToken()) {
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
		"expires_on"   : "12345",
		"not_before"   : "67890",
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
	} else if req.Header.Get(http.CanonicalHeaderKey("Authorization")) != fmt.Sprintf("Bearer %s", spt.OAuthToken()) {
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

func TestApiKeyAuthorization(t *testing.T) {

	headers := make(map[string]interface{})
	queryParameters := make(map[string]interface{})

	dummyAuthHeader := "dummyAuthHeader"
	dummyAuthHeaderValue := "dummyAuthHeaderValue"

	dummyAuthQueryParameter := "dummyAuthQueryParameter"
	dummyAuthQueryParameterValue := "dummyAuthQueryParameterValue"

	headers[dummyAuthHeader] = dummyAuthHeaderValue
	queryParameters[dummyAuthQueryParameter] = dummyAuthQueryParameterValue

	aka := NewAPIKeyAuthorizer(headers, queryParameters)

	req, err := Prepare(mocks.NewRequest(), aka.WithAuthorization())

	if err != nil {
		t.Fatalf("azure: APIKeyAuthorizer#WithAuthorization returned an error (%v)", err)
	} else if req.Header.Get(http.CanonicalHeaderKey(dummyAuthHeader)) != dummyAuthHeaderValue {
		t.Fatalf("azure: APIKeyAuthorizer#WithAuthorization failed to set %s header", dummyAuthHeader)

	} else if req.URL.Query().Get(dummyAuthQueryParameter) != dummyAuthQueryParameterValue {
		t.Fatalf("azure: APIKeyAuthorizer#WithAuthorization failed to set %s query parameter", dummyAuthQueryParameterValue)
	}
}

func TestCognitivesServicesAuthorization(t *testing.T) {
	subscriptionKey := "dummyKey"
	csa := NewCognitiveServicesAuthorizer(subscriptionKey)
	req, err := Prepare(mocks.NewRequest(), csa.WithAuthorization())

	if err != nil {
		t.Fatalf("azure: CognitiveServicesAuthorizer#WithAuthorization returned an error (%v)", err)
	} else if req.Header.Get(http.CanonicalHeaderKey(bingAPISdkHeader)) != golangBingAPISdkHeaderValue {
		t.Fatalf("azure: CognitiveServicesAuthorizer#WithAuthorization failed to set %s header", bingAPISdkHeader)
	} else if req.Header.Get(http.CanonicalHeaderKey(apiKeyAuthorizerHeader)) != subscriptionKey {
		t.Fatalf("azure: CognitiveServicesAuthorizer#WithAuthorization failed to set %s header", apiKeyAuthorizerHeader)
	}
}

func TestBasicAuthorization(t *testing.T) {
	ba := NewBasicAuthorizer("Aladdin", "open sesame")
	req, err := Prepare(mocks.NewRequest(), ba.WithAuthorization())

	if err != nil {
		t.Fatalf("BasicAuthorizer#WithAuthorization returned an error (%v)", err)
	} else if req.Header.Get(http.CanonicalHeaderKey(authorization)) != basic+" QWxhZGRpbjpvcGVuIHNlc2FtZQ==" {
		t.Fatalf("BasicAuthorizer#WithAuthorization failed to set %s header", authorization)
	}
}

func TestBasicAuthorizationPasswordOnly(t *testing.T) {
	ba := NewBasicAuthorizer("", "dummyKey")
	req, err := Prepare(mocks.NewRequest(), ba.WithAuthorization())

	if err != nil {
		t.Fatalf("BasicAuthorizer#WithAuthorization returned an error (%v)", err)
	} else if req.Header.Get(http.CanonicalHeaderKey(authorization)) != basic+" OmR1bW15S2V5" {
		t.Fatalf("BasicAuthorizer#WithAuthorization failed to set %s header", authorization)
	}
}

type mockMTSPTProvider struct {
	p string
	a []string
}

func (m mockMTSPTProvider) PrimaryOAuthToken() string {
	return m.p
}

func (m mockMTSPTProvider) AuxiliaryOAuthTokens() []string {
	return m.a
}

func TestMultitenantAuthorizationOne(t *testing.T) {
	mtSPTProvider := mockMTSPTProvider{
		p: "primary",
		a: []string{TestAuxTenent1},
	}
	mt := NewMultiTenantServicePrincipalTokenAuthorizer(mtSPTProvider)
	req, err := Prepare(mocks.NewRequest(), mt.WithAuthorization())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if primary := req.Header.Get(headerAuthorization); primary != "Bearer primary" {
		t.Fatalf("bad primary authorization header %s", primary)
	}
	if aux := req.Header.Get(headerAuxAuthorization); aux != "Bearer aux1" {
		t.Fatalf("bad auxiliary authorization header %s", aux)
	}
}

func TestMultitenantAuthorizationThree(t *testing.T) {
	mtSPTProvider := mockMTSPTProvider{
		p: "primary",
		a: []string{TestAuxTenent1, TestAuxTenent2, TestAuxTenent3},
	}
	mt := NewMultiTenantBearerAuthorizer(mtSPTProvider)
	req, err := Prepare(mocks.NewRequest(), mt.WithAuthorization())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if primary := mt.TokenProvider().PrimaryOAuthToken(); primary != mtSPTProvider.p {
		t.Fatalf("bad primary authorization token %s", primary)
	}
	if aux := strings.Join(mt.TokenProvider().AuxiliaryOAuthTokens(), ","); aux != strings.Join(mtSPTProvider.a, ",") {
		t.Fatalf("bad auxiliary authorization tokens %s", aux)
	}
	if primary := req.Header.Get(headerAuthorization); primary != "Bearer primary" {
		t.Fatalf("bad primary authorization header %s", primary)
	}
	if aux := req.Header.Get(headerAuxAuthorization); aux != "Bearer aux1, Bearer aux2, Bearer aux3" {
		t.Fatalf("bad auxiliary authorization header %s", aux)
	}
}

func TestMultiTenantServicePrincipalTokenWithAuthorizationRefresh(t *testing.T) {
	multiTenantCfg, err := adal.NewMultiTenantOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID, []string{TestAuxTenent1, TestAuxTenent2, TestAuxTenent3}, adal.OAuthOptions{})
	if err != nil {
		t.Fatalf("azure: adal#NewMultiTenantOAuthConfig returned an error (%v)", err)
	}
	mtSpt, err := adal.NewMultiTenantServicePrincipalToken(multiTenantCfg, "id", "secret", "resource")
	if err != nil {
		t.Fatalf("azure: adal#NewMultiTenantServicePrincipalToken returned an error (%v)", err)
	}

	primaryToken := `{
		"access_token" : "primary token refreshed",
		"expires_in"   : "3600",
		"expires_on"   : "12345",
		"not_before"   : "67890",
		"resource"     : "test",
		"token_type"   : "Bearer"
	}`

	auxToken1 := `{
		"access_token" : "aux token 1 refreshed",
		"expires_in"   : "3600",
		"expires_on"   : "12345",
		"not_before"   : "67890",
		"resource"     : "test",
		"token_type"   : "Bearer"
	}`

	auxToken2 := `{
		"access_token" : "aux token 2 refreshed",
		"expires_in"   : "3600",
		"expires_on"   : "12345",
		"not_before"   : "67890",
		"resource"     : "test",
		"token_type"   : "Bearer"
	}`

	auxToken3 := `{
		"access_token" : "aux token 3 refreshed",
		"expires_in"   : "3600",
		"expires_on"   : "12345",
		"not_before"   : "67890",
		"resource"     : "test",
		"token_type"   : "Bearer"
	}`

	s := mocks.NewSender()
	s.AppendResponse(mocks.NewResponseWithBodyAndStatus(mocks.NewBody(primaryToken), http.StatusOK, "OK"))
	s.AppendResponse(mocks.NewResponseWithBodyAndStatus(mocks.NewBody(auxToken1), http.StatusOK, "OK"))
	s.AppendResponse(mocks.NewResponseWithBodyAndStatus(mocks.NewBody(auxToken2), http.StatusOK, "OK"))
	s.AppendResponse(mocks.NewResponseWithBodyAndStatus(mocks.NewBody(auxToken3), http.StatusOK, "OK"))

	mtSpt.PrimaryToken.SetSender(s)
	for _, aux := range mtSpt.AuxiliaryTokens {
		aux.SetSender(s)
	}

	mta := NewMultiTenantServicePrincipalTokenAuthorizer(mtSpt)
	req, err := Prepare(mocks.NewRequest(), mta.WithAuthorization())
	if err != nil {
		t.Fatalf("azure: multiTenantSPTAuthorizer#WithAuthorization returned an error (%v)", err)
	}
	if ah := req.Header.Get(http.CanonicalHeaderKey("Authorization")); ah != fmt.Sprintf("Bearer %s", mtSpt.PrimaryOAuthToken()) {
		t.Fatal("azure: multiTenantSPTAuthorizer#WithAuthorization failed to set Authorization header for primary token")
	} else if ah != "Bearer primary token refreshed" {
		t.Fatal("azure: multiTenantSPTAuthorizer#WithAuthorization primary token value doesn't match")
	}
	auxTokens := mtSpt.AuxiliaryOAuthTokens()
	for i := range auxTokens {
		auxTokens[i] = fmt.Sprintf("Bearer %s", auxTokens[i])
	}
	auxHeader := req.Header.Get(http.CanonicalHeaderKey(headerAuxAuthorization))
	if auxHeader != strings.Join(auxTokens, ", ") {
		t.Fatal("azure: multiTenantSPTAuthorizer#WithAuthorization failed to set Authorization header for auxiliary tokens")
	}
	for i := range auxTokens {
		if auxTokens[i] != fmt.Sprintf("Bearer aux token %d refreshed", i+1) {
			t.Fatal("azure: multiTenantSPTAuthorizer#WithAuthorization auxiliary token value doesn't match")
		}
	}
}

func TestMultiTenantServicePrincipalTokenWithAuthorizationRefreshFail1(t *testing.T) {
	multiTenantCfg, err := adal.NewMultiTenantOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID, []string{TestAuxTenent1, TestAuxTenent2, TestAuxTenent3}, adal.OAuthOptions{})
	if err != nil {
		t.Fatalf("azure: adal#NewMultiTenantOAuthConfig returned an error (%v)", err)
	}
	mtSpt, err := adal.NewMultiTenantServicePrincipalToken(multiTenantCfg, "id", "secret", "resource")
	if err != nil {
		t.Fatalf("azure: adal#NewMultiTenantServicePrincipalToken returned an error (%v)", err)
	}

	s := mocks.NewSender()
	s.AppendResponse(mocks.NewResponseWithStatus("access denied", http.StatusForbidden))

	mtSpt.PrimaryToken.SetSender(s)
	for _, aux := range mtSpt.AuxiliaryTokens {
		aux.SetSender(s)
	}

	mta := NewMultiTenantServicePrincipalTokenAuthorizer(mtSpt)
	_, err = Prepare(mocks.NewRequest(), mta.WithAuthorization())
	if err == nil {
		t.Fatalf("azure: multiTenantSPTAuthorizer#WithAuthorization unexpected nil error")
	}
}

func TestMultiTenantServicePrincipalTokenWithAuthorizationRefreshFail2(t *testing.T) {
	multiTenantCfg, err := adal.NewMultiTenantOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID, []string{TestAuxTenent1, TestAuxTenent2, TestAuxTenent3}, adal.OAuthOptions{})
	if err != nil {
		t.Fatalf("azure: adal#NewMultiTenantOAuthConfig returned an error (%v)", err)
	}
	mtSpt, err := adal.NewMultiTenantServicePrincipalToken(multiTenantCfg, "id", "secret", "resource")
	if err != nil {
		t.Fatalf("azure: adal#NewMultiTenantServicePrincipalToken returned an error (%v)", err)
	}

	primaryToken := `{
		"access_token" : "primary token refreshed",
		"expires_in"   : "3600",
		"expires_on"   : "test",
		"not_before"   : "test",
		"resource"     : "test",
		"token_type"   : "Bearer"
	}`

	s := mocks.NewSender()
	s.AppendResponse(mocks.NewResponseWithBodyAndStatus(mocks.NewBody(primaryToken), http.StatusOK, "OK"))
	s.AppendResponse(mocks.NewResponseWithStatus("access denied", http.StatusForbidden))

	mtSpt.PrimaryToken.SetSender(s)
	for _, aux := range mtSpt.AuxiliaryTokens {
		aux.SetSender(s)
	}

	mta := NewMultiTenantServicePrincipalTokenAuthorizer(mtSpt)
	_, err = Prepare(mocks.NewRequest(), mta.WithAuthorization())
	if err == nil {
		t.Fatalf("azure: multiTenantSPTAuthorizer#WithAuthorization unexpected nil error")
	}
}
