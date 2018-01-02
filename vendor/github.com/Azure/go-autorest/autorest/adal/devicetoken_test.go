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
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/Azure/go-autorest/autorest/mocks"
)

const (
	TestResource                = "SomeResource"
	TestClientID                = "SomeClientID"
	TestTenantID                = "SomeTenantID"
	TestActiveDirectoryEndpoint = "https://login.test.com/"
)

var (
	testOAuthConfig, _ = NewOAuthConfig(TestActiveDirectoryEndpoint, TestTenantID)
	TestOAuthConfig    = *testOAuthConfig
)

const MockDeviceCodeResponse = `
{
	"device_code": "10000-40-1234567890",
	"user_code": "ABCDEF",
	"verification_url": "http://aka.ms/deviceauth",
	"expires_in": "900",
	"interval": "0"
}
`

const MockDeviceTokenResponse = `{
	"access_token": "accessToken",
	"refresh_token": "refreshToken",
	"expires_in": "1000",
	"expires_on": "2000",
	"not_before": "3000",
	"resource": "resource",
	"token_type": "type"
}
`

func TestDeviceCodeIncludesResource(t *testing.T) {
	sender := mocks.NewSender()
	sender.AppendResponse(mocks.NewResponseWithContent(MockDeviceCodeResponse))

	code, err := InitiateDeviceAuth(sender, TestOAuthConfig, TestClientID, TestResource)
	if err != nil {
		t.Fatalf("adal: unexpected error initiating device auth")
	}

	if code.Resource != TestResource {
		t.Fatalf("adal: InitiateDeviceAuth failed to stash the resource in the DeviceCode struct")
	}
}

func TestDeviceCodeReturnsErrorIfSendingFails(t *testing.T) {
	sender := mocks.NewSender()
	sender.SetError(fmt.Errorf("this is an error"))

	_, err := InitiateDeviceAuth(sender, TestOAuthConfig, TestClientID, TestResource)
	if err == nil || !strings.Contains(err.Error(), errCodeSendingFails) {
		t.Fatalf("adal: failed to get correct error expected(%s) actual(%s)", errCodeSendingFails, err.Error())
	}
}

func TestDeviceCodeReturnsErrorIfBadRequest(t *testing.T) {
	sender := mocks.NewSender()
	body := mocks.NewBody("doesn't matter")
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusBadRequest, "Bad Request"))

	_, err := InitiateDeviceAuth(sender, TestOAuthConfig, TestClientID, TestResource)
	if err == nil || !strings.Contains(err.Error(), errCodeHandlingFails) {
		t.Fatalf("adal: failed to get correct error expected(%s) actual(%s)", errCodeHandlingFails, err.Error())
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

func TestDeviceCodeReturnsErrorIfCannotDeserializeDeviceCode(t *testing.T) {
	gibberishJSON := strings.Replace(MockDeviceCodeResponse, "expires_in", "\":, :gibberish", -1)
	sender := mocks.NewSender()
	body := mocks.NewBody(gibberishJSON)
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK"))

	_, err := InitiateDeviceAuth(sender, TestOAuthConfig, TestClientID, TestResource)
	if err == nil || !strings.Contains(err.Error(), errCodeHandlingFails) {
		t.Fatalf("adal: failed to get correct error expected(%s) actual(%s)", errCodeHandlingFails, err.Error())
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

func TestDeviceCodeReturnsErrorIfEmptyDeviceCode(t *testing.T) {
	sender := mocks.NewSender()
	body := mocks.NewBody("")
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK"))

	_, err := InitiateDeviceAuth(sender, TestOAuthConfig, TestClientID, TestResource)
	if err != ErrDeviceCodeEmpty {
		t.Fatalf("adal: failed to get correct error expected(%s) actual(%s)", ErrDeviceCodeEmpty, err.Error())
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

func deviceCode() *DeviceCode {
	var deviceCode DeviceCode
	_ = json.Unmarshal([]byte(MockDeviceCodeResponse), &deviceCode)
	deviceCode.Resource = TestResource
	deviceCode.ClientID = TestClientID
	return &deviceCode
}

func TestDeviceTokenReturns(t *testing.T) {
	sender := mocks.NewSender()
	body := mocks.NewBody(MockDeviceTokenResponse)
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK"))

	_, err := WaitForUserCompletion(sender, deviceCode())
	if err != nil {
		t.Fatalf("adal: got error unexpectedly")
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

func TestDeviceTokenReturnsErrorIfSendingFails(t *testing.T) {
	sender := mocks.NewSender()
	sender.SetError(fmt.Errorf("this is an error"))

	_, err := WaitForUserCompletion(sender, deviceCode())
	if err == nil || !strings.Contains(err.Error(), errTokenSendingFails) {
		t.Fatalf("adal: failed to get correct error expected(%s) actual(%s)", errTokenSendingFails, err.Error())
	}
}

func TestDeviceTokenReturnsErrorIfServerError(t *testing.T) {
	sender := mocks.NewSender()
	body := mocks.NewBody("")
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusInternalServerError, "Internal Server Error"))

	_, err := WaitForUserCompletion(sender, deviceCode())
	if err == nil || !strings.Contains(err.Error(), errTokenHandlingFails) {
		t.Fatalf("adal: failed to get correct error expected(%s) actual(%s)", errTokenHandlingFails, err.Error())
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

func TestDeviceTokenReturnsErrorIfCannotDeserializeDeviceToken(t *testing.T) {
	gibberishJSON := strings.Replace(MockDeviceTokenResponse, "expires_in", ";:\"gibberish", -1)
	sender := mocks.NewSender()
	body := mocks.NewBody(gibberishJSON)
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK"))

	_, err := WaitForUserCompletion(sender, deviceCode())
	if err == nil || !strings.Contains(err.Error(), errTokenHandlingFails) {
		t.Fatalf("adal: failed to get correct error expected(%s) actual(%s)", errTokenHandlingFails, err.Error())
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

func errorDeviceTokenResponse(message string) string {
	return `{ "error": "` + message + `" }`
}

func TestDeviceTokenReturnsErrorIfAuthorizationPending(t *testing.T) {
	sender := mocks.NewSender()
	body := mocks.NewBody(errorDeviceTokenResponse("authorization_pending"))
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusBadRequest, "Bad Request"))

	_, err := CheckForUserCompletion(sender, deviceCode())
	if err != ErrDeviceAuthorizationPending {
		t.Fatalf("!!!")
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

func TestDeviceTokenReturnsErrorIfSlowDown(t *testing.T) {
	sender := mocks.NewSender()
	body := mocks.NewBody(errorDeviceTokenResponse("slow_down"))
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusBadRequest, "Bad Request"))

	_, err := CheckForUserCompletion(sender, deviceCode())
	if err != ErrDeviceSlowDown {
		t.Fatalf("!!!")
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

type deviceTokenSender struct {
	errorString string
	attempts    int
}

func newDeviceTokenSender(deviceErrorString string) *deviceTokenSender {
	return &deviceTokenSender{errorString: deviceErrorString, attempts: 0}
}

func (s *deviceTokenSender) Do(req *http.Request) (*http.Response, error) {
	var resp *http.Response
	if s.attempts < 1 {
		s.attempts++
		resp = mocks.NewResponseWithContent(errorDeviceTokenResponse(s.errorString))
	} else {
		resp = mocks.NewResponseWithContent(MockDeviceTokenResponse)
	}
	return resp, nil
}

// since the above only exercise CheckForUserCompletion, we repeat the test here,
// but with the intent of showing that WaitForUserCompletion loops properly.
func TestDeviceTokenSucceedsWithIntermediateAuthPending(t *testing.T) {
	sender := newDeviceTokenSender("authorization_pending")

	_, err := WaitForUserCompletion(sender, deviceCode())
	if err != nil {
		t.Fatalf("unexpected error occurred")
	}
}

// same as above but with SlowDown now
func TestDeviceTokenSucceedsWithIntermediateSlowDown(t *testing.T) {
	sender := newDeviceTokenSender("slow_down")

	_, err := WaitForUserCompletion(sender, deviceCode())
	if err != nil {
		t.Fatalf("unexpected error occurred")
	}
}

func TestDeviceTokenReturnsErrorIfAccessDenied(t *testing.T) {
	sender := mocks.NewSender()
	body := mocks.NewBody(errorDeviceTokenResponse("access_denied"))
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusBadRequest, "Bad Request"))

	_, err := WaitForUserCompletion(sender, deviceCode())
	if err != ErrDeviceAccessDenied {
		t.Fatalf("adal: got wrong error expected(%s) actual(%s)", ErrDeviceAccessDenied.Error(), err.Error())
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

func TestDeviceTokenReturnsErrorIfCodeExpired(t *testing.T) {
	sender := mocks.NewSender()
	body := mocks.NewBody(errorDeviceTokenResponse("code_expired"))
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusBadRequest, "Bad Request"))

	_, err := WaitForUserCompletion(sender, deviceCode())
	if err != ErrDeviceCodeExpired {
		t.Fatalf("adal: got wrong error expected(%s) actual(%s)", ErrDeviceCodeExpired.Error(), err.Error())
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

func TestDeviceTokenReturnsErrorForUnknownError(t *testing.T) {
	sender := mocks.NewSender()
	body := mocks.NewBody(errorDeviceTokenResponse("unknown_error"))
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusBadRequest, "Bad Request"))

	_, err := WaitForUserCompletion(sender, deviceCode())
	if err == nil {
		t.Fatalf("failed to get error")
	}
	if err != ErrDeviceGeneric {
		t.Fatalf("adal: got wrong error expected(%s) actual(%s)", ErrDeviceGeneric.Error(), err.Error())
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}

func TestDeviceTokenReturnsErrorIfTokenEmptyAndStatusOK(t *testing.T) {
	sender := mocks.NewSender()
	body := mocks.NewBody("")
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(body, http.StatusOK, "OK"))

	_, err := WaitForUserCompletion(sender, deviceCode())
	if err != ErrOAuthTokenEmpty {
		t.Fatalf("adal: got wrong error expected(%s) actual(%s)", ErrOAuthTokenEmpty.Error(), err.Error())
	}

	if body.IsOpen() {
		t.Fatalf("response body was left open!")
	}
}
