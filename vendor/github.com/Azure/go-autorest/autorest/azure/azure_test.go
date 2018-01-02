package azure

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
	"io/ioutil"
	"net/http"
	"reflect"
	"strconv"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/mocks"
)

const (
	headerAuthorization = "Authorization"
	longDelay           = 5 * time.Second
	retryDelay          = 10 * time.Millisecond
	testLogPrefix       = "azure:"
)

// Use a Client Inspector to set the request identifier.
func ExampleWithClientID() {
	uuid := "71FDB9F4-5E49-4C12-B266-DE7B4FD999A6"
	req, _ := autorest.Prepare(&http.Request{},
		autorest.AsGet(),
		autorest.WithBaseURL("https://microsoft.com/a/b/c/"))

	c := autorest.Client{Sender: mocks.NewSender()}
	c.RequestInspector = WithReturningClientID(uuid)

	autorest.SendWithSender(c, req)
	fmt.Printf("Inspector added the %s header with the value %s\n",
		HeaderClientID, req.Header.Get(HeaderClientID))
	fmt.Printf("Inspector added the %s header with the value %s\n",
		HeaderReturnClientID, req.Header.Get(HeaderReturnClientID))
	// Output:
	// Inspector added the x-ms-client-request-id header with the value 71FDB9F4-5E49-4C12-B266-DE7B4FD999A6
	// Inspector added the x-ms-return-client-request-id header with the value true
}

func TestWithReturningClientIDReturnsError(t *testing.T) {
	var errIn error
	uuid := "71FDB9F4-5E49-4C12-B266-DE7B4FD999A6"
	_, errOut := autorest.Prepare(&http.Request{},
		withErrorPrepareDecorator(&errIn),
		WithReturningClientID(uuid))

	if errOut == nil || errIn != errOut {
		t.Fatalf("azure: WithReturningClientID failed to exit early when receiving an error -- expected (%v), received (%v)",
			errIn, errOut)
	}
}

func TestWithClientID(t *testing.T) {
	uuid := "71FDB9F4-5E49-4C12-B266-DE7B4FD999A6"
	req, _ := autorest.Prepare(&http.Request{},
		WithClientID(uuid))

	if req.Header.Get(HeaderClientID) != uuid {
		t.Fatalf("azure: WithClientID failed to set %s -- expected %s, received %s",
			HeaderClientID, uuid, req.Header.Get(HeaderClientID))
	}
}

func TestWithReturnClientID(t *testing.T) {
	b := false
	req, _ := autorest.Prepare(&http.Request{},
		WithReturnClientID(b))

	if req.Header.Get(HeaderReturnClientID) != strconv.FormatBool(b) {
		t.Fatalf("azure: WithReturnClientID failed to set %s -- expected %s, received %s",
			HeaderClientID, strconv.FormatBool(b), req.Header.Get(HeaderClientID))
	}
}

func TestExtractClientID(t *testing.T) {
	uuid := "71FDB9F4-5E49-4C12-B266-DE7B4FD999A6"
	resp := mocks.NewResponse()
	mocks.SetResponseHeader(resp, HeaderClientID, uuid)

	if ExtractClientID(resp) != uuid {
		t.Fatalf("azure: ExtractClientID failed to extract the %s -- expected %s, received %s",
			HeaderClientID, uuid, ExtractClientID(resp))
	}
}

func TestExtractRequestID(t *testing.T) {
	uuid := "71FDB9F4-5E49-4C12-B266-DE7B4FD999A6"
	resp := mocks.NewResponse()
	mocks.SetResponseHeader(resp, HeaderRequestID, uuid)

	if ExtractRequestID(resp) != uuid {
		t.Fatalf("azure: ExtractRequestID failed to extract the %s -- expected %s, received %s",
			HeaderRequestID, uuid, ExtractRequestID(resp))
	}
}

func TestIsAzureError_ReturnsTrueForAzureError(t *testing.T) {
	if !IsAzureError(&RequestError{}) {
		t.Fatalf("azure: IsAzureError failed to return true for an Azure Service error")
	}
}

func TestIsAzureError_ReturnsFalseForNonAzureError(t *testing.T) {
	if IsAzureError(fmt.Errorf("An Error")) {
		t.Fatalf("azure: IsAzureError return true for an non-Azure Service error")
	}
}

func TestNewErrorWithError_UsesReponseStatusCode(t *testing.T) {
	e := NewErrorWithError(fmt.Errorf("Error"), "packageType", "method", mocks.NewResponseWithStatus("Forbidden", http.StatusForbidden), "message")
	if e.StatusCode != http.StatusForbidden {
		t.Fatalf("azure: NewErrorWithError failed to use the Status Code of the passed Response -- expected %v, received %v", http.StatusForbidden, e.StatusCode)
	}
}

func TestNewErrorWithError_ReturnsUnwrappedError(t *testing.T) {
	e1 := RequestError{}
	e1.ServiceError = &ServiceError{Code: "42", Message: "A Message"}
	e1.StatusCode = 200
	e1.RequestID = "A RequestID"
	e2 := NewErrorWithError(&e1, "packageType", "method", nil, "message")

	if !reflect.DeepEqual(e1, e2) {
		t.Fatalf("azure: NewErrorWithError wrapped an RequestError -- expected %T, received %T", e1, e2)
	}
}

func TestNewErrorWithError_WrapsAnError(t *testing.T) {
	e1 := fmt.Errorf("Inner Error")
	var e2 interface{} = NewErrorWithError(e1, "packageType", "method", nil, "message")

	if _, ok := e2.(RequestError); !ok {
		t.Fatalf("azure: NewErrorWithError failed to wrap a standard error -- received %T", e2)
	}
}

func TestWithErrorUnlessStatusCode_NotAnAzureError(t *testing.T) {
	body := `<html>
		<head>
			<title>IIS Error page</title>
		</head>
		<body>Some non-JSON error page</body>
	</html>`
	r := mocks.NewResponseWithContent(body)
	r.Request = mocks.NewRequest()
	r.StatusCode = http.StatusBadRequest
	r.Status = http.StatusText(r.StatusCode)

	err := autorest.Respond(r,
		WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByClosing())
	ok, _ := err.(*RequestError)
	if ok != nil {
		t.Fatalf("azure: azure.RequestError returned from malformed response: %v", err)
	}

	// the error body should still be there
	defer r.Body.Close()
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != body {
		t.Fatalf("response body is wrong. got=%q exptected=%q", string(b), body)
	}
}

func TestWithErrorUnlessStatusCode_FoundAzureErrorWithoutDetails(t *testing.T) {
	j := `{
		"error": {
			"code": "InternalError",
			"message": "Azure is having trouble right now."
		}
	}`
	uuid := "71FDB9F4-5E49-4C12-B266-DE7B4FD999A6"
	r := mocks.NewResponseWithContent(j)
	mocks.SetResponseHeader(r, HeaderRequestID, uuid)
	r.Request = mocks.NewRequest()
	r.StatusCode = http.StatusInternalServerError
	r.Status = http.StatusText(r.StatusCode)

	err := autorest.Respond(r,
		WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByClosing())

	if err == nil {
		t.Fatalf("azure: returned nil error for proper error response")
	}
	azErr, ok := err.(*RequestError)
	if !ok {
		t.Fatalf("azure: returned error is not azure.RequestError: %T", err)
	}

	expected := "autorest/azure: Service returned an error. Status=500 Code=\"InternalError\" Message=\"Azure is having trouble right now.\""
	if !reflect.DeepEqual(expected, azErr.Error()) {
		t.Fatalf("azure: service error is not unmarshaled properly.\nexpected=%v\ngot=%v", expected, azErr.Error())
	}

	if expected := http.StatusInternalServerError; azErr.StatusCode != expected {
		t.Fatalf("azure: got wrong StatusCode=%d Expected=%d", azErr.StatusCode, expected)
	}
	if expected := uuid; azErr.RequestID != expected {
		t.Fatalf("azure: wrong request ID in error. expected=%q; got=%q", expected, azErr.RequestID)
	}

	_ = azErr.Error()

	// the error body should still be there
	defer r.Body.Close()
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != j {
		t.Fatalf("response body is wrong. got=%q expected=%q", string(b), j)
	}

}

func TestWithErrorUnlessStatusCode_FoundAzureErrorWithDetails(t *testing.T) {
	j := `{
		"error": {
			"code": "InternalError",
			"message": "Azure is having trouble right now.",
			"details": [{"code": "conflict1", "message":"error message1"}, 
						{"code": "conflict2", "message":"error message2"}]
		}
	}`
	uuid := "71FDB9F4-5E49-4C12-B266-DE7B4FD999A6"
	r := mocks.NewResponseWithContent(j)
	mocks.SetResponseHeader(r, HeaderRequestID, uuid)
	r.Request = mocks.NewRequest()
	r.StatusCode = http.StatusInternalServerError
	r.Status = http.StatusText(r.StatusCode)

	err := autorest.Respond(r,
		WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByClosing())

	if err == nil {
		t.Fatalf("azure: returned nil error for proper error response")
	}
	azErr, ok := err.(*RequestError)
	if !ok {
		t.Fatalf("azure: returned error is not azure.RequestError: %T", err)
	}

	if expected := "InternalError"; azErr.ServiceError.Code != expected {
		t.Fatalf("azure: wrong error code. expected=%q; got=%q", expected, azErr.ServiceError.Code)
	}
	if azErr.ServiceError.Message == "" {
		t.Fatalf("azure: error message is not unmarshaled properly")
	}
	b, _ := json.Marshal(*azErr.ServiceError.Details)
	if string(b) != `[{"code":"conflict1","message":"error message1"},{"code":"conflict2","message":"error message2"}]` {
		t.Fatalf("azure: error details is not unmarshaled properly")
	}

	if expected := http.StatusInternalServerError; azErr.StatusCode != expected {
		t.Fatalf("azure: got wrong StatusCode=%v Expected=%d", azErr.StatusCode, expected)
	}
	if expected := uuid; azErr.RequestID != expected {
		t.Fatalf("azure: wrong request ID in error. expected=%q; got=%q", expected, azErr.RequestID)
	}

	_ = azErr.Error()

	// the error body should still be there
	defer r.Body.Close()
	b, err = ioutil.ReadAll(r.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != j {
		t.Fatalf("response body is wrong. got=%q expected=%q", string(b), j)
	}

}

func TestWithErrorUnlessStatusCode_NoAzureError(t *testing.T) {
	j := `{
			"Status":"NotFound"
		}`
	uuid := "71FDB9F4-5E49-4C12-B266-DE7B4FD999A6"
	r := mocks.NewResponseWithContent(j)
	mocks.SetResponseHeader(r, HeaderRequestID, uuid)
	r.Request = mocks.NewRequest()
	r.StatusCode = http.StatusInternalServerError
	r.Status = http.StatusText(r.StatusCode)

	err := autorest.Respond(r,
		WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByClosing())
	if err == nil {
		t.Fatalf("azure: returned nil error for proper error response")
	}
	azErr, ok := err.(*RequestError)
	if !ok {
		t.Fatalf("azure: returned error is not azure.RequestError: %T", err)
	}

	expected := &ServiceError{
		Code:    "Unknown",
		Message: "Unknown service error",
	}

	if !reflect.DeepEqual(expected, azErr.ServiceError) {
		t.Fatalf("azure: service error is not unmarshaled properly. expected=%q\ngot=%q", expected, azErr.ServiceError)
	}

	if expected := http.StatusInternalServerError; azErr.StatusCode != expected {
		t.Fatalf("azure: got wrong StatusCode=%v Expected=%d", azErr.StatusCode, expected)
	}
	if expected := uuid; azErr.RequestID != expected {
		t.Fatalf("azure: wrong request ID in error. expected=%q; got=%q", expected, azErr.RequestID)
	}

	_ = azErr.Error()

	// the error body should still be there
	defer r.Body.Close()
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != j {
		t.Fatalf("response body is wrong. got=%q expected=%q", string(b), j)
	}

}

func TestWithErrorUnlessStatusCode_UnwrappedError(t *testing.T) {
	j := `{
		"target": null,
		"code": "InternalError",
		"message": "Azure is having trouble right now.",
		"details": [{"code": "conflict1", "message":"error message1"},
					{"code": "conflict2", "message":"error message2"}],
		"innererror": []
}`
	uuid := "71FDB9F4-5E49-4C12-B266-DE7B4FD999A6"
	r := mocks.NewResponseWithContent(j)
	mocks.SetResponseHeader(r, HeaderRequestID, uuid)
	r.Request = mocks.NewRequest()
	r.StatusCode = http.StatusInternalServerError
	r.Status = http.StatusText(r.StatusCode)

	err := autorest.Respond(r,
		WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByClosing())

	if err == nil {
		t.Fatal("azure: returned nil error for proper error response")
	}

	azErr, ok := err.(*RequestError)
	if !ok {
		t.Fatalf("returned error is not azure.RequestError: %T", err)
	}

	if expected := http.StatusInternalServerError; azErr.StatusCode != expected {
		t.Logf("Incorrect StatusCode got: %v want: %d", azErr.StatusCode, expected)
		t.Fail()
	}

	if expected := "Azure is having trouble right now."; azErr.ServiceError.Message != expected {
		t.Logf("Incorrect Message\n\tgot:  %q\n\twant: %q", azErr.Message, expected)
		t.Fail()
	}

	if expected := uuid; azErr.RequestID != expected {
		t.Logf("Incorrect request ID\n\tgot:  %q\n\twant: %q", azErr.RequestID, expected)
		t.Fail()
	}

	expectedServiceErrorDetails := `[{"code":"conflict1","message":"error message1"},{"code":"conflict2","message":"error message2"}]`
	if azErr.ServiceError == nil {
		t.Logf("`ServiceError` was nil when it shouldn't have been.")
		t.Fail()
	} else if azErr.ServiceError.Details == nil {
		t.Logf("`ServiceError.Details` was nil when it should have been %q", expectedServiceErrorDetails)
		t.Fail()
	} else if details, _ := json.Marshal(*azErr.ServiceError.Details); expectedServiceErrorDetails != string(details) {
		t.Logf("Error detaisl was not unmarshaled properly.\n\tgot:  %q\n\twant: %q", string(details), expectedServiceErrorDetails)
		t.Fail()
	}

	// the error body should still be there
	defer r.Body.Close()
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Error(err)
	}
	if string(b) != j {
		t.Fatalf("response body is wrong. got=%q expected=%q", string(b), j)
	}

}

func TestRequestErrorString_WithError(t *testing.T) {
	j := `{
		"error": {
			"code": "InternalError",
			"message": "Conflict",
			"details": [{"code": "conflict1", "message":"error message1"}]
		}
	}`
	uuid := "71FDB9F4-5E49-4C12-B266-DE7B4FD999A6"
	r := mocks.NewResponseWithContent(j)
	mocks.SetResponseHeader(r, HeaderRequestID, uuid)
	r.Request = mocks.NewRequest()
	r.StatusCode = http.StatusInternalServerError
	r.Status = http.StatusText(r.StatusCode)

	err := autorest.Respond(r,
		WithErrorUnlessStatusCode(http.StatusOK),
		autorest.ByClosing())

	if err == nil {
		t.Fatalf("azure: returned nil error for proper error response")
	}
	azErr, _ := err.(*RequestError)
	expected := "autorest/azure: Service returned an error. Status=500 Code=\"InternalError\" Message=\"Conflict\" Details=[{\"code\":\"conflict1\",\"message\":\"error message1\"}]"
	if expected != azErr.Error() {
		t.Fatalf("azure: send wrong RequestError.\nexpected=%v\ngot=%v", expected, azErr.Error())
	}
}

func withErrorPrepareDecorator(e *error) autorest.PrepareDecorator {
	return func(p autorest.Preparer) autorest.Preparer {
		return autorest.PreparerFunc(func(r *http.Request) (*http.Request, error) {
			*e = fmt.Errorf("azure: Faux Prepare Error")
			return r, *e
		})
	}
}

func withAsyncResponseDecorator(n int) autorest.SendDecorator {
	i := 0
	return func(s autorest.Sender) autorest.Sender {
		return autorest.SenderFunc(func(r *http.Request) (*http.Response, error) {
			resp, err := s.Do(r)
			if err == nil {
				if i < n {
					resp.StatusCode = http.StatusCreated
					resp.Header = http.Header{}
					resp.Header.Add(http.CanonicalHeaderKey(headerAsyncOperation), mocks.TestURL)
					i++
				} else {
					resp.StatusCode = http.StatusOK
					resp.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
				}
			}
			return resp, err
		})
	}
}

type mockAuthorizer struct{}

func (ma mockAuthorizer) WithAuthorization() autorest.PrepareDecorator {
	return autorest.WithHeader(headerAuthorization, mocks.TestAuthorizationHeader)
}

type mockFailingAuthorizer struct{}

func (mfa mockFailingAuthorizer) WithAuthorization() autorest.PrepareDecorator {
	return func(p autorest.Preparer) autorest.Preparer {
		return autorest.PreparerFunc(func(r *http.Request) (*http.Request, error) {
			return r, fmt.Errorf("ERROR: mockFailingAuthorizer returned expected error")
		})
	}
}

type mockInspector struct {
	wasInvoked bool
}

func (mi *mockInspector) WithInspection() autorest.PrepareDecorator {
	return func(p autorest.Preparer) autorest.Preparer {
		return autorest.PreparerFunc(func(r *http.Request) (*http.Request, error) {
			mi.wasInvoked = true
			return p.Prepare(r)
		})
	}
}

func (mi *mockInspector) ByInspecting() autorest.RespondDecorator {
	return func(r autorest.Responder) autorest.Responder {
		return autorest.ResponderFunc(func(resp *http.Response) error {
			mi.wasInvoked = true
			return r.Respond(resp)
		})
	}
}
