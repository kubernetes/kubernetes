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
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/mocks"
)

func TestGetAsyncOperation_ReturnsAzureAsyncOperationHeader(t *testing.T) {
	r := newAsynchronousResponse()

	if getAsyncOperation(r) != mocks.TestAzureAsyncURL {
		t.Fatalf("azure: getAsyncOperation failed to extract the Azure-AsyncOperation header -- expected %v, received %v", mocks.TestURL, getAsyncOperation(r))
	}
}

func TestGetAsyncOperation_ReturnsEmptyStringIfHeaderIsAbsent(t *testing.T) {
	r := mocks.NewResponse()

	if len(getAsyncOperation(r)) != 0 {
		t.Fatalf("azure: getAsyncOperation failed to return empty string when the Azure-AsyncOperation header is absent -- received %v", getAsyncOperation(r))
	}
}

func TestHasSucceeded_ReturnsTrueForSuccess(t *testing.T) {
	if !hasSucceeded(operationSucceeded) {
		t.Fatal("azure: hasSucceeded failed to return true for success")
	}
}

func TestHasSucceeded_ReturnsFalseOtherwise(t *testing.T) {
	if hasSucceeded("not a success string") {
		t.Fatal("azure: hasSucceeded returned true for a non-success")
	}
}

func TestHasTerminated_ReturnsTrueForValidTerminationStates(t *testing.T) {
	for _, state := range []string{operationSucceeded, operationCanceled, operationFailed} {
		if !hasTerminated(state) {
			t.Fatalf("azure: hasTerminated failed to return true for the '%s' state", state)
		}
	}
}

func TestHasTerminated_ReturnsFalseForUnknownStates(t *testing.T) {
	if hasTerminated("not a known state") {
		t.Fatal("azure: hasTerminated returned true for an unknown state")
	}
}

func TestOperationError_ErrorReturnsAString(t *testing.T) {
	s := (ServiceError{Code: "server code", Message: "server error"}).Error()
	if s == "" {
		t.Fatalf("azure: operationError#Error failed to return an error")
	}
	if !strings.Contains(s, "server code") || !strings.Contains(s, "server error") {
		t.Fatalf("azure: operationError#Error returned a malformed error -- error='%v'", s)
	}
}

func TestOperationResource_StateReturnsState(t *testing.T) {
	if (operationResource{Status: "state"}).state() != "state" {
		t.Fatalf("azure: operationResource#state failed to return the correct state")
	}
}

func TestOperationResource_HasSucceededReturnsFalseIfNotSuccess(t *testing.T) {
	if (operationResource{Status: "not a success string"}).hasSucceeded() {
		t.Fatalf("azure: operationResource#hasSucceeded failed to return false for a canceled operation")
	}
}

func TestOperationResource_HasSucceededReturnsTrueIfSuccessful(t *testing.T) {
	if !(operationResource{Status: operationSucceeded}).hasSucceeded() {
		t.Fatalf("azure: operationResource#hasSucceeded failed to return true for a successful operation")
	}
}

func TestOperationResource_HasTerminatedReturnsTrueForKnownStates(t *testing.T) {
	for _, state := range []string{operationSucceeded, operationCanceled, operationFailed} {
		if !(operationResource{Status: state}).hasTerminated() {
			t.Fatalf("azure: operationResource#hasTerminated failed to return true for the '%s' state", state)
		}
	}
}

func TestOperationResource_HasTerminatedReturnsFalseForUnknownStates(t *testing.T) {
	if (operationResource{Status: "not a known state"}).hasTerminated() {
		t.Fatalf("azure: operationResource#hasTerminated returned true for a non-terminal operation")
	}
}

func TestProvisioningStatus_StateReturnsState(t *testing.T) {
	if (provisioningStatus{Properties: provisioningProperties{"state"}}).state() != "state" {
		t.Fatalf("azure: provisioningStatus#state failed to return the correct state")
	}
}

func TestProvisioningStatus_HasSucceededReturnsFalseIfNotSuccess(t *testing.T) {
	if (provisioningStatus{Properties: provisioningProperties{"not a success string"}}).hasSucceeded() {
		t.Fatalf("azure: provisioningStatus#hasSucceeded failed to return false for a canceled operation")
	}
}

func TestProvisioningStatus_HasSucceededReturnsTrueIfSuccessful(t *testing.T) {
	if !(provisioningStatus{Properties: provisioningProperties{operationSucceeded}}).hasSucceeded() {
		t.Fatalf("azure: provisioningStatus#hasSucceeded failed to return true for a successful operation")
	}
}

func TestProvisioningStatus_HasTerminatedReturnsTrueForKnownStates(t *testing.T) {
	for _, state := range []string{operationSucceeded, operationCanceled, operationFailed} {
		if !(provisioningStatus{Properties: provisioningProperties{state}}).hasTerminated() {
			t.Fatalf("azure: provisioningStatus#hasTerminated failed to return true for the '%s' state", state)
		}
	}
}

func TestProvisioningStatus_HasTerminatedReturnsFalseForUnknownStates(t *testing.T) {
	if (provisioningStatus{Properties: provisioningProperties{"not a known state"}}).hasTerminated() {
		t.Fatalf("azure: provisioningStatus#hasTerminated returned true for a non-terminal operation")
	}
}

func TestPollingState_HasSucceededReturnsFalseIfNotSuccess(t *testing.T) {
	if (pollingState{state: "not a success string"}).hasSucceeded() {
		t.Fatalf("azure: pollingState#hasSucceeded failed to return false for a canceled operation")
	}
}

func TestPollingState_HasSucceededReturnsTrueIfSuccessful(t *testing.T) {
	if !(pollingState{state: operationSucceeded}).hasSucceeded() {
		t.Fatalf("azure: pollingState#hasSucceeded failed to return true for a successful operation")
	}
}

func TestPollingState_HasTerminatedReturnsTrueForKnownStates(t *testing.T) {
	for _, state := range []string{operationSucceeded, operationCanceled, operationFailed} {
		if !(pollingState{state: state}).hasTerminated() {
			t.Fatalf("azure: pollingState#hasTerminated failed to return true for the '%s' state", state)
		}
	}
}

func TestPollingState_HasTerminatedReturnsFalseForUnknownStates(t *testing.T) {
	if (pollingState{state: "not a known state"}).hasTerminated() {
		t.Fatalf("azure: pollingState#hasTerminated returned true for a non-terminal operation")
	}
}

func TestUpdatePollingState_ReturnsAnErrorIfOneOccurs(t *testing.T) {
	resp := mocks.NewResponseWithContent(operationResourceIllegal)
	err := updatePollingState(resp, &pollingState{})
	if err == nil {
		t.Fatalf("azure: updatePollingState failed to return an error after a JSON parsing error")
	}
}

func TestUpdatePollingState_ReturnsTerminatedForKnownProvisioningStates(t *testing.T) {
	for _, state := range []string{operationSucceeded, operationCanceled, operationFailed} {
		resp := mocks.NewResponseWithContent(fmt.Sprintf(pollingStateFormat, state))
		resp.StatusCode = 42
		ps := &pollingState{responseFormat: usesProvisioningStatus}
		updatePollingState(resp, ps)
		if !ps.hasTerminated() {
			t.Fatalf("azure: updatePollingState failed to return a terminating pollingState for the '%s' state", state)
		}
	}
}

func TestUpdatePollingState_ReturnsSuccessForSuccessfulProvisioningState(t *testing.T) {
	resp := mocks.NewResponseWithContent(fmt.Sprintf(pollingStateFormat, operationSucceeded))
	resp.StatusCode = 42
	ps := &pollingState{responseFormat: usesProvisioningStatus}
	updatePollingState(resp, ps)
	if !ps.hasSucceeded() {
		t.Fatalf("azure: updatePollingState failed to return a successful pollingState for the '%s' state", operationSucceeded)
	}
}

func TestUpdatePollingState_ReturnsInProgressForAllOtherProvisioningStates(t *testing.T) {
	s := "not a recognized state"
	resp := mocks.NewResponseWithContent(fmt.Sprintf(pollingStateFormat, s))
	resp.StatusCode = 42
	ps := &pollingState{responseFormat: usesProvisioningStatus}
	updatePollingState(resp, ps)
	if ps.hasTerminated() {
		t.Fatalf("azure: updatePollingState returned terminated for unknown state '%s'", s)
	}
}

func TestUpdatePollingState_ReturnsSuccessWhenProvisioningStateFieldIsAbsentForSuccessStatusCodes(t *testing.T) {
	for _, sc := range []int{http.StatusOK, http.StatusCreated, http.StatusNoContent} {
		resp := mocks.NewResponseWithContent(pollingStateEmpty)
		resp.StatusCode = sc
		ps := &pollingState{responseFormat: usesProvisioningStatus}
		updatePollingState(resp, ps)
		if !ps.hasSucceeded() {
			t.Fatalf("azure: updatePollingState failed to return success when the provisionState field is absent for Status Code %d", sc)
		}
	}
}

func TestUpdatePollingState_ReturnsInProgressWhenProvisioningStateFieldIsAbsentForAccepted(t *testing.T) {
	resp := mocks.NewResponseWithContent(pollingStateEmpty)
	resp.StatusCode = http.StatusAccepted
	ps := &pollingState{responseFormat: usesProvisioningStatus}
	updatePollingState(resp, ps)
	if ps.hasTerminated() {
		t.Fatalf("azure: updatePollingState returned terminated when the provisionState field is absent for Status Code Accepted")
	}
}

func TestUpdatePollingState_ReturnsFailedWhenProvisioningStateFieldIsAbsentForUnknownStatusCodes(t *testing.T) {
	resp := mocks.NewResponseWithContent(pollingStateEmpty)
	resp.StatusCode = 42
	ps := &pollingState{responseFormat: usesProvisioningStatus}
	updatePollingState(resp, ps)
	if !ps.hasTerminated() || ps.hasSucceeded() {
		t.Fatalf("azure: updatePollingState did not return failed when the provisionState field is absent for an unknown Status Code")
	}
}

func TestUpdatePollingState_ReturnsTerminatedForKnownOperationResourceStates(t *testing.T) {
	for _, state := range []string{operationSucceeded, operationCanceled, operationFailed} {
		resp := mocks.NewResponseWithContent(fmt.Sprintf(operationResourceFormat, state))
		resp.StatusCode = 42
		ps := &pollingState{responseFormat: usesOperationResponse}
		updatePollingState(resp, ps)
		if !ps.hasTerminated() {
			t.Fatalf("azure: updatePollingState failed to return a terminating pollingState for the '%s' state", state)
		}
	}
}

func TestUpdatePollingState_ReturnsSuccessForSuccessfulOperationResourceState(t *testing.T) {
	resp := mocks.NewResponseWithContent(fmt.Sprintf(operationResourceFormat, operationSucceeded))
	resp.StatusCode = 42
	ps := &pollingState{responseFormat: usesOperationResponse}
	updatePollingState(resp, ps)
	if !ps.hasSucceeded() {
		t.Fatalf("azure: updatePollingState failed to return a successful pollingState for the '%s' state", operationSucceeded)
	}
}

func TestUpdatePollingState_ReturnsInProgressForAllOtherOperationResourceStates(t *testing.T) {
	s := "not a recognized state"
	resp := mocks.NewResponseWithContent(fmt.Sprintf(operationResourceFormat, s))
	resp.StatusCode = 42
	ps := &pollingState{responseFormat: usesOperationResponse}
	updatePollingState(resp, ps)
	if ps.hasTerminated() {
		t.Fatalf("azure: updatePollingState returned terminated for unknown state '%s'", s)
	}
}

func TestUpdatePollingState_CopiesTheResponseBody(t *testing.T) {
	s := fmt.Sprintf(pollingStateFormat, operationSucceeded)
	resp := mocks.NewResponseWithContent(s)
	resp.StatusCode = 42
	ps := &pollingState{responseFormat: usesOperationResponse}
	updatePollingState(resp, ps)
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("azure: updatePollingState failed to replace the http.Response Body -- Error='%v'", err)
	}
	if string(b) != s {
		t.Fatalf("azure: updatePollingState failed to copy the http.Response Body -- Expected='%s' Received='%s'", s, string(b))
	}
}

func TestUpdatePollingState_ClosesTheOriginalResponseBody(t *testing.T) {
	resp := mocks.NewResponse()
	b := resp.Body.(*mocks.Body)
	ps := &pollingState{responseFormat: usesProvisioningStatus}
	updatePollingState(resp, ps)
	if b.IsOpen() {
		t.Fatal("azure: updatePollingState failed to close the original http.Response Body")
	}
}

func TestUpdatePollingState_FailsWhenResponseLacksRequest(t *testing.T) {
	resp := newAsynchronousResponse()
	resp.Request = nil

	ps := pollingState{}
	err := updatePollingState(resp, &ps)
	if err == nil {
		t.Fatal("azure: updatePollingState failed to return an error when the http.Response lacked the original http.Request")
	}
}

func TestUpdatePollingState_SetsTheResponseFormatWhenUsingTheAzureAsyncOperationHeader(t *testing.T) {
	ps := pollingState{}
	updatePollingState(newAsynchronousResponse(), &ps)

	if ps.responseFormat != usesOperationResponse {
		t.Fatal("azure: updatePollingState failed to set the correct response format when using the Azure-AsyncOperation header")
	}
}

func TestUpdatePollingState_SetsTheResponseFormatWhenUsingTheAzureAsyncOperationHeaderIsMissing(t *testing.T) {
	resp := newAsynchronousResponse()
	resp.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))

	ps := pollingState{}
	updatePollingState(resp, &ps)

	if ps.responseFormat != usesProvisioningStatus {
		t.Fatal("azure: updatePollingState failed to set the correct response format when the Azure-AsyncOperation header is absent")
	}
}

func TestUpdatePollingState_DoesNotChangeAnExistingReponseFormat(t *testing.T) {
	resp := newAsynchronousResponse()
	resp.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))

	ps := pollingState{responseFormat: usesOperationResponse}
	updatePollingState(resp, &ps)

	if ps.responseFormat != usesOperationResponse {
		t.Fatal("azure: updatePollingState failed to leave an existing response format setting")
	}
}

func TestUpdatePollingState_PrefersTheAzureAsyncOperationHeader(t *testing.T) {
	resp := newAsynchronousResponse()

	ps := pollingState{}
	updatePollingState(resp, &ps)

	if ps.uri != mocks.TestAzureAsyncURL {
		t.Fatal("azure: updatePollingState failed to prefer the Azure-AsyncOperation header")
	}
}

func TestUpdatePollingState_PrefersLocationWhenTheAzureAsyncOperationHeaderMissing(t *testing.T) {
	resp := newAsynchronousResponse()
	resp.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))

	ps := pollingState{}
	updatePollingState(resp, &ps)

	if ps.uri != mocks.TestLocationURL {
		t.Fatal("azure: updatePollingState failed to prefer the Location header when the Azure-AsyncOperation header is missing")
	}
}

func TestUpdatePollingState_UsesTheObjectLocationIfAsyncHeadersAreMissing(t *testing.T) {
	resp := newAsynchronousResponse()
	resp.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	resp.Header.Del(http.CanonicalHeaderKey(autorest.HeaderLocation))
	resp.Request.Method = http.MethodPatch

	ps := pollingState{}
	updatePollingState(resp, &ps)

	if ps.uri != mocks.TestURL {
		t.Fatal("azure: updatePollingState failed to use the Object URL when the asynchronous headers are missing")
	}
}

func TestUpdatePollingState_RecognizesLowerCaseHTTPVerbs(t *testing.T) {
	for _, m := range []string{strings.ToLower(http.MethodPatch), strings.ToLower(http.MethodPut), strings.ToLower(http.MethodGet)} {
		resp := newAsynchronousResponse()
		resp.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
		resp.Header.Del(http.CanonicalHeaderKey(autorest.HeaderLocation))
		resp.Request.Method = m

		ps := pollingState{}
		updatePollingState(resp, &ps)

		if ps.uri != mocks.TestURL {
			t.Fatalf("azure: updatePollingState failed to recognize the lower-case HTTP verb '%s'", m)
		}
	}
}

func TestUpdatePollingState_ReturnsAnErrorIfAsyncHeadersAreMissingForANewOrDeletedObject(t *testing.T) {
	resp := newAsynchronousResponse()
	resp.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	resp.Header.Del(http.CanonicalHeaderKey(autorest.HeaderLocation))

	for _, m := range []string{http.MethodDelete, http.MethodPost} {
		resp.Request.Method = m
		err := updatePollingState(resp, &pollingState{})
		if err == nil {
			t.Fatalf("azure: updatePollingState failed to return an error even though it could not determine the polling URL for Method '%s'", m)
		}
	}
}

func TestNewPollingRequest_FailsWhenResponseLacksRequest(t *testing.T) {
	resp := newAsynchronousResponse()
	resp.Request = nil

	_, err := newPollingRequest(resp, pollingState{})
	if err == nil {
		t.Fatal("azure: newPollingRequest failed to return an error when the http.Response lacked the original http.Request")
	}
}

func TestNewPollingRequest_ReturnsAnErrorWhenPrepareFails(t *testing.T) {
	_, err := newPollingRequest(newAsynchronousResponse(), pollingState{responseFormat: usesOperationResponse, uri: mocks.TestBadURL})
	if err == nil {
		t.Fatal("azure: newPollingRequest failed to return an error when Prepare fails")
	}
}

func TestNewPollingRequest_DoesNotReturnARequestWhenPrepareFails(t *testing.T) {
	req, _ := newPollingRequest(newAsynchronousResponse(), pollingState{responseFormat: usesOperationResponse, uri: mocks.TestBadURL})
	if req != nil {
		t.Fatal("azure: newPollingRequest returned an http.Request when Prepare failed")
	}
}

func TestNewPollingRequest_ReturnsAGetRequest(t *testing.T) {
	req, _ := newPollingRequest(newAsynchronousResponse(), pollingState{responseFormat: usesOperationResponse, uri: mocks.TestAzureAsyncURL})
	if req.Method != "GET" {
		t.Fatalf("azure: newPollingRequest did not create an HTTP GET request -- actual method %v", req.Method)
	}
}

func TestDoPollForAsynchronous_IgnoresUnspecifiedStatusCodes(t *testing.T) {
	client := mocks.NewSender()

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Duration(0)))

	if client.Attempts() != 1 {
		t.Fatalf("azure: DoPollForAsynchronous polled for unspecified status code")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_PollsForSpecifiedStatusCodes(t *testing.T) {
	client := mocks.NewSender()
	client.AppendResponse(newAsynchronousResponse())

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() != 2 {
		t.Fatalf("azure: DoPollForAsynchronous failed to poll for specified status code")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_CanBeCanceled(t *testing.T) {
	cancel := make(chan struct{})
	delay := 5 * time.Second

	r1 := newAsynchronousResponse()

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(newOperationResourceResponse("Busy"), -1)

	var wg sync.WaitGroup
	wg.Add(1)
	start := time.Now()
	go func() {
		req := mocks.NewRequest()
		req.Cancel = cancel

		wg.Done()

		r, _ := autorest.SendWithSender(client, req,
			DoPollForAsynchronous(10*time.Second))
		autorest.Respond(r,
			autorest.ByClosing())
	}()
	wg.Wait()
	close(cancel)
	time.Sleep(5 * time.Millisecond)
	if time.Since(start) >= delay {
		t.Fatalf("azure: DoPollForAsynchronous failed to cancel")
	}
}

func TestDoPollForAsynchronous_ClosesAllNonreturnedResponseBodiesWhenPolling(t *testing.T) {
	r1 := newAsynchronousResponse()
	b1 := r1.Body.(*mocks.Body)
	r2 := newOperationResourceResponse("busy")
	b2 := r2.Body.(*mocks.Body)
	r3 := newOperationResourceResponse(operationSucceeded)
	b3 := r3.Body.(*mocks.Body)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendResponse(r3)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if b1.IsOpen() || b2.IsOpen() || b3.IsOpen() {
		t.Fatalf("azure: DoPollForAsynchronous did not close unreturned response bodies")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_LeavesLastResponseBodyOpen(t *testing.T) {
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceResponse(operationSucceeded)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendResponse(r3)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	b, err := ioutil.ReadAll(r.Body)
	if len(b) <= 0 || err != nil {
		t.Fatalf("azure: DoPollForAsynchronous did not leave open the body of the last response - Error='%v'", err)
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_DoesNotPollIfOriginalRequestReturnedAnError(t *testing.T) {
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendResponse(r2)
	client.SetError(fmt.Errorf("Faux Error"))

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() != 1 {
		t.Fatalf("azure: DoPollForAsynchronous tried to poll after receiving an error")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_DoesNotPollIfCreatingOperationRequestFails(t *testing.T) {
	r1 := newAsynchronousResponse()
	mocks.SetResponseHeader(r1, http.CanonicalHeaderKey(headerAsyncOperation), mocks.TestBadURL)
	r2 := newOperationResourceResponse("busy")

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() > 1 {
		t.Fatalf("azure: DoPollForAsynchronous polled with an invalidly formed operation request")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_StopsPollingAfterAnError(t *testing.T) {
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.SetError(fmt.Errorf("Faux Error"))
	client.SetEmitErrorAfter(2)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() > 3 {
		t.Fatalf("azure: DoPollForAsynchronous failed to stop polling after receiving an error")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_ReturnsPollingError(t *testing.T) {
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(newAsynchronousResponse(), 5)
	client.SetError(fmt.Errorf("Faux Error"))
	client.SetEmitErrorAfter(1)

	r, err := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if err == nil {
		t.Fatalf("azure: DoPollForAsynchronous failed to return error from polling")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_PollsForStatusAccepted(t *testing.T) {
	r1 := newAsynchronousResponse()
	r1.Status = "202 Accepted"
	r1.StatusCode = http.StatusAccepted
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceResponse(operationCanceled)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() < 4 {
		t.Fatalf("azure: DoPollForAsynchronous stopped polling before receiving a terminated OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_PollsForStatusCreated(t *testing.T) {
	r1 := newAsynchronousResponse()
	r1.Status = "201 Created"
	r1.StatusCode = http.StatusCreated
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceResponse(operationCanceled)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() < 4 {
		t.Fatalf("azure: DoPollForAsynchronous stopped polling before receiving a terminated OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_PollsUntilProvisioningStatusTerminates(t *testing.T) {
	r1 := newAsynchronousResponse()
	r1.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	r2 := newProvisioningStatusResponse("busy")
	r2.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	r3 := newProvisioningStatusResponse(operationCanceled)
	r3.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() < 4 {
		t.Fatalf("azure: DoPollForAsynchronous stopped polling before receiving a terminated OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_PollsUntilProvisioningStatusSucceeds(t *testing.T) {
	r1 := newAsynchronousResponse()
	r1.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	r2 := newProvisioningStatusResponse("busy")
	r2.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	r3 := newProvisioningStatusResponse(operationSucceeded)
	r3.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() < 4 {
		t.Fatalf("azure: DoPollForAsynchronous stopped polling before receiving a terminated OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_PollsUntilOperationResourceHasTerminated(t *testing.T) {
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceResponse(operationCanceled)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() < 4 {
		t.Fatalf("azure: DoPollForAsynchronous stopped polling before receiving a terminated OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_PollsUntilOperationResourceHasSucceeded(t *testing.T) {
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceResponse(operationSucceeded)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() < 4 {
		t.Fatalf("azure: DoPollForAsynchronous stopped polling before receiving a terminated OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_StopsPollingWhenOperationResourceHasTerminated(t *testing.T) {
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceResponse(operationCanceled)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 2)

	r, _ := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() > 4 {
		t.Fatalf("azure: DoPollForAsynchronous failed to stop after receiving a terminated OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_ReturnsAnErrorForCanceledOperations(t *testing.T) {
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceErrorResponse(operationCanceled)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, err := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if err == nil || !strings.Contains(fmt.Sprintf("%v", err), "Canceled") {
		t.Fatalf("azure: DoPollForAsynchronous failed to return an appropriate error for a canceled OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_ReturnsAnErrorForFailedOperations(t *testing.T) {
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceErrorResponse(operationFailed)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, err := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if err == nil || !strings.Contains(fmt.Sprintf("%v", err), "Failed") {
		t.Fatalf("azure: DoPollForAsynchronous failed to return an appropriate error for a canceled OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_WithNilURI(t *testing.T) {
	r1 := newAsynchronousResponse()
	r1.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	r1.Header.Del(http.CanonicalHeaderKey(autorest.HeaderLocation))

	r2 := newOperationResourceResponse("busy")
	r2.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	r2.Header.Del(http.CanonicalHeaderKey(autorest.HeaderLocation))

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendResponse(r2)

	req, _ := http.NewRequest("POST", "https://microsoft.com/a/b/c/", mocks.NewBody(""))
	r, err := autorest.SendWithSender(client, req,
		DoPollForAsynchronous(time.Millisecond))

	if err == nil {
		t.Fatalf("azure: DoPollForAsynchronous failed to return error for nil URI. got: nil; want: Azure Polling Error - Unable to obtain polling URI for POST")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_ReturnsAnUnknownErrorForFailedOperations(t *testing.T) {
	// Return unknown error if error not present in last response
	r1 := newAsynchronousResponse()
	r1.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	r2 := newProvisioningStatusResponse("busy")
	r2.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	r3 := newProvisioningStatusResponse(operationFailed)
	r3.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, err := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	expected := makeLongRunningOperationErrorString("Unknown", "None")
	if err.Error() != expected {
		t.Fatalf("azure: DoPollForAsynchronous failed to return an appropriate error message for an unknown error. \n expected=%q \n got=%q",
			expected, err.Error())
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_ReturnsErrorForLastErrorResponse(t *testing.T) {
	// Return error code and message if error present in last response
	r1 := newAsynchronousResponse()
	r1.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	r2 := newProvisioningStatusResponse("busy")
	r2.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))
	r3 := newAsynchronousResponseWithError()
	r3.Header.Del(http.CanonicalHeaderKey(headerAsyncOperation))

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, err := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	expected := makeLongRunningOperationErrorString("InvalidParameter", "tom-service-DISCOVERY-server-base-v1.core.local' is not a valid captured VHD blob name prefix.")
	if err.Error() != expected {
		t.Fatalf("azure: DoPollForAsynchronous failed to return an appropriate error message for an unknown error. \n expected=%q \n got=%q",
			expected, err.Error())
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_ReturnsOperationResourceErrorForFailedOperations(t *testing.T) {
	// Return Operation resource response with error code and message in last operation resource response
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceErrorResponse(operationFailed)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, err := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	expected := makeLongRunningOperationErrorString("BadArgument", "The provided database 'foo' has an invalid username.")
	if err.Error() != expected {
		t.Fatalf("azure: DoPollForAsynchronous failed to return an appropriate error message for a failed Operations. \n expected=%q \n got=%q",
			expected, err.Error())
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_ReturnsErrorForFirstPutRequest(t *testing.T) {
	// Return 400 bad response with error code and message in first put
	r1 := newAsynchronousResponseWithError()
	client := mocks.NewSender()
	client.AppendResponse(r1)

	res, err := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))
	if err != nil {
		t.Fatalf("azure: DoPollForAsynchronous failed to return an appropriate error message for a failed Operations. \n expected=%q \n got=%q",
			errorResponse, err.Error())
	}

	err = autorest.Respond(res,
		WithErrorUnlessStatusCode(http.StatusAccepted, http.StatusCreated, http.StatusOK),
		autorest.ByClosing())

	reqError, ok := err.(*RequestError)
	if !ok {
		t.Fatalf("azure: returned error is not azure.RequestError: %T", err)
	}

	expected := &RequestError{
		ServiceError: &ServiceError{
			Code:    "InvalidParameter",
			Message: "tom-service-DISCOVERY-server-base-v1.core.local' is not a valid captured VHD blob name prefix.",
		},
		DetailedError: autorest.DetailedError{
			StatusCode: 400,
		},
	}
	if !reflect.DeepEqual(reqError, expected) {
		t.Fatalf("azure: wrong error. expected=%q\ngot=%q", expected, reqError)
	}

	defer res.Body.Close()
	b, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != errorResponse {
		t.Fatalf("azure: Response body is wrong. got=%q expected=%q", string(b), errorResponse)
	}

}

func TestDoPollForAsynchronous_ReturnsNoErrorForSuccessfulOperations(t *testing.T) {
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceErrorResponse(operationSucceeded)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)

	r, err := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if err != nil {
		t.Fatalf("azure: DoPollForAsynchronous returned an error for a successful OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

func TestDoPollForAsynchronous_StopsPollingIfItReceivesAnInvalidOperationResource(t *testing.T) {
	r1 := newAsynchronousResponse()
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceResponse("busy")
	r3.Body = mocks.NewBody(operationResourceIllegal)
	r4 := newOperationResourceResponse(operationSucceeded)

	client := mocks.NewSender()
	client.AppendResponse(r1)
	client.AppendAndRepeatResponse(r2, 2)
	client.AppendAndRepeatResponse(r3, 1)
	client.AppendAndRepeatResponse(r4, 1)

	r, err := autorest.SendWithSender(client, mocks.NewRequest(),
		DoPollForAsynchronous(time.Millisecond))

	if client.Attempts() > 4 {
		t.Fatalf("azure: DoPollForAsynchronous failed to stop polling after receiving an invalid OperationResource")
	}
	if err == nil {
		t.Fatalf("azure: DoPollForAsynchronous failed to return an error after receving an invalid OperationResource")
	}

	autorest.Respond(r,
		autorest.ByClosing())
}

const (
	operationResourceIllegal = `
	This is not JSON and should fail...badly.
	`
	pollingStateFormat = `
	{
		"unused" : {
			"somefield" : 42
		},
		"properties" : {
			"provisioningState": "%s"
		}
	}
	`

	errorResponse = `
	{
		"error" : {
			"code" : "InvalidParameter",
			"message" : "tom-service-DISCOVERY-server-base-v1.core.local' is not a valid captured VHD blob name prefix."
		}
	}
	`

	pollingStateEmpty = `
	{
		"unused" : {
			"somefield" : 42
		},
		"properties" : {
		}
	}
	`

	operationResourceFormat = `
	{
		"id": "/subscriptions/id/locations/westus/operationsStatus/sameguid",
		"name": "sameguid",
		"status" : "%s",
		"startTime" : "2006-01-02T15:04:05Z",
		"endTime" : "2006-01-02T16:04:05Z",
		"percentComplete" : 50.00,

		"properties" : {}
	}
	`

	operationResourceErrorFormat = `
	{
		"id": "/subscriptions/id/locations/westus/operationsStatus/sameguid",
		"name": "sameguid",
		"status" : "%s",
		"startTime" : "2006-01-02T15:04:05Z",
		"endTime" : "2006-01-02T16:04:05Z",
		"percentComplete" : 50.00,

		"properties" : {},
		"error" : {
			"code" : "BadArgument",
			"message" : "The provided database 'foo' has an invalid username."
		}
	}
	`
)

func newAsynchronousResponse() *http.Response {
	r := mocks.NewResponseWithStatus("201 Created", http.StatusCreated)
	r.Body = mocks.NewBody(fmt.Sprintf(pollingStateFormat, operationInProgress))
	mocks.SetResponseHeader(r, http.CanonicalHeaderKey(headerAsyncOperation), mocks.TestAzureAsyncURL)
	mocks.SetResponseHeader(r, http.CanonicalHeaderKey(autorest.HeaderLocation), mocks.TestLocationURL)
	mocks.SetRetryHeader(r, retryDelay)
	r.Request = mocks.NewRequestForURL(mocks.TestURL)
	return r
}

func newAsynchronousResponseWithError() *http.Response {
	r := mocks.NewResponseWithStatus("400 Bad Request", http.StatusBadRequest)
	mocks.SetRetryHeader(r, retryDelay)
	r.Request = mocks.NewRequestForURL(mocks.TestURL)
	r.Body = mocks.NewBody(errorResponse)
	return r
}

func newOperationResourceResponse(status string) *http.Response {
	r := newAsynchronousResponse()
	r.Body = mocks.NewBody(fmt.Sprintf(operationResourceFormat, status))
	return r
}

func newOperationResourceErrorResponse(status string) *http.Response {
	r := newAsynchronousResponse()
	r.Body = mocks.NewBody(fmt.Sprintf(operationResourceErrorFormat, status))
	return r
}

func newProvisioningStatusResponse(status string) *http.Response {
	r := newAsynchronousResponse()
	r.Body = mocks.NewBody(fmt.Sprintf(pollingStateFormat, status))
	return r
}

func makeLongRunningOperationErrorString(code string, message string) string {
	return fmt.Sprintf("Long running operation terminated with status 'Failed': Code=%q Message=%q", code, message)
}
