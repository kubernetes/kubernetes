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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/mocks"
)

func TestCreateFromInvalidRequestVerb(t *testing.T) {
	resp := mocks.NewResponseWithBodyAndStatus(nil, http.StatusOK, "some status")
	resp.Request = mocks.NewRequestWithParams(http.MethodGet, mocks.TestURL, nil)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

// DELETE
func TestCreateDeleteTracker201Success(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodDelete, nil), http.StatusCreated, nil)
	mocks.SetLocationHeader(resp, mocks.TestLocationURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingLocation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestLocationURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreateDeleteTracker201FailNoLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodDelete, nil), http.StatusCreated, nil)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

func TestCreateDeleteTracker201FailBadLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodDelete, nil), http.StatusCreated, nil)
	mocks.SetLocationHeader(resp, mocks.TestBadURL)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

func TestCreateDeleteTracker202SuccessAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodDelete, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != "" {
		t.Fatal("expected empty GET URL")
	}
}

func TestCreateDeleteTracker202SuccessLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodDelete, nil), http.StatusAccepted, nil)
	mocks.SetLocationHeader(resp, mocks.TestLocationURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingLocation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestLocationURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreateDeleteTracker202SuccessBoth(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodDelete, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	mocks.SetLocationHeader(resp, mocks.TestLocationURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestLocationURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreateDeleteTracker202SuccessBadLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodDelete, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	mocks.SetLocationHeader(resp, mocks.TestBadURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != "" {
		t.Fatal("expected empty GET URL")
	}
}

func TestCreateDeleteTracker202FailBadAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodDelete, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestBadURL)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

func TestCreateDeleteTracker202FailBadLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodDelete, nil), http.StatusAccepted, nil)
	mocks.SetLocationHeader(resp, mocks.TestBadURL)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

// PATCH

func TestCreatePatchTracker201Success(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPatch, nil), http.StatusCreated, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePatchTracker201SuccessNoHeaders(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPatch, nil), http.StatusCreated, nil)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingRequestURI {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePatchTracker201FailBadAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPatch, nil), http.StatusCreated, nil)
	setAsyncOpHeader(resp, mocks.TestBadURL)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

func TestCreatePatchTracker202SuccessAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPatch, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePatchTracker202SuccessLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPatch, nil), http.StatusAccepted, nil)
	mocks.SetLocationHeader(resp, mocks.TestLocationURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingLocation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePatchTracker202SuccessBoth(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPatch, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	mocks.SetLocationHeader(resp, mocks.TestLocationURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePatchTracker202FailBadAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPatch, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestBadURL)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

func TestCreatePatchTracker202FailBadLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPatch, nil), http.StatusAccepted, nil)
	mocks.SetLocationHeader(resp, mocks.TestBadURL)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

// POST

func TestCreatePostTracker201Success(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusCreated, nil)
	mocks.SetLocationHeader(resp, mocks.TestLocationURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingLocation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestLocationURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePostTracker201FailNoHeader(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusCreated, nil)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil err")
	}
}

func TestCreatePostTracker201FailBadHeader(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusCreated, nil)
	mocks.SetLocationHeader(resp, mocks.TestBadURL)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil err")
	}
}

func TestCreatePostTracker202SuccessAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != "" {
		t.Fatal("expected empty final GET URL")
	}
}

func TestCreatePostTracker202SuccessLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusAccepted, nil)
	mocks.SetLocationHeader(resp, mocks.TestLocationURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingLocation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestLocationURL {
		t.Fatalf("wrong final GET URI: %s", pt.finalGetURL())
	}
}

func TestCreatePostTracker202SuccessBoth(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	mocks.SetLocationHeader(resp, mocks.TestLocationURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestLocationURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePostTracker202SuccessBadLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	mocks.SetLocationHeader(resp, mocks.TestBadURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != "" {
		t.Fatal("expected empty final GET URL")
	}
}

func TestCreatePostTracker202FailBadAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestBadURL)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

func TestCreatePostTracker202FailBadLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusAccepted, nil)
	_, err := createPollingTracker(resp)
	mocks.SetLocationHeader(resp, mocks.TestBadURL)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

// PUT

func TestCreatePutTracker201SuccessAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusCreated, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePutTracker201SuccessNoHeaders(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusCreated, nil)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingRequestURI {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePutTracker201FailBadAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusCreated, nil)
	setAsyncOpHeader(resp, mocks.TestBadURL)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

func TestCreatePutTracker202SuccessAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePutTracker202SuccessLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusAccepted, nil)
	mocks.SetLocationHeader(resp, mocks.TestLocationURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingLocation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != resp.Request.URL.String() {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePutTracker202SuccessBoth(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	mocks.SetLocationHeader(resp, mocks.TestLocationURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != resp.Request.URL.String() {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePutTracker202SuccessBadLocation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestAzureAsyncURL)
	mocks.SetLocationHeader(resp, mocks.TestBadURL)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pt.pollingMethod() != PollingAsyncOperation {
		t.Fatalf("wrong polling method: %s", pt.pollingMethod())
	}
	if pt.finalGetURL() != mocks.TestURL {
		t.Fatalf("wrong final GET URL: %s", pt.finalGetURL())
	}
}

func TestCreatePutTracker202FailBadAsyncOp(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusAccepted, nil)
	setAsyncOpHeader(resp, mocks.TestBadURL)
	_, err := createPollingTracker(resp)
	if err == nil {
		t.Fatal("unexpected nil error")
	}
}

func TestPollPutTrackerSuccessNoHeaders(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusAccepted, nil)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	sender := mocks.NewSender()
	sender.AppendResponse(newProvisioningStatusResponse("InProgress"))
	err = pt.pollForStatus(context.Background(), sender)
	if err != nil {
		t.Fatalf("failed to poll for status: %v", err)
	}
	err = pt.checkForErrors()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
}

func TestPollPutTrackerFailNoHeadersEmptyBody(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusAccepted, nil)
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	sender := mocks.NewSender()
	sender.AppendResponse(mocks.NewResponseWithBodyAndStatus(&mocks.Body{}, http.StatusOK, "status ok"))
	err = pt.pollForStatus(context.Background(), sender)
	if err != nil {
		t.Fatalf("failed to poll for status: %v", err)
	}
	err = pt.checkForErrors()
	if err == nil {
		t.Fatalf("unexpected nil error")
	}
}

// errors

func TestAsyncPollingReturnsWrappedError(t *testing.T) {
	resp := newSimpleAsyncResp()
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	sender := mocks.NewSender()
	sender.AppendResponse(newOperationResourceErrorResponse("Failed"))
	err = pt.pollForStatus(context.Background(), sender)
	if err == nil {
		t.Fatal("unexpected nil polling error")
	}
	err = pt.pollingError()
	if err == nil {
		t.Fatal("unexpected nil polling error")
	}
	if se, ok := err.(*ServiceError); !ok {
		t.Fatal("incorrect error type")
	} else if se.Code == "" {
		t.Fatal("empty service error code")
	} else if se.Message == "" {
		t.Fatal("empty service error message")
	}
}

func TestLocationPollingReturnsWrappedError(t *testing.T) {
	resp := newSimpleLocationResp()
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	sender := mocks.NewSender()
	sender.AppendResponse(newProvisioningStatusErrorResponse("Failed"))
	err = pt.pollForStatus(context.Background(), sender)
	if err == nil {
		t.Fatal("unexpected nil polling error")
	}
	err = pt.pollingError()
	if err == nil {
		t.Fatal("unexpected nil polling error")
	}
	if se, ok := err.(*ServiceError); !ok {
		t.Fatal("incorrect error type")
	} else if se.Code == "" {
		t.Fatal("empty service error code")
	} else if se.Message == "" {
		t.Fatal("empty service error message")
	}
}

func TestLocationPollingReturnsUnwrappedError(t *testing.T) {
	resp := newSimpleLocationResp()
	pt, err := createPollingTracker(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	sender := mocks.NewSender()
	sender.AppendResponse(newProvisioningStatusUnwrappedErrorResponse("Failed"))
	err = pt.pollForStatus(context.Background(), sender)
	if err == nil {
		t.Fatal("unexpected nil polling error")
	}
	err = pt.pollingError()
	if err == nil {
		t.Fatal("unexpected nil polling error")
	}
	if se, ok := err.(*ServiceError); !ok {
		t.Fatal("incorrect error type")
	} else if se.Code == "" {
		t.Fatal("empty service error code")
	} else if se.Message == "" {
		t.Fatal("empty service error message")
	}
}

func TestFuture_PollsUntilProvisioningStatusSucceeds(t *testing.T) {
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceResponse(operationSucceeded)

	sender := mocks.NewSender()
	ctx := context.Background()
	sender.AppendAndRepeatResponse(r2, 2)
	sender.AppendResponse(r3)

	future, err := NewFutureFromResponse(newSimpleAsyncResp())
	if err != nil {
		t.Fatalf("failed to create future: %v", err)
	}

	for done, err := future.DoneWithContext(ctx, sender); !done; done, err = future.DoneWithContext(ctx, sender) {
		if future.PollingMethod() != PollingAsyncOperation {
			t.Fatalf("wrong future polling method: %s", future.PollingMethod())
		}
		if err != nil {
			t.Fatalf("polling Done failed: %v", err)
		}
		delay, ok := future.GetPollingDelay()
		if !ok {
			t.Fatalf("expected Retry-After value")
		}
		time.Sleep(delay)
	}

	if sender.Attempts() < sender.NumResponses() {
		t.Fatalf("stopped polling before receiving a terminated OperationResource")
	}

	autorest.Respond(future.Response(),
		autorest.ByClosing())
}

func TestFuture_MarshallingSuccess(t *testing.T) {
	future, err := NewFutureFromResponse(newSimpleAsyncResp())
	if err != nil {
		t.Fatalf("failed to create future: %v", err)
	}

	data, err := json.Marshal(future)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var future2 Future
	err = json.Unmarshal(data, &future2)
	if err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if reflect.DeepEqual(future.pt, future2.pt) {
		t.Fatalf("marshalling unexpected match")
	}

	// these fields don't get marshalled so nil them before deep comparison
	future.pt.(*pollingTrackerPut).resp = nil
	future.pt.(*pollingTrackerPut).rawBody = nil
	if !reflect.DeepEqual(future.pt, future2.pt) {
		t.Fatalf("marshalling futures don't match")
	}
}

func TestFuture_MarshallingWithError(t *testing.T) {
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceErrorResponse(operationFailed)

	sender := mocks.NewSender()
	sender.AppendAndRepeatResponse(r2, 2)
	sender.AppendResponse(r3)
	client := autorest.Client{
		PollingDelay:    1 * time.Second,
		PollingDuration: autorest.DefaultPollingDuration,
		RetryAttempts:   autorest.DefaultRetryAttempts,
		RetryDuration:   1 * time.Second,
		Sender:          sender,
	}

	future, err := NewFutureFromResponse(newSimpleAsyncResp())
	if err != nil {
		t.Fatalf("failed to create future: %v", err)
	}

	err = future.WaitForCompletionRef(context.Background(), client)
	if err == nil {
		t.Fatal("expected non-nil error")
	}

	data, err := json.Marshal(future)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var future2 Future
	err = json.Unmarshal(data, &future2)
	if err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if reflect.DeepEqual(future.pt, future2.pt) {
		t.Fatalf("marshalling unexpected match")
	}

	// these fields don't get marshalled so nil them before deep comparison
	future.pt.(*pollingTrackerPut).resp = nil
	future.pt.(*pollingTrackerPut).rawBody = nil
	if !reflect.DeepEqual(future.pt, future2.pt) {
		t.Fatalf("marshalling futures don't match")
	}
}

func TestFuture_CreateFromFailedOperation(t *testing.T) {
	_, err := NewFutureFromResponse(newAsyncResponseWithError(http.MethodPut))
	if err == nil {
		t.Fatal("expected non-nil error")
	}
}

func TestFuture_WaitForCompletionRef(t *testing.T) {
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceResponse(operationSucceeded)

	sender := mocks.NewSender()
	sender.AppendAndRepeatResponse(r2, 2)
	sender.AppendResponse(r3)
	client := autorest.Client{
		PollingDelay:    1 * time.Second,
		PollingDuration: autorest.DefaultPollingDuration,
		RetryAttempts:   autorest.DefaultRetryAttempts,
		RetryDuration:   1 * time.Second,
		Sender:          sender,
	}

	future, err := NewFutureFromResponse(newSimpleAsyncResp())
	if err != nil {
		t.Fatalf("failed to create future: %v", err)
	}

	err = future.WaitForCompletionRef(context.Background(), client)
	if err != nil {
		t.Fatalf("WaitForCompletion returned non-nil error")
	}

	if sender.Attempts() < sender.NumResponses() {
		t.Fatalf("stopped polling before receiving a terminated OperationResource")
	}

	autorest.Respond(future.Response(),
		autorest.ByClosing())
}

func TestFuture_WaitForCompletionRefWithRetryAfter(t *testing.T) {
	r2 := newOperationResourceResponse("busy")
	r3 := newOperationResourceResponse(operationSucceeded)

	sender := mocks.NewSender()
	sender.AppendAndRepeatResponse(r2, 2)
	sender.AppendResponse(r3)
	client := autorest.Client{
		PollingDelay:    1 * time.Second,
		PollingDuration: autorest.DefaultPollingDuration,
		RetryAttempts:   autorest.DefaultRetryAttempts,
		RetryDuration:   1 * time.Second,
		Sender:          sender,
	}

	future, err := NewFutureFromResponse(newSimpleAsyncRespWithRetryAfter())
	if err != nil {
		t.Fatalf("failed to create future: %v", err)
	}

	err = future.WaitForCompletionRef(context.Background(), client)
	if err != nil {
		t.Fatalf("WaitForCompletion returned non-nil error")
	}

	if sender.Attempts() < sender.NumResponses() {
		t.Fatalf("stopped polling before receiving a terminated OperationResource")
	}

	autorest.Respond(future.Response(),
		autorest.ByClosing())
}

func TestFuture_WaitForCompletionTimedOut(t *testing.T) {
	r2 := newProvisioningStatusResponse("busy")

	sender := mocks.NewSender()
	sender.AppendAndRepeatResponseWithDelay(r2, 1*time.Second, 5)

	future, err := NewFutureFromResponse(newSimpleAsyncResp())
	if err != nil {
		t.Fatalf("failed to create future: %v", err)
	}

	client := autorest.Client{
		PollingDelay:    autorest.DefaultPollingDelay,
		PollingDuration: 2 * time.Second,
		RetryAttempts:   autorest.DefaultRetryAttempts,
		RetryDuration:   1 * time.Second,
		Sender:          sender,
	}

	err = future.WaitForCompletionRef(context.Background(), client)
	if err == nil {
		t.Fatalf("WaitForCompletion returned nil error, should have timed out")
	}
}

func TestFuture_WaitForCompletionRetriesExceeded(t *testing.T) {
	r1 := newProvisioningStatusResponse("InProgress")

	sender := mocks.NewSender()
	sender.AppendResponse(r1)
	sender.AppendAndRepeatError(errors.New("transient network failure"), autorest.DefaultRetryAttempts+1)

	future, err := NewFutureFromResponse(newSimpleAsyncResp())
	if err != nil {
		t.Fatalf("failed to create future: %v", err)
	}

	client := autorest.Client{
		PollingDelay:    autorest.DefaultPollingDelay,
		PollingDuration: autorest.DefaultPollingDuration,
		RetryAttempts:   autorest.DefaultRetryAttempts,
		RetryDuration:   100 * time.Millisecond,
		Sender:          sender,
	}

	err = future.WaitForCompletionRef(context.Background(), client)
	if err == nil {
		t.Fatalf("WaitForCompletion returned nil error, should have errored out")
	}
}

func TestFuture_WaitForCompletionCancelled(t *testing.T) {
	r1 := newProvisioningStatusResponse("InProgress")

	sender := mocks.NewSender()
	sender.AppendAndRepeatResponseWithDelay(r1, 1*time.Second, 5)

	future, err := NewFutureFromResponse(newSimpleAsyncResp())
	if err != nil {
		t.Fatalf("failed to create future: %v", err)
	}

	client := autorest.Client{
		PollingDelay:    autorest.DefaultPollingDelay,
		PollingDuration: autorest.DefaultPollingDuration,
		RetryAttempts:   autorest.DefaultRetryAttempts,
		RetryDuration:   autorest.DefaultRetryDuration,
		Sender:          sender,
	}

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(2 * time.Second)
		cancel()
	}()

	err = future.WaitForCompletionRef(ctx, client)
	if err == nil {
		t.Fatalf("WaitForCompletion returned nil error, should have been cancelled")
	}
}

func TestFuture_GetResultFromNonAsyncOperation(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusOK, mocks.NewBody(someResource))
	future, err := NewFutureFromResponse(resp)
	if err != nil {
		t.Fatalf("failed to create tracker: %v", err)
	}
	if pm := future.PollingMethod(); pm != PollingUnknown {
		t.Fatalf("wrong polling method: %s", pm)
	}
	done, err := future.DoneWithContext(context.Background(), nil)
	if err != nil {
		t.Fatalf("failed to check status: %v", err)
	}
	if !done {
		t.Fatal("operation should be done")
	}
	res, err := future.GetResult(nil)
	if err != nil {
		t.Fatalf("failed to get result: %v", err)
	}
	if res != resp {
		t.Fatal("result and response don't match")
	}
}

func TestFuture_GetResultNonTerminal(t *testing.T) {
	resp := newAsyncResp(newAsyncReq(http.MethodDelete, nil), http.StatusAccepted, mocks.NewBody(fmt.Sprintf(operationResourceFormat, operationInProgress)))
	mocks.SetResponseHeader(resp, headerAsyncOperation, mocks.TestAzureAsyncURL)
	future, err := NewFutureFromResponse(resp)
	if err != nil {
		t.Fatalf("failed to create future: %v", err)
	}
	res, err := future.GetResult(nil)
	if err == nil {
		t.Fatal("expected non-nil error")
	}
	if res != nil {
		t.Fatal("expected nil result")
	}
}

const (
	operationResourceIllegal = `
	This is not JSON and should fail...badly.
	`

	// returned from LROs that use Location header
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

	// returned from LROs that use Location header
	errorResponse = `
	{
		"error" : {
			"code" : "InvalidParameter",
			"message" : "tom-service-DISCOVERY-server-base-v1.core.local' is not a valid captured VHD blob name prefix."
		}
	}
	`

	// returned from LROs that use Location header
	unwrappedErrorResponse = `
	{
		"code" : "InvalidParameter",
		"message" : "tom-service-DISCOVERY-server-base-v1.core.local' is not a valid captured VHD blob name prefix."
	}
	`

	// returned from LROs that use Location header
	pollingStateEmpty = `
	{
		"unused" : {
			"somefield" : 42
		},
		"properties" : {
		}
	}
	`

	// returned from LROs that use Azure-AsyncOperation header
	operationResourceFormat = `
	{
		"id": "/subscriptions/id/locations/westus/operationsStatus/sameguid",
		"name": "sameguid",
		"status" : "%s",
		"startTime" : "2006-01-02T15:04:05Z",
		"endTime" : "2006-01-02T16:04:05Z",
		"percentComplete" : 50.00,
		"properties" : {
			"foo": "bar"
		}
	}
	`

	// returned from LROs that use Azure-AsyncOperation header
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

	// returned from an operation marked as LRO but really isn't
	someResource = `
	{
		"id": "/subscriptions/guid/resourceGroups/rg/providers/something/else/thing",
		"name": "thing",
		"type": "Imaginary.type",
		"location": "Central US",
		"properties": {}
	}
	`
)

// creates an async request with the specified body.
func newAsyncReq(reqMethod string, body *mocks.Body) *http.Request {
	return mocks.NewRequestWithParams(reqMethod, mocks.TestURL, body)
}

// creates an async response with the specified body.
// the req param is the originating LRO request.
func newAsyncResp(req *http.Request, statusCode int, body *mocks.Body) *http.Response {
	status := "Unknown"
	switch statusCode {
	case http.StatusOK, http.StatusNoContent:
		status = "Completed"
	case http.StatusCreated:
		status = "Creating"
	case http.StatusAccepted:
		status = "In progress"
	case http.StatusBadRequest:
		status = "Bad request"
	}
	r := mocks.NewResponseWithBodyAndStatus(body, statusCode, status)
	r.Request = req
	return r
}

// creates a simple LRO response, PUT/201 with Azure-AsyncOperation header
func newSimpleAsyncResp() *http.Response {
	r := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusCreated, mocks.NewBody(fmt.Sprintf(operationResourceFormat, operationInProgress)))
	mocks.SetResponseHeader(r, headerAsyncOperation, mocks.TestAzureAsyncURL)
	return r
}

func newSimpleAsyncRespWithRetryAfter() *http.Response {
	r := newAsyncResp(newAsyncReq(http.MethodPut, nil), http.StatusCreated, mocks.NewBody(fmt.Sprintf(operationResourceFormat, operationInProgress)))
	mocks.SetResponseHeader(r, headerAsyncOperation, mocks.TestAzureAsyncURL)
	mocks.SetResponseHeader(r, autorest.HeaderRetryAfter, "1")
	return r
}

// creates a simple LRO response, POST/201 with Location header
func newSimpleLocationResp() *http.Response {
	r := newAsyncResp(newAsyncReq(http.MethodPost, nil), http.StatusCreated, mocks.NewBody(fmt.Sprintf(pollingStateFormat, operationInProgress)))
	mocks.SetResponseHeader(r, autorest.HeaderLocation, mocks.TestLocationURL)
	return r
}

// creates an async response that contains an error (HTTP 400 + error response body)
func newAsyncResponseWithError(reqMethod string) *http.Response {
	return newAsyncResp(newAsyncReq(reqMethod, nil), http.StatusBadRequest, mocks.NewBody(errorResponse))
}

// creates a LRO polling response using the operation resource format (Azure-AsyncOperation LROs)
func newOperationResourceResponse(status string) *http.Response {
	r := mocks.NewResponseWithBodyAndStatus(mocks.NewBody(fmt.Sprintf(operationResourceFormat, status)), http.StatusOK, status)
	mocks.SetRetryHeader(r, retryDelay)
	return r
}

// creates a LRO polling error response using the operation resource format (Azure-AsyncOperation LROs)
func newOperationResourceErrorResponse(status string) *http.Response {
	return mocks.NewResponseWithBodyAndStatus(mocks.NewBody(fmt.Sprintf(operationResourceErrorFormat, status)), http.StatusBadRequest, status)
}

// creates a LRO polling response using the provisioning state format (Location LROs)
func newProvisioningStatusResponse(status string) *http.Response {
	r := mocks.NewResponseWithBodyAndStatus(mocks.NewBody(fmt.Sprintf(pollingStateFormat, status)), http.StatusOK, status)
	mocks.SetRetryHeader(r, retryDelay)
	return r
}

// creates a LRO polling error response using the provisioning state format (Location LROs)
func newProvisioningStatusErrorResponse(status string) *http.Response {
	return mocks.NewResponseWithBodyAndStatus(mocks.NewBody(errorResponse), http.StatusBadRequest, status)
}

// creates a LRO polling unwrapped error response using the provisioning state format (Location LROs)
func newProvisioningStatusUnwrappedErrorResponse(status string) *http.Response {
	return mocks.NewResponseWithBodyAndStatus(mocks.NewBody(unwrappedErrorResponse), http.StatusBadRequest, status)
}

// adds the Azure-AsyncOperation header with the specified location to the response
func setAsyncOpHeader(resp *http.Response, location string) {
	mocks.SetResponseHeader(resp, http.CanonicalHeaderKey(headerAsyncOperation), location)
}
