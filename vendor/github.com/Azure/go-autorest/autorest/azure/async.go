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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/Azure/go-autorest/autorest"
)

const (
	headerAsyncOperation = "Azure-AsyncOperation"
)

const (
	operationInProgress string = "InProgress"
	operationCanceled   string = "Canceled"
	operationFailed     string = "Failed"
	operationSucceeded  string = "Succeeded"
)

var pollingCodes = [...]int{http.StatusNoContent, http.StatusAccepted, http.StatusCreated, http.StatusOK}

// Future provides a mechanism to access the status and results of an asynchronous request.
// Since futures are stateful they should be passed by value to avoid race conditions.
type Future struct {
	req *http.Request // legacy
	pt  pollingTracker
}

// NewFuture returns a new Future object initialized with the specified request.
// Deprecated: Please use NewFutureFromResponse instead.
func NewFuture(req *http.Request) Future {
	return Future{req: req}
}

// NewFutureFromResponse returns a new Future object initialized
// with the initial response from an asynchronous operation.
func NewFutureFromResponse(resp *http.Response) (Future, error) {
	pt, err := createPollingTracker(resp)
	if err != nil {
		return Future{}, err
	}
	return Future{pt: pt}, nil
}

// Response returns the last HTTP response.
func (f Future) Response() *http.Response {
	if f.pt == nil {
		return nil
	}
	return f.pt.latestResponse()
}

// Status returns the last status message of the operation.
func (f Future) Status() string {
	if f.pt == nil {
		return ""
	}
	return f.pt.pollingStatus()
}

// PollingMethod returns the method used to monitor the status of the asynchronous operation.
func (f Future) PollingMethod() PollingMethodType {
	if f.pt == nil {
		return PollingUnknown
	}
	return f.pt.pollingMethod()
}

// Done queries the service to see if the operation has completed.
func (f *Future) Done(sender autorest.Sender) (bool, error) {
	// support for legacy Future implementation
	if f.req != nil {
		resp, err := sender.Do(f.req)
		if err != nil {
			return false, err
		}
		pt, err := createPollingTracker(resp)
		if err != nil {
			return false, err
		}
		f.pt = pt
		f.req = nil
	}
	// end legacy
	if f.pt == nil {
		return false, autorest.NewError("Future", "Done", "future is not initialized")
	}
	if f.pt.hasTerminated() {
		return true, f.pt.pollingError()
	}
	if err := f.pt.pollForStatus(sender); err != nil {
		return false, err
	}
	if err := f.pt.checkForErrors(); err != nil {
		return f.pt.hasTerminated(), err
	}
	if err := f.pt.updatePollingState(f.pt.provisioningStateApplicable()); err != nil {
		return false, err
	}
	if err := f.pt.initPollingMethod(); err != nil {
		return false, err
	}
	if err := f.pt.updatePollingMethod(); err != nil {
		return false, err
	}
	return f.pt.hasTerminated(), f.pt.pollingError()
}

// GetPollingDelay returns a duration the application should wait before checking
// the status of the asynchronous request and true; this value is returned from
// the service via the Retry-After response header.  If the header wasn't returned
// then the function returns the zero-value time.Duration and false.
func (f Future) GetPollingDelay() (time.Duration, bool) {
	if f.pt == nil {
		return 0, false
	}
	resp := f.pt.latestResponse()
	if resp == nil {
		return 0, false
	}

	retry := resp.Header.Get(autorest.HeaderRetryAfter)
	if retry == "" {
		return 0, false
	}

	d, err := time.ParseDuration(retry + "s")
	if err != nil {
		panic(err)
	}

	return d, true
}

// WaitForCompletion will return when one of the following conditions is met: the long
// running operation has completed, the provided context is cancelled, or the client's
// polling duration has been exceeded.  It will retry failed polling attempts based on
// the retry value defined in the client up to the maximum retry attempts.
// Deprecated: Please use WaitForCompletionRef() instead.
func (f Future) WaitForCompletion(ctx context.Context, client autorest.Client) error {
	return f.WaitForCompletionRef(ctx, client)
}

// WaitForCompletionRef will return when one of the following conditions is met: the long
// running operation has completed, the provided context is cancelled, or the client's
// polling duration has been exceeded.  It will retry failed polling attempts based on
// the retry value defined in the client up to the maximum retry attempts.
func (f *Future) WaitForCompletionRef(ctx context.Context, client autorest.Client) error {
	if d := client.PollingDuration; d != 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, d)
		defer cancel()
	}

	done, err := f.Done(client)
	for attempts := 0; !done; done, err = f.Done(client) {
		if attempts >= client.RetryAttempts {
			return autorest.NewErrorWithError(err, "Future", "WaitForCompletion", f.pt.latestResponse(), "the number of retries has been exceeded")
		}
		// we want delayAttempt to be zero in the non-error case so
		// that DelayForBackoff doesn't perform exponential back-off
		var delayAttempt int
		var delay time.Duration
		if err == nil {
			// check for Retry-After delay, if not present use the client's polling delay
			var ok bool
			delay, ok = f.GetPollingDelay()
			if !ok {
				delay = client.PollingDelay
			}
		} else {
			// there was an error polling for status so perform exponential
			// back-off based on the number of attempts using the client's retry
			// duration.  update attempts after delayAttempt to avoid off-by-one.
			delayAttempt = attempts
			delay = client.RetryDuration
			attempts++
		}
		// wait until the delay elapses or the context is cancelled
		delayElapsed := autorest.DelayForBackoff(delay, delayAttempt, ctx.Done())
		if !delayElapsed {
			return autorest.NewErrorWithError(ctx.Err(), "Future", "WaitForCompletion", f.pt.latestResponse(), "context has been cancelled")
		}
	}
	return err
}

// MarshalJSON implements the json.Marshaler interface.
func (f Future) MarshalJSON() ([]byte, error) {
	return json.Marshal(f.pt)
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (f *Future) UnmarshalJSON(data []byte) error {
	// unmarshal into JSON object to determine the tracker type
	obj := map[string]interface{}{}
	err := json.Unmarshal(data, &obj)
	if err != nil {
		return err
	}
	if obj["method"] == nil {
		return autorest.NewError("Future", "UnmarshalJSON", "missing 'method' property")
	}
	method := obj["method"].(string)
	switch strings.ToUpper(method) {
	case http.MethodDelete:
		f.pt = &pollingTrackerDelete{}
	case http.MethodPatch:
		f.pt = &pollingTrackerPatch{}
	case http.MethodPost:
		f.pt = &pollingTrackerPost{}
	case http.MethodPut:
		f.pt = &pollingTrackerPut{}
	default:
		return autorest.NewError("Future", "UnmarshalJSON", "unsupoorted method '%s'", method)
	}
	// now unmarshal into the tracker
	return json.Unmarshal(data, &f.pt)
}

// PollingURL returns the URL used for retrieving the status of the long-running operation.
func (f Future) PollingURL() string {
	if f.pt == nil {
		return ""
	}
	return f.pt.pollingURL()
}

// GetResult should be called once polling has completed successfully.
// It makes the final GET call to retrieve the resultant payload.
func (f Future) GetResult(sender autorest.Sender) (*http.Response, error) {
	if f.pt.finalGetURL() == "" {
		// we can end up in this situation if the async operation returns a 200
		// with no polling URLs.  in that case return the response which should
		// contain the JSON payload (only do this for successful terminal cases).
		if lr := f.pt.latestResponse(); lr != nil && f.pt.hasSucceeded() {
			return lr, nil
		}
		return nil, autorest.NewError("Future", "GetResult", "missing URL for retrieving result")
	}
	req, err := http.NewRequest(http.MethodGet, f.pt.finalGetURL(), nil)
	if err != nil {
		return nil, err
	}
	return sender.Do(req)
}

type pollingTracker interface {
	// these methods can differ per tracker

	// checks the response headers and status code to determine the polling mechanism
	updatePollingMethod() error

	// checks the response for tracker-specific error conditions
	checkForErrors() error

	// returns true if provisioning state should be checked
	provisioningStateApplicable() bool

	// methods common to all trackers

	// initializes a tracker's polling URL and method, called for each iteration.
	// these values can be overridden by each polling tracker as required.
	initPollingMethod() error

	// initializes the tracker's internal state, call this when the tracker is created
	initializeState() error

	// makes an HTTP request to check the status of the LRO
	pollForStatus(sender autorest.Sender) error

	// updates internal tracker state, call this after each call to pollForStatus
	updatePollingState(provStateApl bool) error

	// returns the error response from the service, can be nil
	pollingError() error

	// returns the polling method being used
	pollingMethod() PollingMethodType

	// returns the state of the LRO as returned from the service
	pollingStatus() string

	// returns the URL used for polling status
	pollingURL() string

	// returns the URL used for the final GET to retrieve the resource
	finalGetURL() string

	// returns true if the LRO is in a terminal state
	hasTerminated() bool

	// returns true if the LRO is in a failed terminal state
	hasFailed() bool

	// returns true if the LRO is in a successful terminal state
	hasSucceeded() bool

	// returns the cached HTTP response after a call to pollForStatus(), can be nil
	latestResponse() *http.Response
}

type pollingTrackerBase struct {
	// resp is the last response, either from the submission of the LRO or from polling
	resp *http.Response

	// method is the HTTP verb, this is needed for deserialization
	Method string `json:"method"`

	// rawBody is the raw JSON response body
	rawBody map[string]interface{}

	// denotes if polling is using async-operation or location header
	Pm PollingMethodType `json:"pollingMethod"`

	// the URL to poll for status
	URI string `json:"pollingURI"`

	// the state of the LRO as returned from the service
	State string `json:"lroState"`

	// the URL to GET for the final result
	FinalGetURI string `json:"resultURI"`

	// used to hold an error object returned from the service
	Err *ServiceError `json:"error,omitempty"`
}

func (pt *pollingTrackerBase) initializeState() error {
	// determine the initial polling state based on response body and/or HTTP status
	// code.  this is applicable to the initial LRO response, not polling responses!
	pt.Method = pt.resp.Request.Method
	if err := pt.updateRawBody(); err != nil {
		return err
	}
	switch pt.resp.StatusCode {
	case http.StatusOK:
		if ps := pt.getProvisioningState(); ps != nil {
			pt.State = *ps
			if pt.hasFailed() {
				pt.updateErrorFromResponse()
				return pt.pollingError()
			}
		} else {
			pt.State = operationSucceeded
		}
	case http.StatusCreated:
		if ps := pt.getProvisioningState(); ps != nil {
			pt.State = *ps
		} else {
			pt.State = operationInProgress
		}
	case http.StatusAccepted:
		pt.State = operationInProgress
	case http.StatusNoContent:
		pt.State = operationSucceeded
	default:
		pt.State = operationFailed
		pt.updateErrorFromResponse()
		return pt.pollingError()
	}
	return pt.initPollingMethod()
}

func (pt pollingTrackerBase) getProvisioningState() *string {
	if pt.rawBody != nil && pt.rawBody["properties"] != nil {
		p := pt.rawBody["properties"].(map[string]interface{})
		if ps := p["provisioningState"]; ps != nil {
			s := ps.(string)
			return &s
		}
	}
	return nil
}

func (pt *pollingTrackerBase) updateRawBody() error {
	pt.rawBody = map[string]interface{}{}
	if pt.resp.ContentLength != 0 {
		defer pt.resp.Body.Close()
		b, err := ioutil.ReadAll(pt.resp.Body)
		if err != nil {
			return autorest.NewErrorWithError(err, "pollingTrackerBase", "updateRawBody", nil, "failed to read response body")
		}
		// put the body back so it's available to other callers
		pt.resp.Body = ioutil.NopCloser(bytes.NewReader(b))
		if err = json.Unmarshal(b, &pt.rawBody); err != nil {
			return autorest.NewErrorWithError(err, "pollingTrackerBase", "updateRawBody", nil, "failed to unmarshal response body")
		}
	}
	return nil
}

func (pt *pollingTrackerBase) pollForStatus(sender autorest.Sender) error {
	req, err := http.NewRequest(http.MethodGet, pt.URI, nil)
	if err != nil {
		return autorest.NewErrorWithError(err, "pollingTrackerBase", "pollForStatus", nil, "failed to create HTTP request")
	}
	// attach the context from the original request if available (it will be absent for deserialized futures)
	if pt.resp != nil {
		req = req.WithContext(pt.resp.Request.Context())
	}
	pt.resp, err = sender.Do(req)
	if err != nil {
		return autorest.NewErrorWithError(err, "pollingTrackerBase", "pollForStatus", nil, "failed to send HTTP request")
	}
	if autorest.ResponseHasStatusCode(pt.resp, pollingCodes[:]...) {
		// reset the service error on success case
		pt.Err = nil
		err = pt.updateRawBody()
	} else {
		// check response body for error content
		pt.updateErrorFromResponse()
		err = pt.pollingError()
	}
	return err
}

// attempts to unmarshal a ServiceError type from the response body.
// if that fails then make a best attempt at creating something meaningful.
// NOTE: this assumes that the async operation has failed.
func (pt *pollingTrackerBase) updateErrorFromResponse() {
	var err error
	if pt.resp.ContentLength != 0 {
		type respErr struct {
			ServiceError *ServiceError `json:"error"`
		}
		re := respErr{}
		defer pt.resp.Body.Close()
		var b []byte
		if b, err = ioutil.ReadAll(pt.resp.Body); err != nil {
			goto Default
		}
		if err = json.Unmarshal(b, &re); err != nil {
			goto Default
		}
		// unmarshalling the error didn't yield anything, try unwrapped error
		if re.ServiceError == nil {
			err = json.Unmarshal(b, &re.ServiceError)
			if err != nil {
				goto Default
			}
		}
		// the unmarshaller will ensure re.ServiceError is non-nil
		// even if there was no content unmarshalled so check the code.
		if re.ServiceError.Code != "" {
			pt.Err = re.ServiceError
			return
		}
	}
Default:
	se := &ServiceError{
		Code:    pt.pollingStatus(),
		Message: "The async operation failed.",
	}
	if err != nil {
		se.InnerError = make(map[string]interface{})
		se.InnerError["unmarshalError"] = err.Error()
	}
	// stick the response body into the error object in hopes
	// it contains something useful to help diagnose the failure.
	if len(pt.rawBody) > 0 {
		se.AdditionalInfo = []map[string]interface{}{
			pt.rawBody,
		}
	}
	pt.Err = se
}

func (pt *pollingTrackerBase) updatePollingState(provStateApl bool) error {
	if pt.Pm == PollingAsyncOperation && pt.rawBody["status"] != nil {
		pt.State = pt.rawBody["status"].(string)
	} else {
		if pt.resp.StatusCode == http.StatusAccepted {
			pt.State = operationInProgress
		} else if provStateApl {
			if ps := pt.getProvisioningState(); ps != nil {
				pt.State = *ps
			} else {
				pt.State = operationSucceeded
			}
		} else {
			return autorest.NewError("pollingTrackerBase", "updatePollingState", "the response from the async operation has an invalid status code")
		}
	}
	// if the operation has failed update the error state
	if pt.hasFailed() {
		pt.updateErrorFromResponse()
	}
	return nil
}

func (pt pollingTrackerBase) pollingError() error {
	if pt.Err == nil {
		return nil
	}
	return pt.Err
}

func (pt pollingTrackerBase) pollingMethod() PollingMethodType {
	return pt.Pm
}

func (pt pollingTrackerBase) pollingStatus() string {
	return pt.State
}

func (pt pollingTrackerBase) pollingURL() string {
	return pt.URI
}

func (pt pollingTrackerBase) finalGetURL() string {
	return pt.FinalGetURI
}

func (pt pollingTrackerBase) hasTerminated() bool {
	return strings.EqualFold(pt.State, operationCanceled) || strings.EqualFold(pt.State, operationFailed) || strings.EqualFold(pt.State, operationSucceeded)
}

func (pt pollingTrackerBase) hasFailed() bool {
	return strings.EqualFold(pt.State, operationCanceled) || strings.EqualFold(pt.State, operationFailed)
}

func (pt pollingTrackerBase) hasSucceeded() bool {
	return strings.EqualFold(pt.State, operationSucceeded)
}

func (pt pollingTrackerBase) latestResponse() *http.Response {
	return pt.resp
}

// error checking common to all trackers
func (pt pollingTrackerBase) baseCheckForErrors() error {
	// for Azure-AsyncOperations the response body cannot be nil or empty
	if pt.Pm == PollingAsyncOperation {
		if pt.resp.Body == nil || pt.resp.ContentLength == 0 {
			return autorest.NewError("pollingTrackerBase", "baseCheckForErrors", "for Azure-AsyncOperation response body cannot be nil")
		}
		if pt.rawBody["status"] == nil {
			return autorest.NewError("pollingTrackerBase", "baseCheckForErrors", "missing status property in Azure-AsyncOperation response body")
		}
	}
	return nil
}

// default initialization of polling URL/method.  each verb tracker will update this as required.
func (pt *pollingTrackerBase) initPollingMethod() error {
	if ao, err := getURLFromAsyncOpHeader(pt.resp); err != nil {
		return err
	} else if ao != "" {
		pt.URI = ao
		pt.Pm = PollingAsyncOperation
		return nil
	}
	if lh, err := getURLFromLocationHeader(pt.resp); err != nil {
		return err
	} else if lh != "" {
		pt.URI = lh
		pt.Pm = PollingLocation
		return nil
	}
	// it's ok if we didn't find a polling header, this will be handled elsewhere
	return nil
}

// DELETE

type pollingTrackerDelete struct {
	pollingTrackerBase
}

func (pt *pollingTrackerDelete) updatePollingMethod() error {
	// for 201 the Location header is required
	if pt.resp.StatusCode == http.StatusCreated {
		if lh, err := getURLFromLocationHeader(pt.resp); err != nil {
			return err
		} else if lh == "" {
			return autorest.NewError("pollingTrackerDelete", "updateHeaders", "missing Location header in 201 response")
		} else {
			pt.URI = lh
		}
		pt.Pm = PollingLocation
		pt.FinalGetURI = pt.URI
	}
	// for 202 prefer the Azure-AsyncOperation header but fall back to Location if necessary
	if pt.resp.StatusCode == http.StatusAccepted {
		ao, err := getURLFromAsyncOpHeader(pt.resp)
		if err != nil {
			return err
		} else if ao != "" {
			pt.URI = ao
			pt.Pm = PollingAsyncOperation
		}
		// if the Location header is invalid and we already have a polling URL
		// then we don't care if the Location header URL is malformed.
		if lh, err := getURLFromLocationHeader(pt.resp); err != nil && pt.URI == "" {
			return err
		} else if lh != "" {
			if ao == "" {
				pt.URI = lh
				pt.Pm = PollingLocation
			}
			// when both headers are returned we use the value in the Location header for the final GET
			pt.FinalGetURI = lh
		}
		// make sure a polling URL was found
		if pt.URI == "" {
			return autorest.NewError("pollingTrackerPost", "updateHeaders", "didn't get any suitable polling URLs in 202 response")
		}
	}
	return nil
}

func (pt pollingTrackerDelete) checkForErrors() error {
	return pt.baseCheckForErrors()
}

func (pt pollingTrackerDelete) provisioningStateApplicable() bool {
	return pt.resp.StatusCode == http.StatusOK || pt.resp.StatusCode == http.StatusNoContent
}

// PATCH

type pollingTrackerPatch struct {
	pollingTrackerBase
}

func (pt *pollingTrackerPatch) updatePollingMethod() error {
	// by default we can use the original URL for polling and final GET
	if pt.URI == "" {
		pt.URI = pt.resp.Request.URL.String()
	}
	if pt.FinalGetURI == "" {
		pt.FinalGetURI = pt.resp.Request.URL.String()
	}
	if pt.Pm == PollingUnknown {
		pt.Pm = PollingRequestURI
	}
	// for 201 it's permissible for no headers to be returned
	if pt.resp.StatusCode == http.StatusCreated {
		if ao, err := getURLFromAsyncOpHeader(pt.resp); err != nil {
			return err
		} else if ao != "" {
			pt.URI = ao
			pt.Pm = PollingAsyncOperation
		}
	}
	// for 202 prefer the Azure-AsyncOperation header but fall back to Location if necessary
	// note the absense of the "final GET" mechanism for PATCH
	if pt.resp.StatusCode == http.StatusAccepted {
		ao, err := getURLFromAsyncOpHeader(pt.resp)
		if err != nil {
			return err
		} else if ao != "" {
			pt.URI = ao
			pt.Pm = PollingAsyncOperation
		}
		if ao == "" {
			if lh, err := getURLFromLocationHeader(pt.resp); err != nil {
				return err
			} else if lh == "" {
				return autorest.NewError("pollingTrackerPatch", "updateHeaders", "didn't get any suitable polling URLs in 202 response")
			} else {
				pt.URI = lh
				pt.Pm = PollingLocation
			}
		}
	}
	return nil
}

func (pt pollingTrackerPatch) checkForErrors() error {
	return pt.baseCheckForErrors()
}

func (pt pollingTrackerPatch) provisioningStateApplicable() bool {
	return pt.resp.StatusCode == http.StatusOK || pt.resp.StatusCode == http.StatusCreated
}

// POST

type pollingTrackerPost struct {
	pollingTrackerBase
}

func (pt *pollingTrackerPost) updatePollingMethod() error {
	// 201 requires Location header
	if pt.resp.StatusCode == http.StatusCreated {
		if lh, err := getURLFromLocationHeader(pt.resp); err != nil {
			return err
		} else if lh == "" {
			return autorest.NewError("pollingTrackerPost", "updateHeaders", "missing Location header in 201 response")
		} else {
			pt.URI = lh
			pt.FinalGetURI = lh
			pt.Pm = PollingLocation
		}
	}
	// for 202 prefer the Azure-AsyncOperation header but fall back to Location if necessary
	if pt.resp.StatusCode == http.StatusAccepted {
		ao, err := getURLFromAsyncOpHeader(pt.resp)
		if err != nil {
			return err
		} else if ao != "" {
			pt.URI = ao
			pt.Pm = PollingAsyncOperation
		}
		// if the Location header is invalid and we already have a polling URL
		// then we don't care if the Location header URL is malformed.
		if lh, err := getURLFromLocationHeader(pt.resp); err != nil && pt.URI == "" {
			return err
		} else if lh != "" {
			if ao == "" {
				pt.URI = lh
				pt.Pm = PollingLocation
			}
			// when both headers are returned we use the value in the Location header for the final GET
			pt.FinalGetURI = lh
		}
		// make sure a polling URL was found
		if pt.URI == "" {
			return autorest.NewError("pollingTrackerPost", "updateHeaders", "didn't get any suitable polling URLs in 202 response")
		}
	}
	return nil
}

func (pt pollingTrackerPost) checkForErrors() error {
	return pt.baseCheckForErrors()
}

func (pt pollingTrackerPost) provisioningStateApplicable() bool {
	return pt.resp.StatusCode == http.StatusOK || pt.resp.StatusCode == http.StatusNoContent
}

// PUT

type pollingTrackerPut struct {
	pollingTrackerBase
}

func (pt *pollingTrackerPut) updatePollingMethod() error {
	// by default we can use the original URL for polling and final GET
	if pt.URI == "" {
		pt.URI = pt.resp.Request.URL.String()
	}
	if pt.FinalGetURI == "" {
		pt.FinalGetURI = pt.resp.Request.URL.String()
	}
	if pt.Pm == PollingUnknown {
		pt.Pm = PollingRequestURI
	}
	// for 201 it's permissible for no headers to be returned
	if pt.resp.StatusCode == http.StatusCreated {
		if ao, err := getURLFromAsyncOpHeader(pt.resp); err != nil {
			return err
		} else if ao != "" {
			pt.URI = ao
			pt.Pm = PollingAsyncOperation
		}
	}
	// for 202 prefer the Azure-AsyncOperation header but fall back to Location if necessary
	if pt.resp.StatusCode == http.StatusAccepted {
		ao, err := getURLFromAsyncOpHeader(pt.resp)
		if err != nil {
			return err
		} else if ao != "" {
			pt.URI = ao
			pt.Pm = PollingAsyncOperation
		}
		// if the Location header is invalid and we already have a polling URL
		// then we don't care if the Location header URL is malformed.
		if lh, err := getURLFromLocationHeader(pt.resp); err != nil && pt.URI == "" {
			return err
		} else if lh != "" {
			if ao == "" {
				pt.URI = lh
				pt.Pm = PollingLocation
			}
			// when both headers are returned we use the value in the Location header for the final GET
			pt.FinalGetURI = lh
		}
		// make sure a polling URL was found
		if pt.URI == "" {
			return autorest.NewError("pollingTrackerPut", "updateHeaders", "didn't get any suitable polling URLs in 202 response")
		}
	}
	return nil
}

func (pt pollingTrackerPut) checkForErrors() error {
	err := pt.baseCheckForErrors()
	if err != nil {
		return err
	}
	// if there are no LRO headers then the body cannot be empty
	ao, err := getURLFromAsyncOpHeader(pt.resp)
	if err != nil {
		return err
	}
	lh, err := getURLFromLocationHeader(pt.resp)
	if err != nil {
		return err
	}
	if ao == "" && lh == "" && len(pt.rawBody) == 0 {
		return autorest.NewError("pollingTrackerPut", "checkForErrors", "the response did not contain a body")
	}
	return nil
}

func (pt pollingTrackerPut) provisioningStateApplicable() bool {
	return pt.resp.StatusCode == http.StatusOK || pt.resp.StatusCode == http.StatusCreated
}

// creates a polling tracker based on the verb of the original request
func createPollingTracker(resp *http.Response) (pollingTracker, error) {
	var pt pollingTracker
	switch strings.ToUpper(resp.Request.Method) {
	case http.MethodDelete:
		pt = &pollingTrackerDelete{pollingTrackerBase: pollingTrackerBase{resp: resp}}
	case http.MethodPatch:
		pt = &pollingTrackerPatch{pollingTrackerBase: pollingTrackerBase{resp: resp}}
	case http.MethodPost:
		pt = &pollingTrackerPost{pollingTrackerBase: pollingTrackerBase{resp: resp}}
	case http.MethodPut:
		pt = &pollingTrackerPut{pollingTrackerBase: pollingTrackerBase{resp: resp}}
	default:
		return nil, autorest.NewError("azure", "createPollingTracker", "unsupported HTTP method %s", resp.Request.Method)
	}
	if err := pt.initializeState(); err != nil {
		return pt, err
	}
	// this initializes the polling header values, we do this during creation in case the
	// initial response send us invalid values; this way the API call will return a non-nil
	// error (not doing this means the error shows up in Future.Done)
	return pt, pt.updatePollingMethod()
}

// gets the polling URL from the Azure-AsyncOperation header.
// ensures the URL is well-formed and absolute.
func getURLFromAsyncOpHeader(resp *http.Response) (string, error) {
	s := resp.Header.Get(http.CanonicalHeaderKey(headerAsyncOperation))
	if s == "" {
		return "", nil
	}
	if !isValidURL(s) {
		return "", autorest.NewError("azure", "getURLFromAsyncOpHeader", "invalid polling URL '%s'", s)
	}
	return s, nil
}

// gets the polling URL from the Location header.
// ensures the URL is well-formed and absolute.
func getURLFromLocationHeader(resp *http.Response) (string, error) {
	s := resp.Header.Get(http.CanonicalHeaderKey(autorest.HeaderLocation))
	if s == "" {
		return "", nil
	}
	if !isValidURL(s) {
		return "", autorest.NewError("azure", "getURLFromLocationHeader", "invalid polling URL '%s'", s)
	}
	return s, nil
}

// verify that the URL is valid and absolute
func isValidURL(s string) bool {
	u, err := url.Parse(s)
	return err == nil && u.IsAbs()
}

// DoPollForAsynchronous returns a SendDecorator that polls if the http.Response is for an Azure
// long-running operation. It will delay between requests for the duration specified in the
// RetryAfter header or, if the header is absent, the passed delay. Polling may be canceled via
// the context associated with the http.Request.
// Deprecated: Prefer using Futures to allow for non-blocking async operations.
func DoPollForAsynchronous(delay time.Duration) autorest.SendDecorator {
	return func(s autorest.Sender) autorest.Sender {
		return autorest.SenderFunc(func(r *http.Request) (*http.Response, error) {
			resp, err := s.Do(r)
			if err != nil {
				return resp, err
			}
			if !autorest.ResponseHasStatusCode(resp, pollingCodes[:]...) {
				return resp, nil
			}
			future, err := NewFutureFromResponse(resp)
			if err != nil {
				return resp, err
			}
			// retry until either the LRO completes or we receive an error
			var done bool
			for done, err = future.Done(s); !done && err == nil; done, err = future.Done(s) {
				// check for Retry-After delay, if not present use the specified polling delay
				if pd, ok := future.GetPollingDelay(); ok {
					delay = pd
				}
				// wait until the delay elapses or the context is cancelled
				if delayElapsed := autorest.DelayForBackoff(delay, 0, r.Context().Done()); !delayElapsed {
					return future.Response(),
						autorest.NewErrorWithError(r.Context().Err(), "azure", "DoPollForAsynchronous", future.Response(), "context has been cancelled")
				}
			}
			return future.Response(), err
		})
	}
}

// PollingMethodType defines a type used for enumerating polling mechanisms.
type PollingMethodType string

const (
	// PollingAsyncOperation indicates the polling method uses the Azure-AsyncOperation header.
	PollingAsyncOperation PollingMethodType = "AsyncOperation"

	// PollingLocation indicates the polling method uses the Location header.
	PollingLocation PollingMethodType = "Location"

	// PollingRequestURI indicates the polling method uses the original request URI.
	PollingRequestURI PollingMethodType = "RequestURI"

	// PollingUnknown indicates an unknown polling method and is the default value.
	PollingUnknown PollingMethodType = ""
)

// AsyncOpIncompleteError is the type that's returned from a future that has not completed.
type AsyncOpIncompleteError struct {
	// FutureType is the name of the type composed of a azure.Future.
	FutureType string
}

// Error returns an error message including the originating type name of the error.
func (e AsyncOpIncompleteError) Error() string {
	return fmt.Sprintf("%s: asynchronous operation has not completed", e.FutureType)
}

// NewAsyncOpIncompleteError creates a new AsyncOpIncompleteError with the specified parameters.
func NewAsyncOpIncompleteError(futureType string) AsyncOpIncompleteError {
	return AsyncOpIncompleteError{
		FutureType: futureType,
	}
}
