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
	"strings"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/date"
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
	req  *http.Request
	resp *http.Response
	ps   pollingState
}

// NewFuture returns a new Future object initialized with the specified request.
func NewFuture(req *http.Request) Future {
	return Future{req: req}
}

// Response returns the last HTTP response or nil if there isn't one.
func (f Future) Response() *http.Response {
	return f.resp
}

// Status returns the last status message of the operation.
func (f Future) Status() string {
	if f.ps.State == "" {
		return "Unknown"
	}
	return f.ps.State
}

// PollingMethod returns the method used to monitor the status of the asynchronous operation.
func (f Future) PollingMethod() PollingMethodType {
	return f.ps.PollingMethod
}

// Done queries the service to see if the operation has completed.
func (f *Future) Done(sender autorest.Sender) (bool, error) {
	// exit early if this future has terminated
	if f.ps.hasTerminated() {
		return true, f.errorInfo()
	}
	resp, err := sender.Do(f.req)
	f.resp = resp
	if err != nil {
		return false, err
	}

	if !autorest.ResponseHasStatusCode(resp, pollingCodes[:]...) {
		// check response body for error content
		if resp.Body != nil {
			type respErr struct {
				ServiceError ServiceError `json:"error"`
			}
			re := respErr{}

			defer resp.Body.Close()
			b, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				return false, err
			}
			err = json.Unmarshal(b, &re)
			if err != nil {
				return false, err
			}
			return false, re.ServiceError
		}

		// try to return something meaningful
		return false, ServiceError{
			Code:    fmt.Sprintf("%v", resp.StatusCode),
			Message: resp.Status,
		}
	}

	err = updatePollingState(resp, &f.ps)
	if err != nil {
		return false, err
	}

	if f.ps.hasTerminated() {
		return true, f.errorInfo()
	}

	f.req, err = newPollingRequest(f.ps)
	return false, err
}

// GetPollingDelay returns a duration the application should wait before checking
// the status of the asynchronous request and true; this value is returned from
// the service via the Retry-After response header.  If the header wasn't returned
// then the function returns the zero-value time.Duration and false.
func (f Future) GetPollingDelay() (time.Duration, bool) {
	if f.resp == nil {
		return 0, false
	}

	retry := f.resp.Header.Get(autorest.HeaderRetryAfter)
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
func (f Future) WaitForCompletion(ctx context.Context, client autorest.Client) error {
	ctx, cancel := context.WithTimeout(ctx, client.PollingDuration)
	defer cancel()

	done, err := f.Done(client)
	for attempts := 0; !done; done, err = f.Done(client) {
		if attempts >= client.RetryAttempts {
			return autorest.NewErrorWithError(err, "azure", "WaitForCompletion", f.resp, "the number of retries has been exceeded")
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
			return autorest.NewErrorWithError(ctx.Err(), "azure", "WaitForCompletion", f.resp, "context has been cancelled")
		}
	}
	return err
}

// if the operation failed the polling state will contain
// error information and implements the error interface
func (f *Future) errorInfo() error {
	if !f.ps.hasSucceeded() {
		return f.ps
	}
	return nil
}

// MarshalJSON implements the json.Marshaler interface.
func (f Future) MarshalJSON() ([]byte, error) {
	return json.Marshal(&f.ps)
}

// UnmarshalJSON implements the json.Unmarshaler interface.
func (f *Future) UnmarshalJSON(data []byte) error {
	err := json.Unmarshal(data, &f.ps)
	if err != nil {
		return err
	}
	f.req, err = newPollingRequest(f.ps)
	return err
}

// PollingURL returns the URL used for retrieving the status of the long-running operation.
// For LROs that use the Location header the final URL value is used to retrieve the result.
func (f Future) PollingURL() string {
	return f.ps.URI
}

// DoPollForAsynchronous returns a SendDecorator that polls if the http.Response is for an Azure
// long-running operation. It will delay between requests for the duration specified in the
// RetryAfter header or, if the header is absent, the passed delay. Polling may be canceled by
// closing the optional channel on the http.Request.
func DoPollForAsynchronous(delay time.Duration) autorest.SendDecorator {
	return func(s autorest.Sender) autorest.Sender {
		return autorest.SenderFunc(func(r *http.Request) (resp *http.Response, err error) {
			resp, err = s.Do(r)
			if err != nil {
				return resp, err
			}
			if !autorest.ResponseHasStatusCode(resp, pollingCodes[:]...) {
				return resp, nil
			}

			ps := pollingState{}
			for err == nil {
				err = updatePollingState(resp, &ps)
				if err != nil {
					break
				}
				if ps.hasTerminated() {
					if !ps.hasSucceeded() {
						err = ps
					}
					break
				}

				r, err = newPollingRequest(ps)
				if err != nil {
					return resp, err
				}
				r = r.WithContext(resp.Request.Context())

				delay = autorest.GetRetryAfter(resp, delay)
				resp, err = autorest.SendWithSender(s, r,
					autorest.AfterDelay(delay))
			}

			return resp, err
		})
	}
}

func getAsyncOperation(resp *http.Response) string {
	return resp.Header.Get(http.CanonicalHeaderKey(headerAsyncOperation))
}

func hasSucceeded(state string) bool {
	return strings.EqualFold(state, operationSucceeded)
}

func hasTerminated(state string) bool {
	return strings.EqualFold(state, operationCanceled) || strings.EqualFold(state, operationFailed) || strings.EqualFold(state, operationSucceeded)
}

func hasFailed(state string) bool {
	return strings.EqualFold(state, operationFailed)
}

type provisioningTracker interface {
	state() string
	hasSucceeded() bool
	hasTerminated() bool
}

type operationResource struct {
	// Note:
	// 	The specification states services should return the "id" field. However some return it as
	// 	"operationId".
	ID              string                 `json:"id"`
	OperationID     string                 `json:"operationId"`
	Name            string                 `json:"name"`
	Status          string                 `json:"status"`
	Properties      map[string]interface{} `json:"properties"`
	OperationError  ServiceError           `json:"error"`
	StartTime       date.Time              `json:"startTime"`
	EndTime         date.Time              `json:"endTime"`
	PercentComplete float64                `json:"percentComplete"`
}

func (or operationResource) state() string {
	return or.Status
}

func (or operationResource) hasSucceeded() bool {
	return hasSucceeded(or.state())
}

func (or operationResource) hasTerminated() bool {
	return hasTerminated(or.state())
}

type provisioningProperties struct {
	ProvisioningState string `json:"provisioningState"`
}

type provisioningStatus struct {
	Properties        provisioningProperties `json:"properties,omitempty"`
	ProvisioningError ServiceError           `json:"error,omitempty"`
}

func (ps provisioningStatus) state() string {
	return ps.Properties.ProvisioningState
}

func (ps provisioningStatus) hasSucceeded() bool {
	return hasSucceeded(ps.state())
}

func (ps provisioningStatus) hasTerminated() bool {
	return hasTerminated(ps.state())
}

func (ps provisioningStatus) hasProvisioningError() bool {
	// code and message are required fields so only check them
	return len(ps.ProvisioningError.Code) > 0 ||
		len(ps.ProvisioningError.Message) > 0
}

// PollingMethodType defines a type used for enumerating polling mechanisms.
type PollingMethodType string

const (
	// PollingAsyncOperation indicates the polling method uses the Azure-AsyncOperation header.
	PollingAsyncOperation PollingMethodType = "AsyncOperation"

	// PollingLocation indicates the polling method uses the Location header.
	PollingLocation PollingMethodType = "Location"

	// PollingUnknown indicates an unknown polling method and is the default value.
	PollingUnknown PollingMethodType = ""
)

type pollingState struct {
	PollingMethod PollingMethodType `json:"pollingMethod"`
	URI           string            `json:"uri"`
	State         string            `json:"state"`
	ServiceError  *ServiceError     `json:"error,omitempty"`
}

func (ps pollingState) hasSucceeded() bool {
	return hasSucceeded(ps.State)
}

func (ps pollingState) hasTerminated() bool {
	return hasTerminated(ps.State)
}

func (ps pollingState) hasFailed() bool {
	return hasFailed(ps.State)
}

func (ps pollingState) Error() string {
	s := fmt.Sprintf("Long running operation terminated with status '%s'", ps.State)
	if ps.ServiceError != nil {
		s = fmt.Sprintf("%s: %+v", s, *ps.ServiceError)
	}
	return s
}

//	updatePollingState maps the operation status -- retrieved from either a provisioningState
// 	field, the status field of an OperationResource, or inferred from the HTTP status code --
// 	into a well-known states. Since the process begins from the initial request, the state
//	always comes from either a the provisioningState returned or is inferred from the HTTP
//	status code. Subsequent requests will read an Azure OperationResource object if the
//	service initially returned the Azure-AsyncOperation header. The responseFormat field notes
//	the expected response format.
func updatePollingState(resp *http.Response, ps *pollingState) error {
	// Determine the response shape
	// -- The first response will always be a provisioningStatus response; only the polling requests,
	//    depending on the header returned, may be something otherwise.
	var pt provisioningTracker
	if ps.PollingMethod == PollingAsyncOperation {
		pt = &operationResource{}
	} else {
		pt = &provisioningStatus{}
	}

	// If this is the first request (that is, the polling response shape is unknown), determine how
	// to poll and what to expect
	if ps.PollingMethod == PollingUnknown {
		req := resp.Request
		if req == nil {
			return autorest.NewError("azure", "updatePollingState", "Azure Polling Error - Original HTTP request is missing")
		}

		// Prefer the Azure-AsyncOperation header
		ps.URI = getAsyncOperation(resp)
		if ps.URI != "" {
			ps.PollingMethod = PollingAsyncOperation
		} else {
			ps.PollingMethod = PollingLocation
		}

		// Else, use the Location header
		if ps.URI == "" {
			ps.URI = autorest.GetLocation(resp)
		}

		// Lastly, requests against an existing resource, use the last request URI
		if ps.URI == "" {
			m := strings.ToUpper(req.Method)
			if m == http.MethodPatch || m == http.MethodPut || m == http.MethodGet {
				ps.URI = req.URL.String()
			}
		}
	}

	// Read and interpret the response (saving the Body in case no polling is necessary)
	b := &bytes.Buffer{}
	err := autorest.Respond(resp,
		autorest.ByCopying(b),
		autorest.ByUnmarshallingJSON(pt),
		autorest.ByClosing())
	resp.Body = ioutil.NopCloser(b)
	if err != nil {
		return err
	}

	// Interpret the results
	// -- Terminal states apply regardless
	// -- Unknown states are per-service inprogress states
	// -- Otherwise, infer state from HTTP status code
	if pt.hasTerminated() {
		ps.State = pt.state()
	} else if pt.state() != "" {
		ps.State = operationInProgress
	} else {
		switch resp.StatusCode {
		case http.StatusAccepted:
			ps.State = operationInProgress

		case http.StatusNoContent, http.StatusCreated, http.StatusOK:
			ps.State = operationSucceeded

		default:
			ps.State = operationFailed
		}
	}

	if strings.EqualFold(ps.State, operationInProgress) && ps.URI == "" {
		return autorest.NewError("azure", "updatePollingState", "Azure Polling Error - Unable to obtain polling URI for %s %s", resp.Request.Method, resp.Request.URL)
	}

	// For failed operation, check for error code and message in
	// -- Operation resource
	// -- Response
	// -- Otherwise, Unknown
	if ps.hasFailed() {
		if or, ok := pt.(*operationResource); ok {
			ps.ServiceError = &or.OperationError
		} else if p, ok := pt.(*provisioningStatus); ok && p.hasProvisioningError() {
			ps.ServiceError = &p.ProvisioningError
		} else {
			ps.ServiceError = &ServiceError{
				Code:    "Unknown",
				Message: "None",
			}
		}
	}
	return nil
}

func newPollingRequest(ps pollingState) (*http.Request, error) {
	reqPoll, err := autorest.Prepare(&http.Request{},
		autorest.AsGet(),
		autorest.WithBaseURL(ps.URI))
	if err != nil {
		return nil, autorest.NewErrorWithError(err, "azure", "newPollingRequest", nil, "Failure creating poll request to %s", ps.URI)
	}

	return reqPoll, nil
}

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
