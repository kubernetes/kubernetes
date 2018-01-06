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
			pollingCodes := []int{http.StatusAccepted, http.StatusCreated, http.StatusOK}
			if !autorest.ResponseHasStatusCode(resp, pollingCodes...) {
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

				r, err = newPollingRequest(resp, ps)
				if err != nil {
					return resp, err
				}

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
	return state == operationSucceeded
}

func hasTerminated(state string) bool {
	switch state {
	case operationCanceled, operationFailed, operationSucceeded:
		return true
	default:
		return false
	}
}

func hasFailed(state string) bool {
	return state == operationFailed
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
	return ps.ProvisioningError != ServiceError{}
}

type pollingResponseFormat string

const (
	usesOperationResponse  pollingResponseFormat = "OperationResponse"
	usesProvisioningStatus pollingResponseFormat = "ProvisioningStatus"
	formatIsUnknown        pollingResponseFormat = ""
)

type pollingState struct {
	responseFormat pollingResponseFormat
	uri            string
	state          string
	code           string
	message        string
}

func (ps pollingState) hasSucceeded() bool {
	return hasSucceeded(ps.state)
}

func (ps pollingState) hasTerminated() bool {
	return hasTerminated(ps.state)
}

func (ps pollingState) hasFailed() bool {
	return hasFailed(ps.state)
}

func (ps pollingState) Error() string {
	return fmt.Sprintf("Long running operation terminated with status '%s': Code=%q Message=%q", ps.state, ps.code, ps.message)
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
	if ps.responseFormat == usesOperationResponse {
		pt = &operationResource{}
	} else {
		pt = &provisioningStatus{}
	}

	// If this is the first request (that is, the polling response shape is unknown), determine how
	// to poll and what to expect
	if ps.responseFormat == formatIsUnknown {
		req := resp.Request
		if req == nil {
			return autorest.NewError("azure", "updatePollingState", "Azure Polling Error - Original HTTP request is missing")
		}

		// Prefer the Azure-AsyncOperation header
		ps.uri = getAsyncOperation(resp)
		if ps.uri != "" {
			ps.responseFormat = usesOperationResponse
		} else {
			ps.responseFormat = usesProvisioningStatus
		}

		// Else, use the Location header
		if ps.uri == "" {
			ps.uri = autorest.GetLocation(resp)
		}

		// Lastly, requests against an existing resource, use the last request URI
		if ps.uri == "" {
			m := strings.ToUpper(req.Method)
			if m == http.MethodPatch || m == http.MethodPut || m == http.MethodGet {
				ps.uri = req.URL.String()
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
		ps.state = pt.state()
	} else if pt.state() != "" {
		ps.state = operationInProgress
	} else {
		switch resp.StatusCode {
		case http.StatusAccepted:
			ps.state = operationInProgress

		case http.StatusNoContent, http.StatusCreated, http.StatusOK:
			ps.state = operationSucceeded

		default:
			ps.state = operationFailed
		}
	}

	if ps.state == operationInProgress && ps.uri == "" {
		return autorest.NewError("azure", "updatePollingState", "Azure Polling Error - Unable to obtain polling URI for %s %s", resp.Request.Method, resp.Request.URL)
	}

	// For failed operation, check for error code and message in
	// -- Operation resource
	// -- Response
	// -- Otherwise, Unknown
	if ps.hasFailed() {
		if ps.responseFormat == usesOperationResponse {
			or := pt.(*operationResource)
			ps.code = or.OperationError.Code
			ps.message = or.OperationError.Message
		} else {
			p := pt.(*provisioningStatus)
			if p.hasProvisioningError() {
				ps.code = p.ProvisioningError.Code
				ps.message = p.ProvisioningError.Message
			} else {
				ps.code = "Unknown"
				ps.message = "None"
			}
		}
	}
	return nil
}

func newPollingRequest(resp *http.Response, ps pollingState) (*http.Request, error) {
	req := resp.Request
	if req == nil {
		return nil, autorest.NewError("azure", "newPollingRequest", "Azure Polling Error - Original HTTP request is missing")
	}

	reqPoll, err := autorest.Prepare(&http.Request{Cancel: req.Cancel},
		autorest.AsGet(),
		autorest.WithBaseURL(ps.uri))
	if err != nil {
		return nil, autorest.NewErrorWithError(err, "azure", "newPollingRequest", nil, "Failure creating poll request to %s", ps.uri)
	}

	return reqPoll, nil
}
