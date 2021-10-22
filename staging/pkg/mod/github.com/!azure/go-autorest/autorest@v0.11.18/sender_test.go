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
	"bytes"
	"context"
	"fmt"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"sync"
	"testing"
	"time"

	"github.com/Azure/go-autorest/autorest/mocks"
)

func ExampleSendWithSender() {
	r := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)
	mocks.SetAcceptedHeaders(r)

	client := mocks.NewSender()
	client.AppendAndRepeatResponse(r, 10)

	logger := log.New(os.Stdout, "autorest: ", 0)
	na := NullAuthorizer{}

	req, _ := Prepare(&http.Request{},
		AsGet(),
		WithBaseURL("https://microsoft.com/a/b/c/"),
		na.WithAuthorization())

	r, _ = SendWithSender(client, req,
		WithLogging(logger),
		DoErrorIfStatusCode(http.StatusAccepted),
		DoCloseIfError(),
		DoRetryForAttempts(5, time.Duration(0)))

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	// Output:
	// autorest: Sending GET https://microsoft.com/a/b/c/
	// autorest: GET https://microsoft.com/a/b/c/ received 202 Accepted
	// autorest: Sending GET https://microsoft.com/a/b/c/
	// autorest: GET https://microsoft.com/a/b/c/ received 202 Accepted
	// autorest: Sending GET https://microsoft.com/a/b/c/
	// autorest: GET https://microsoft.com/a/b/c/ received 202 Accepted
	// autorest: Sending GET https://microsoft.com/a/b/c/
	// autorest: GET https://microsoft.com/a/b/c/ received 202 Accepted
	// autorest: Sending GET https://microsoft.com/a/b/c/
	// autorest: GET https://microsoft.com/a/b/c/ received 202 Accepted
}

func ExampleDoRetryForAttempts() {
	client := mocks.NewSender()
	client.SetAndRepeatError(fmt.Errorf("Faux Error"), 10)

	// Retry with backoff -- ensure returned Bodies are closed
	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoCloseIfError(),
		DoRetryForAttempts(5, time.Duration(0)))

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	fmt.Printf("Retry stopped after %d attempts", client.Attempts())
	// Output: Retry stopped after 5 attempts
}

func ExampleDoErrorIfStatusCode() {
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(mocks.NewResponseWithStatus("204 NoContent", http.StatusNoContent), 10)

	// Chain decorators to retry the request, up to five times, if the status code is 204
	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoErrorIfStatusCode(http.StatusNoContent),
		DoCloseIfError(),
		DoRetryForAttempts(5, time.Duration(0)))

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	fmt.Printf("Retry stopped after %d attempts with code %s", client.Attempts(), r.Status)
	// Output: Retry stopped after 5 attempts with code 204 NoContent
}

func TestSendWithSenderRunsDecoratorsInOrder(t *testing.T) {
	client := mocks.NewSender()
	s := ""

	r, err := SendWithSender(client, mocks.NewRequest(),
		withMessage(&s, "a"),
		withMessage(&s, "b"),
		withMessage(&s, "c"))
	if err != nil {
		t.Fatalf("autorest: SendWithSender returned an error (%v)", err)
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if s != "abc" {
		t.Fatalf("autorest: SendWithSender invoke decorators out of order; expected 'abc', received '%s'", s)
	}
}

func TestCreateSender(t *testing.T) {
	f := false

	s := CreateSender(
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					f = true
					return nil, nil
				})
			}
		})())
	s.Do(&http.Request{})

	if !f {
		t.Fatal("autorest: CreateSender failed to apply supplied decorator")
	}
}

func TestSend(t *testing.T) {
	f := false

	Send(&http.Request{},
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					f = true
					return nil, nil
				})
			}
		})())

	if !f {
		t.Fatal("autorest: Send failed to apply supplied decorator")
	}
}

func TestAfterDelayWaits(t *testing.T) {
	client := mocks.NewSender()

	d := 2 * time.Second

	tt := time.Now()
	r, _ := SendWithSender(client, mocks.NewRequest(),
		AfterDelay(d))
	s := time.Since(tt)
	if s < d {
		t.Fatal("autorest: AfterDelay failed to wait for at least the specified duration")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestAfterDelay_Cancels(t *testing.T) {
	client := mocks.NewSender()
	ctx, cancel := context.WithCancel(context.Background())
	delay := 5 * time.Second

	var wg sync.WaitGroup
	wg.Add(1)
	start := time.Now()
	end := time.Now()
	var err error
	go func() {
		req := mocks.NewRequest()
		req = req.WithContext(ctx)
		_, err = SendWithSender(client, req,
			AfterDelay(delay))
		end = time.Now()
		wg.Done()
	}()
	cancel()
	wg.Wait()
	time.Sleep(5 * time.Millisecond)
	if end.Sub(start) >= delay {
		t.Fatal("autorest: AfterDelay elapsed")
	}
	if err == nil {
		t.Fatal("autorest: AfterDelay didn't cancel")
	}
}

func TestAfterDelayDoesNotWaitTooLong(t *testing.T) {
	client := mocks.NewSender()

	d := 5 * time.Millisecond
	start := time.Now()
	r, _ := SendWithSender(client, mocks.NewRequest(),
		AfterDelay(d))

	if time.Since(start) > (5 * d) {
		t.Fatal("autorest: AfterDelay waited too long (exceeded 5 times specified duration)")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestAsIs(t *testing.T) {
	client := mocks.NewSender()

	r1 := mocks.NewResponse()
	client.AppendResponse(r1)

	r2, err := SendWithSender(client, mocks.NewRequest(),
		AsIs())
	if err != nil {
		t.Fatalf("autorest: AsIs returned an unexpected error (%v)", err)
	} else if !reflect.DeepEqual(r1, r2) {
		t.Fatalf("autorest: AsIs modified the response -- received %v, expected %v", r2, r1)
	}

	Respond(r1,
		ByDiscardingBody(),
		ByClosing())
	Respond(r2,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoCloseIfError(t *testing.T) {
	client := mocks.NewSender()
	client.AppendResponse(mocks.NewResponseWithStatus("400 BadRequest", http.StatusBadRequest))

	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoErrorIfStatusCode(http.StatusBadRequest),
		DoCloseIfError())

	if r.Body.(*mocks.Body).IsOpen() {
		t.Fatal("autorest: Expected DoCloseIfError to close response body -- it was left open")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoCloseIfErrorAcceptsNilResponse(t *testing.T) {
	client := mocks.NewSender()

	SendWithSender(client, mocks.NewRequest(),
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					resp, err := s.Do(r)
					if err != nil {
						resp.Body.Close()
					}
					return nil, fmt.Errorf("Faux Error")
				})
			}
		})(),
		DoCloseIfError())
}

func TestDoCloseIfErrorAcceptsNilBody(t *testing.T) {
	client := mocks.NewSender()

	SendWithSender(client, mocks.NewRequest(),
		(func() SendDecorator {
			return func(s Sender) Sender {
				return SenderFunc(func(r *http.Request) (*http.Response, error) {
					resp, err := s.Do(r)
					if err != nil {
						resp.Body.Close()
					}
					resp.Body = nil
					return resp, fmt.Errorf("Faux Error")
				})
			}
		})(),
		DoCloseIfError())
}

func TestDoErrorIfStatusCode(t *testing.T) {
	client := mocks.NewSender()
	client.AppendResponse(mocks.NewResponseWithStatus("400 BadRequest", http.StatusBadRequest))

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoErrorIfStatusCode(http.StatusBadRequest),
		DoCloseIfError())
	if err == nil {
		t.Fatal("autorest: DoErrorIfStatusCode failed to emit an error for passed code")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoErrorIfStatusCodeIgnoresStatusCodes(t *testing.T) {
	client := mocks.NewSender()
	client.AppendResponse(newAcceptedResponse())

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoErrorIfStatusCode(http.StatusBadRequest),
		DoCloseIfError())
	if err != nil {
		t.Fatal("autorest: DoErrorIfStatusCode failed to ignore a status code")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoErrorUnlessStatusCode(t *testing.T) {
	client := mocks.NewSender()
	client.AppendResponse(mocks.NewResponseWithStatus("400 BadRequest", http.StatusBadRequest))

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoErrorUnlessStatusCode(http.StatusAccepted),
		DoCloseIfError())
	if err == nil {
		t.Fatal("autorest: DoErrorUnlessStatusCode failed to emit an error for an unknown status code")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoErrorUnlessStatusCodeIgnoresStatusCodes(t *testing.T) {
	client := mocks.NewSender()
	client.AppendResponse(newAcceptedResponse())

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoErrorUnlessStatusCode(http.StatusAccepted),
		DoCloseIfError())
	if err != nil {
		t.Fatal("autorest: DoErrorUnlessStatusCode emitted an error for a knonwn status code")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoRetryForAttemptsStopsAfterSuccess(t *testing.T) {
	client := mocks.NewSender()

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoRetryForAttempts(5, time.Duration(0)))
	if client.Attempts() != 1 {
		t.Fatalf("autorest: DoRetryForAttempts failed to stop after success -- expected attempts %v, actual %v",
			1, client.Attempts())
	}
	if err != nil {
		t.Fatalf("autorest: DoRetryForAttempts returned an unexpected error (%v)", err)
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoRetryForAttemptsStopsAfterAttempts(t *testing.T) {
	client := mocks.NewSender()
	client.SetAndRepeatError(fmt.Errorf("Faux Error"), 10)

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoRetryForAttempts(5, time.Duration(0)),
		DoCloseIfError())
	if err == nil {
		t.Fatal("autorest: Mock client failed to emit errors")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if client.Attempts() != 5 {
		t.Fatal("autorest: DoRetryForAttempts failed to stop after specified number of attempts")
	}
}

func TestDoRetryForAttemptsReturnsResponse(t *testing.T) {
	client := mocks.NewSender()
	client.SetError(fmt.Errorf("Faux Error"))

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoRetryForAttempts(1, time.Duration(0)))
	if err == nil {
		t.Fatal("autorest: Mock client failed to emit errors")
	}

	if r == nil {
		t.Fatal("autorest: DoRetryForAttempts failed to return the underlying response")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoRetryForDurationStopsAfterSuccess(t *testing.T) {
	client := mocks.NewSender()

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoRetryForDuration(10*time.Millisecond, time.Duration(0)))
	if client.Attempts() != 1 {
		t.Fatalf("autorest: DoRetryForDuration failed to stop after success -- expected attempts %v, actual %v",
			1, client.Attempts())
	}
	if err != nil {
		t.Fatalf("autorest: DoRetryForDuration returned an unexpected error (%v)", err)
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoRetryForDurationStopsAfterDuration(t *testing.T) {
	client := mocks.NewSender()
	client.SetAndRepeatError(fmt.Errorf("Faux Error"), -1)

	d := 5 * time.Millisecond
	start := time.Now()
	r, err := SendWithSender(client, mocks.NewRequest(),
		DoRetryForDuration(d, time.Duration(0)),
		DoCloseIfError())
	if err == nil {
		t.Fatal("autorest: Mock client failed to emit errors")
	}

	if time.Since(start) < d {
		t.Fatal("autorest: DoRetryForDuration failed stopped too soon")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoRetryForDurationStopsWithinReason(t *testing.T) {
	client := mocks.NewSender()
	client.SetAndRepeatError(fmt.Errorf("Faux Error"), -1)

	d := 5 * time.Second
	start := time.Now()
	r, err := SendWithSender(client, mocks.NewRequest(),
		DoRetryForDuration(d, time.Duration(0)),
		DoCloseIfError())
	if err == nil {
		t.Fatal("autorest: Mock client failed to emit errors")
	}

	if time.Since(start) > (5 * d) {
		t.Fatal("autorest: DoRetryForDuration failed stopped soon enough (exceeded 5 times specified duration)")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoRetryForDurationReturnsResponse(t *testing.T) {
	client := mocks.NewSender()
	client.SetAndRepeatError(fmt.Errorf("Faux Error"), -1)

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoRetryForDuration(10*time.Millisecond, time.Duration(0)),
		DoCloseIfError())
	if err == nil {
		t.Fatal("autorest: Mock client failed to emit errors")
	}

	if r == nil {
		t.Fatal("autorest: DoRetryForDuration failed to return the underlying response")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDelayForBackoff(t *testing.T) {
	d := 2 * time.Second
	start := time.Now()
	DelayForBackoff(d, 0, nil)
	if time.Since(start) < d {
		t.Fatal("autorest: DelayForBackoff did not delay as long as expected")
	}
}

func TestDelayForBackoffWithCap(t *testing.T) {
	d := 2 * time.Second
	start := time.Now()
	DelayForBackoffWithCap(d, 1*time.Second, 0, nil)
	if time.Since(start) >= d {
		t.Fatal("autorest: DelayForBackoffWithCap delayed for too long")
	}
}

func TestDelayForBackoff_Cancels(t *testing.T) {
	cancel := make(chan struct{})
	delay := 5 * time.Second

	var wg sync.WaitGroup
	wg.Add(1)
	start := time.Now()
	go func() {
		wg.Done()
		DelayForBackoff(delay, 0, cancel)
	}()
	wg.Wait()
	close(cancel)
	time.Sleep(5 * time.Millisecond)
	if time.Since(start) >= delay {
		t.Fatal("autorest: DelayForBackoff failed to cancel")
	}
}

func TestDelayForBackoffWithinReason(t *testing.T) {
	d := 5 * time.Second
	maxCoefficient := 2
	start := time.Now()
	DelayForBackoff(d, 0, nil)
	if time.Since(start) > (time.Duration(maxCoefficient) * d) {

		t.Fatalf("autorest: DelayForBackoff delayed too long (exceeded %d times the specified duration)", maxCoefficient)
	}
}

func TestDoPollForStatusCodes_IgnoresUnspecifiedStatusCodes(t *testing.T) {
	client := mocks.NewSender()

	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoPollForStatusCodes(time.Duration(0), time.Duration(0)))

	if client.Attempts() != 1 {
		t.Fatalf("autorest: Sender#DoPollForStatusCodes polled for unspecified status code")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoPollForStatusCodes_PollsForSpecifiedStatusCodes(t *testing.T) {
	client := mocks.NewSender()
	client.AppendResponse(newAcceptedResponse())

	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoPollForStatusCodes(time.Millisecond, time.Millisecond, http.StatusAccepted))

	if client.Attempts() != 2 {
		t.Fatalf("autorest: Sender#DoPollForStatusCodes failed to poll for specified status code")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoPollForStatusCodes_CanBeCanceled(t *testing.T) {
	cancel := make(chan struct{})
	delay := 5 * time.Second

	r := mocks.NewResponse()
	mocks.SetAcceptedHeaders(r)
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(r, 100)

	var wg sync.WaitGroup
	wg.Add(1)
	start := time.Now()
	go func() {
		wg.Done()
		r, _ := SendWithSender(client, mocks.NewRequest(),
			DoPollForStatusCodes(time.Millisecond, time.Millisecond, http.StatusAccepted))
		Respond(r,
			ByDiscardingBody(),
			ByClosing())
	}()
	wg.Wait()
	close(cancel)
	time.Sleep(5 * time.Millisecond)
	if time.Since(start) >= delay {
		t.Fatalf("autorest: Sender#DoPollForStatusCodes failed to cancel")
	}
}

func TestDoPollForStatusCodes_ClosesAllNonreturnedResponseBodiesWhenPolling(t *testing.T) {
	resp := newAcceptedResponse()

	client := mocks.NewSender()
	client.AppendAndRepeatResponse(resp, 2)

	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoPollForStatusCodes(time.Millisecond, time.Millisecond, http.StatusAccepted))

	if resp.Body.(*mocks.Body).IsOpen() || resp.Body.(*mocks.Body).CloseAttempts() < 2 {
		t.Fatalf("autorest: Sender#DoPollForStatusCodes did not close unreturned response bodies")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoPollForStatusCodes_LeavesLastResponseBodyOpen(t *testing.T) {
	client := mocks.NewSender()
	client.AppendResponse(newAcceptedResponse())

	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoPollForStatusCodes(time.Millisecond, time.Millisecond, http.StatusAccepted))

	if !r.Body.(*mocks.Body).IsOpen() {
		t.Fatalf("autorest: Sender#DoPollForStatusCodes did not leave open the body of the last response")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoPollForStatusCodes_StopsPollingAfterAnError(t *testing.T) {
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(newAcceptedResponse(), 5)
	client.SetError(fmt.Errorf("Faux Error"))
	client.SetEmitErrorAfter(1)

	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoPollForStatusCodes(time.Millisecond, time.Millisecond, http.StatusAccepted))

	if client.Attempts() > 2 {
		t.Fatalf("autorest: Sender#DoPollForStatusCodes failed to stop polling after receiving an error")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoPollForStatusCodes_ReturnsPollingError(t *testing.T) {
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(newAcceptedResponse(), 5)
	client.SetError(fmt.Errorf("Faux Error"))
	client.SetEmitErrorAfter(1)

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoPollForStatusCodes(time.Millisecond, time.Millisecond, http.StatusAccepted))

	if err == nil {
		t.Fatalf("autorest: Sender#DoPollForStatusCodes failed to return error from polling")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestWithLogging_Logs(t *testing.T) {
	buf := &bytes.Buffer{}
	logger := log.New(buf, "autorest: ", 0)
	client := mocks.NewSender()

	r, _ := SendWithSender(client, &http.Request{},
		WithLogging(logger))

	if buf.String() == "" {
		t.Fatal("autorest: Sender#WithLogging failed to log the request")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestWithLogging_HandlesMissingResponse(t *testing.T) {
	buf := &bytes.Buffer{}
	logger := log.New(buf, "autorest: ", 0)
	client := mocks.NewSender()
	client.AppendResponse(nil)
	client.SetError(fmt.Errorf("Faux Error"))

	r, err := SendWithSender(client, &http.Request{},
		WithLogging(logger))

	if r != nil || err == nil {
		t.Fatal("autorest: Sender#WithLogging returned a valid response -- expecting nil")
	}
	if buf.String() == "" {
		t.Fatal("autorest: Sender#WithLogging failed to log the request for a nil response")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())
}

func TestDoRetryForStatusCodesWithSuccess(t *testing.T) {
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(mocks.NewResponseWithStatus("408 Request Timeout", http.StatusRequestTimeout), 2)
	client.AppendResponse(mocks.NewResponseWithStatus("200 OK", http.StatusOK))

	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoRetryForStatusCodes(5, time.Duration(2*time.Second), http.StatusRequestTimeout),
	)

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if client.Attempts() != 3 {
		t.Fatalf("autorest: Sender#DoRetryForStatusCodes -- Got: StatusCode %v in %v attempts; Want: StatusCode 200 OK in 2 attempts -- ",
			r.Status, client.Attempts()-1)
	}
}

func TestDoRetryForStatusCodesWithNoSuccess(t *testing.T) {
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(mocks.NewResponseWithStatus("504 Gateway Timeout", http.StatusGatewayTimeout), 5)

	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoRetryForStatusCodes(2, time.Duration(2*time.Second), http.StatusGatewayTimeout),
	)
	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if client.Attempts() != 3 {
		t.Fatalf("autorest: Sender#DoRetryForStatusCodes -- Got: failed stop after %v retry attempts; Want: Stop after 2 retry attempts",
			client.Attempts()-1)
	}
}

func TestDoRetryForStatusCodes_CodeNotInRetryList(t *testing.T) {
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(mocks.NewResponseWithStatus("204 No Content", http.StatusNoContent), 1)

	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoRetryForStatusCodes(6, time.Duration(2*time.Second), http.StatusGatewayTimeout),
	)

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if client.Attempts() != 1 || r.Status != "204 No Content" {
		t.Fatalf("autorest: Sender#DoRetryForStatusCodes -- Got: Retry attempts %v for StatusCode %v; Want: 0 attempts for StatusCode 204",
			client.Attempts(), r.Status)
	}
}

func TestDoRetryForStatusCodes_RequestBodyReadError(t *testing.T) {
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(mocks.NewResponseWithStatus("204 No Content", http.StatusNoContent), 2)

	r, err := SendWithSender(client, mocks.NewRequestWithCloseBody(),
		DoRetryForStatusCodes(6, time.Duration(2*time.Second), http.StatusGatewayTimeout),
	)

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if err == nil || client.Attempts() != 0 {
		t.Fatalf("autorest: Sender#DoRetryForStatusCodes -- Got: Not failed for request body read error; Want: Failed for body read error - %v", err)
	}
}

func newAcceptedResponse() *http.Response {
	resp := mocks.NewResponseWithStatus("202 Accepted", http.StatusAccepted)
	mocks.SetAcceptedHeaders(resp)
	return resp
}

func TestDelayWithRetryAfterWithSuccess(t *testing.T) {
	Count429AsRetry = false
	defer func() { Count429AsRetry = true }()
	after, retries := 2, 2
	totalSecs := after * retries

	client := mocks.NewSender()
	resp := mocks.NewResponseWithStatus("429 Too many requests", http.StatusTooManyRequests)
	mocks.SetResponseHeader(resp, "Retry-After", fmt.Sprintf("%v", after))
	client.AppendAndRepeatResponse(resp, retries)
	client.AppendResponse(mocks.NewResponseWithStatus("200 OK", http.StatusOK))

	d := time.Second * time.Duration(totalSecs)
	start := time.Now()
	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoRetryForStatusCodes(1, time.Duration(time.Second), http.StatusTooManyRequests),
	)

	if time.Since(start) < d {
		t.Fatal("autorest: DelayWithRetryAfter failed stopped too soon")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if r.StatusCode != http.StatusOK {
		t.Fatalf("autorest: Sender#DelayWithRetryAfterWithSuccess -- got status code %d, wanted 200", r.StatusCode)
	}
	if client.Attempts() != 3 {
		t.Fatalf("autorest: Sender#DelayWithRetryAfterWithSuccess -- Got: StatusCode %v in %v attempts; Want: StatusCode 200 OK in 2 attempts -- ",
			r.Status, client.Attempts()-1)
	}
}

func TestDelayWithRetryAfterWithFail(t *testing.T) {
	after, retries := 2, 2
	totalSecs := after * retries

	client := mocks.NewSender()
	resp := mocks.NewResponseWithStatus("429 Too many requests", http.StatusTooManyRequests)
	mocks.SetResponseHeader(resp, "Retry-After", fmt.Sprintf("%v", after))
	client.AppendAndRepeatResponse(resp, retries)
	client.AppendResponse(mocks.NewResponseWithStatus("200 OK", http.StatusOK))

	d := time.Second * time.Duration(totalSecs)
	start := time.Now()
	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoRetryForStatusCodes(1, time.Duration(time.Second), http.StatusTooManyRequests),
	)

	if time.Since(start) < d {
		t.Fatal("autorest: DelayWithRetryAfter failed stopped too soon")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if r.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("autorest: Sender#DelayWithRetryAfterWithFail -- got status code %d, wanted 429", r.StatusCode)
	}
	if client.Attempts() != 2 {
		t.Fatalf("autorest: Sender#DelayWithRetryAfterWithFail -- Got: StatusCode %v in %v attempts; Want: StatusCode 429 OK in 1 attempt -- ",
			r.Status, client.Attempts()-1)
	}
}

func TestDelayWithRetryAfterWithSuccessDateTime(t *testing.T) {
	resumeAt := time.Now().Add(2 * time.Second).Round(time.Second)

	client := mocks.NewSender()
	resp := mocks.NewResponseWithStatus("503 Service temporarily unavailable", http.StatusServiceUnavailable)
	mocks.SetResponseHeader(resp, "Retry-After", resumeAt.Format(time.RFC1123))
	client.AppendResponse(resp)
	client.AppendResponse(mocks.NewResponseWithStatus("200 OK", http.StatusOK))

	r, _ := SendWithSender(client, mocks.NewRequest(),
		DoRetryForStatusCodes(1, time.Duration(time.Second), http.StatusServiceUnavailable),
	)

	if time.Now().Before(resumeAt) {
		t.Fatal("autorest: DelayWithRetryAfter failed stopped too soon")
	}

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if client.Attempts() != 2 {
		t.Fatalf("autorest: Sender#DelayWithRetryAfter -- Got: StatusCode %v in %v attempts; Want: StatusCode 200 OK in 2 attempts -- ",
			r.Status, client.Attempts()-1)
	}
}

type temporaryError struct {
	message string
}

func (te temporaryError) Error() string {
	return te.message
}

func (te temporaryError) Timeout() bool {
	return true
}

func (te temporaryError) Temporary() bool {
	return true
}

func TestDoRetryForStatusCodes_NilResponseTemporaryError(t *testing.T) {
	client := mocks.NewSender()
	client.AppendResponse(nil)
	client.SetError(temporaryError{message: "faux error"})

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoRetryForStatusCodes(3, time.Duration(1*time.Second), StatusCodesForRetry...),
	)

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if err != nil || client.Attempts() != 2 {
		t.Fatalf("autorest: Sender#TestDoRetryForStatusCodes_NilResponseTemporaryError -- Got: non-nil error or wrong number of attempts - %v", err)
	}
}

func TestDoRetryForStatusCodes_NilResponseTemporaryError2(t *testing.T) {
	client := mocks.NewSender()
	client.AppendResponse(nil)
	client.SetError(fmt.Errorf("faux error"))

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoRetryForStatusCodes(3, time.Duration(1*time.Second), StatusCodesForRetry...),
	)

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if err != nil || client.Attempts() != 2 {
		t.Fatalf("autorest: Sender#TestDoRetryForStatusCodes_NilResponseTemporaryError2 -- Got: nil error or wrong number of attempts - %v", err)
	}
}

type fatalError struct {
	message string
}

func (fe fatalError) Error() string {
	return fe.message
}

func (fe fatalError) Timeout() bool {
	return false
}

func (fe fatalError) Temporary() bool {
	return false
}

func TestDoRetryForStatusCodes_NilResponseFatalError(t *testing.T) {
	const retryAttempts = 3
	client := mocks.NewSender()
	client.AppendAndRepeatResponse(nil, retryAttempts+1)
	client.SetAndRepeatError(fatalError{"fatal error"}, retryAttempts+1)

	r, err := SendWithSender(client, mocks.NewRequest(),
		DoRetryForStatusCodes(retryAttempts, time.Duration(1*time.Second), StatusCodesForRetry...),
	)

	Respond(r,
		ByDiscardingBody(),
		ByClosing())

	if err == nil || client.Attempts() < retryAttempts+1 {
		t.Fatalf("autorest: Sender#TestDoRetryForStatusCodes_NilResponseFatalError -- Got: nil error or wrong number of attempts - %v", err)
	}
}

func TestDoRetryForStatusCodes_Cancel429(t *testing.T) {
	Count429AsRetry = false
	defer func() { Count429AsRetry = true }()
	retries := 6
	client := mocks.NewSender()
	resp := mocks.NewResponseWithStatus("429 Too many requests", http.StatusTooManyRequests)
	client.AppendAndRepeatResponse(resp, retries)

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(retries/2)*time.Second)
	defer cancel()
	req := mocks.NewRequest().WithContext(ctx)
	r, err := SendWithSender(client, req,
		DoRetryForStatusCodes(1, time.Duration(time.Second), http.StatusTooManyRequests),
	)

	if err == nil {
		t.Fatal("unexpected nil-error")
	}
	if r == nil {
		t.Fatal("unexpected nil response")
	}
	if r.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("expected status code 429, got: %d", r.StatusCode)
	}
	if client.Attempts() >= retries {
		t.Fatalf("too many attempts: %d", client.Attempts())
	}
}

func TestDoRetryForStatusCodes_Race(t *testing.T) {
	// cannot use the mock sender as it's not safe for concurrent use
	s := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	defer s.Close()

	sender := DecorateSender(s.Client(),
		DoRetryForStatusCodes(0, 0, http.StatusRequestTimeout))

	runs := 2
	errCh := make(chan error, runs)

	for i := 0; i < runs; i++ {
		go func() {
			req, _ := http.NewRequest(http.MethodGet, s.URL, nil)
			// cannot use testing.T.Fatal inside a goroutine, send error down channel
			_, err := sender.Do(req)
			errCh <- err
		}()
	}
	for i := 0; i < runs; i++ {
		err := <-errCh
		if err != nil {
			t.Fatal(err)
		}
	}
	close(errCh)
}

func TestGetSendDecorators(t *testing.T) {
	sd := GetSendDecorators(context.Background())
	if l := len(sd); l != 0 {
		t.Fatalf("expected zero length but got %d", l)
	}
	sd = GetSendDecorators(context.Background(), DoCloseIfError(), DoErrorIfStatusCode())
	if l := len(sd); l != 2 {
		t.Fatalf("expected length of two but got %d", l)
	}
}

func TestWithSendDecorators(t *testing.T) {
	ctx := WithSendDecorators(context.Background(), []SendDecorator{DoRetryForAttempts(5, 5*time.Second)})
	sd := GetSendDecorators(ctx)
	if l := len(sd); l != 1 {
		t.Fatalf("expected length of one but got %d", l)
	}
	sd = GetSendDecorators(ctx, DoCloseIfError(), DoErrorIfStatusCode())
	if l := len(sd); l != 1 {
		t.Fatalf("expected length of one but got %d", l)
	}
}
