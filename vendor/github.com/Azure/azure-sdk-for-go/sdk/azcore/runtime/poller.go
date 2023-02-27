//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package runtime

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/exported"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/log"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/pollers"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/pollers/async"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/pollers/body"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/pollers/loc"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/pollers/op"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/shared"
)

// FinalStateVia is the enumerated type for the possible final-state-via values.
type FinalStateVia = pollers.FinalStateVia

const (
	// FinalStateViaAzureAsyncOp indicates the final payload comes from the Azure-AsyncOperation URL.
	FinalStateViaAzureAsyncOp = pollers.FinalStateViaAzureAsyncOp

	// FinalStateViaLocation indicates the final payload comes from the Location URL.
	FinalStateViaLocation = pollers.FinalStateViaLocation

	// FinalStateViaOriginalURI indicates the final payload comes from the original URL.
	FinalStateViaOriginalURI = pollers.FinalStateViaOriginalURI

	// FinalStateViaOpLocation indicates the final payload comes from the Operation-Location URL.
	FinalStateViaOpLocation = pollers.FinalStateViaOpLocation
)

// NewPollerOptions contains the optional parameters for NewPoller.
type NewPollerOptions[T any] struct {
	// FinalStateVia contains the final-state-via value for the LRO.
	FinalStateVia FinalStateVia

	// Response contains a preconstructed response type.
	// The final payload will be unmarshaled into it and returned.
	Response *T

	// Handler[T] contains a custom polling implementation.
	Handler PollingHandler[T]
}

// NewPoller creates a Poller based on the provided initial response.
func NewPoller[T any](resp *http.Response, pl exported.Pipeline, options *NewPollerOptions[T]) (*Poller[T], error) {
	if options == nil {
		options = &NewPollerOptions[T]{}
	}
	result := options.Response
	if result == nil {
		result = new(T)
	}
	if options.Handler != nil {
		return &Poller[T]{
			op:     options.Handler,
			resp:   resp,
			result: result,
		}, nil
	}

	defer resp.Body.Close()
	// this is a back-stop in case the swagger is incorrect (i.e. missing one or more status codes for success).
	// ideally the codegen should return an error if the initial response failed and not even create a poller.
	if !pollers.StatusCodeValid(resp) {
		return nil, errors.New("the operation failed or was cancelled")
	}

	// determine the polling method
	var opr PollingHandler[T]
	var err error
	if async.Applicable(resp) {
		// async poller must be checked first as it can also have a location header
		opr, err = async.New[T](pl, resp, options.FinalStateVia)
	} else if op.Applicable(resp) {
		// op poller must be checked before loc as it can also have a location header
		opr, err = op.New[T](pl, resp, options.FinalStateVia)
	} else if loc.Applicable(resp) {
		opr, err = loc.New[T](pl, resp)
	} else if body.Applicable(resp) {
		// must test body poller last as it's a subset of the other pollers.
		// TODO: this is ambiguous for PATCH/PUT if it returns a 200 with no polling headers (sync completion)
		opr, err = body.New[T](pl, resp)
	} else if m := resp.Request.Method; resp.StatusCode == http.StatusAccepted && (m == http.MethodDelete || m == http.MethodPost) {
		// if we get here it means we have a 202 with no polling headers.
		// for DELETE and POST this is a hard error per ARM RPC spec.
		return nil, errors.New("response is missing polling URL")
	} else {
		opr, err = pollers.NewNopPoller[T](resp)
	}

	if err != nil {
		return nil, err
	}
	return &Poller[T]{
		op:     opr,
		resp:   resp,
		result: result,
	}, nil
}

// NewPollerFromResumeTokenOptions contains the optional parameters for NewPollerFromResumeToken.
type NewPollerFromResumeTokenOptions[T any] struct {
	// Response contains a preconstructed response type.
	// The final payload will be unmarshaled into it and returned.
	Response *T

	// Handler[T] contains a custom polling implementation.
	Handler PollingHandler[T]
}

// NewPollerFromResumeToken creates a Poller from a resume token string.
func NewPollerFromResumeToken[T any](token string, pl exported.Pipeline, options *NewPollerFromResumeTokenOptions[T]) (*Poller[T], error) {
	if options == nil {
		options = &NewPollerFromResumeTokenOptions[T]{}
	}
	result := options.Response
	if result == nil {
		result = new(T)
	}

	if err := pollers.IsTokenValid[T](token); err != nil {
		return nil, err
	}
	raw, err := pollers.ExtractToken(token)
	if err != nil {
		return nil, err
	}
	var asJSON map[string]interface{}
	if err := json.Unmarshal(raw, &asJSON); err != nil {
		return nil, err
	}

	opr := options.Handler
	// now rehydrate the poller based on the encoded poller type
	if async.CanResume(asJSON) {
		opr, _ = async.New[T](pl, nil, "")
	} else if body.CanResume(asJSON) {
		opr, _ = body.New[T](pl, nil)
	} else if loc.CanResume(asJSON) {
		opr, _ = loc.New[T](pl, nil)
	} else if op.CanResume(asJSON) {
		opr, _ = op.New[T](pl, nil, "")
	} else if opr != nil {
		log.Writef(log.EventLRO, "Resuming custom poller %T.", opr)
	} else {
		return nil, fmt.Errorf("unhandled poller token %s", string(raw))
	}
	if err := json.Unmarshal(raw, &opr); err != nil {
		return nil, err
	}
	return &Poller[T]{
		op:     opr,
		result: result,
	}, nil
}

// PollingHandler[T] abstracts the differences among poller implementations.
type PollingHandler[T any] interface {
	// Done returns true if the LRO has reached a terminal state.
	Done() bool

	// Poll fetches the latest state of the LRO.
	Poll(context.Context) (*http.Response, error)

	// Result is called once the LRO has reached a terminal state. It populates the out parameter
	// with the result of the operation.
	Result(ctx context.Context, out *T) error
}

// Poller encapsulates a long-running operation, providing polling facilities until the operation reaches a terminal state.
type Poller[T any] struct {
	op     PollingHandler[T]
	resp   *http.Response
	err    error
	result *T
	done   bool
}

// PollUntilDoneOptions contains the optional values for the Poller[T].PollUntilDone() method.
type PollUntilDoneOptions struct {
	// Frequency is the time to wait between polling intervals in absence of a Retry-After header. Allowed minimum is one second.
	// Pass zero to accept the default value (30s).
	Frequency time.Duration
}

// PollUntilDone will poll the service endpoint until a terminal state is reached, an error is received, or the context expires.
// It internally uses Poll(), Done(), and Result() in its polling loop, sleeping for the specified duration between intervals.
// options: pass nil to accept the default values.
// NOTE: the default polling frequency is 30 seconds which works well for most operations.  However, some operations might
// benefit from a shorter or longer duration.
func (p *Poller[T]) PollUntilDone(ctx context.Context, options *PollUntilDoneOptions) (T, error) {
	if options == nil {
		options = &PollUntilDoneOptions{}
	}
	cp := *options
	if cp.Frequency == 0 {
		cp.Frequency = 30 * time.Second
	}

	if cp.Frequency < time.Second {
		return *new(T), errors.New("polling frequency minimum is one second")
	}

	start := time.Now()
	logPollUntilDoneExit := func(v interface{}) {
		log.Writef(log.EventLRO, "END PollUntilDone() for %T: %v, total time: %s", p.op, v, time.Since(start))
	}
	log.Writef(log.EventLRO, "BEGIN PollUntilDone() for %T", p.op)
	if p.resp != nil {
		// initial check for a retry-after header existing on the initial response
		if retryAfter := shared.RetryAfter(p.resp); retryAfter > 0 {
			log.Writef(log.EventLRO, "initial Retry-After delay for %s", retryAfter.String())
			if err := shared.Delay(ctx, retryAfter); err != nil {
				logPollUntilDoneExit(err)
				return *new(T), err
			}
		}
	}
	// begin polling the endpoint until a terminal state is reached
	for {
		resp, err := p.Poll(ctx)
		if err != nil {
			logPollUntilDoneExit(err)
			return *new(T), err
		}
		if p.Done() {
			logPollUntilDoneExit("succeeded")
			return p.Result(ctx)
		}
		d := cp.Frequency
		if retryAfter := shared.RetryAfter(resp); retryAfter > 0 {
			log.Writef(log.EventLRO, "Retry-After delay for %s", retryAfter.String())
			d = retryAfter
		} else {
			log.Writef(log.EventLRO, "delay for %s", d.String())
		}
		if err = shared.Delay(ctx, d); err != nil {
			logPollUntilDoneExit(err)
			return *new(T), err
		}
	}
}

// Poll fetches the latest state of the LRO.  It returns an HTTP response or error.
// If Poll succeeds, the poller's state is updated and the HTTP response is returned.
// If Poll fails, the poller's state is unmodified and the error is returned.
// Calling Poll on an LRO that has reached a terminal state will return the last HTTP response.
func (p *Poller[T]) Poll(ctx context.Context) (*http.Response, error) {
	if p.Done() {
		// the LRO has reached a terminal state, don't poll again
		return p.resp, nil
	}
	resp, err := p.op.Poll(ctx)
	if err != nil {
		return nil, err
	}
	p.resp = resp
	return p.resp, nil
}

// Done returns true if the LRO has reached a terminal state.
// Once a terminal state is reached, call Result().
func (p *Poller[T]) Done() bool {
	return p.op.Done()
}

// Result returns the result of the LRO and is meant to be used in conjunction with Poll and Done.
// If the LRO completed successfully, a populated instance of T is returned.
// If the LRO failed or was canceled, an *azcore.ResponseError error is returned.
// Calling this on an LRO in a non-terminal state will return an error.
func (p *Poller[T]) Result(ctx context.Context) (T, error) {
	if !p.Done() {
		return *new(T), errors.New("poller is in a non-terminal state")
	}
	if p.done {
		// the result has already been retrieved, return the cached value
		if p.err != nil {
			return *new(T), p.err
		}
		return *p.result, nil
	}
	err := p.op.Result(ctx, p.result)
	var respErr *exported.ResponseError
	if errors.As(err, &respErr) {
		// the LRO failed. record the error
		p.err = err
	} else if err != nil {
		// the call to Result failed, don't cache anything in this case
		return *new(T), err
	}
	p.done = true
	if p.err != nil {
		return *new(T), p.err
	}
	return *p.result, nil
}

// ResumeToken returns a value representing the poller that can be used to resume
// the LRO at a later time. ResumeTokens are unique per service operation.
// The token's format should be considered opaque and is subject to change.
// Calling this on an LRO in a terminal state will return an error.
func (p *Poller[T]) ResumeToken() (string, error) {
	if p.Done() {
		return "", errors.New("poller is in a terminal state")
	}
	tk, err := pollers.NewResumeToken[T](p.op)
	if err != nil {
		return "", err
	}
	return tk, err
}
