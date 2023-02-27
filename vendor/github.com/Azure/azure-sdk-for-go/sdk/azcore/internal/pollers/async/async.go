//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package async

import (
	"context"
	"errors"
	"fmt"
	"net/http"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/exported"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/log"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/pollers"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/shared"
)

// see https://github.com/Azure/azure-resource-manager-rpc/blob/master/v1.0/async-api-reference.md

// Applicable returns true if the LRO is using Azure-AsyncOperation.
func Applicable(resp *http.Response) bool {
	return resp.Header.Get(shared.HeaderAzureAsync) != ""
}

// CanResume returns true if the token can rehydrate this poller type.
func CanResume(token map[string]interface{}) bool {
	_, ok := token["asyncURL"]
	return ok
}

// Poller is an LRO poller that uses the Azure-AsyncOperation pattern.
type Poller[T any] struct {
	pl exported.Pipeline

	resp *http.Response

	// The URL from Azure-AsyncOperation header.
	AsyncURL string `json:"asyncURL"`

	// The URL from Location header.
	LocURL string `json:"locURL"`

	// The URL from the initial LRO request.
	OrigURL string `json:"origURL"`

	// The HTTP method from the initial LRO request.
	Method string `json:"method"`

	// The value of final-state-via from swagger, can be the empty string.
	FinalState pollers.FinalStateVia `json:"finalState"`

	// The LRO's current state.
	CurState string `json:"state"`
}

// New creates a new Poller from the provided initial response and final-state type.
// Pass nil for response to create an empty Poller for rehydration.
func New[T any](pl exported.Pipeline, resp *http.Response, finalState pollers.FinalStateVia) (*Poller[T], error) {
	if resp == nil {
		log.Write(log.EventLRO, "Resuming Azure-AsyncOperation poller.")
		return &Poller[T]{pl: pl}, nil
	}
	log.Write(log.EventLRO, "Using Azure-AsyncOperation poller.")
	asyncURL := resp.Header.Get(shared.HeaderAzureAsync)
	if asyncURL == "" {
		return nil, errors.New("response is missing Azure-AsyncOperation header")
	}
	if !pollers.IsValidURL(asyncURL) {
		return nil, fmt.Errorf("invalid polling URL %s", asyncURL)
	}
	p := &Poller[T]{
		pl:         pl,
		resp:       resp,
		AsyncURL:   asyncURL,
		LocURL:     resp.Header.Get(shared.HeaderLocation),
		OrigURL:    resp.Request.URL.String(),
		Method:     resp.Request.Method,
		FinalState: finalState,
		CurState:   pollers.StatusInProgress,
	}
	return p, nil
}

// Done returns true if the LRO is in a terminal state.
func (p *Poller[T]) Done() bool {
	return pollers.IsTerminalState(p.CurState)
}

// Poll retrieves the current state of the LRO.
func (p *Poller[T]) Poll(ctx context.Context) (*http.Response, error) {
	err := pollers.PollHelper(ctx, p.AsyncURL, p.pl, func(resp *http.Response) (string, error) {
		state, err := pollers.GetStatus(resp)
		if err != nil {
			return "", err
		} else if state == "" {
			return "", errors.New("the response did not contain a status")
		}
		p.resp = resp
		p.CurState = state
		return p.CurState, nil
	})
	if err != nil {
		return nil, err
	}
	return p.resp, nil
}

func (p *Poller[T]) Result(ctx context.Context, out *T) error {
	if p.resp.StatusCode == http.StatusNoContent {
		return nil
	} else if pollers.Failed(p.CurState) {
		return exported.NewResponseError(p.resp)
	}
	var req *exported.Request
	var err error
	if p.Method == http.MethodPatch || p.Method == http.MethodPut {
		// for PATCH and PUT, the final GET is on the original resource URL
		req, err = exported.NewRequest(ctx, http.MethodGet, p.OrigURL)
	} else if p.Method == http.MethodPost {
		if p.FinalState == pollers.FinalStateViaAzureAsyncOp {
			// no final GET required
		} else if p.FinalState == pollers.FinalStateViaOriginalURI {
			req, err = exported.NewRequest(ctx, http.MethodGet, p.OrigURL)
		} else if p.LocURL != "" {
			// ideally FinalState would be set to "location" but it isn't always.
			// must check last due to more permissive condition.
			req, err = exported.NewRequest(ctx, http.MethodGet, p.LocURL)
		}
	}
	if err != nil {
		return err
	}

	// if a final GET request has been created, execute it
	if req != nil {
		resp, err := p.pl.Do(req)
		if err != nil {
			return err
		}
		p.resp = resp
	}

	return pollers.ResultHelper(p.resp, pollers.Failed(p.CurState), out)
}
