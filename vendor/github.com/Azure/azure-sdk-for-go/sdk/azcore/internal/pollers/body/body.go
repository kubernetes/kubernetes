//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package body

import (
	"context"
	"errors"
	"net/http"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/exported"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/log"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/internal/pollers"
)

// Kind is the identifier of this type in a resume token.
const kind = "body"

// Applicable returns true if the LRO is using no headers, just provisioning state.
// This is only applicable to PATCH and PUT methods and assumes no polling headers.
func Applicable(resp *http.Response) bool {
	// we can't check for absense of headers due to some misbehaving services
	// like redis that return a Location header but don't actually use that protocol
	return resp.Request.Method == http.MethodPatch || resp.Request.Method == http.MethodPut
}

// CanResume returns true if the token can rehydrate this poller type.
func CanResume(token map[string]interface{}) bool {
	t, ok := token["type"]
	if !ok {
		return false
	}
	tt, ok := t.(string)
	if !ok {
		return false
	}
	return tt == kind
}

// Poller is an LRO poller that uses the Body pattern.
type Poller[T any] struct {
	pl exported.Pipeline

	resp *http.Response

	// The poller's type, used for resume token processing.
	Type string `json:"type"`

	// The URL for polling.
	PollURL string `json:"pollURL"`

	// The LRO's current state.
	CurState string `json:"state"`
}

// New creates a new Poller from the provided initial response.
// Pass nil for response to create an empty Poller for rehydration.
func New[T any](pl exported.Pipeline, resp *http.Response) (*Poller[T], error) {
	if resp == nil {
		log.Write(log.EventLRO, "Resuming Body poller.")
		return &Poller[T]{pl: pl}, nil
	}
	log.Write(log.EventLRO, "Using Body poller.")
	p := &Poller[T]{
		pl:      pl,
		resp:    resp,
		Type:    kind,
		PollURL: resp.Request.URL.String(),
	}
	// default initial state to InProgress.  depending on the HTTP
	// status code and provisioning state, we might change the value.
	curState := pollers.StatusInProgress
	provState, err := pollers.GetProvisioningState(resp)
	if err != nil && !errors.Is(err, pollers.ErrNoBody) {
		return nil, err
	}
	if resp.StatusCode == http.StatusCreated && provState != "" {
		// absense of provisioning state is ok for a 201, means the operation is in progress
		curState = provState
	} else if resp.StatusCode == http.StatusOK {
		if provState != "" {
			curState = provState
		} else if provState == "" {
			// for a 200, absense of provisioning state indicates success
			curState = pollers.StatusSucceeded
		}
	} else if resp.StatusCode == http.StatusNoContent {
		curState = pollers.StatusSucceeded
	}
	p.CurState = curState
	return p, nil
}

func (p *Poller[T]) Done() bool {
	return pollers.IsTerminalState(p.CurState)
}

func (p *Poller[T]) Poll(ctx context.Context) (*http.Response, error) {
	err := pollers.PollHelper(ctx, p.PollURL, p.pl, func(resp *http.Response) (string, error) {
		if resp.StatusCode == http.StatusNoContent {
			p.resp = resp
			p.CurState = pollers.StatusSucceeded
			return p.CurState, nil
		}
		state, err := pollers.GetProvisioningState(resp)
		if errors.Is(err, pollers.ErrNoBody) {
			// a missing response body in non-204 case is an error
			return "", err
		} else if state == "" {
			// a response body without provisioning state is considered terminal success
			state = pollers.StatusSucceeded
		} else if err != nil {
			return "", err
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
	return pollers.ResultHelper(p.resp, pollers.Failed(p.CurState), out)
}
