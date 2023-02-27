//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package op

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

// Applicable returns true if the LRO is using Operation-Location.
func Applicable(resp *http.Response) bool {
	return resp.Header.Get(shared.HeaderOperationLocation) != ""
}

// CanResume returns true if the token can rehydrate this poller type.
func CanResume(token map[string]interface{}) bool {
	_, ok := token["oplocURL"]
	return ok
}

// Poller is an LRO poller that uses the Operation-Location pattern.
type Poller[T any] struct {
	pl   exported.Pipeline
	resp *http.Response

	OpLocURL   string                `json:"oplocURL"`
	LocURL     string                `json:"locURL"`
	OrigURL    string                `json:"origURL"`
	Method     string                `json:"method"`
	FinalState pollers.FinalStateVia `json:"finalState"`
	CurState   string                `json:"state"`
}

// New creates a new Poller from the provided initial response.
// Pass nil for response to create an empty Poller for rehydration.
func New[T any](pl exported.Pipeline, resp *http.Response, finalState pollers.FinalStateVia) (*Poller[T], error) {
	if resp == nil {
		log.Write(log.EventLRO, "Resuming Operation-Location poller.")
		return &Poller[T]{pl: pl}, nil
	}
	log.Write(log.EventLRO, "Using Operation-Location poller.")
	opURL := resp.Header.Get(shared.HeaderOperationLocation)
	if opURL == "" {
		return nil, errors.New("response is missing Operation-Location header")
	}
	if !pollers.IsValidURL(opURL) {
		return nil, fmt.Errorf("invalid Operation-Location URL %s", opURL)
	}
	locURL := resp.Header.Get(shared.HeaderLocation)
	// Location header is optional
	if locURL != "" && !pollers.IsValidURL(locURL) {
		return nil, fmt.Errorf("invalid Location URL %s", locURL)
	}
	// default initial state to InProgress.  if the
	// service sent us a status then use that instead.
	curState := pollers.StatusInProgress
	status, err := pollers.GetStatus(resp)
	if err != nil && !errors.Is(err, pollers.ErrNoBody) {
		return nil, err
	}
	if status != "" {
		curState = status
	}

	return &Poller[T]{
		pl:         pl,
		resp:       resp,
		OpLocURL:   opURL,
		LocURL:     locURL,
		OrigURL:    resp.Request.URL.String(),
		Method:     resp.Request.Method,
		FinalState: finalState,
		CurState:   curState,
	}, nil
}

func (p *Poller[T]) Done() bool {
	return pollers.IsTerminalState(p.CurState)
}

func (p *Poller[T]) Poll(ctx context.Context) (*http.Response, error) {
	err := pollers.PollHelper(ctx, p.OpLocURL, p.pl, func(resp *http.Response) (string, error) {
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
	var req *exported.Request
	var err error
	if p.FinalState == pollers.FinalStateViaLocation && p.LocURL != "" {
		req, err = exported.NewRequest(ctx, http.MethodGet, p.LocURL)
	} else if p.FinalState == pollers.FinalStateViaOpLocation && p.Method == http.MethodPost {
		// no final GET required, terminal response should have it
	} else if rl, rlErr := pollers.GetResourceLocation(p.resp); rlErr != nil && !errors.Is(rlErr, pollers.ErrNoBody) {
		return rlErr
	} else if rl != "" {
		req, err = exported.NewRequest(ctx, http.MethodGet, rl)
	} else if p.Method == http.MethodPatch || p.Method == http.MethodPut {
		req, err = exported.NewRequest(ctx, http.MethodGet, p.OrigURL)
	} else if p.Method == http.MethodPost && p.LocURL != "" {
		req, err = exported.NewRequest(ctx, http.MethodGet, p.LocURL)
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
