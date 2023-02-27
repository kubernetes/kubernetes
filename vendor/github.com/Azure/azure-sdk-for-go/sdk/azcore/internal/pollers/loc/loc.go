//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package loc

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

// Kind is the identifier of this type in a resume token.
const kind = "loc"

// Applicable returns true if the LRO is using Location.
func Applicable(resp *http.Response) bool {
	return resp.Header.Get(shared.HeaderLocation) != ""
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

// Poller is an LRO poller that uses the Location pattern.
type Poller[T any] struct {
	pl   exported.Pipeline
	resp *http.Response

	Type     string `json:"type"`
	PollURL  string `json:"pollURL"`
	CurState string `json:"state"`
}

// New creates a new Poller from the provided initial response.
// Pass nil for response to create an empty Poller for rehydration.
func New[T any](pl exported.Pipeline, resp *http.Response) (*Poller[T], error) {
	if resp == nil {
		log.Write(log.EventLRO, "Resuming Location poller.")
		return &Poller[T]{pl: pl}, nil
	}
	log.Write(log.EventLRO, "Using Location poller.")
	locURL := resp.Header.Get(shared.HeaderLocation)
	if locURL == "" {
		return nil, errors.New("response is missing Location header")
	}
	if !pollers.IsValidURL(locURL) {
		return nil, fmt.Errorf("invalid polling URL %s", locURL)
	}
	return &Poller[T]{
		pl:       pl,
		resp:     resp,
		Type:     kind,
		PollURL:  locURL,
		CurState: pollers.StatusInProgress,
	}, nil
}

func (p *Poller[T]) Done() bool {
	return pollers.IsTerminalState(p.CurState)
}

func (p *Poller[T]) Poll(ctx context.Context) (*http.Response, error) {
	err := pollers.PollHelper(ctx, p.PollURL, p.pl, func(resp *http.Response) (string, error) {
		// location polling can return an updated polling URL
		if h := resp.Header.Get(shared.HeaderLocation); h != "" {
			p.PollURL = h
		}
		// if provisioning state is available, use that.  this is only
		// for some ARM LRO scenarios (e.g. DELETE with a Location header)
		// so if it's missing then use HTTP status code.
		provState, _ := pollers.GetProvisioningState(resp)
		p.resp = resp
		if provState != "" {
			p.CurState = provState
		} else if resp.StatusCode == http.StatusAccepted {
			p.CurState = pollers.StatusInProgress
		} else if resp.StatusCode > 199 && resp.StatusCode < 300 {
			// any 2xx other than a 202 indicates success
			p.CurState = pollers.StatusSucceeded
		} else {
			p.CurState = pollers.StatusFailed
		}
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
