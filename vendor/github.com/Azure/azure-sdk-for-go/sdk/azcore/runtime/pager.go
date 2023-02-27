//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package runtime

import (
	"context"
	"encoding/json"
	"errors"
)

// PagingHandler contains the required data for constructing a Pager.
type PagingHandler[T any] struct {
	// More returns a boolean indicating if there are more pages to fetch.
	// It uses the provided page to make the determination.
	More func(T) bool

	// Fetcher fetches the first and subsequent pages.
	Fetcher func(context.Context, *T) (T, error)
}

// Pager provides operations for iterating over paged responses.
type Pager[T any] struct {
	current   *T
	handler   PagingHandler[T]
	firstPage bool
}

// NewPager creates an instance of Pager using the specified PagingHandler.
// Pass a non-nil T for firstPage if the first page has already been retrieved.
func NewPager[T any](handler PagingHandler[T]) *Pager[T] {
	return &Pager[T]{
		handler:   handler,
		firstPage: true,
	}
}

// More returns true if there are more pages to retrieve.
func (p *Pager[T]) More() bool {
	if p.current != nil {
		return p.handler.More(*p.current)
	}
	return true
}

// NextPage advances the pager to the next page.
func (p *Pager[T]) NextPage(ctx context.Context) (T, error) {
	var resp T
	var err error
	if p.current != nil {
		if p.firstPage {
			// we get here if it's an LRO-pager, we already have the first page
			p.firstPage = false
			return *p.current, nil
		} else if !p.handler.More(*p.current) {
			return *new(T), errors.New("no more pages")
		}
		resp, err = p.handler.Fetcher(ctx, p.current)
	} else {
		// non-LRO case, first page
		p.firstPage = false
		resp, err = p.handler.Fetcher(ctx, nil)
	}
	if err != nil {
		return *new(T), err
	}
	p.current = &resp
	return *p.current, nil
}

// UnmarshalJSON implements the json.Unmarshaler interface for Pager[T].
func (p *Pager[T]) UnmarshalJSON(data []byte) error {
	return json.Unmarshal(data, &p.current)
}
