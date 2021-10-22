// Copyright 2016 Google LLC.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package iterator_test

import (
	"context"
	"encoding/json"
	"math"
	"reflect"
	"testing"

	"google.golang.org/api/iterator"
	itest "google.golang.org/api/iterator/testing"
)

// Service represents the implementation of a Google API's List method.
// We want to test against a large range of possible valid behaviors.
// All the behaviors this can generate are valid under the spec for
// Google API paging.
type service struct {
	// End of the sequence. end-1 is the last value returned.
	end int
	// Maximum number of items to return in one RPC. Also the default page size.
	// If zero, max is unlimited.
	max int
	// If true, return two empty pages before each RPC that returns items, and
	// two zero pages at the end. E.g. if end = 5, max = 2 and the pageSize
	// parameter to List is zero, then the number of items returned in
	// successive RPCS is:
	//    0 0 2 0 0 2 0 0 1 0 0
	// Note that this implies that the RPC returning the last items will have a
	// non-empty page token.
	zeroes bool
}

// List simulates an API List RPC. It returns integers in the range [0, s.end).
func (s *service) List(pageSize int, pageToken string) ([]int, string, error) {
	max := s.max
	if max == 0 {
		max = math.MaxInt64
	}
	// Never give back any more than s.max.
	if pageSize <= 0 || pageSize > max {
		pageSize = max
	}
	state := &listState{}
	if pageToken != "" {
		if err := json.Unmarshal([]byte(pageToken), state); err != nil {
			return nil, "", err
		}
	}
	ints := state.advance(pageSize, s.end, s.zeroes)
	if state.Start == s.end && (!s.zeroes || state.NumZeroes == 2) {
		pageToken = ""
	} else {
		bytes, err := json.Marshal(state)
		if err != nil {
			return nil, "", err
		}
		pageToken = string(bytes)
	}
	return ints, pageToken, nil
}

type listState struct {
	Start     int // where to start this page
	NumZeroes int // number of consecutive empty pages before this
}

func (s *listState) advance(pageSize, end int, zeroes bool) []int {
	var page []int
	if zeroes && s.NumZeroes != 2 {
		// Return a zero page.
	} else {
		for i := s.Start; i < end && len(page) < pageSize; i++ {
			page = append(page, i)
		}
	}
	s.Start += len(page)
	if len(page) == 0 {
		s.NumZeroes++
	} else {
		s.NumZeroes = 0
	}
	return page
}

func TestServiceList(t *testing.T) {
	for _, test := range []struct {
		svc      service
		pageSize int
		want     [][]int
	}{
		{service{end: 0}, 0, [][]int{nil}},
		{service{end: 5}, 0, [][]int{{0, 1, 2, 3, 4}}},
		{service{end: 5}, 8, [][]int{{0, 1, 2, 3, 4}}},
		{service{end: 5}, 2, [][]int{{0, 1}, {2, 3}, {4}}},
		{service{end: 5, max: 2}, 0, [][]int{{0, 1}, {2, 3}, {4}}},
		{service{end: 5, max: 2}, 1, [][]int{{0}, {1}, {2}, {3}, {4}}},
		{service{end: 5, max: 2}, 10, [][]int{{0, 1}, {2, 3}, {4}}},
		{service{end: 5, zeroes: true}, 0, [][]int{nil, nil, {0, 1, 2, 3, 4}, nil, nil}},
		{service{end: 5, max: 3, zeroes: true}, 0, [][]int{nil, nil, {0, 1, 2}, nil, nil, {3, 4}, nil, nil}},
	} {
		var got [][]int
		token := ""
		for {
			items, nextToken, err := test.svc.List(test.pageSize, token)
			if err != nil {
				t.Fatalf("%v, %d: %v", test.svc, test.pageSize, err)
			}
			got = append(got, items)
			if nextToken == "" {
				break
			}
			token = nextToken
		}
		if !reflect.DeepEqual(got, test.want) {
			t.Errorf("%v, %d: got %v, want %v", test.svc, test.pageSize, got, test.want)
		}
	}
}

type Client struct{ s *service }

// ItemIterator is a sample implementation of a standard iterator.
type ItemIterator struct {
	pageInfo *iterator.PageInfo
	nextFunc func() error
	s        *service
	items    []int
}

// PageInfo returns a PageInfo, which supports pagination.
func (it *ItemIterator) PageInfo() *iterator.PageInfo { return it.pageInfo }

// Items is a sample implementation of an iterator-creating method.
func (c *Client) Items(ctx context.Context) *ItemIterator {
	it := &ItemIterator{s: c.s}
	it.pageInfo, it.nextFunc = iterator.NewPageInfo(
		it.fetch,
		func() int { return len(it.items) },
		func() interface{} { b := it.items; it.items = nil; return b })
	return it
}

func (it *ItemIterator) fetch(pageSize int, pageToken string) (string, error) {
	items, tok, err := it.s.List(pageSize, pageToken)
	it.items = append(it.items, items...)
	return tok, err
}

func (it *ItemIterator) Next() (int, error) {
	if err := it.nextFunc(); err != nil {
		return 0, err
	}
	item := it.items[0]
	it.items = it.items[1:]
	return item, nil
}

func TestNext(t *testing.T) {
	// Test the iterator's Next method with a variety of different service behaviors.
	// This is primarily a test of PageInfo.next.
	for _, svc := range []service{
		{end: 0},
		{end: 5},
		{end: 5, max: 1},
		{end: 5, max: 2},
		{end: 5, zeroes: true},
		{end: 5, max: 2, zeroes: true},
	} {
		client := &Client{&svc}

		msg, ok := itest.TestIterator(
			seq(0, svc.end),
			func() interface{} { return client.Items(ctx) },
			func(it interface{}) (interface{}, error) { return it.(*ItemIterator).Next() })
		if !ok {
			t.Errorf("%+v: %s", svc, msg)
		}
	}
}

// TODO(jba): test setting PageInfo.MaxSize
// TODO(jba): test setting PageInfo.Token

// Verify that, for an iterator that uses PageInfo.next to implement its Next
// method, using Next and NextPage together result in an error.
func TestNextWithNextPage(t *testing.T) {
	client := &Client{&service{end: 11}}
	var items []int

	// Calling Next before NextPage.
	it := client.Items(ctx)
	it.Next()
	_, err := iterator.NewPager(it, 1, "").NextPage(&items)
	if err == nil {
		t.Error("NextPage after Next: got nil, want error")
	}
	_, err = it.Next()
	if err == nil {
		t.Error("Next after NextPage: got nil, want error")
	}

	// Next between two calls to NextPage.
	it = client.Items(ctx)
	p := iterator.NewPager(it, 1, "")
	p.NextPage(&items)
	_, err = it.Next()
	if err == nil {
		t.Error("Next after NextPage: got nil, want error")
	}
	_, err = p.NextPage(&items)
	if err == nil {
		t.Error("second NextPage after Next: got nil, want error")
	}
}

// Verify that we turn various potential reflection panics into errors.
func TestNextPageReflectionErrors(t *testing.T) {
	client := &Client{&service{end: 1}}
	p := iterator.NewPager(client.Items(ctx), 1, "")

	// Passing the nil interface value.
	_, err := p.NextPage(nil)
	if err == nil {
		t.Error("nil: got nil, want error")
	}

	// Passing a non-slice.
	_, err = p.NextPage(17)
	if err == nil {
		t.Error("non-slice: got nil, want error")
	}

	// Passing a slice of the wrong type.
	var bools []bool
	_, err = p.NextPage(&bools)
	if err == nil {
		t.Error("wrong type: got nil, want error")
	}

	// Using a slice of the right type, but not passing a pointer to it.
	var ints []int
	_, err = p.NextPage(ints)
	if err == nil {
		t.Error("not a pointer: got nil, want error")
	}
}

// seq returns a slice containing the values in [from, to).
func seq(from, to int) []int {
	var r []int
	for i := from; i < to; i++ {
		r = append(r, i)
	}
	return r
}
