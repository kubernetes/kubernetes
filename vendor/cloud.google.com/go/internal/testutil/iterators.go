/*
Copyright 2016 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package testutil

import (
	"fmt"
	"reflect"
)

// TestIteratorNext tests the Next method of a standard client iterator (see
// https://github.com/GoogleCloudPlatform/google-cloud-go/wiki/Iterator-Guidelines).
//
// This function assumes that an iterator has already been created, and that
// the underlying sequence to be iterated over already exists. want should be a
// slice that contains the elements of this sequence. It must contain at least
// one element.
//
// done is the special error value that signals that the iterator has provided all the items.
//
// next is a function that returns the result of calling Next on the iterator. It can usually be
// defined as
//     func() (interface{}, error) { return iter.Next() }
//
// TestIteratorNext checks that the iterator returns all the elements of want
// in order, followed by (zero, done). It also confirms that subsequent calls
// to next also return (zero, done).
//
// On success, TestIteratorNext returns ("", true). On failure, it returns a
// suitable error message and false.
func TestIteratorNext(want interface{}, done error, next func() (interface{}, error)) (string, bool) {
	wVal := reflect.ValueOf(want)
	if wVal.Kind() != reflect.Slice {
		return "'want' must be a slice", false
	}
	for i := 0; i < wVal.Len(); i++ {
		got, err := next()
		if err != nil {
			return fmt.Sprintf("#%d: got %v, expected an item", i, err), false
		}
		w := wVal.Index(i).Interface()
		if !reflect.DeepEqual(got, w) {
			return fmt.Sprintf("#%d: got %+v, want %+v", i, got, w), false
		}
	}
	// We now should see (<zero value of item type>, done), no matter how many
	// additional calls we make.
	zero := reflect.Zero(wVal.Type().Elem()).Interface()
	for i := 0; i < 3; i++ {
		got, err := next()
		if err != done {
			return fmt.Sprintf("at end: got error %v, want done", err), false
		}
		// Since err == done, got should be zero.
		if got != zero {
			return fmt.Sprintf("got %+v with done, want zero %T", got, zero), false
		}
	}
	return "", true
}

// PagingIterator describes the standard client iterator pattern with paging as best as possible in Go.
// See https://github.com/GoogleCloudPlatform/google-cloud-go/wiki/Iterator-Guidelines.
type PagingIterator interface {
	SetPageSize(int)
	SetPageToken(string)
	NextPageToken() string
	// NextPage() ([]T, error)
}

// TestIteratorNextPageExact tests the NextPage method of a standard client
// iterator with paging (see PagingIterator).
//
// This function assumes that the underlying sequence to be iterated over
// already exists. want should be a slice that contains the elements of this
// sequence. It must contain at least three elements, in order to test
// non-trivial paging behavior.
//
// done is the special error value that signals that the iterator has provided all the items.
//
// defaultPageSize is the page size to use when the user has not called SetPageSize, or calls
// it with a value <= 0.
//
// newIter should return a new iterator each time it is called.
//
// nextPage should return the result of calling NextPage on the iterator. It can usually be
// defined as
//     func(i testutil.PagingIterator) (interface{}, error) { return i.(*<iteratorType>).NextPage() }
//
// TestIteratorNextPageExact checks that the iterator returns all the elements
// of want in order, divided into pages of the exactly the right size. It
// confirms that if the last page is partial, done is returned along with it,
// and in any case, done is returned subsequently along with a zero-length
// slice.
//
// On success, TestIteratorNextPageExact returns ("", true). On failure, it returns a
// suitable error message and false.
func TestIteratorNextPageExact(want interface{}, done error, defaultPageSize int, newIter func() PagingIterator, nextPage func(PagingIterator) (interface{}, error)) (string, bool) {
	wVal := reflect.ValueOf(want)
	if wVal.Kind() != reflect.Slice {
		return "'want' must be a slice", false
	}
	if wVal.Len() < 3 {
		return "need at least 3 values for 'want' to effectively test paging", false
	}
	const doNotSetPageSize = -999
	for _, pageSize := range []int{doNotSetPageSize, -7, 0, 1, 2, wVal.Len(), wVal.Len() + 10} {
		adjustedPageSize := int(pageSize)
		if pageSize <= 0 {
			adjustedPageSize = int(defaultPageSize)
		}
		// Create the pages we expect to see.
		var wantPages []interface{}
		for i, j := 0, adjustedPageSize; i < wVal.Len(); i, j = j, j+adjustedPageSize {
			if j > wVal.Len() {
				j = wVal.Len()
			}
			wantPages = append(wantPages, wVal.Slice(i, j).Interface())
		}
		for _, usePageToken := range []bool{false, true} {
			it := newIter()
			if pageSize != doNotSetPageSize {
				it.SetPageSize(pageSize)
			}
			for i, wantPage := range wantPages {
				gotPage, err := nextPage(it)
				if err != nil && err != done {
					return fmt.Sprintf("usePageToken %v, pageSize %d, #%d: got %v, expected a page",
						usePageToken, pageSize, i, err), false
				}
				if !reflect.DeepEqual(gotPage, wantPage) {
					return fmt.Sprintf("usePageToken %v, pageSize %d, #%d:\ngot  %v\nwant %+v",
						usePageToken, pageSize, i, gotPage, wantPage), false
				}
				// If the last page is partial, NextPage must return done.
				if reflect.ValueOf(gotPage).Len() < adjustedPageSize && err != done {
					return fmt.Sprintf("usePageToken %v, pageSize %d, #%d: expected done on partial page, got %v",
						usePageToken, pageSize, i, err), false
				}
				if usePageToken {
					// Pretend that we are displaying a paginated listing on the web, and the next
					// page may be served by a different process.
					// Empty page token implies done, and vice versa.
					if (it.NextPageToken() == "") != (err == done) {
						return fmt.Sprintf("pageSize %d: next page token = %q and err = %v; expected empty page token iff done",
							pageSize, it.NextPageToken(), err), false
					}
					if err == nil {
						token := it.NextPageToken()
						it = newIter()
						it.SetPageSize(pageSize)
						it.SetPageToken(token)
					}
				}
			}
			// We now should see (<zero-length or nil slice>, done), no matter how many
			// additional calls we make.
			for i := 0; i < 3; i++ {
				gotPage, err := nextPage(it)
				if err != done {
					return fmt.Sprintf("usePageToken %v, pageSize %d, at end: got error %v, want done",
						usePageToken, pageSize, err), false
				}
				pVal := reflect.ValueOf(gotPage)
				if pVal.Kind() != reflect.Slice || pVal.Len() != 0 {
					return fmt.Sprintf("usePageToken %v, pageSize %d, at end: got %+v with done, want zero-length slice",
						usePageToken, pageSize, gotPage), false
				}
			}
		}
	}
	return "", true
}
