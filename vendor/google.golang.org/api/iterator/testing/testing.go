// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package testing provides support functions for testing iterators conforming
// to the standard pattern.
// See package google.golang.org/api/iterator and
// https://github.com/GoogleCloudPlatform/gcloud-golang/wiki/Iterator-Guidelines.
package testing

import (
	"fmt"
	"reflect"

	"google.golang.org/api/iterator"
)

// TestIterator tests the Next method of a standard iterator. It assumes that
// the underlying sequence to be iterated over already exists.
//
// The want argument should be a slice that contains the elements of this
// sequence. It may be an empty slice, but it must not be the nil interface
// value. The elements must be comparable with reflect.DeepEqual.
//
// The create function should create and return a new iterator.
// It will typically look like
//    func() interface{} { return client.Items(ctx) }
//
// The next function takes the return value of create and should return the
// result of calling Next on the iterator. It can usually be defined as
//     func(it interface{}) (interface{}, error) { return it.(*ItemIterator).Next() }
//
// TestIterator checks that the iterator returns all the elements of want
// in order, followed by (zero, done). It also confirms that subsequent calls
// to next also return (zero, done).
//
// If the iterator implements the method
//     PageInfo() *iterator.PageInfo
// then exact pagination with iterator.Pager is also tested. Pagination testing
// will be more informative if the want slice contains at least three elements.
//
// On success, TestIterator returns ("", true). On failure, it returns a
// suitable error message and false.
func TestIterator(want interface{}, create func() interface{}, next func(interface{}) (interface{}, error)) (string, bool) {
	vWant := reflect.ValueOf(want)
	if vWant.Kind() != reflect.Slice {
		return "'want' must be a slice", false
	}
	it := create()
	msg, ok := testNext(vWant, it, next)
	if !ok {
		return msg, ok
	}
	if _, ok := it.(iterator.Pageable); !ok || vWant.Len() == 0 {
		return "", true
	}
	return testPaging(vWant, create, next)
}

// Check that the iterator returns vWant, the desired sequence.
func testNext(vWant reflect.Value, it interface{}, next func(interface{}) (interface{}, error)) (string, bool) {
	for i := 0; i < vWant.Len(); i++ {
		got, err := next(it)
		if err != nil {
			return fmt.Sprintf("#%d: got %v, expected an item", i, err), false
		}
		w := vWant.Index(i).Interface()
		if !reflect.DeepEqual(got, w) {
			return fmt.Sprintf("#%d: got %+v, want %+v", i, got, w), false
		}
	}
	// We now should see (<zero value of item type>, done), no matter how many
	// additional calls we make.
	zero := reflect.Zero(vWant.Type().Elem()).Interface()
	for i := 0; i < 3; i++ {
		got, err := next(it)
		if err != iterator.Done {
			return fmt.Sprintf("at end: got error %v, want iterator.Done", err), false
		}
		// Since err == iterator.Done, got should be zero.
		if got != zero {
			return fmt.Sprintf("got %+v with done, want zero %T", got, zero), false
		}
	}
	return "", true
}

// Test the iterator's behavior when used with iterator.Pager.
func testPaging(vWant reflect.Value, create func() interface{}, next func(interface{}) (interface{}, error)) (string, bool) {
	// Test page sizes that are smaller, equal to, and greater than the length
	// of the expected sequence.
	for _, pageSize := range []int{1, 2, vWant.Len(), vWant.Len() + 10} {
		wantPages := wantedPages(vWant, pageSize)
		// Test the Pager in two ways.
		// First, by creating a single Pager and calling NextPage in a loop,
		// ignoring the page token except for detecting the end of the
		// iteration.
		it := create().(iterator.Pageable)
		pager := iterator.NewPager(it, pageSize, "")
		msg, ok := testPager(fmt.Sprintf("ignore page token, pageSize %d", pageSize),
			vWant.Type(), wantPages,
			func(_ string, pagep interface{}) (string, error) {
				return pager.NextPage(pagep)
			})
		if !ok {
			return msg, false
		}
		// Second, by creating a new Pager for each page, passing in the page
		// token from the previous page, as would be done in a web handler.
		it = create().(iterator.Pageable)
		msg, ok = testPager(fmt.Sprintf("use page token, pageSize %d", pageSize),
			vWant.Type(), wantPages,
			func(pageToken string, pagep interface{}) (string, error) {
				return iterator.NewPager(it, pageSize, pageToken).NextPage(pagep)
			})
		if !ok {
			return msg, false
		}
	}
	return "", true
}

// Create the pages we expect to see.
func wantedPages(vWant reflect.Value, pageSize int) []interface{} {
	var pages []interface{}
	for i, j := 0, pageSize; i < vWant.Len(); i, j = j, j+pageSize {
		if j > vWant.Len() {
			j = vWant.Len()
		}
		pages = append(pages, vWant.Slice(i, j).Interface())
	}
	return pages
}

func testPager(prefix string, sliceType reflect.Type, wantPages []interface{},
	nextPage func(pageToken string, pagep interface{}) (string, error)) (string, bool) {
	tok := ""
	var err error
	for i := 0; i < len(wantPages)+1; i++ {
		vpagep := reflect.New(sliceType)
		tok, err = nextPage(tok, vpagep.Interface())
		if err != nil {
			return fmt.Sprintf("%s, page #%d: got error %v", prefix, i, err), false
		}
		if i == len(wantPages) {
			// Allow one empty page at the end.
			if vpagep.Elem().Len() != 0 || tok != "" {
				return fmt.Sprintf("%s: did not get one empty page at end", prefix), false
			}
			break
		}
		if msg, ok := compareSlices(vpagep.Elem(), reflect.ValueOf(wantPages[i])); !ok {
			return fmt.Sprintf("%s, page #%d:\n%s", prefix, i, msg), false
		}
		if tok == "" {
			if i != len(wantPages)-1 {
				return fmt.Sprintf("%s, page #%d: got empty page token", prefix, i), false
			}
			break
		}
	}
	return "", true
}

// Compare two slices element-by-element. If they are equal, return ("", true).
// Otherwise, return a description of the difference and false.
func compareSlices(vgot, vwant reflect.Value) (string, bool) {
	if got, want := vgot.Len(), vwant.Len(); got != want {
		return fmt.Sprintf("got %d items, want %d", got, want), false
	}
	for i := 0; i < vgot.Len(); i++ {
		if got, want := vgot.Index(i).Interface(), vwant.Index(i).Interface(); !reflect.DeepEqual(got, want) {
			return fmt.Sprintf("got[%d] = %+v\nwant   = %+v", i, got, want), false
		}
	}
	return "", true
}
