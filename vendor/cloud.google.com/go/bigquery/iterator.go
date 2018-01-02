// Copyright 2015 Google Inc. All Rights Reserved.
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

package bigquery

import (
	"errors"
	"fmt"

	"golang.org/x/net/context"
)

// A pageFetcher returns a page of rows, starting from the row specified by token.
type pageFetcher interface {
	fetch(ctx context.Context, s service, token string) (*readDataResult, error)
}

// Iterator provides access to the result of a BigQuery lookup.
// Next must be called before the first call to Get.
type Iterator struct {
	service service

	err error // contains any error encountered during calls to Next.

	// Once Next has been called at least once, schema has the result schema, rs contains the current
	// page of data, and nextToken contains the token for fetching the next
	// page (empty if there is no more data to be fetched).
	schema    Schema
	rs        [][]Value
	nextToken string

	// The remaining fields contain enough information to fetch the current
	// page of data, and determine which row of data from this page is the
	// current row.

	pf        pageFetcher
	pageToken string

	// The offset from the start of the current page to the current row.
	// For a new iterator, this is -1.
	offset int64
}

func newIterator(s service, pf pageFetcher) *Iterator {
	return &Iterator{
		service: s,
		pf:      pf,
		offset:  -1,
	}
}

// fetchPage loads the current page of data from the server.
// The contents of rs and nextToken are replaced with the loaded data.
// If there is an error while fetching, the error is stored in it.err and false is returned.
func (it *Iterator) fetchPage(ctx context.Context) bool {
	var res *readDataResult
	var err error
	for {
		res, err = it.pf.fetch(ctx, it.service, it.pageToken)
		if err != errIncompleteJob {
			break
		}
	}

	if err != nil {
		it.err = err
		return false
	}

	it.schema = res.schema
	it.rs = res.rows
	it.nextToken = res.pageToken
	return true
}

// getEnoughData loads new data into rs until offset no longer points beyond the end of rs.
func (it *Iterator) getEnoughData(ctx context.Context) bool {
	if len(it.rs) == 0 {
		// Either we have not yet fetched any pages, or we are iterating over an empty dataset.
		// In the former case, we should fetch a page of data, so that we can depend on the resultant nextToken.
		// In the latter case, it is harmless to fetch a page of data.
		if !it.fetchPage(ctx) {
			return false
		}
	}

	for it.offset >= int64(len(it.rs)) {
		// If offset is still outside the bounds of the loaded data,
		// but there are no more pages of data to fetch, then we have
		// failed to satisfy the offset.
		if it.nextToken == "" {
			return false
		}

		// offset cannot be satisfied with the currently loaded data,
		// so we fetch the next page.  We no longer need the existing
		// cached rows, so we remove them and update the offset to be
		// relative to the new page that we're about to fetch.
		// NOTE: we can't just set offset to 0, because after
		// marshalling/unmarshalling, it's possible for the offset to
		// point arbitrarily far beyond the end of rs.
		// This can happen if the server returns a different size
		// results page before and after marshalling.
		it.offset -= int64(len(it.rs))
		it.pageToken = it.nextToken
		if !it.fetchPage(ctx) {
			return false
		}
	}
	return true
}

// Next advances the Iterator to the next row, making that row available
// via the Get method.
// Next must be called before the first call to Get or Schema, and blocks until data is available.
// Next returns false when there are no more rows available, either because
// the end of the output was reached, or because there was an error (consult
// the Err method to determine which).
func (it *Iterator) Next(ctx context.Context) bool {
	if it.err != nil {
		return false
	}

	// Advance offset to where we want it to be for the next call to Get.
	it.offset++

	// offset may now point beyond the end of rs, so we fetch data
	// until offset is within its bounds again.  If there are no more
	// results available, offset will be left pointing beyond the bounds
	// of rs.
	// At the end of this method, rs will contain at least one element
	// unless the dataset we are iterating over is empty.
	return it.getEnoughData(ctx)
}

// Err returns the last error encountered by Next, or nil for no error.
func (it *Iterator) Err() error {
	return it.err
}

// verifyState checks that the iterator is pointing to a valid row.
func (it *Iterator) verifyState() error {
	if it.err != nil {
		return fmt.Errorf("called on iterator in error state: %v", it.err)
	}

	// If Next has been called, then offset should always index into a
	// valid row in rs, as long as there is still data available.
	if it.offset >= int64(len(it.rs)) || it.offset < 0 {
		return errors.New("called without preceding successful call to Next")
	}

	return nil
}

// Get loads the current row into dst, which must implement ValueLoader.
func (it *Iterator) Get(dst interface{}) error {
	if err := it.verifyState(); err != nil {
		return fmt.Errorf("Get %v", err)
	}

	if dst, ok := dst.(ValueLoader); ok {
		return dst.Load(it.rs[it.offset])
	}
	return errors.New("Get called with unsupported argument type")
}

// Schema returns the schema of the result rows.
func (it *Iterator) Schema() (Schema, error) {
	if err := it.verifyState(); err != nil {
		return nil, fmt.Errorf("Schema %v", err)
	}

	return it.schema, nil
}
