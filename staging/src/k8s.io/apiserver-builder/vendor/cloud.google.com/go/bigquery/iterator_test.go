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
	"reflect"
	"testing"

	"golang.org/x/net/context"
)

type fetchResponse struct {
	result *readDataResult // The result to return.
	err    error           // The error to return.
}

// pageFetcherStub services fetch requests by returning data from an in-memory list of values.
type pageFetcherStub struct {
	fetchResponses map[string]fetchResponse

	err error
}

func (pf *pageFetcherStub) fetch(ctx context.Context, s service, token string) (*readDataResult, error) {
	call, ok := pf.fetchResponses[token]
	if !ok {
		pf.err = fmt.Errorf("Unexpected page token: %q", token)
	}
	return call.result, call.err
}

func TestIterator(t *testing.T) {
	fetchFailure := errors.New("fetch failure")

	testCases := []struct {
		desc            string
		alreadyConsumed int64 // amount to advance offset before commencing reading.
		fetchResponses  map[string]fetchResponse
		want            []ValueList
		wantErr         error
		wantSchema      Schema
	}{
		{
			desc: "Iteration over single empty page",
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{},
						schema:    Schema{},
					},
				},
			},
			want:       []ValueList{},
			wantSchema: Schema{},
		},
		{
			desc: "Iteration over single page",
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{{1, 2}, {11, 12}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
			},
			want: []ValueList{{1, 2}, {11, 12}},
			wantSchema: Schema{
				{Type: IntegerFieldType},
				{Type: IntegerFieldType},
			},
		},
		{
			desc: "Iteration over single page with different schema",
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{{"1", 2}, {"11", 12}},
						schema: Schema{
							{Type: StringFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
			},
			want: []ValueList{{"1", 2}, {"11", 12}},
			wantSchema: Schema{
				{Type: StringFieldType},
				{Type: IntegerFieldType},
			},
		},
		{
			desc: "Iteration over two pages",
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "a",
						rows:      [][]Value{{1, 2}, {11, 12}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
				"a": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{{101, 102}, {111, 112}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
			},
			want: []ValueList{{1, 2}, {11, 12}, {101, 102}, {111, 112}},
			wantSchema: Schema{
				{Type: IntegerFieldType},
				{Type: IntegerFieldType},
			},
		},
		{
			desc: "Server response includes empty page",
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "a",
						rows:      [][]Value{{1, 2}, {11, 12}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
				"a": {
					result: &readDataResult{
						pageToken: "b",
						rows:      [][]Value{},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
				"b": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{{101, 102}, {111, 112}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
			},
			want: []ValueList{{1, 2}, {11, 12}, {101, 102}, {111, 112}},
			wantSchema: Schema{
				{Type: IntegerFieldType},
				{Type: IntegerFieldType},
			},
		},
		{
			desc: "Fetch error",
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "a",
						rows:      [][]Value{{1, 2}, {11, 12}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
				"a": {
					// We returns some data from this fetch, but also an error.
					// So the end result should include only data from the previous fetch.
					err: fetchFailure,
					result: &readDataResult{
						pageToken: "b",
						rows:      [][]Value{{101, 102}, {111, 112}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
			},
			want:    []ValueList{{1, 2}, {11, 12}},
			wantErr: fetchFailure,
			wantSchema: Schema{
				{Type: IntegerFieldType},
				{Type: IntegerFieldType},
			},
		},
		{
			desc:            "Skip over a single element",
			alreadyConsumed: 1,
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "a",
						rows:      [][]Value{{1, 2}, {11, 12}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
				"a": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{{101, 102}, {111, 112}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
			},
			want: []ValueList{{11, 12}, {101, 102}, {111, 112}},
			wantSchema: Schema{
				{Type: IntegerFieldType},
				{Type: IntegerFieldType},
			},
		},
		{
			desc:            "Skip over an entire page",
			alreadyConsumed: 2,
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "a",
						rows:      [][]Value{{1, 2}, {11, 12}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
				"a": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{{101, 102}, {111, 112}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
			},
			want: []ValueList{{101, 102}, {111, 112}},
			wantSchema: Schema{
				{Type: IntegerFieldType},
				{Type: IntegerFieldType},
			},
		},
		{
			desc:            "Skip beyond start of second page",
			alreadyConsumed: 3,
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "a",
						rows:      [][]Value{{1, 2}, {11, 12}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
				"a": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{{101, 102}, {111, 112}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
			},
			want: []ValueList{{111, 112}},
			wantSchema: Schema{
				{Type: IntegerFieldType},
				{Type: IntegerFieldType},
			},
		},
		{
			desc:            "Skip beyond all data",
			alreadyConsumed: 4,
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "a",
						rows:      [][]Value{{1, 2}, {11, 12}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
				"a": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{{101, 102}, {111, 112}},
						schema: Schema{
							{Type: IntegerFieldType},
							{Type: IntegerFieldType},
						},
					},
				},
			},
			// In this test case, Next will return false on its first call,
			// so we won't even attempt to call Get.
			want:       []ValueList{},
			wantSchema: Schema{},
		},
	}

	for _, tc := range testCases {
		pf := &pageFetcherStub{
			fetchResponses: tc.fetchResponses,
		}
		it := newIterator(nil, pf)
		it.offset += tc.alreadyConsumed

		values, schema, err := consumeIterator(it)
		if err != nil {
			t.Fatalf("%s: %v", tc.desc, err)
		}

		if (len(values) != 0 || len(tc.want) != 0) && !reflect.DeepEqual(values, tc.want) {
			t.Errorf("%s: values:\ngot: %v\nwant:%v", tc.desc, values, tc.want)
		}
		if it.Err() != tc.wantErr {
			t.Errorf("%s: iterator.Err:\ngot: %v\nwant: %v", tc.desc, it.Err(), tc.wantErr)
		}
		if (len(schema) != 0 || len(tc.wantSchema) != 0) && !reflect.DeepEqual(schema, tc.wantSchema) {
			t.Errorf("%s: iterator.Schema:\ngot: %v\nwant: %v", tc.desc, schema, tc.wantSchema)
		}
	}
}

// consumeIterator reads the schema and all values from an iterator and returns them.
func consumeIterator(it *Iterator) ([]ValueList, Schema, error) {
	var got []ValueList
	var schema Schema
	for it.Next(context.Background()) {
		var vals ValueList
		var err error
		if err = it.Get(&vals); err != nil {
			return nil, Schema{}, fmt.Errorf("err calling Get: %v", err)
		}
		got = append(got, vals)
		if schema, err = it.Schema(); err != nil {
			return nil, Schema{}, fmt.Errorf("err calling Schema: %v", err)
		}
	}

	return got, schema, nil
}

func TestGetBeforeNext(t *testing.T) {
	// TODO: once mashalling/unmarshalling of iterators is implemented, do a similar test for unmarshalled iterators.
	pf := &pageFetcherStub{
		fetchResponses: map[string]fetchResponse{
			"": {
				result: &readDataResult{
					pageToken: "",
					rows:      [][]Value{{1, 2}, {11, 12}},
				},
			},
		},
	}
	it := newIterator(nil, pf)
	var vals ValueList
	if err := it.Get(&vals); err == nil {
		t.Errorf("Expected error calling Get before Next")
	}
}

type delayedPageFetcher struct {
	pageFetcherStub
	delayCount int
}

func (pf *delayedPageFetcher) fetch(ctx context.Context, s service, token string) (*readDataResult, error) {
	if pf.delayCount > 0 {
		pf.delayCount--
		return nil, errIncompleteJob
	}
	return pf.pageFetcherStub.fetch(ctx, s, token)
}

func TestIterateIncompleteJob(t *testing.T) {
	want := []ValueList{{1, 2}, {11, 12}, {101, 102}, {111, 112}}
	pf := pageFetcherStub{
		fetchResponses: map[string]fetchResponse{
			"": {
				result: &readDataResult{
					pageToken: "a",
					rows:      [][]Value{{1, 2}, {11, 12}},
				},
			},
			"a": {
				result: &readDataResult{
					pageToken: "",
					rows:      [][]Value{{101, 102}, {111, 112}},
				},
			},
		},
	}
	dpf := &delayedPageFetcher{
		pageFetcherStub: pf,
		delayCount:      1,
	}
	it := newIterator(nil, dpf)

	values, _, err := consumeIterator(it)
	if err != nil {
		t.Fatal(err)
	}

	if (len(values) != 0 || len(want) != 0) && !reflect.DeepEqual(values, want) {
		t.Errorf("values: got:\n%v\nwant:\n%v", values, want)
	}
	if it.Err() != nil {
		t.Fatalf("iterator.Err: got:\n%v", it.Err())
	}
	if dpf.delayCount != 0 {
		t.Errorf("delayCount: got: %v, want: 0", dpf.delayCount)
	}
}

func TestGetDuringErrorState(t *testing.T) {
	pf := &pageFetcherStub{
		fetchResponses: map[string]fetchResponse{
			"": {err: errors.New("bang")},
		},
	}
	it := newIterator(nil, pf)
	var vals ValueList
	it.Next(context.Background())
	if it.Err() == nil {
		t.Errorf("Expected error after calling Next")
	}
	if err := it.Get(&vals); err == nil {
		t.Errorf("Expected error calling Get when iterator has a non-nil error.")
	}
}

func TestGetAfterFinished(t *testing.T) {
	testCases := []struct {
		alreadyConsumed int64 // amount to advance offset before commencing reading.
		fetchResponses  map[string]fetchResponse
		want            []ValueList
	}{
		{
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{{1, 2}, {11, 12}},
					},
				},
			},
			want: []ValueList{{1, 2}, {11, 12}},
		},
		{
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{},
					},
				},
			},
			want: []ValueList{},
		},
		{
			alreadyConsumed: 100,
			fetchResponses: map[string]fetchResponse{
				"": {
					result: &readDataResult{
						pageToken: "",
						rows:      [][]Value{{1, 2}, {11, 12}},
					},
				},
			},
			want: []ValueList{},
		},
	}

	for _, tc := range testCases {
		pf := &pageFetcherStub{
			fetchResponses: tc.fetchResponses,
		}
		it := newIterator(nil, pf)
		it.offset += tc.alreadyConsumed

		values, _, err := consumeIterator(it)
		if err != nil {
			t.Fatal(err)
		}

		if (len(values) != 0 || len(tc.want) != 0) && !reflect.DeepEqual(values, tc.want) {
			t.Errorf("values: got:\n%v\nwant:\n%v", values, tc.want)
		}
		if it.Err() != nil {
			t.Fatalf("iterator.Err: got:\n%v\nwant:\n:nil", it.Err())
		}
		// Try calling Get again.
		var vals ValueList
		if err := it.Get(&vals); err == nil {
			t.Errorf("Expected error calling Get when there are no more values")
		}
	}
}
