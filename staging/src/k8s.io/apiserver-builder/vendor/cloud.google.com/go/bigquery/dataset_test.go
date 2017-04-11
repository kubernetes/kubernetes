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
	"reflect"
	"testing"

	"golang.org/x/net/context"
)

// readServiceStub services read requests by returning data from an in-memory list of values.
type listTablesServiceStub struct {
	expectedProject, expectedDataset string
	values                           [][]*Table        // contains pages of tables.
	pageTokens                       map[string]string // maps incoming page token to returned page token.

	service
}

func (s *listTablesServiceStub) listTables(ctx context.Context, projectID, datasetID, pageToken string) ([]*Table, string, error) {
	if projectID != s.expectedProject {
		return nil, "", errors.New("wrong project id")
	}
	if datasetID != s.expectedDataset {
		return nil, "", errors.New("wrong dataset id")
	}

	tables := s.values[0]
	s.values = s.values[1:]
	return tables, s.pageTokens[pageToken], nil
}

func TestListTables(t *testing.T) {
	t1 := &Table{ProjectID: "p1", DatasetID: "d1", TableID: "t1"}
	t2 := &Table{ProjectID: "p1", DatasetID: "d1", TableID: "t2"}
	t3 := &Table{ProjectID: "p1", DatasetID: "d1", TableID: "t3"}
	testCases := []struct {
		data       [][]*Table
		pageTokens map[string]string
		want       []*Table
	}{
		{
			data:       [][]*Table{{t1, t2}, {t3}},
			pageTokens: map[string]string{"": "a", "a": ""},
			want:       []*Table{t1, t2, t3},
		},
		{
			data:       [][]*Table{{t1, t2}, {t3}},
			pageTokens: map[string]string{"": ""}, // no more pages after first one.
			want:       []*Table{t1, t2},
		},
	}

	for _, tc := range testCases {
		c := &Client{
			service: &listTablesServiceStub{
				expectedProject: "x",
				expectedDataset: "y",
				values:          tc.data,
				pageTokens:      tc.pageTokens,
			},
			projectID: "x",
		}
		got, err := c.Dataset("y").ListTables(context.Background())
		if err != nil {
			t.Errorf("err calling ListTables: %v", err)
			continue
		}

		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("reading: got:\n%v\nwant:\n%v", got, tc.want)
		}
	}
}

func TestListTablesError(t *testing.T) {
	c := &Client{
		service: &listTablesServiceStub{
			expectedProject: "x",
			expectedDataset: "y",
		},
		projectID: "x",
	}
	// Test that service read errors are propagated back to the caller.
	// Passing "not y" as the dataset id will cause the service to return an error.
	_, err := c.Dataset("not y").ListTables(context.Background())
	if err == nil {
		// Read should not return an error; only Err should.
		t.Errorf("ListTables expected: non-nil err, got: nil")
	}
}
