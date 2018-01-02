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
	"fmt"
	"net/http"
	"reflect"
	"testing"
	"time"

	"cloud.google.com/go/internal/testutil"
	"golang.org/x/net/context"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
)

func TestIntegration(t *testing.T) {
	if testing.Short() {
		t.Skip("Integration tests skipped in short mode")
	}

	ctx := context.Background()
	ts := testutil.TokenSource(ctx, Scope)
	if ts == nil {
		t.Skip("Integration tests skipped. See CONTRIBUTING.md for details")
	}

	projID := testutil.ProjID()
	c, err := NewClient(ctx, projID, option.WithTokenSource(ts))
	if err != nil {
		t.Fatal(err)
	}
	ds := c.Dataset("bigquery_integration_test")
	if err := ds.Create(ctx); err != nil && !hasStatusCode(err, http.StatusConflict) { // AlreadyExists is 409
		t.Fatal(err)
	}
	schema := Schema([]*FieldSchema{
		{Name: "name", Type: StringFieldType},
		{Name: "num", Type: IntegerFieldType},
	})
	table := ds.Table("t1")
	// Delete the table in case it already exists. (Ignore errors.)
	table.Delete(ctx)
	// Create the table.
	err = table.Create(ctx, schema, TableExpiration(time.Now().Add(5*time.Minute)))
	if err != nil {
		t.Fatal(err)
	}
	// Check table metadata.
	md, err := table.Metadata(ctx)
	if err != nil {
		t.Fatal(err)
	}
	// TODO(jba): check md more thorougly.
	if got, want := md.ID, fmt.Sprintf("%s:%s.%s", projID, ds.id, table.TableID); got != want {
		t.Errorf("metadata.ID: got %q, want %q", got, want)
	}
	if got, want := md.Type, RegularTable; got != want {
		t.Errorf("metadata.Type: got %v, want %v", got, want)
	}

	// List tables in the dataset.
	tables, err := ds.ListTables(ctx)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := len(tables), 1; got != want {
		t.Fatalf("ListTables: got %d, want %d", got, want)
	}
	want := *table
	if got := tables[0]; !reflect.DeepEqual(got, &want) {
		t.Errorf("ListTables: got %v, want %v", got, &want)
	}

	// Populate the table.
	upl := table.NewUploader()
	var rows []*ValuesSaver
	for i, name := range []string{"a", "b", "c"} {
		rows = append(rows, &ValuesSaver{
			Schema:   schema,
			InsertID: name,
			Row:      []Value{name, i},
		})
	}
	if err := upl.Put(ctx, rows); err != nil {
		t.Fatal(err)
	}

	checkRead := func(src ReadSource) {
		it, err := c.Read(ctx, src)
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; it.Next(ctx); i++ {
			var vals ValueList
			if err := it.Get(&vals); err != nil {
				t.Fatal(err)
			}
			if got, want := vals, rows[i].Row; !reflect.DeepEqual([]Value(got), want) {
				t.Errorf("got %v, want %v", got, want)
			}
		}
	}
	// Read the table.
	checkRead(table)

	// Query the table.
	q := &Query{
		Q:                "select name, num from t1",
		DefaultProjectID: projID,
		DefaultDatasetID: ds.id,
	}
	checkRead(q)

	// Query the long way.
	dest := &Table{}
	job1, err := c.Copy(ctx, dest, q, WriteTruncate)
	if err != nil {
		t.Fatal(err)
	}
	job2, err := c.JobFromID(ctx, job1.ID())
	if err != nil {
		t.Fatal(err)
	}
	// TODO(jba): poll status until job is done
	_, err = job2.Status(ctx)
	if err != nil {
		t.Fatal(err)
	}

	checkRead(job2)

	// TODO(jba): patch the table
}

func hasStatusCode(err error, code int) bool {
	if e, ok := err.(*googleapi.Error); ok && e.Code == code {
		return true
	}
	return false
}
