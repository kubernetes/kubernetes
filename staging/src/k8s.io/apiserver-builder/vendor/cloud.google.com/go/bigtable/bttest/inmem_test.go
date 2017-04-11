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

package bttest

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/net/context"
	btapb "google.golang.org/genproto/googleapis/bigtable/admin/v2"
	btpb "google.golang.org/genproto/googleapis/bigtable/v2"
)

func TestConcurrentMutationsReadModifyAndGC(t *testing.T) {
	s := &server{
		tables: make(map[string]*table),
	}
	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := s.CreateTable(
		ctx,
		&btapb.CreateTableRequest{Parent: "cluster", TableId: "t"}); err != nil {
		t.Fatal(err)
	}
	const name = `cluster/tables/t`
	tbl := s.tables[name]
	req := &btapb.ModifyColumnFamiliesRequest{
		Name: name,
		Modifications: []*btapb.ModifyColumnFamiliesRequest_Modification{
			{
				Id:  "cf",
				Mod: &btapb.ModifyColumnFamiliesRequest_Modification_Create{Create: &btapb.ColumnFamily{}},
			},
		},
	}
	_, err := s.ModifyColumnFamilies(ctx, req)
	if err != nil {
		t.Fatal(err)
	}
	req = &btapb.ModifyColumnFamiliesRequest{
		Name: name,
		Modifications: []*btapb.ModifyColumnFamiliesRequest_Modification{
			{
				Id: "cf",
				Mod: &btapb.ModifyColumnFamiliesRequest_Modification_Update{
					Update: &btapb.ColumnFamily{GcRule: &btapb.GcRule{Rule: &btapb.GcRule_MaxNumVersions{MaxNumVersions: 1}}}},
			},
		},
	}
	if _, err := s.ModifyColumnFamilies(ctx, req); err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup
	var ts int64
	ms := func() []*btpb.Mutation {
		return []*btpb.Mutation{
			{
				Mutation: &btpb.Mutation_SetCell_{
					SetCell: &btpb.Mutation_SetCell{
						FamilyName:      "cf",
						ColumnQualifier: []byte(`col`),
						TimestampMicros: atomic.AddInt64(&ts, 1000),
					},
				},
			},
		}
	}

	rmw := func() *btpb.ReadModifyWriteRowRequest {
		return &btpb.ReadModifyWriteRowRequest{
			TableName: name,
			RowKey:    []byte(fmt.Sprint(rand.Intn(100))),
			Rules: []*btpb.ReadModifyWriteRule{
				{
					FamilyName:      "cf",
					ColumnQualifier: []byte("col"),
					Rule:            &btpb.ReadModifyWriteRule_IncrementAmount{IncrementAmount: 1},
				},
			},
		}
	}
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ctx.Err() == nil {
				req := &btpb.MutateRowRequest{
					TableName: name,
					RowKey:    []byte(fmt.Sprint(rand.Intn(100))),
					Mutations: ms(),
				}
				s.MutateRow(ctx, req)
			}
		}()
		wg.Add(1)
		go func() {
			defer wg.Done()
			for ctx.Err() == nil {
				_, _ = s.ReadModifyWriteRow(ctx, rmw())
			}
		}()

		wg.Add(1)
		go func() {
			defer wg.Done()
			tbl.gc()
		}()
	}
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(100 * time.Millisecond):
		t.Error("Concurrent mutations and GCs haven't completed after 100ms")
	}
}

func TestCreateTableWithFamily(t *testing.T) {
	// The Go client currently doesn't support creating a table with column families
	// in one operation but it is allowed by the API. This must still be supported by the
	// fake server so this test lives here instead of in the main bigtable
	// integration test.
	s := &server{
		tables: make(map[string]*table),
	}
	ctx := context.Background()
	newTbl := btapb.Table{
		ColumnFamilies: map[string]*btapb.ColumnFamily{
			"cf1": {GcRule: &btapb.GcRule{Rule: &btapb.GcRule_MaxNumVersions{MaxNumVersions: 123}}},
			"cf2": {GcRule: &btapb.GcRule{Rule: &btapb.GcRule_MaxNumVersions{MaxNumVersions: 456}}},
		},
	}
	cTbl, err := s.CreateTable(ctx, &btapb.CreateTableRequest{Parent: "cluster", TableId: "t", Table: &newTbl})
	if err != nil {
		t.Fatalf("Creating table: %v", err)
	}
	tbl, err := s.GetTable(ctx, &btapb.GetTableRequest{Name: cTbl.Name})
	if err != nil {
		t.Fatalf("Getting table: %v", err)
	}
	cf := tbl.ColumnFamilies["cf1"]
	if cf == nil {
		t.Fatalf("Missing col family cf1")
	}
	if got, want := cf.GcRule.GetMaxNumVersions(), int32(123); got != want {
		t.Errorf("Invalid MaxNumVersions: wanted:%d, got:%d", want, got)
	}
	cf = tbl.ColumnFamilies["cf2"]
	if cf == nil {
		t.Fatalf("Missing col family cf2")
	}
	if got, want := cf.GcRule.GetMaxNumVersions(), int32(456); got != want {
		t.Errorf("Invalid MaxNumVersions: wanted:%d, got:%d", want, got)
	}
}
