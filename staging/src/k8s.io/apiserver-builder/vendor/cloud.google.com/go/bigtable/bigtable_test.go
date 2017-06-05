/*
Copyright 2015 Google Inc. All Rights Reserved.

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

package bigtable

import (
	"flag"
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"

	"cloud.google.com/go/bigtable/bttest"
	"golang.org/x/net/context"
	"google.golang.org/api/option"
	"google.golang.org/grpc"
)

func TestPrefix(t *testing.T) {
	tests := []struct {
		prefix, succ string
	}{
		{"", ""},
		{"\xff", ""}, // when used, "" means Infinity
		{"x\xff", "y"},
		{"\xfe", "\xff"},
	}
	for _, tc := range tests {
		got := prefixSuccessor(tc.prefix)
		if got != tc.succ {
			t.Errorf("prefixSuccessor(%q) = %q, want %s", tc.prefix, got, tc.succ)
			continue
		}
		r := PrefixRange(tc.prefix)
		if tc.succ == "" && r.limit != "" {
			t.Errorf("PrefixRange(%q) got limit %q", tc.prefix, r.limit)
		}
		if tc.succ != "" && r.limit != tc.succ {
			t.Errorf("PrefixRange(%q) got limit %q, want %q", tc.prefix, r.limit, tc.succ)
		}
	}
}

var useProd = flag.String("use_prod", "", `if set to "proj,instance,table", run integration test against production`)

func TestClientIntegration(t *testing.T) {
	start := time.Now()
	lastCheckpoint := start
	checkpoint := func(s string) {
		n := time.Now()
		t.Logf("[%s] %v since start, %v since last checkpoint", s, n.Sub(start), n.Sub(lastCheckpoint))
		lastCheckpoint = n
	}

	proj, instance, table := "proj", "instance", "mytable"
	var clientOpts []option.ClientOption
	timeout := 20 * time.Second
	if *useProd == "" {
		srv, err := bttest.NewServer("127.0.0.1:0")
		if err != nil {
			t.Fatal(err)
		}
		defer srv.Close()
		t.Logf("bttest.Server running on %s", srv.Addr)
		conn, err := grpc.Dial(srv.Addr, grpc.WithInsecure())
		if err != nil {
			t.Fatalf("grpc.Dial: %v", err)
		}
		clientOpts = []option.ClientOption{option.WithGRPCConn(conn)}
	} else {
		t.Logf("Running test against production")
		a := strings.SplitN(*useProd, ",", 3)
		proj, instance, table = a[0], a[1], a[2]
		timeout = 5 * time.Minute
	}

	ctx, _ := context.WithTimeout(context.Background(), timeout)

	client, err := NewClient(ctx, proj, instance, clientOpts...)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer client.Close()
	checkpoint("dialed Client")

	adminClient, err := NewAdminClient(ctx, proj, instance, clientOpts...)
	if err != nil {
		t.Fatalf("NewAdminClient: %v", err)
	}
	defer adminClient.Close()
	checkpoint("dialed AdminClient")

	// Delete the table at the end of the test.
	// Do this even before creating the table so that if this is running
	// against production and CreateTable fails there's a chance of cleaning it up.
	defer adminClient.DeleteTable(ctx, table)

	if err := adminClient.CreateTable(ctx, table); err != nil {
		t.Fatalf("Creating table: %v", err)
	}
	checkpoint("created table")
	if err := adminClient.CreateColumnFamily(ctx, table, "follows"); err != nil {
		t.Fatalf("Creating column family: %v", err)
	}
	checkpoint(`created "follows" column family`)

	tbl := client.Open(table)

	// Insert some data.
	initialData := map[string][]string{
		"wmckinley":   {"tjefferson"},
		"gwashington": {"jadams"},
		"tjefferson":  {"gwashington", "jadams"}, // wmckinley set conditionally below
		"jadams":      {"gwashington", "tjefferson"},
	}
	for row, ss := range initialData {
		mut := NewMutation()
		for _, name := range ss {
			mut.Set("follows", name, 0, []byte("1"))
		}
		if err := tbl.Apply(ctx, row, mut); err != nil {
			t.Errorf("Mutating row %q: %v", row, err)
		}
	}
	checkpoint("inserted initial data")

	// Do a conditional mutation with a complex filter.
	mutTrue := NewMutation()
	mutTrue.Set("follows", "wmckinley", 0, []byte("1"))
	filter := ChainFilters(ColumnFilter("gwash[iz].*"), ValueFilter("."))
	mut := NewCondMutation(filter, mutTrue, nil)
	if err := tbl.Apply(ctx, "tjefferson", mut); err != nil {
		t.Errorf("Conditionally mutating row: %v", err)
	}
	// Do a second condition mutation with a filter that does not match,
	// and thus no changes should be made.
	mutTrue = NewMutation()
	mutTrue.DeleteRow()
	filter = ColumnFilter("snoop.dogg")
	mut = NewCondMutation(filter, mutTrue, nil)
	if err := tbl.Apply(ctx, "tjefferson", mut); err != nil {
		t.Errorf("Conditionally mutating row: %v", err)
	}
	checkpoint("did two conditional mutations")

	// Fetch a row.
	row, err := tbl.ReadRow(ctx, "jadams")
	if err != nil {
		t.Fatalf("Reading a row: %v", err)
	}
	wantRow := Row{
		"follows": []ReadItem{
			{Row: "jadams", Column: "follows:gwashington", Value: []byte("1")},
			{Row: "jadams", Column: "follows:tjefferson", Value: []byte("1")},
		},
	}
	for _, ris := range row {
		sort.Sort(byColumn(ris))
	}
	if !reflect.DeepEqual(row, wantRow) {
		t.Errorf("Read row mismatch.\n got %#v\nwant %#v", row, wantRow)
	}
	checkpoint("tested ReadRow")

	// Do a bunch of reads with filters.
	readTests := []struct {
		desc   string
		rr     RowRange
		filter Filter // may be nil

		// We do the read, grab all the cells, turn them into "<row>-<col>-<val>",
		// sort that list, and join with a comma.
		want string
	}{
		{
			desc: "read all, unfiltered",
			rr:   RowRange{},
			want: "gwashington-jadams-1,jadams-gwashington-1,jadams-tjefferson-1,tjefferson-gwashington-1,tjefferson-jadams-1,tjefferson-wmckinley-1,wmckinley-tjefferson-1",
		},
		{
			desc: "read with InfiniteRange, unfiltered",
			rr:   InfiniteRange("tjefferson"),
			want: "tjefferson-gwashington-1,tjefferson-jadams-1,tjefferson-wmckinley-1,wmckinley-tjefferson-1",
		},
		{
			desc: "read with NewRange, unfiltered",
			rr:   NewRange("gargamel", "hubbard"),
			want: "gwashington-jadams-1",
		},
		{
			desc: "read with PrefixRange, unfiltered",
			rr:   PrefixRange("jad"),
			want: "jadams-gwashington-1,jadams-tjefferson-1",
		},
		{
			desc: "read with SingleRow, unfiltered",
			rr:   SingleRow("wmckinley"),
			want: "wmckinley-tjefferson-1",
		},
		{
			desc:   "read all, with ColumnFilter",
			rr:     RowRange{},
			filter: ColumnFilter(".*j.*"), // matches "jadams" and "tjefferson"
			want:   "gwashington-jadams-1,jadams-tjefferson-1,tjefferson-jadams-1,wmckinley-tjefferson-1",
		},
	}
	for _, tc := range readTests {
		var opts []ReadOption
		if tc.filter != nil {
			opts = append(opts, RowFilter(tc.filter))
		}
		var elt []string
		err := tbl.ReadRows(context.Background(), tc.rr, func(r Row) bool {
			for _, ris := range r {
				for _, ri := range ris {
					elt = append(elt, formatReadItem(ri))
				}
			}
			return true
		}, opts...)
		if err != nil {
			t.Errorf("%s: %v", tc.desc, err)
			continue
		}
		sort.Strings(elt)
		if got := strings.Join(elt, ","); got != tc.want {
			t.Errorf("%s: wrong reads.\n got %q\nwant %q", tc.desc, got, tc.want)
		}
	}
	// Read a RowList
	var elt []string
	keys := RowList{"wmckinley", "gwashington", "jadams"}
	want := "gwashington-jadams-1,jadams-gwashington-1,jadams-tjefferson-1,wmckinley-tjefferson-1"
	err = tbl.ReadRows(ctx, keys, func(r Row) bool {
		for _, ris := range r {
			for _, ri := range ris {
				elt = append(elt, formatReadItem(ri))
			}
		}
		return true
	})
	if err != nil {
		t.Errorf("read RowList: %v", err)
	}

	sort.Strings(elt)
	if got := strings.Join(elt, ","); got != want {
		t.Errorf("bulk read: wrong reads.\n got %q\nwant %q", got, want)
	}
	checkpoint("tested ReadRows in a few ways")

	// Do a scan and stop part way through.
	// Verify that the ReadRows callback doesn't keep running.
	stopped := false
	err = tbl.ReadRows(ctx, InfiniteRange(""), func(r Row) bool {
		if r.Key() < "h" {
			return true
		}
		if !stopped {
			stopped = true
			return false
		}
		t.Errorf("ReadRows kept scanning to row %q after being told to stop", r.Key())
		return false
	})
	if err != nil {
		t.Errorf("Partial ReadRows: %v", err)
	}
	checkpoint("did partial ReadRows test")

	// Delete a row and check it goes away.
	mut = NewMutation()
	mut.DeleteRow()
	if err := tbl.Apply(ctx, "wmckinley", mut); err != nil {
		t.Errorf("Apply DeleteRow: %v", err)
	}
	row, err = tbl.ReadRow(ctx, "wmckinley")
	if err != nil {
		t.Fatalf("Reading a row after DeleteRow: %v", err)
	}
	if len(row) != 0 {
		t.Fatalf("Read non-zero row after DeleteRow: %v", row)
	}
	checkpoint("exercised DeleteRow")

	// Check ReadModifyWrite.

	if err := adminClient.CreateColumnFamily(ctx, table, "counter"); err != nil {
		t.Fatalf("Creating column family: %v", err)
	}

	appendRMW := func(b []byte) *ReadModifyWrite {
		rmw := NewReadModifyWrite()
		rmw.AppendValue("counter", "likes", b)
		return rmw
	}
	incRMW := func(n int64) *ReadModifyWrite {
		rmw := NewReadModifyWrite()
		rmw.Increment("counter", "likes", n)
		return rmw
	}
	rmwSeq := []struct {
		desc string
		rmw  *ReadModifyWrite
		want []byte
	}{
		{
			desc: "append #1",
			rmw:  appendRMW([]byte{0, 0, 0}),
			want: []byte{0, 0, 0},
		},
		{
			desc: "append #2",
			rmw:  appendRMW([]byte{0, 0, 0, 0, 17}), // the remaining 40 bits to make a big-endian 17
			want: []byte{0, 0, 0, 0, 0, 0, 0, 17},
		},
		{
			desc: "increment",
			rmw:  incRMW(8),
			want: []byte{0, 0, 0, 0, 0, 0, 0, 25},
		},
	}
	for _, step := range rmwSeq {
		row, err := tbl.ApplyReadModifyWrite(ctx, "gwashington", step.rmw)
		if err != nil {
			t.Fatalf("ApplyReadModifyWrite %+v: %v", step.rmw, err)
		}
		clearTimestamps(row)
		wantRow := Row{"counter": []ReadItem{{Row: "gwashington", Column: "counter:likes", Value: step.want}}}
		if !reflect.DeepEqual(row, wantRow) {
			t.Fatalf("After %s,\n got %v\nwant %v", step.desc, row, wantRow)
		}
	}
	checkpoint("tested ReadModifyWrite")

	// Test arbitrary timestamps more thoroughly.
	if err := adminClient.CreateColumnFamily(ctx, table, "ts"); err != nil {
		t.Fatalf("Creating column family: %v", err)
	}
	const numVersions = 4
	mut = NewMutation()
	for i := 0; i < numVersions; i++ {
		// Timestamps are used in thousands because the server
		// only permits that granularity.
		mut.Set("ts", "col", Timestamp(i*1000), []byte(fmt.Sprintf("val-%d", i)))
	}
	if err := tbl.Apply(ctx, "testrow", mut); err != nil {
		t.Fatalf("Mutating row: %v", err)
	}
	r, err := tbl.ReadRow(ctx, "testrow")
	if err != nil {
		t.Fatalf("Reading row: %v", err)
	}
	wantRow = Row{"ts": []ReadItem{
		// These should be returned in descending timestamp order.
		{Row: "testrow", Column: "ts:col", Timestamp: 3000, Value: []byte("val-3")},
		{Row: "testrow", Column: "ts:col", Timestamp: 2000, Value: []byte("val-2")},
		{Row: "testrow", Column: "ts:col", Timestamp: 1000, Value: []byte("val-1")},
		{Row: "testrow", Column: "ts:col", Timestamp: 0, Value: []byte("val-0")},
	}}
	if !reflect.DeepEqual(r, wantRow) {
		t.Errorf("Cell with multiple versions,\n got %v\nwant %v", r, wantRow)
	}
	// Do the same read, but filter to the latest two versions.
	r, err = tbl.ReadRow(ctx, "testrow", RowFilter(LatestNFilter(2)))
	if err != nil {
		t.Fatalf("Reading row: %v", err)
	}
	wantRow = Row{"ts": []ReadItem{
		{Row: "testrow", Column: "ts:col", Timestamp: 3000, Value: []byte("val-3")},
		{Row: "testrow", Column: "ts:col", Timestamp: 2000, Value: []byte("val-2")},
	}}
	if !reflect.DeepEqual(r, wantRow) {
		t.Errorf("Cell with multiple versions and LatestNFilter(2),\n got %v\nwant %v", r, wantRow)
	}
	// Delete the cell with timestamp 2000 and repeat the last read,
	// checking that we get ts 3000 and ts 1000.
	mut = NewMutation()
	mut.DeleteTimestampRange("ts", "col", 2000, 3000) // half-open interval
	if err := tbl.Apply(ctx, "testrow", mut); err != nil {
		t.Fatalf("Mutating row: %v", err)
	}
	r, err = tbl.ReadRow(ctx, "testrow", RowFilter(LatestNFilter(2)))
	if err != nil {
		t.Fatalf("Reading row: %v", err)
	}
	wantRow = Row{"ts": []ReadItem{
		{Row: "testrow", Column: "ts:col", Timestamp: 3000, Value: []byte("val-3")},
		{Row: "testrow", Column: "ts:col", Timestamp: 1000, Value: []byte("val-1")},
	}}
	if !reflect.DeepEqual(r, wantRow) {
		t.Errorf("Cell with multiple versions and LatestNFilter(2), after deleting timestamp 2000,\n got %v\nwant %v", r, wantRow)
	}
	checkpoint("tested multiple versions in a cell")

	// Do highly concurrent reads/writes.
	// TODO(dsymonds): Raise this to 1000 when https://github.com/grpc/grpc-go/issues/205 is resolved.
	const maxConcurrency = 100
	var wg sync.WaitGroup
	for i := 0; i < maxConcurrency; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			switch r := rand.Intn(100); { // r ∈ [0,100)
			case 0 <= r && r < 30:
				// Do a read.
				_, err := tbl.ReadRow(ctx, "testrow", RowFilter(LatestNFilter(1)))
				if err != nil {
					t.Errorf("Concurrent read: %v", err)
				}
			case 30 <= r && r < 100:
				// Do a write.
				mut := NewMutation()
				mut.Set("ts", "col", 0, []byte("data"))
				if err := tbl.Apply(ctx, "testrow", mut); err != nil {
					t.Errorf("Concurrent write: %v", err)
				}
			}
		}()
	}
	wg.Wait()
	checkpoint("tested high concurrency")

	// Large reads, writes and scans.
	bigBytes := make([]byte, 3<<20) // 3 MB is large, but less than current gRPC max of 4 MB.
	nonsense := []byte("lorem ipsum dolor sit amet, ")
	fill(bigBytes, nonsense)
	mut = NewMutation()
	mut.Set("ts", "col", 0, bigBytes)
	if err := tbl.Apply(ctx, "bigrow", mut); err != nil {
		t.Errorf("Big write: %v", err)
	}
	r, err = tbl.ReadRow(ctx, "bigrow")
	if err != nil {
		t.Errorf("Big read: %v", err)
	}
	wantRow = Row{"ts": []ReadItem{
		{Row: "bigrow", Column: "ts:col", Value: bigBytes},
	}}
	if !reflect.DeepEqual(r, wantRow) {
		t.Errorf("Big read returned incorrect bytes: %v", r)
	}
	// Now write 1000 rows, each with 82 KB values, then scan them all.
	medBytes := make([]byte, 82<<10)
	fill(medBytes, nonsense)
	sem := make(chan int, 50) // do up to 50 mutations at a time.
	for i := 0; i < 1000; i++ {
		mut := NewMutation()
		mut.Set("ts", "big-scan", 0, medBytes)
		row := fmt.Sprintf("row-%d", i)
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			sem <- 1
			if err := tbl.Apply(ctx, row, mut); err != nil {
				t.Errorf("Preparing large scan: %v", err)
			}
		}()
	}
	wg.Wait()
	n := 0
	err = tbl.ReadRows(ctx, PrefixRange("row-"), func(r Row) bool {
		for _, ris := range r {
			for _, ri := range ris {
				n += len(ri.Value)
			}
		}
		return true
	}, RowFilter(ColumnFilter("big-scan")))
	if err != nil {
		t.Errorf("Doing large scan: %v", err)
	}
	if want := 1000 * len(medBytes); n != want {
		t.Errorf("Large scan returned %d bytes, want %d", n, want)
	}
	// Scan a subset of the 1000 rows that we just created, using a LimitRows ReadOption.
	rc := 0
	wantRc := 3
	err = tbl.ReadRows(ctx, PrefixRange("row-"), func(r Row) bool {
		rc++
		return true
	}, LimitRows(int64(wantRc)))
	if rc != wantRc {
		t.Errorf("Scan with row limit returned %d rows, want %d", rc, wantRc)
	}
	checkpoint("tested big read/write/scan")

	// Test bulk mutations
	if err := adminClient.CreateColumnFamily(ctx, table, "bulk"); err != nil {
		t.Fatalf("Creating column family: %v", err)
	}
	bulkData := map[string][]string{
		"red sox":  {"2004", "2007", "2013"},
		"patriots": {"2001", "2003", "2004", "2014"},
		"celtics":  {"1981", "1984", "1986", "2008"},
	}
	var rowKeys []string
	var muts []*Mutation
	for row, ss := range bulkData {
		mut := NewMutation()
		for _, name := range ss {
			mut.Set("bulk", name, 0, []byte("1"))
		}
		rowKeys = append(rowKeys, row)
		muts = append(muts, mut)
	}
	status, err := tbl.ApplyBulk(ctx, rowKeys, muts)
	if err != nil {
		t.Fatalf("Bulk mutating rows %q: %v", rowKeys, err)
	}
	if status != nil {
		t.Errorf("non-nil errors: %v", err)
	}
	checkpoint("inserted bulk data")

	// Read each row back
	for rowKey, ss := range bulkData {
		row, err := tbl.ReadRow(ctx, rowKey)
		if err != nil {
			t.Fatalf("Reading a bulk row: %v", err)
		}
		for _, ris := range row {
			sort.Sort(byColumn(ris))
		}
		var wantItems []ReadItem
		for _, val := range ss {
			wantItems = append(wantItems, ReadItem{Row: rowKey, Column: "bulk:" + val, Value: []byte("1")})
		}
		wantRow := Row{"bulk": wantItems}
		if !reflect.DeepEqual(row, wantRow) {
			t.Errorf("Read row mismatch.\n got %#v\nwant %#v", row, wantRow)
		}
	}
	checkpoint("tested reading from bulk insert")

	// Test bulk write errors.
	// Note: Setting timestamps as ServerTime makes sure the mutations are not retried on error.
	badMut := NewMutation()
	badMut.Set("badfamily", "col", ServerTime, nil)
	badMut2 := NewMutation()
	badMut2.Set("badfamily2", "goodcol", ServerTime, []byte("1"))
	status, err = tbl.ApplyBulk(ctx, []string{"badrow", "badrow2"}, []*Mutation{badMut, badMut2})
	if err != nil {
		t.Fatalf("Bulk mutating rows %q: %v", rowKeys, err)
	}
	if status == nil {
		t.Errorf("No errors for bad bulk mutation")
	} else if status[0] == nil || status[1] == nil {
		t.Errorf("No error for bad bulk mutation")
	}
}

func formatReadItem(ri ReadItem) string {
	// Use the column qualifier only to make the test data briefer.
	col := ri.Column[strings.Index(ri.Column, ":")+1:]
	return fmt.Sprintf("%s-%s-%s", ri.Row, col, ri.Value)
}

func fill(b, sub []byte) {
	for len(b) > len(sub) {
		n := copy(b, sub)
		b = b[n:]
	}
}

type byColumn []ReadItem

func (b byColumn) Len() int           { return len(b) }
func (b byColumn) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byColumn) Less(i, j int) bool { return b[i].Column < b[j].Column }

func clearTimestamps(r Row) {
	for _, ris := range r {
		for i := range ris {
			ris[i].Timestamp = 0
		}
	}
}
