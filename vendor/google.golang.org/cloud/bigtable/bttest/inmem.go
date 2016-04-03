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

/*
Package bttest contains test helpers for working with the bigtable package.

To use a Server, create it, and then connect to it with no security:
(The project/zone/cluster values are ignored.)
	srv, err := bttest.NewServer()
	...
	conn, err := grpc.Dial(srv.Addr, grpc.WithInsecure())
	...
	client, err := bigtable.NewClient(ctx, proj, zone, cluster,
		cloud.WithBaseGRPC(conn))
	...
*/
package bttest // import "google.golang.org/cloud/bigtable/bttest"

import (
	"encoding/binary"
	"fmt"
	"log"
	"math/rand"
	"net"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/context"
	btdpb "google.golang.org/cloud/bigtable/internal/data_proto"
	emptypb "google.golang.org/cloud/bigtable/internal/empty"
	btspb "google.golang.org/cloud/bigtable/internal/service_proto"
	bttdpb "google.golang.org/cloud/bigtable/internal/table_data_proto"
	bttspb "google.golang.org/cloud/bigtable/internal/table_service_proto"
	"google.golang.org/grpc"
)

// Server is an in-memory Cloud Bigtable fake.
// It is unauthenticated, and only a rough approximation.
type Server struct {
	Addr string

	l   net.Listener
	srv *grpc.Server
	s   *server
}

// server is the real implementation of the fake.
// It is a separate and unexported type so the API won't be cluttered with
// methods that are only relevant to the fake's implementation.
type server struct {
	mu     sync.Mutex
	tables map[string]*table // keyed by fully qualified name
	gcc    chan int          // set when gcloop starts, closed when server shuts down

	// Any unimplemented methods will cause a panic.
	bttspb.BigtableTableServiceServer
	btspb.BigtableServiceServer
}

// NewServer creates a new Server. The Server will be listening for gRPC connections
// at the address named by the Addr field, without TLS.
func NewServer() (*Server, error) {
	l, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}

	s := &Server{
		Addr: l.Addr().String(),
		l:    l,
		srv:  grpc.NewServer(),
		s: &server{
			tables: make(map[string]*table),
		},
	}
	bttspb.RegisterBigtableTableServiceServer(s.srv, s.s)
	btspb.RegisterBigtableServiceServer(s.srv, s.s)

	go s.srv.Serve(s.l)

	return s, nil
}

// Close shuts down the server.
func (s *Server) Close() {
	s.s.mu.Lock()
	if s.s.gcc != nil {
		close(s.s.gcc)
	}
	s.s.mu.Unlock()

	s.srv.Stop()
	s.l.Close()
}

func (s *server) CreateTable(ctx context.Context, req *bttspb.CreateTableRequest) (*bttdpb.Table, error) {
	tbl := req.Name + "/tables/" + req.TableId

	s.mu.Lock()
	if _, ok := s.tables[tbl]; ok {
		s.mu.Unlock()
		return nil, fmt.Errorf("table %q already exists", tbl)
	}
	s.tables[tbl] = newTable()
	s.mu.Unlock()

	return &bttdpb.Table{Name: tbl}, nil
}

func (s *server) ListTables(ctx context.Context, req *bttspb.ListTablesRequest) (*bttspb.ListTablesResponse, error) {
	res := &bttspb.ListTablesResponse{}
	prefix := req.Name + "/tables/"

	s.mu.Lock()
	for tbl := range s.tables {
		if strings.HasPrefix(tbl, prefix) {
			res.Tables = append(res.Tables, &bttdpb.Table{Name: tbl})
		}
	}
	s.mu.Unlock()

	return res, nil
}

func (s *server) GetTable(ctx context.Context, req *bttspb.GetTableRequest) (*bttdpb.Table, error) {
	tbl := req.Name

	s.mu.Lock()
	tblIns, ok := s.tables[tbl]
	s.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("table %q not found", tbl)
	}

	return &bttdpb.Table{
		Name:           tbl,
		ColumnFamilies: toColumnFamilies(tblIns.families),
	}, nil
}

func (s *server) DeleteTable(ctx context.Context, req *bttspb.DeleteTableRequest) (*emptypb.Empty, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.tables[req.Name]; !ok {
		return nil, fmt.Errorf("no such table %q", req.Name)
	}
	delete(s.tables, req.Name)
	return &emptypb.Empty{}, nil
}

func (s *server) CreateColumnFamily(ctx context.Context, req *bttspb.CreateColumnFamilyRequest) (*bttdpb.ColumnFamily, error) {
	s.mu.Lock()
	tbl, ok := s.tables[req.Name]
	s.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("no such table %q", req.Name)
	}

	// Check it is unique and record it.
	fam := req.ColumnFamilyId
	tbl.mu.Lock()
	defer tbl.mu.Unlock()
	if _, ok := tbl.families[fam]; ok {
		return nil, fmt.Errorf("family %q already exists", fam)
	}
	newcf := &columnFamily{
		name: req.Name + "/columnFamilies/" + fam,
	}
	tbl.families[fam] = newcf
	return newcf.proto(), nil
}

func (s *server) UpdateColumnFamily(ctx context.Context, req *bttdpb.ColumnFamily) (*bttdpb.ColumnFamily, error) {
	index := strings.Index(req.Name, "/columnFamilies/")
	if index == -1 {
		return nil, fmt.Errorf("bad family name %q", req.Name)
	}
	tblName := req.Name[:index]
	fam := req.Name[index+len("/columnFamilies/"):]

	s.mu.Lock()
	tbl, ok := s.tables[tblName]
	s.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("no such table %q", req.Name)
	}

	tbl.mu.Lock()
	defer tbl.mu.Unlock()

	// Check it is unique and record it.
	if _, ok := tbl.families[fam]; !ok {
		return nil, fmt.Errorf("no such family %q", fam)
	}

	newcf := &columnFamily{
		name:   req.Name,
		gcRule: req.GcRule,
	}
	// assume that we ALWAYS want to replace by the new setting
	// we may need partial update through
	tbl.families[fam] = newcf
	s.needGC()
	return newcf.proto(), nil
}

func (s *server) ReadRows(req *btspb.ReadRowsRequest, stream btspb.BigtableService_ReadRowsServer) error {
	s.mu.Lock()
	tbl, ok := s.tables[req.TableName]
	s.mu.Unlock()
	if !ok {
		return fmt.Errorf("no such table %q", req.TableName)
	}

	var start, end string // half-open interval
	switch targ := req.Target.(type) {
	case *btspb.ReadRowsRequest_RowRange:
		start, end = string(targ.RowRange.StartKey), string(targ.RowRange.EndKey)
	case *btspb.ReadRowsRequest_RowKey:
		// A single row read is simply an edge case.
		start = string(targ.RowKey)
		end = start + "\x00"
	default:
		return fmt.Errorf("unknown ReadRowsRequest.Target oneof %T", targ)
	}

	// Get rows to stream back.
	tbl.mu.RLock()
	si, ei := 0, len(tbl.rows) // half-open interval
	if start != "" {
		si = sort.Search(len(tbl.rows), func(i int) bool { return tbl.rows[i].key >= start })
	}
	if end != "" {
		ei = sort.Search(len(tbl.rows), func(i int) bool { return tbl.rows[i].key >= end })
	}
	if si >= ei {
		tbl.mu.RUnlock()
		return nil
	}
	rows := make([]*row, ei-si)
	copy(rows, tbl.rows[si:ei])
	tbl.mu.RUnlock()

	for _, r := range rows {
		if err := streamRow(stream, r, req.Filter); err != nil {
			return err
		}
	}

	return nil
}

func streamRow(stream btspb.BigtableService_ReadRowsServer, r *row, f *btdpb.RowFilter) error {
	r.mu.Lock()
	nr := r.copy()
	r.mu.Unlock()
	r = nr

	filterRow(f, r)

	rrr := &btspb.ReadRowsResponse{
		RowKey: []byte(r.key),
	}
	for col, cells := range r.cells {
		i := strings.Index(col, ":") // guaranteed to exist
		fam, col := col[:i], col[i+1:]
		if len(cells) == 0 {
			continue
		}
		// TODO(dsymonds): Apply transformers.
		colm := &btdpb.Column{
			Qualifier: []byte(col),
			// Cells is populated below.
		}
		for _, cell := range cells {
			colm.Cells = append(colm.Cells, &btdpb.Cell{
				TimestampMicros: cell.ts,
				Value:           cell.value,
			})
		}
		rrr.Chunks = append(rrr.Chunks, &btspb.ReadRowsResponse_Chunk{
			Chunk: &btspb.ReadRowsResponse_Chunk_RowContents{&btdpb.Family{
				Name:    fam,
				Columns: []*btdpb.Column{colm},
			}},
		})
	}
	rrr.Chunks = append(rrr.Chunks, &btspb.ReadRowsResponse_Chunk{Chunk: &btspb.ReadRowsResponse_Chunk_CommitRow{true}})
	return stream.Send(rrr)
}

// filterRow modifies a row with the given filter.
func filterRow(f *btdpb.RowFilter, r *row) {
	if f == nil {
		return
	}
	// Handle filters that apply beyond just including/excluding cells.
	switch f := f.Filter.(type) {
	case *btdpb.RowFilter_Chain_:
		for _, sub := range f.Chain.Filters {
			filterRow(sub, r)
		}
		return
	case *btdpb.RowFilter_Interleave_:
		srs := make([]*row, 0, len(f.Interleave.Filters))
		for _, sub := range f.Interleave.Filters {
			sr := r.copy()
			filterRow(sub, sr)
			srs = append(srs, sr)
		}
		// merge
		// TODO(dsymonds): is this correct?
		r.cells = make(map[string][]cell)
		for _, sr := range srs {
			for col, cs := range sr.cells {
				r.cells[col] = append(r.cells[col], cs...)
			}
		}
		for _, cs := range r.cells {
			sort.Sort(byDescTS(cs))
		}
		return
	case *btdpb.RowFilter_CellsPerColumnLimitFilter:
		lim := int(f.CellsPerColumnLimitFilter)
		for col, cs := range r.cells {
			if len(cs) > lim {
				r.cells[col] = cs[:lim]
			}
		}
		return
	}

	// Any other case, operate on a per-cell basis.
	for key, cs := range r.cells {
		i := strings.Index(key, ":") // guaranteed to exist
		fam, col := key[:i], key[i+1:]
		r.cells[key] = filterCells(f, fam, col, cs)
	}
}

func filterCells(f *btdpb.RowFilter, fam, col string, cs []cell) []cell {
	var ret []cell
	for _, cell := range cs {
		if includeCell(f, fam, col, cell) {
			ret = append(ret, cell)
		}
	}
	return ret
}

func includeCell(f *btdpb.RowFilter, fam, col string, cell cell) bool {
	if f == nil {
		return true
	}
	// TODO(dsymonds): Implement many more filters.
	switch f := f.Filter.(type) {
	default:
		log.Printf("WARNING: don't know how to handle filter of type %T (ignoring it)", f)
		return true
	case *btdpb.RowFilter_FamilyNameRegexFilter:
		pat := string(f.FamilyNameRegexFilter)
		rx, err := regexp.Compile(pat)
		if err != nil {
			log.Printf("Bad family_name_regex_filter pattern %q: %v", pat, err)
			return false
		}
		return rx.MatchString(fam)
	case *btdpb.RowFilter_ColumnQualifierRegexFilter:
		pat := string(f.ColumnQualifierRegexFilter)
		rx, err := regexp.Compile(pat)
		if err != nil {
			log.Printf("Bad column_qualifier_regex_filter pattern %q: %v", pat, err)
			return false
		}
		return rx.MatchString(col)
	case *btdpb.RowFilter_ValueRegexFilter:
		pat := string(f.ValueRegexFilter)
		rx, err := regexp.Compile(pat)
		if err != nil {
			log.Printf("Bad value_regex_filter pattern %q: %v", pat, err)
			return false
		}
		return rx.Match(cell.value)
	}
}

func (s *server) MutateRow(ctx context.Context, req *btspb.MutateRowRequest) (*emptypb.Empty, error) {
	s.mu.Lock()
	tbl, ok := s.tables[req.TableName]
	s.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("no such table %q", req.TableName)
	}

	r := tbl.mutableRow(string(req.RowKey))
	r.mu.Lock()
	defer r.mu.Unlock()

	if err := applyMutations(tbl, r, req.Mutations); err != nil {
		return nil, err
	}
	return &emptypb.Empty{}, nil
}

func (s *server) CheckAndMutateRow(ctx context.Context, req *btspb.CheckAndMutateRowRequest) (*btspb.CheckAndMutateRowResponse, error) {
	s.mu.Lock()
	tbl, ok := s.tables[req.TableName]
	s.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("no such table %q", req.TableName)
	}

	res := &btspb.CheckAndMutateRowResponse{}

	r := tbl.mutableRow(string(req.RowKey))
	r.mu.Lock()
	defer r.mu.Unlock()

	// Figure out which mutation to apply.
	whichMut := false
	if req.PredicateFilter == nil {
		// Use true_mutations iff row contains any cells.
		whichMut = len(r.cells) > 0
	} else {
		// Use true_mutations iff any cells in the row match the filter.
		// TODO(dsymonds): This could be cheaper.
		nr := r.copy()
		filterRow(req.PredicateFilter, nr)
		for _, cs := range nr.cells {
			if len(cs) > 0 {
				whichMut = true
				break
			}
		}
		// TODO(dsymonds): Figure out if this is supposed to be set
		// even when there's no predicate filter.
		res.PredicateMatched = whichMut
	}
	muts := req.FalseMutations
	if whichMut {
		muts = req.TrueMutations
	}

	if err := applyMutations(tbl, r, muts); err != nil {
		return nil, err
	}
	return res, nil
}

// applyMutations applies a sequence of mutations to a row.
// It assumes r.mu is locked.
func applyMutations(tbl *table, r *row, muts []*btdpb.Mutation) error {
	for _, mut := range muts {
		switch mut := mut.Mutation.(type) {
		default:
			return fmt.Errorf("can't handle mutation type %T", mut)
		case *btdpb.Mutation_SetCell_:
			set := mut.SetCell
			tbl.mu.RLock()
			_, famOK := tbl.families[set.FamilyName]
			tbl.mu.RUnlock()
			if !famOK {
				return fmt.Errorf("unknown family %q", set.FamilyName)
			}
			ts := set.TimestampMicros
			if ts == -1 { // bigtable.ServerTime
				ts = time.Now().UnixNano() / 1e3
				ts -= ts % 1000 // round to millisecond granularity
			}
			if !tbl.validTimestamp(ts) {
				return fmt.Errorf("invalid timestamp %d", ts)
			}
			col := fmt.Sprintf("%s:%s", set.FamilyName, set.ColumnQualifier)

			cs := r.cells[col]
			newCell := cell{ts: ts, value: set.Value}
			replaced := false
			for i, cell := range cs {
				if cell.ts == newCell.ts {
					cs[i] = newCell
					replaced = true
					break
				}
			}
			if !replaced {
				cs = append(cs, newCell)
			}
			sort.Sort(byDescTS(cs))
			r.cells[col] = cs
		case *btdpb.Mutation_DeleteFromColumn_:
			del := mut.DeleteFromColumn
			col := fmt.Sprintf("%s:%s", del.FamilyName, del.ColumnQualifier)

			cs := r.cells[col]
			if del.TimeRange != nil {
				tsr := del.TimeRange
				if !tbl.validTimestamp(tsr.StartTimestampMicros) {
					return fmt.Errorf("invalid timestamp %d", tsr.StartTimestampMicros)
				}
				if !tbl.validTimestamp(tsr.EndTimestampMicros) {
					return fmt.Errorf("invalid timestamp %d", tsr.EndTimestampMicros)
				}
				// Find half-open interval to remove.
				// Cells are in descending timestamp order,
				// so the predicates to sort.Search are inverted.
				si, ei := 0, len(cs)
				if tsr.StartTimestampMicros > 0 {
					ei = sort.Search(len(cs), func(i int) bool { return cs[i].ts < tsr.StartTimestampMicros })
				}
				if tsr.EndTimestampMicros > 0 {
					si = sort.Search(len(cs), func(i int) bool { return cs[i].ts < tsr.EndTimestampMicros })
				}
				if si < ei {
					copy(cs[si:], cs[ei:])
					cs = cs[:len(cs)-(ei-si)]
				}
			} else {
				cs = nil
			}
			if len(cs) == 0 {
				delete(r.cells, col)
			} else {
				r.cells[col] = cs
			}
		case *btdpb.Mutation_DeleteFromRow_:
			r.cells = make(map[string][]cell)
		}
	}
	return nil
}

func (s *server) ReadModifyWriteRow(ctx context.Context, req *btspb.ReadModifyWriteRowRequest) (*btdpb.Row, error) {
	s.mu.Lock()
	tbl, ok := s.tables[req.TableName]
	s.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("no such table %q", req.TableName)
	}

	updates := make(map[string]cell) // copy of updated cells; keyed by full column name

	r := tbl.mutableRow(string(req.RowKey))
	r.mu.Lock()
	defer r.mu.Unlock()
	// Assume all mutations apply to the most recent version of the cell.
	// TODO(dsymonds): Verify this assumption and document it in the proto.
	for _, rule := range req.Rules {
		tbl.mu.RLock()
		_, famOK := tbl.families[rule.FamilyName]
		tbl.mu.RUnlock()
		if !famOK {
			return nil, fmt.Errorf("unknown family %q", rule.FamilyName)
		}

		key := fmt.Sprintf("%s:%s", rule.FamilyName, rule.ColumnQualifier)

		newCell := false
		if len(r.cells[key]) == 0 {
			r.cells[key] = []cell{{
			// TODO(dsymonds): should this set a timestamp?
			}}
			newCell = true
		}
		cell := &r.cells[key][0]

		switch rule := rule.Rule.(type) {
		default:
			return nil, fmt.Errorf("unknown RMW rule oneof %T", rule)
		case *btdpb.ReadModifyWriteRule_AppendValue:
			cell.value = append(cell.value, rule.AppendValue...)
		case *btdpb.ReadModifyWriteRule_IncrementAmount:
			var v int64
			if !newCell {
				if len(cell.value) != 8 {
					return nil, fmt.Errorf("increment on non-64-bit value")
				}
				v = int64(binary.BigEndian.Uint64(cell.value))
			}
			v += rule.IncrementAmount
			var val [8]byte
			binary.BigEndian.PutUint64(val[:], uint64(v))
			cell.value = val[:]
		}
		updates[key] = *cell
	}

	res := &btdpb.Row{
		Key: req.RowKey,
	}
	for col, cell := range updates {
		i := strings.Index(col, ":")
		fam, qual := col[:i], col[i+1:]
		var f *btdpb.Family
		for _, ff := range res.Families {
			if ff.Name == fam {
				f = ff
				break
			}
		}
		if f == nil {
			f = &btdpb.Family{Name: fam}
			res.Families = append(res.Families, f)
		}
		f.Columns = append(f.Columns, &btdpb.Column{
			Qualifier: []byte(qual),
			Cells: []*btdpb.Cell{{
				Value: cell.value,
			}},
		})
	}
	return res, nil
}

// needGC is invoked whenever the server needs gcloop running.
func (s *server) needGC() {
	s.mu.Lock()
	if s.gcc == nil {
		s.gcc = make(chan int)
		go s.gcloop(s.gcc)
	}
	s.mu.Unlock()
}

func (s *server) gcloop(done <-chan int) {
	const (
		minWait = 500  // ms
		maxWait = 1500 // ms
	)

	for {
		// Wait for a random time interval.
		d := time.Duration(minWait+rand.Intn(maxWait-minWait)) * time.Millisecond
		select {
		case <-time.After(d):
		case <-done:
			return // server has been closed
		}

		// Do a GC pass over all tables.
		var tables []*table
		s.mu.Lock()
		for _, tbl := range s.tables {
			tables = append(tables, tbl)
		}
		s.mu.Unlock()
		for _, tbl := range tables {
			tbl.gc()
		}
	}
}

type table struct {
	mu       sync.RWMutex
	families map[string]*columnFamily // keyed by plain family name
	rows     []*row                   // sorted by row key
	rowIndex map[string]*row          // indexed by row key
}

func newTable() *table {
	return &table{
		families: make(map[string]*columnFamily),
		rowIndex: make(map[string]*row),
	}
}

func (t *table) validTimestamp(ts int64) bool {
	// Assume millisecond granularity is required.
	return ts%1000 == 0
}

func (t *table) mutableRow(row string) *row {
	// Try fast path first.
	t.mu.RLock()
	r := t.rowIndex[row]
	t.mu.RUnlock()
	if r != nil {
		return r
	}

	// We probably need to create the row.
	t.mu.Lock()
	r = t.rowIndex[row]
	if r == nil {
		r = newRow(row)
		t.rowIndex[row] = r
		t.rows = append(t.rows, r)
		sort.Sort(byRowKey(t.rows)) // yay, inefficient!
	}
	t.mu.Unlock()
	return r
}

func (t *table) gc() {
	// This method doesn't add or remove rows, so we only need a read lock for the table.
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Gather GC rules we'll apply.
	rules := make(map[string]*bttdpb.GcRule) // keyed by "fam"
	for fam, cf := range t.families {
		if cf.gcRule != nil {
			rules[fam] = cf.gcRule
		}
	}
	if len(rules) == 0 {
		return
	}

	for _, r := range t.rows {
		r.mu.Lock()
		r.gc(rules)
		r.mu.Unlock()
	}
}

type byRowKey []*row

func (b byRowKey) Len() int           { return len(b) }
func (b byRowKey) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byRowKey) Less(i, j int) bool { return b[i].key < b[j].key }

type row struct {
	key string

	mu    sync.Mutex
	cells map[string][]cell // keyed by full column name; cells are in descending timestamp order
}

func newRow(key string) *row {
	return &row{
		key:   key,
		cells: make(map[string][]cell),
	}
}

// copy returns a copy of the row.
// Cell values are aliased.
// r.mu should be held.
func (r *row) copy() *row {
	nr := &row{
		key:   r.key,
		cells: make(map[string][]cell, len(r.cells)),
	}
	for col, cs := range r.cells {
		// Copy the []cell slice, but not the []byte inside each cell.
		nr.cells[col] = append([]cell(nil), cs...)
	}
	return nr
}

// gc applies the given GC rules to the row.
// r.mu should be held.
func (r *row) gc(rules map[string]*bttdpb.GcRule) {
	for col, cs := range r.cells {
		fam := col[:strings.Index(col, ":")]
		rule, ok := rules[fam]
		if !ok {
			continue
		}
		r.cells[col] = applyGC(cs, rule)
	}
}

var gcTypeWarn sync.Once

// applyGC applies the given GC rule to the cells.
func applyGC(cells []cell, rule *bttdpb.GcRule) []cell {
	switch rule := rule.Rule.(type) {
	default:
		// TODO(dsymonds): Support GcRule_Intersection_
		gcTypeWarn.Do(func() {
			log.Printf("Unsupported GC rule type %T", rule)
		})
	case *bttdpb.GcRule_Union_:
		for _, sub := range rule.Union.Rules {
			cells = applyGC(cells, sub)
		}
		return cells
	case *bttdpb.GcRule_MaxAge:
		// Timestamps are in microseconds.
		cutoff := time.Now().UnixNano() / 1e3
		cutoff -= rule.MaxAge.Seconds * 1e6
		cutoff -= int64(rule.MaxAge.Nanos) / 1e3
		// The slice of cells in in descending timestamp order.
		// This sort.Search will return the index of the first cell whose timestamp is chronologically before the cutoff.
		si := sort.Search(len(cells), func(i int) bool { return cells[i].ts < cutoff })
		if si < len(cells) {
			log.Printf("bttest: GC MaxAge(%v) deleted %d cells.", rule.MaxAge, len(cells)-si)
		}
		return cells[:si]
	case *bttdpb.GcRule_MaxNumVersions:
		n := int(rule.MaxNumVersions)
		if len(cells) > n {
			log.Printf("bttest: GC MaxNumVersions(%d) deleted %d cells.", n, len(cells)-n)
			cells = cells[:n]
		}
		return cells
	}
	return cells
}

type cell struct {
	ts    int64
	value []byte
}

type byDescTS []cell

func (b byDescTS) Len() int           { return len(b) }
func (b byDescTS) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }
func (b byDescTS) Less(i, j int) bool { return b[i].ts > b[j].ts }

type columnFamily struct {
	name   string
	gcRule *bttdpb.GcRule
}

func (c *columnFamily) proto() *bttdpb.ColumnFamily {
	return &bttdpb.ColumnFamily{
		Name:   c.name,
		GcRule: c.gcRule,
	}
}

func toColumnFamilies(families map[string]*columnFamily) map[string]*bttdpb.ColumnFamily {
	f := make(map[string]*bttdpb.ColumnFamily)
	for k, v := range families {
		f[k] = v.proto()
	}
	return f
}
