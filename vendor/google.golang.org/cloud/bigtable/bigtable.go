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

package bigtable // import "google.golang.org/cloud/bigtable"

import (
	"fmt"
	"io"
	"strconv"
	"time"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	"google.golang.org/cloud"
	btdpb "google.golang.org/cloud/bigtable/internal/data_proto"
	btspb "google.golang.org/cloud/bigtable/internal/service_proto"
	"google.golang.org/cloud/internal/transport"
	"google.golang.org/grpc"
)

const prodAddr = "bigtable.googleapis.com:443"

// Client is a client for reading and writing data to tables in a cluster.
//
// A Client is safe to use concurrently, except for its Close method.
type Client struct {
	conn   *grpc.ClientConn
	client btspb.BigtableServiceClient

	project, zone, cluster string
}

// NewClient creates a new Client for a given project, zone and cluster.
func NewClient(ctx context.Context, project, zone, cluster string, opts ...cloud.ClientOption) (*Client, error) {
	o := []cloud.ClientOption{
		cloud.WithEndpoint(prodAddr),
		cloud.WithScopes(Scope),
		cloud.WithUserAgent(clientUserAgent),
	}
	o = append(o, opts...)
	conn, err := transport.DialGRPC(ctx, o...)
	if err != nil {
		return nil, fmt.Errorf("dialing: %v", err)
	}
	return &Client{
		conn:   conn,
		client: btspb.NewBigtableServiceClient(conn),

		project: project,
		zone:    zone,
		cluster: cluster,
	}, nil
}

// Close closes the Client.
func (c *Client) Close() {
	c.conn.Close()
}

func (c *Client) fullTableName(table string) string {
	return fmt.Sprintf("projects/%s/zones/%s/clusters/%s/tables/%s", c.project, c.zone, c.cluster, table)
}

// A Table refers to a table.
//
// A Table is safe to use concurrently.
type Table struct {
	c     *Client
	table string
}

// Open opens a table.
func (c *Client) Open(table string) *Table {
	return &Table{
		c:     c,
		table: table,
	}
}

// TODO(dsymonds): Read method that returns a sequence of ReadItems.

// ReadRows reads rows from a table. f is called for each row.
// If f returns false, the stream is shut down and ReadRows returns.
// f owns its argument, and f is called serially.
//
// By default, the yielded rows will contain all values in all cells.
// Use RowFilter to limit the cells returned.
func (t *Table) ReadRows(ctx context.Context, arg RowRange, f func(Row) bool, opts ...ReadOption) error {
	req := &btspb.ReadRowsRequest{
		TableName: t.c.fullTableName(t.table),
		Target:    &btspb.ReadRowsRequest_RowRange{arg.proto()},
	}
	for _, opt := range opts {
		opt.set(req)
	}
	ctx, cancel := context.WithCancel(ctx) // for aborting the stream
	defer cancel()
	stream, err := t.c.client.ReadRows(ctx, req)
	if err != nil {
		return err
	}
	cr := new(chunkReader)
	for {
		res, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		if row := cr.process(res); row != nil {
			if !f(row) {
				// Cancel and drain stream.
				cancel()
				for {
					if _, err := stream.Recv(); err != nil {
						// The stream has ended. We don't return an error
						// because the caller has intentionally interrupted the scan.
						return nil
					}
				}
			}
		}
	}
	return nil
}

// ReadRow is a convenience implementation of a single-row reader.
// A missing row will return a zero-length map and a nil error.
func (t *Table) ReadRow(ctx context.Context, row string, opts ...ReadOption) (Row, error) {
	var r Row
	err := t.ReadRows(ctx, SingleRow(row), func(rr Row) bool {
		r = rr
		return true
	}, opts...)
	return r, err
}

type chunkReader struct {
	partial map[string]Row // incomplete rows
}

// process handles a single btspb.ReadRowsResponse.
// If it completes a row, that row is returned.
func (cr *chunkReader) process(rrr *btspb.ReadRowsResponse) Row {
	if cr.partial == nil {
		cr.partial = make(map[string]Row)
	}
	row := string(rrr.RowKey)
	r := cr.partial[row]
	if r == nil {
		r = make(Row)
		cr.partial[row] = r
	}
	for _, chunk := range rrr.Chunks {
		switch c := chunk.Chunk.(type) {
		case *btspb.ReadRowsResponse_Chunk_ResetRow:
			r = make(Row)
			cr.partial[row] = r
			continue
		case *btspb.ReadRowsResponse_Chunk_CommitRow:
			delete(cr.partial, row)
			if len(r) == 0 {
				// Treat zero-content commits as absent.
				continue
			}
			return r // assume that this is the last chunk
		case *btspb.ReadRowsResponse_Chunk_RowContents:
			decodeFamilyProto(r, row, c.RowContents)
		}
	}
	return nil
}

// decodeFamilyProto adds the cell data from f to the given row.
func decodeFamilyProto(r Row, row string, f *btdpb.Family) {
	fam := f.Name // does not have colon
	for _, col := range f.Columns {
		for _, cell := range col.Cells {
			ri := ReadItem{
				Row:       row,
				Column:    fmt.Sprintf("%s:%s", fam, col.Qualifier),
				Timestamp: Timestamp(cell.TimestampMicros),
				Value:     cell.Value,
			}
			r[fam] = append(r[fam], ri)
		}
	}
}

// A RowRange is used to describe the rows to be read.
// A RowRange is a half-open interval [Start, Limit) encompassing
// all the rows with keys at least as large as Start, and less than Limit.
// (Bigtable string comparison is the same as Go's.)
// A RowRange can be unbounded, encompassing all keys at least as large as Start.
type RowRange struct {
	start string
	limit string
}

// NewRange returns the new RowRange [begin, end).
func NewRange(begin, end string) RowRange {
	return RowRange{
		start: begin,
		limit: end,
	}
}

// Unbounded tests whether a RowRange is unbounded.
func (r RowRange) Unbounded() bool {
	return r.limit == ""
}

// Contains says whether the RowRange contains the key.
func (r RowRange) Contains(row string) bool {
	return r.start <= row && (r.limit == "" || r.limit > row)
}

// String provides a printable description of a RowRange.
func (r RowRange) String() string {
	a := strconv.Quote(r.start)
	if r.Unbounded() {
		return fmt.Sprintf("[%s,∞)", a)
	}
	return fmt.Sprintf("[%s,%q)", a, r.limit)
}

func (r RowRange) proto() *btdpb.RowRange {
	if r.Unbounded() {
		return &btdpb.RowRange{StartKey: []byte(r.start)}
	}
	return &btdpb.RowRange{
		StartKey: []byte(r.start),
		EndKey:   []byte(r.limit),
	}
}

// SingleRow returns a RowRange for reading a single row.
func SingleRow(row string) RowRange {
	return RowRange{
		start: row,
		limit: row + "\x00",
	}
}

// PrefixRange returns a RowRange consisting of all keys starting with the prefix.
func PrefixRange(prefix string) RowRange {
	return RowRange{
		start: prefix,
		limit: prefixSuccessor(prefix),
	}
}

// InfiniteRange returns the RowRange consisting of all keys at least as
// large as start.
func InfiniteRange(start string) RowRange {
	return RowRange{
		start: start,
		limit: "",
	}
}

// prefixSuccessor returns the lexically smallest string greater than the
// prefix, if it exists, or "" otherwise.  In either case, it is the string
// needed for the Limit of a RowRange.
func prefixSuccessor(prefix string) string {
	if prefix == "" {
		return "" // infinite range
	}
	n := len(prefix)
	for n--; n >= 0 && prefix[n] == '\xff'; n-- {
	}
	if n == -1 {
		return ""
	}
	ans := []byte(prefix[:n])
	ans = append(ans, prefix[n]+1)
	return string(ans)
}

// A ReadOption is an optional argument to ReadRows.
type ReadOption interface {
	set(req *btspb.ReadRowsRequest)
}

// RowFilter returns a ReadOption that applies f to the contents of read rows.
func RowFilter(f Filter) ReadOption { return rowFilter{f} }

type rowFilter struct{ f Filter }

func (rf rowFilter) set(req *btspb.ReadRowsRequest) { req.Filter = rf.f.proto() }

// LimitRows returns a ReadOption that will limit the number of rows to be read.
func LimitRows(limit int64) ReadOption { return limitRows{limit} }

type limitRows struct{ limit int64 }

func (lr limitRows) set(req *btspb.ReadRowsRequest) { req.NumRowsLimit = lr.limit }

// A Row is returned by ReadRow. The map is keyed by column family (the prefix
// of the column name before the colon). The values are the returned ReadItems
// for that column family in the order returned by Read.
type Row map[string][]ReadItem

// Key returns the row's key, or "" if the row is empty.
func (r Row) Key() string {
	for _, items := range r {
		if len(items) > 0 {
			return items[0].Row
		}
	}
	return ""
}

// A ReadItem is returned by Read. A ReadItem contains data from a specific row and column.
type ReadItem struct {
	Row, Column string
	Timestamp   Timestamp
	Value       []byte
}

// Apply applies a Mutation to a specific row.
func (t *Table) Apply(ctx context.Context, row string, m *Mutation, opts ...ApplyOption) error {
	after := func(res proto.Message) {
		for _, o := range opts {
			o.after(res)
		}
	}

	if m.cond == nil {
		req := &btspb.MutateRowRequest{
			TableName: t.c.fullTableName(t.table),
			RowKey:    []byte(row),
			Mutations: m.ops,
		}
		res, err := t.c.client.MutateRow(ctx, req)
		if err == nil {
			after(res)
		}
		return err
	}
	req := &btspb.CheckAndMutateRowRequest{
		TableName:       t.c.fullTableName(t.table),
		RowKey:          []byte(row),
		PredicateFilter: m.cond.proto(),
	}
	if m.mtrue != nil {
		req.TrueMutations = m.mtrue.ops
	}
	if m.mfalse != nil {
		req.FalseMutations = m.mfalse.ops
	}
	res, err := t.c.client.CheckAndMutateRow(ctx, req)
	if err == nil {
		after(res)
	}
	return err
}

// An ApplyOption is an optional argument to Apply.
type ApplyOption interface {
	after(res proto.Message)
}

type applyAfterFunc func(res proto.Message)

func (a applyAfterFunc) after(res proto.Message) { a(res) }

// GetCondMutationResult returns an ApplyOption that reports whether the conditional
// mutation's condition matched.
func GetCondMutationResult(matched *bool) ApplyOption {
	return applyAfterFunc(func(res proto.Message) {
		if res, ok := res.(*btspb.CheckAndMutateRowResponse); ok {
			*matched = res.PredicateMatched
		}
	})
}

// Mutation represents a set of changes for a single row of a table.
type Mutation struct {
	ops []*btdpb.Mutation

	// for conditional mutations
	cond          Filter
	mtrue, mfalse *Mutation
}

// NewMutation returns a new mutation.
func NewMutation() *Mutation {
	return new(Mutation)
}

// NewCondMutation returns a conditional mutation.
// The given row filter determines which mutation is applied:
// If the filter matches any cell in the row, mtrue is applied;
// otherwise, mfalse is applied.
// Either given mutation may be nil.
func NewCondMutation(cond Filter, mtrue, mfalse *Mutation) *Mutation {
	return &Mutation{cond: cond, mtrue: mtrue, mfalse: mfalse}
}

// Set sets a value in a specified column, with the given timestamp.
// The timestamp will be truncated to millisecond resolution.
// A timestamp of ServerTime means to use the server timestamp.
func (m *Mutation) Set(family, column string, ts Timestamp, value []byte) {
	if ts != ServerTime {
		// Truncate to millisecond resolution, since that's the default table config.
		// TODO(dsymonds): Provide a way to override this behaviour.
		ts -= ts % 1000
	}
	m.ops = append(m.ops, &btdpb.Mutation{Mutation: &btdpb.Mutation_SetCell_{&btdpb.Mutation_SetCell{
		FamilyName:      family,
		ColumnQualifier: []byte(column),
		TimestampMicros: int64(ts),
		Value:           value,
	}}})
}

// DeleteCellsInColumn will delete all the cells whose columns are family:column.
func (m *Mutation) DeleteCellsInColumn(family, column string) {
	m.ops = append(m.ops, &btdpb.Mutation{Mutation: &btdpb.Mutation_DeleteFromColumn_{&btdpb.Mutation_DeleteFromColumn{
		FamilyName:      family,
		ColumnQualifier: []byte(column),
	}}})
}

// DeleteTimestampRange deletes all cells whose columns are family:column
// and whose timestamps are in the half-open interval [start, end).
// If end is zero, it will be interpreted as infinity.
func (m *Mutation) DeleteTimestampRange(family, column string, start, end Timestamp) {
	m.ops = append(m.ops, &btdpb.Mutation{Mutation: &btdpb.Mutation_DeleteFromColumn_{&btdpb.Mutation_DeleteFromColumn{
		FamilyName:      family,
		ColumnQualifier: []byte(column),
		TimeRange: &btdpb.TimestampRange{
			StartTimestampMicros: int64(start),
			EndTimestampMicros:   int64(end),
		},
	}}})
}

// DeleteCellsInFamily will delete all the cells whose columns are family:*.
func (m *Mutation) DeleteCellsInFamily(family string) {
	m.ops = append(m.ops, &btdpb.Mutation{Mutation: &btdpb.Mutation_DeleteFromFamily_{&btdpb.Mutation_DeleteFromFamily{
		FamilyName: family,
	}}})
}

// DeleteRow deletes the entire row.
func (m *Mutation) DeleteRow() {
	m.ops = append(m.ops, &btdpb.Mutation{Mutation: &btdpb.Mutation_DeleteFromRow_{&btdpb.Mutation_DeleteFromRow{}}})
}

// Timestamp is in units of microseconds since 1 January 1970.
type Timestamp int64

// ServerTime is a specific Timestamp that may be passed to (*Mutation).Set.
// It indicates that the server's timestamp should be used.
const ServerTime Timestamp = -1

// Time converts a time.Time into a Timestamp.
func Time(t time.Time) Timestamp { return Timestamp(t.UnixNano() / 1e3) }

// Now returns the Timestamp representation of the current time on the client.
func Now() Timestamp { return Time(time.Now()) }

// Time converts a Timestamp into a time.Time.
func (ts Timestamp) Time() time.Time { return time.Unix(0, int64(ts)*1e3) }

// ApplyReadModifyWrite applies a ReadModifyWrite to a specific row.
// It returns the newly written cells.
func (t *Table) ApplyReadModifyWrite(ctx context.Context, row string, m *ReadModifyWrite) (Row, error) {
	req := &btspb.ReadModifyWriteRowRequest{
		TableName: t.c.fullTableName(t.table),
		RowKey:    []byte(row),
		Rules:     m.ops,
	}
	res, err := t.c.client.ReadModifyWriteRow(ctx, req)
	if err != nil {
		return nil, err
	}
	r := make(Row)
	for _, fam := range res.Families { // res is *btdpb.Row, fam is *btdpb.Family
		decodeFamilyProto(r, row, fam)
	}
	return r, nil
}

// ReadModifyWrite represents a set of operations on a single row of a table.
// It is like Mutation but for non-idempotent changes.
// When applied, these operations operate on the latest values of the row's cells,
// and result in a new value being written to the relevant cell with a timestamp
// that is max(existing timestamp, current server time).
//
// The application of a ReadModifyWrite is atomic; concurrent ReadModifyWrites will
// be executed serially by the server.
type ReadModifyWrite struct {
	ops []*btdpb.ReadModifyWriteRule
}

// NewReadModifyWrite returns a new ReadModifyWrite.
func NewReadModifyWrite() *ReadModifyWrite { return new(ReadModifyWrite) }

// AppendValue appends a value to a specific cell's value.
// If the cell is unset, it will be treated as an empty value.
func (m *ReadModifyWrite) AppendValue(family, column string, v []byte) {
	m.ops = append(m.ops, &btdpb.ReadModifyWriteRule{
		FamilyName:      family,
		ColumnQualifier: []byte(column),
		Rule:            &btdpb.ReadModifyWriteRule_AppendValue{v},
	})
}

// Increment interprets the value in a specific cell as a 64-bit big-endian signed integer,
// and adds a value to it. If the cell is unset, it will be treated as zero.
// If the cell is set and is not an 8-byte value, the entire ApplyReadModifyWrite
// operation will fail.
func (m *ReadModifyWrite) Increment(family, column string, delta int64) {
	m.ops = append(m.ops, &btdpb.ReadModifyWriteRule{
		FamilyName:      family,
		ColumnQualifier: []byte(column),
		Rule:            &btdpb.ReadModifyWriteRule_IncrementAmount{delta},
	})
}
