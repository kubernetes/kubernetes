// Copyright 2014 Google Inc. All Rights Reserved.
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

package datastore

import (
	"encoding/base64"
	"errors"
	"fmt"
	"math"
	"reflect"
	"strconv"
	"strings"

	wrapperspb "github.com/golang/protobuf/ptypes/wrappers"
	"golang.org/x/net/context"
	pb "google.golang.org/genproto/googleapis/datastore/v1"
)

type operator int

const (
	lessThan operator = iota + 1
	lessEq
	equal
	greaterEq
	greaterThan

	keyFieldName = "__key__"
)

var operatorToProto = map[operator]pb.PropertyFilter_Operator{
	lessThan:    pb.PropertyFilter_LESS_THAN,
	lessEq:      pb.PropertyFilter_LESS_THAN_OR_EQUAL,
	equal:       pb.PropertyFilter_EQUAL,
	greaterEq:   pb.PropertyFilter_GREATER_THAN_OR_EQUAL,
	greaterThan: pb.PropertyFilter_GREATER_THAN,
}

// filter is a conditional filter on query results.
type filter struct {
	FieldName string
	Op        operator
	Value     interface{}
}

type sortDirection bool

const (
	ascending  sortDirection = false
	descending sortDirection = true
)

var sortDirectionToProto = map[sortDirection]pb.PropertyOrder_Direction{
	ascending:  pb.PropertyOrder_ASCENDING,
	descending: pb.PropertyOrder_DESCENDING,
}

// order is a sort order on query results.
type order struct {
	FieldName string
	Direction sortDirection
}

// NewQuery creates a new Query for a specific entity kind.
//
// An empty kind means to return all entities, including entities created and
// managed by other App Engine features, and is called a kindless query.
// Kindless queries cannot include filters or sort orders on property values.
func NewQuery(kind string) *Query {
	return &Query{
		kind:  kind,
		limit: -1,
	}
}

// Query represents a datastore query.
type Query struct {
	kind       string
	ancestor   *Key
	filter     []filter
	order      []order
	projection []string

	distinct bool
	keysOnly bool
	eventual bool
	limit    int32
	offset   int32
	start    []byte
	end      []byte

	trans *Transaction

	err error
}

func (q *Query) clone() *Query {
	x := *q
	// Copy the contents of the slice-typed fields to a new backing store.
	if len(q.filter) > 0 {
		x.filter = make([]filter, len(q.filter))
		copy(x.filter, q.filter)
	}
	if len(q.order) > 0 {
		x.order = make([]order, len(q.order))
		copy(x.order, q.order)
	}
	return &x
}

// Ancestor returns a derivative query with an ancestor filter.
// The ancestor should not be nil.
func (q *Query) Ancestor(ancestor *Key) *Query {
	q = q.clone()
	if ancestor == nil {
		q.err = errors.New("datastore: nil query ancestor")
		return q
	}
	q.ancestor = ancestor
	return q
}

// EventualConsistency returns a derivative query that returns eventually
// consistent results.
// It only has an effect on ancestor queries.
func (q *Query) EventualConsistency() *Query {
	q = q.clone()
	q.eventual = true
	return q
}

// Transaction returns a derivative query that is associated with the given
// transaction.
//
// All reads performed as part of the transaction will come from a single
// consistent snapshot. Furthermore, if the transaction is set to a
// serializable isolation level, another transaction cannot concurrently modify
// the data that is read or modified by this transaction.
func (q *Query) Transaction(t *Transaction) *Query {
	q = q.clone()
	q.trans = t
	return q
}

// Filter returns a derivative query with a field-based filter.
// The filterStr argument must be a field name followed by optional space,
// followed by an operator, one of ">", "<", ">=", "<=", or "=".
// Fields are compared against the provided value using the operator.
// Multiple filters are AND'ed together.
// Field names which contain spaces, quote marks, or operator characters
// should be passed as quoted Go string literals as returned by strconv.Quote
// or the fmt package's %q verb.
func (q *Query) Filter(filterStr string, value interface{}) *Query {
	q = q.clone()
	filterStr = strings.TrimSpace(filterStr)
	if filterStr == "" {
		q.err = fmt.Errorf("datastore: invalid filter %q", filterStr)
		return q
	}
	f := filter{
		FieldName: strings.TrimRight(filterStr, " ><=!"),
		Value:     value,
	}
	switch op := strings.TrimSpace(filterStr[len(f.FieldName):]); op {
	case "<=":
		f.Op = lessEq
	case ">=":
		f.Op = greaterEq
	case "<":
		f.Op = lessThan
	case ">":
		f.Op = greaterThan
	case "=":
		f.Op = equal
	default:
		q.err = fmt.Errorf("datastore: invalid operator %q in filter %q", op, filterStr)
		return q
	}
	var err error
	f.FieldName, err = unquote(f.FieldName)
	if err != nil {
		q.err = fmt.Errorf("datastore: invalid syntax for quoted field name %q", f.FieldName)
		return q
	}
	q.filter = append(q.filter, f)
	return q
}

// Order returns a derivative query with a field-based sort order. Orders are
// applied in the order they are added. The default order is ascending; to sort
// in descending order prefix the fieldName with a minus sign (-).
// Field names which contain spaces, quote marks, or the minus sign
// should be passed as quoted Go string literals as returned by strconv.Quote
// or the fmt package's %q verb.
func (q *Query) Order(fieldName string) *Query {
	q = q.clone()
	fieldName, dir := strings.TrimSpace(fieldName), ascending
	if strings.HasPrefix(fieldName, "-") {
		fieldName, dir = strings.TrimSpace(fieldName[1:]), descending
	} else if strings.HasPrefix(fieldName, "+") {
		q.err = fmt.Errorf("datastore: invalid order: %q", fieldName)
		return q
	}
	fieldName, err := unquote(fieldName)
	if err != nil {
		q.err = fmt.Errorf("datastore: invalid syntax for quoted field name %q", fieldName)
		return q
	}
	if fieldName == "" {
		q.err = errors.New("datastore: empty order")
		return q
	}
	q.order = append(q.order, order{
		Direction: dir,
		FieldName: fieldName,
	})
	return q
}

// unquote optionally interprets s as a double-quoted or backquoted Go
// string literal if it begins with the relevant character.
func unquote(s string) (string, error) {
	if s == "" || (s[0] != '`' && s[0] != '"') {
		return s, nil
	}
	return strconv.Unquote(s)
}

// Project returns a derivative query that yields only the given fields. It
// cannot be used with KeysOnly.
func (q *Query) Project(fieldNames ...string) *Query {
	q = q.clone()
	q.projection = append([]string(nil), fieldNames...)
	return q
}

// Distinct returns a derivative query that yields de-duplicated entities with
// respect to the set of projected fields. It is only used for projection
// queries.
func (q *Query) Distinct() *Query {
	q = q.clone()
	q.distinct = true
	return q
}

// KeysOnly returns a derivative query that yields only keys, not keys and
// entities. It cannot be used with projection queries.
func (q *Query) KeysOnly() *Query {
	q = q.clone()
	q.keysOnly = true
	return q
}

// Limit returns a derivative query that has a limit on the number of results
// returned. A negative value means unlimited.
func (q *Query) Limit(limit int) *Query {
	q = q.clone()
	if limit < math.MinInt32 || limit > math.MaxInt32 {
		q.err = errors.New("datastore: query limit overflow")
		return q
	}
	q.limit = int32(limit)
	return q
}

// Offset returns a derivative query that has an offset of how many keys to
// skip over before returning results. A negative value is invalid.
func (q *Query) Offset(offset int) *Query {
	q = q.clone()
	if offset < 0 {
		q.err = errors.New("datastore: negative query offset")
		return q
	}
	if offset > math.MaxInt32 {
		q.err = errors.New("datastore: query offset overflow")
		return q
	}
	q.offset = int32(offset)
	return q
}

// Start returns a derivative query with the given start point.
func (q *Query) Start(c Cursor) *Query {
	q = q.clone()
	q.start = c.cc
	return q
}

// End returns a derivative query with the given end point.
func (q *Query) End(c Cursor) *Query {
	q = q.clone()
	q.end = c.cc
	return q
}

// toProto converts the query to a protocol buffer.
func (q *Query) toProto(req *pb.RunQueryRequest) error {
	if len(q.projection) != 0 && q.keysOnly {
		return errors.New("datastore: query cannot both project and be keys-only")
	}
	dst := &pb.Query{}
	if q.kind != "" {
		dst.Kind = []*pb.KindExpression{{Name: q.kind}}
	}
	if q.projection != nil {
		for _, propertyName := range q.projection {
			dst.Projection = append(dst.Projection, &pb.Projection{Property: &pb.PropertyReference{Name: propertyName}})
		}

		if q.distinct {
			for _, propertyName := range q.projection {
				dst.DistinctOn = append(dst.DistinctOn, &pb.PropertyReference{Name: propertyName})
			}
		}
	}
	if q.keysOnly {
		dst.Projection = []*pb.Projection{{Property: &pb.PropertyReference{Name: keyFieldName}}}
	}

	var filters []*pb.Filter
	for _, qf := range q.filter {
		if qf.FieldName == "" {
			return errors.New("datastore: empty query filter field name")
		}
		v, err := interfaceToProto(reflect.ValueOf(qf.Value).Interface(), false)
		if err != nil {
			return fmt.Errorf("datastore: bad query filter value type: %v", err)
		}
		op, ok := operatorToProto[qf.Op]
		if !ok {
			return errors.New("datastore: unknown query filter operator")
		}
		xf := &pb.PropertyFilter{
			Op:       op,
			Property: &pb.PropertyReference{Name: qf.FieldName},
			Value:    v,
		}
		filters = append(filters, &pb.Filter{
			FilterType: &pb.Filter_PropertyFilter{xf},
		})
	}

	if q.ancestor != nil {
		filters = append(filters, &pb.Filter{
			FilterType: &pb.Filter_PropertyFilter{&pb.PropertyFilter{
				Property: &pb.PropertyReference{Name: "__key__"},
				Op:       pb.PropertyFilter_HAS_ANCESTOR,
				Value:    &pb.Value{ValueType: &pb.Value_KeyValue{keyToProto(q.ancestor)}},
			}}})
	}

	if len(filters) == 1 {
		dst.Filter = filters[0]
	} else if len(filters) > 1 {
		dst.Filter = &pb.Filter{FilterType: &pb.Filter_CompositeFilter{&pb.CompositeFilter{
			Op:      pb.CompositeFilter_AND,
			Filters: filters,
		}}}
	}

	for _, qo := range q.order {
		if qo.FieldName == "" {
			return errors.New("datastore: empty query order field name")
		}
		xo := &pb.PropertyOrder{
			Property:  &pb.PropertyReference{Name: qo.FieldName},
			Direction: sortDirectionToProto[qo.Direction],
		}
		dst.Order = append(dst.Order, xo)
	}
	if q.limit >= 0 {
		dst.Limit = &wrapperspb.Int32Value{q.limit}
	}
	dst.Offset = q.offset
	dst.StartCursor = q.start
	dst.EndCursor = q.end

	if t := q.trans; t != nil {
		if t.id == nil {
			return errExpiredTransaction
		}
		if q.eventual {
			return errors.New("datastore: cannot use EventualConsistency query in a transaction")
		}
		req.ReadOptions = &pb.ReadOptions{
			ConsistencyType: &pb.ReadOptions_Transaction{t.id},
		}
	}

	if q.eventual {
		req.ReadOptions = &pb.ReadOptions{&pb.ReadOptions_ReadConsistency_{pb.ReadOptions_EVENTUAL}}
	}

	req.QueryType = &pb.RunQueryRequest_Query{dst}
	return nil
}

// Count returns the number of results for the given query.
//
// The running time and number of API calls made by Count scale linearly with
// with the sum of the query's offset and limit. Unless the result count is
// expected to be small, it is best to specify a limit; otherwise Count will
// continue until it finishes counting or the provided context expires.
func (c *Client) Count(ctx context.Context, q *Query) (int, error) {
	// Check that the query is well-formed.
	if q.err != nil {
		return 0, q.err
	}

	// Create a copy of the query, with keysOnly true (if we're not a projection,
	// since the two are incompatible).
	newQ := q.clone()
	newQ.keysOnly = len(newQ.projection) == 0

	// Create an iterator and use it to walk through the batches of results
	// directly.
	it := c.Run(ctx, newQ)
	n := 0
	for {
		err := it.nextBatch()
		if err == Done {
			return n, nil
		}
		if err != nil {
			return 0, err
		}
		n += len(it.results)
	}
}

// GetAll runs the provided query in the given context and returns all keys
// that match that query, as well as appending the values to dst.
//
// dst must have type *[]S or *[]*S or *[]P, for some struct type S or some non-
// interface, non-pointer type P such that P or *P implements PropertyLoadSaver.
//
// As a special case, *PropertyList is an invalid type for dst, even though a
// PropertyList is a slice of structs. It is treated as invalid to avoid being
// mistakenly passed when *[]PropertyList was intended.
//
// The keys returned by GetAll will be in a 1-1 correspondence with the entities
// added to dst.
//
// If q is a ``keys-only'' query, GetAll ignores dst and only returns the keys.
//
// The running time and number of API calls made by GetAll scale linearly with
// with the sum of the query's offset and limit. Unless the result count is
// expected to be small, it is best to specify a limit; otherwise GetAll will
// continue until it finishes collecting results or the provided context
// expires.
func (c *Client) GetAll(ctx context.Context, q *Query, dst interface{}) ([]*Key, error) {
	var (
		dv               reflect.Value
		mat              multiArgType
		elemType         reflect.Type
		errFieldMismatch error
	)
	if !q.keysOnly {
		dv = reflect.ValueOf(dst)
		if dv.Kind() != reflect.Ptr || dv.IsNil() {
			return nil, ErrInvalidEntityType
		}
		dv = dv.Elem()
		mat, elemType = checkMultiArg(dv)
		if mat == multiArgTypeInvalid || mat == multiArgTypeInterface {
			return nil, ErrInvalidEntityType
		}
	}

	var keys []*Key
	for t := c.Run(ctx, q); ; {
		k, e, err := t.next()
		if err == Done {
			break
		}
		if err != nil {
			return keys, err
		}
		if !q.keysOnly {
			ev := reflect.New(elemType)
			if elemType.Kind() == reflect.Map {
				// This is a special case. The zero values of a map type are
				// not immediately useful; they have to be make'd.
				//
				// Funcs and channels are similar, in that a zero value is not useful,
				// but even a freshly make'd channel isn't useful: there's no fixed
				// channel buffer size that is always going to be large enough, and
				// there's no goroutine to drain the other end. Theoretically, these
				// types could be supported, for example by sniffing for a constructor
				// method or requiring prior registration, but for now it's not a
				// frequent enough concern to be worth it. Programmers can work around
				// it by explicitly using Iterator.Next instead of the Query.GetAll
				// convenience method.
				x := reflect.MakeMap(elemType)
				ev.Elem().Set(x)
			}
			if err = loadEntity(ev.Interface(), e); err != nil {
				if _, ok := err.(*ErrFieldMismatch); ok {
					// We continue loading entities even in the face of field mismatch errors.
					// If we encounter any other error, that other error is returned. Otherwise,
					// an ErrFieldMismatch is returned.
					errFieldMismatch = err
				} else {
					return keys, err
				}
			}
			if mat != multiArgTypeStructPtr {
				ev = ev.Elem()
			}
			dv.Set(reflect.Append(dv, ev))
		}
		keys = append(keys, k)
	}
	return keys, errFieldMismatch
}

// Run runs the given query in the given context.
func (c *Client) Run(ctx context.Context, q *Query) *Iterator {
	if q.err != nil {
		return &Iterator{err: q.err}
	}
	t := &Iterator{
		ctx:          ctx,
		client:       c,
		limit:        q.limit,
		offset:       q.offset,
		keysOnly:     q.keysOnly,
		pageCursor:   q.start,
		entityCursor: q.start,
		req: &pb.RunQueryRequest{
			ProjectId: c.dataset,
		},
	}
	if ns := ctxNamespace(ctx); ns != "" {
		t.req.PartitionId = &pb.PartitionId{
			NamespaceId: ns,
		}
	}
	if err := q.toProto(t.req); err != nil {
		t.err = err
	}
	return t
}

// Iterator is the result of running a query.
type Iterator struct {
	ctx    context.Context
	client *Client
	err    error

	// results is the list of EntityResults still to be iterated over from the
	// most recent API call. It will be nil if no requests have yet been issued.
	results []*pb.EntityResult
	// req is the request to send. It may be modified and used multiple times.
	req *pb.RunQueryRequest

	// limit is the limit on the number of results this iterator should return.
	// The zero value is used to prevent further fetches from the server.
	// A negative value means unlimited.
	limit int32
	// offset is the number of results that still need to be skipped.
	offset int32
	// keysOnly records whether the query was keys-only (skip entity loading).
	keysOnly bool

	// pageCursor is the compiled cursor for the next batch/page of result.
	// TODO(djd): Can we delete this in favour of paging with the last
	// entityCursor from each batch?
	pageCursor []byte
	// entityCursor is the compiled cursor of the next result.
	entityCursor []byte
}

// Done is returned when a query iteration has completed.
var Done = errors.New("datastore: query has no more results")

// Next returns the key of the next result. When there are no more results,
// Done is returned as the error.
//
// If the query is not keys only and dst is non-nil, it also loads the entity
// stored for that key into the struct pointer or PropertyLoadSaver dst, with
// the same semantics and possible errors as for the Get function.
func (t *Iterator) Next(dst interface{}) (*Key, error) {
	k, e, err := t.next()
	if err != nil {
		return nil, err
	}
	if dst != nil && !t.keysOnly {
		err = loadEntity(dst, e)
	}
	return k, err
}

func (t *Iterator) next() (*Key, *pb.Entity, error) {
	// Fetch additional batches while there are no more results.
	for t.err == nil && len(t.results) == 0 {
		t.err = t.nextBatch()
	}
	if t.err != nil {
		return nil, nil, t.err
	}

	// Extract the next result, update cursors, and parse the entity's key.
	e := t.results[0]
	t.results = t.results[1:]
	t.entityCursor = e.Cursor
	if len(t.results) == 0 {
		t.entityCursor = t.pageCursor // At the end of the batch.
	}
	if e.Entity.Key == nil {
		return nil, nil, errors.New("datastore: internal error: server did not return a key")
	}
	k, err := protoToKey(e.Entity.Key)
	if err != nil || k.Incomplete() {
		return nil, nil, errors.New("datastore: internal error: server returned an invalid key")
	}

	return k, e.Entity, nil
}

// nextBatch makes a single call to the server for a batch of results.
func (t *Iterator) nextBatch() error {
	if t.limit == 0 {
		return Done // Short-circuits the zero-item response.
	}

	// Adjust the query with the latest start cursor, limit and offset.
	q := t.req.GetQuery()
	q.StartCursor = t.pageCursor
	q.Offset = t.offset
	if t.limit >= 0 {
		q.Limit = &wrapperspb.Int32Value{t.limit}
	} else {
		q.Limit = nil
	}

	// Run the query.
	resp, err := t.client.client.RunQuery(t.ctx, t.req)
	if err != nil {
		return err
	}

	// Adjust any offset from skipped results.
	skip := resp.Batch.SkippedResults
	if skip < 0 {
		return errors.New("datastore: internal error: negative number of skipped_results")
	}
	t.offset -= skip
	if t.offset < 0 {
		return errors.New("datastore: internal error: query skipped too many results")
	}
	if t.offset > 0 && len(resp.Batch.EntityResults) > 0 {
		return errors.New("datastore: internal error: query returned results before requested offset")
	}

	// Adjust the limit.
	if t.limit >= 0 {
		t.limit -= int32(len(resp.Batch.EntityResults))
		if t.limit < 0 {
			return errors.New("datastore: internal error: query returned more results than the limit")
		}
	}

	// If there are no more results available, set limit to zero to prevent
	// further fetches. Otherwise, check that there is a next page cursor available.
	if resp.Batch.MoreResults != pb.QueryResultBatch_NOT_FINISHED {
		t.limit = 0
	} else if resp.Batch.EndCursor == nil {
		return errors.New("datastore: internal error: server did not return a cursor")
	}

	// Update cursors.
	// If any results were skipped, use the SkippedCursor as the next entity cursor.
	if skip > 0 {
		t.entityCursor = resp.Batch.SkippedCursor
	} else {
		t.entityCursor = q.StartCursor
	}
	t.pageCursor = resp.Batch.EndCursor

	t.results = resp.Batch.EntityResults
	return nil
}

// Cursor returns a cursor for the iterator's current location.
func (t *Iterator) Cursor() (Cursor, error) {
	// If there is still an offset, we need to the skip those results first.
	for t.err == nil && t.offset > 0 {
		t.err = t.nextBatch()
	}

	if t.err != nil && t.err != Done {
		return Cursor{}, t.err
	}

	return Cursor{t.entityCursor}, nil
}

// Cursor is an iterator's position. It can be converted to and from an opaque
// string. A cursor can be used from different HTTP requests, but only with a
// query with the same kind, ancestor, filter and order constraints.
//
// The zero Cursor can be used to indicate that there is no start and/or end
// constraint for a query.
type Cursor struct {
	cc []byte
}

// String returns a base-64 string representation of a cursor.
func (c Cursor) String() string {
	if c.cc == nil {
		return ""
	}

	return strings.TrimRight(base64.URLEncoding.EncodeToString(c.cc), "=")
}

// Decode decodes a cursor from its base-64 string representation.
func DecodeCursor(s string) (Cursor, error) {
	if s == "" {
		return Cursor{}, nil
	}
	if n := len(s) % 4; n != 0 {
		s += strings.Repeat("=", 4-n)
	}
	b, err := base64.URLEncoding.DecodeString(s)
	if err != nil {
		return Cursor{}, err
	}
	return Cursor{b}, nil
}
