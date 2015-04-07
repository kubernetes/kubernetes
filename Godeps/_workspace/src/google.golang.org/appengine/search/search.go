// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

/*
Package search provides a client for App Engine's search service.

Indexes contains documents, and a document's contents are a mapping from case-
sensitive field names to values. In Go, documents are represented by struct
pointers, and the valid types for a struct's fields are:
  - string,
  - search.Atom,
  - search.HTML,
  - time.Time (stored with millisecond precision),
  - float64 (value between -2,147,483,647 and 2,147,483,647 inclusive),
  - appengine.GeoPoint.

Documents can also be represented by any type implementing the FieldLoadSaver
interface.

Example code:

	type Doc struct {
		Author   string
		Comment  string
		Creation time.Time
	}

	index, err := search.Open("comments")
	if err != nil {
		return err
	}
	newID, err := index.Put(c, "", &Doc{
		Author:   "gopher",
		Comment:  "the truth of the matter",
		Creation: time.Now(),
	})
	if err != nil {
		return err
	}

Searching an index for a query will result in an iterator. As with an iterator
from package datastore, pass a destination struct to Next to decode the next
result. Next will return Done when the iterator is exhausted.

	for t := index.Search(c, "Comment:truth", nil); ; {
		var doc Doc
		id, err := t.Next(&doc)
		if err == search.Done {
			break
		}
		if err != nil {
			return err
		}
		fmt.Fprintf(w, "%s -> %#v\n", id, doc)
	}

Call List to iterate over documents.

	for t := index.List(c, nil); ; {
		var doc Doc
		id, err := t.Next(&doc)
		if err == search.Done {
			break
		}
		if err != nil {
			return err
		}
		fmt.Fprintf(w, "%s -> %#v\n", id, doc)
	}

A single document can also be retrieved by its ID. Pass a destination struct
to Get to hold the resulting document.

	var doc Doc
	err := index.Get(c, id, &doc)
	if err != nil {
		return err
	}

Queries are expressed as strings, plus some optional parameters. The query
language is described at
https://cloud.google.com/appengine/docs/go/search/query_strings

Note that in Go, field names come from the struct field definition and begin
with an upper case letter.
*/
package search

// TODO: let Put specify the document language: "en", "fr", etc. Also: order_id?? storage??
// TODO: Index.GetAll (or Iterator.GetAll)?
// TODO: struct <-> protobuf tests.
// TODO: enforce Python's MIN_NUMBER_VALUE and MIN_DATE (which would disallow a zero
// time.Time)? _MAXIMUM_STRING_LENGTH?

import (
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/search"
)

var (
	// ErrInvalidDocumentType is returned when methods like Put, Get or Next
	// are passed a dst or src argument of invalid type.
	ErrInvalidDocumentType = errors.New("search: invalid document type")

	// ErrNoSuchDocument is returned when no document was found for a given ID.
	ErrNoSuchDocument = errors.New("search: no such document")
)

// ErrFieldMismatch is returned when a field is to be loaded into a different
// than the one it was stored from, or when a field is missing or unexported in
// the destination struct.
type ErrFieldMismatch struct {
	FieldName string
	Reason    string
}

func (e *ErrFieldMismatch) Error() string {
	return fmt.Sprintf("search: cannot load field %q: %s", e.FieldName, e.Reason)
}

// Atom is a document field whose contents are indexed as a single indivisible
// string.
type Atom string

// HTML is a document field whose contents are indexed as HTML. Only text nodes
// are indexed: "foo<b>bar" will be treated as "foobar".
type HTML string

// validIndexNameOrDocID is the Go equivalent of Python's
// _ValidateVisiblePrintableAsciiNotReserved.
func validIndexNameOrDocID(s string) bool {
	if strings.HasPrefix(s, "!") {
		return false
	}
	for _, c := range s {
		if c < 0x21 || 0x7f <= c {
			return false
		}
	}
	return true
}

var (
	fieldNameRE = regexp.MustCompile(`^[A-Z][A-Za-z0-9_]*$`)
	languageRE  = regexp.MustCompile(`^[a-z]{2}$`)
)

// validFieldName is the Go equivalent of Python's _CheckFieldName.
func validFieldName(s string) bool {
	return len(s) <= 500 && fieldNameRE.MatchString(s)
}

// validDocRank checks that the ranks is in the range [0, 2^31).
func validDocRank(r int) bool {
	return 0 <= r && r <= (1<<31-1)
}

// validLanguage checks that a language looks like ISO 639-1.
func validLanguage(s string) bool {
	return languageRE.MatchString(s)
}

// validFloat checks that f is in the range [-2147483647, 2147483647].
func validFloat(f float64) bool {
	return -(1<<31-1) <= f && f <= (1<<31-1)
}

// Index is an index of documents.
type Index struct {
	spec pb.IndexSpec
}

// orderIDEpoch forms the basis for populating OrderId on documents.
var orderIDEpoch = time.Date(2011, 1, 1, 0, 0, 0, 0, time.UTC)

// Open opens the index with the given name. The index is created if it does
// not already exist.
//
// The name is a human-readable ASCII string. It must contain no whitespace
// characters and not start with "!".
func Open(name string) (*Index, error) {
	if !validIndexNameOrDocID(name) {
		return nil, fmt.Errorf("search: invalid index name %q", name)
	}
	return &Index{
		spec: pb.IndexSpec{
			Name: &name,
		},
	}, nil
}

// Put saves src to the index. If id is empty, a new ID is allocated by the
// service and returned. If id is not empty, any existing index entry for that
// ID is replaced.
//
// The ID is a human-readable ASCII string. It must contain no whitespace
// characters and not start with "!".
//
// src must be a non-nil struct pointer or implement the FieldLoadSaver
// interface.
func (x *Index) Put(c appengine.Context, id string, src interface{}) (string, error) {
	fields, meta, err := saveDoc(src)
	if err != nil {
		return "", err
	}
	d := &pb.Document{
		Field:   fields,
		OrderId: proto.Int32(int32(time.Since(orderIDEpoch).Seconds())),
	}
	if meta != nil {
		if meta.Rank != 0 {
			if !validDocRank(meta.Rank) {
				return "", fmt.Errorf("search: invalid rank %d, must be [0, 2^31)", meta.Rank)
			}
			*d.OrderId = int32(meta.Rank)
		}
	}
	if id != "" {
		if !validIndexNameOrDocID(id) {
			return "", fmt.Errorf("search: invalid ID %q", id)
		}
		d.Id = proto.String(id)
	}
	req := &pb.IndexDocumentRequest{
		Params: &pb.IndexDocumentParams{
			Document:  []*pb.Document{d},
			IndexSpec: &x.spec,
		},
	}
	res := &pb.IndexDocumentResponse{}
	if err := c.Call("search", "IndexDocument", req, res, nil); err != nil {
		return "", err
	}
	if len(res.Status) > 0 {
		if s := res.Status[0]; s.GetCode() != pb.SearchServiceError_OK {
			return "", fmt.Errorf("search: %s: %s", s.GetCode(), s.GetErrorDetail())
		}
	}
	if len(res.Status) != 1 || len(res.DocId) != 1 {
		return "", fmt.Errorf("search: internal error: wrong number of results (%d Statuses, %d DocIDs)",
			len(res.Status), len(res.DocId))
	}
	return res.DocId[0], nil
}

// Get loads the document with the given ID into dst.
//
// The ID is a human-readable ASCII string. It must be non-empty, contain no
// whitespace characters and not start with "!".
//
// dst must be a non-nil struct pointer or implement the FieldLoadSaver
// interface.
//
// ErrFieldMismatch is returned when a field is to be loaded into a different
// type than the one it was stored from, or when a field is missing or
// unexported in the destination struct. ErrFieldMismatch is only returned if
// dst is a struct pointer. It is up to the callee to decide whether this error
// is fatal, recoverable, or ignorable.
func (x *Index) Get(c appengine.Context, id string, dst interface{}) error {
	if id == "" || !validIndexNameOrDocID(id) {
		return fmt.Errorf("search: invalid ID %q", id)
	}
	req := &pb.ListDocumentsRequest{
		Params: &pb.ListDocumentsParams{
			IndexSpec:  &x.spec,
			StartDocId: proto.String(id),
			Limit:      proto.Int32(1),
		},
	}
	res := &pb.ListDocumentsResponse{}
	if err := c.Call("search", "ListDocuments", req, res, nil); err != nil {
		return err
	}
	if res.Status == nil || res.Status.GetCode() != pb.SearchServiceError_OK {
		return fmt.Errorf("search: %s: %s", res.Status.GetCode(), res.Status.GetErrorDetail())
	}
	if len(res.Document) != 1 || res.Document[0].GetId() != id {
		return ErrNoSuchDocument
	}
	metadata := &DocumentMetadata{
		Rank: int(res.Document[0].GetOrderId()),
	}
	return loadDoc(dst, res.Document[0].Field, nil, metadata)
}

// Delete deletes a document from the index.
func (x *Index) Delete(c appengine.Context, id string) error {
	req := &pb.DeleteDocumentRequest{
		Params: &pb.DeleteDocumentParams{
			DocId:     []string{id},
			IndexSpec: &x.spec,
		},
	}
	res := &pb.DeleteDocumentResponse{}
	if err := c.Call("search", "DeleteDocument", req, res, nil); err != nil {
		return err
	}
	if len(res.Status) != 1 {
		return fmt.Errorf("search: internal error: wrong number of results (%d)", len(res.Status))
	}
	if s := res.Status[0]; s.GetCode() != pb.SearchServiceError_OK {
		return fmt.Errorf("search: %s: %s", s.GetCode(), s.GetErrorDetail())
	}
	return nil
}

// List lists all of the documents in an index. The documents are returned in
// increasing ID order.
func (x *Index) List(c appengine.Context, opts *ListOptions) *Iterator {
	t := &Iterator{
		c:             c,
		index:         x,
		count:         -1,
		listInclusive: true,
		more:          moreList,
		limit:         -1,
	}
	if opts != nil {
		t.listStartID = opts.StartID
		if opts.Limit > 0 {
			t.limit = opts.Limit
		}
		t.idsOnly = opts.IDsOnly
	}
	return t
}

func moreList(t *Iterator) error {
	req := &pb.ListDocumentsRequest{
		Params: &pb.ListDocumentsParams{
			IndexSpec: &t.index.spec,
		},
	}
	if t.listStartID != "" {
		req.Params.StartDocId = &t.listStartID
		req.Params.IncludeStartDoc = &t.listInclusive
	}
	if t.limit > 0 {
		req.Params.Limit = proto.Int32(int32(t.limit))
	}
	if t.idsOnly {
		req.Params.KeysOnly = &t.idsOnly
	}

	res := &pb.ListDocumentsResponse{}
	if err := t.c.Call("search", "ListDocuments", req, res, nil); err != nil {
		return err
	}
	if res.Status == nil || res.Status.GetCode() != pb.SearchServiceError_OK {
		return fmt.Errorf("search: %s: %s", res.Status.GetCode(), res.Status.GetErrorDetail())
	}
	t.listRes = res.Document
	t.listStartID, t.listInclusive, t.more = "", false, nil
	if len(res.Document) != 0 {
		if id := res.Document[len(res.Document)-1].GetId(); id != "" {
			t.listStartID, t.more = id, moreList
		}
	}
	return nil
}

// ListOptions are the options for listing documents in an index. Passing a nil
// *ListOptions is equivalent to using the default values.
type ListOptions struct {
	// StartID is the inclusive lower bound for the ID of the returned
	// documents. The zero value means all documents will be returned.
	StartID string

	// Limit is the maximum number of documents to return. The zero value
	// indicates no limit.
	Limit int

	// IDsOnly indicates that only document IDs should be returned for the list
	// operation; no document fields are populated.
	IDsOnly bool
}

// Search searches the index for the given query.
func (x *Index) Search(c appengine.Context, query string, opts *SearchOptions) *Iterator {
	t := &Iterator{
		c:           c,
		index:       x,
		searchQuery: query,
		more:        moreSearch,
		limit:       -1,
	}
	if opts != nil {
		if opts.Limit > 0 {
			t.limit = opts.Limit
		}
		t.fields = opts.Fields
		t.idsOnly = opts.IDsOnly
		t.sort = opts.Sort
		t.exprs = opts.Expressions
	}
	return t
}

func moreSearch(t *Iterator) error {
	req := &pb.SearchRequest{
		Params: &pb.SearchParams{
			IndexSpec:  &t.index.spec,
			Query:      &t.searchQuery,
			CursorType: pb.SearchParams_SINGLE.Enum(),
			FieldSpec: &pb.FieldSpec{
				Name: t.fields,
			},
		},
	}
	if t.limit > 0 {
		req.Params.Limit = proto.Int32(int32(t.limit))
	}
	if t.idsOnly {
		req.Params.KeysOnly = &t.idsOnly
	}
	if t.sort != nil {
		if err := sortToProto(t.sort, req.Params); err != nil {
			return err
		}
	}
	for _, e := range t.exprs {
		req.Params.FieldSpec.Expression = append(req.Params.FieldSpec.Expression, &pb.FieldSpec_Expression{
			Name:       proto.String(e.Name),
			Expression: proto.String(e.Expr),
		})
	}

	if t.searchCursor != nil {
		req.Params.Cursor = t.searchCursor
	}
	res := &pb.SearchResponse{}
	if err := t.c.Call("search", "Search", req, res, nil); err != nil {
		return err
	}
	if res.Status == nil || res.Status.GetCode() != pb.SearchServiceError_OK {
		return fmt.Errorf("search: %s: %s", res.Status.GetCode(), res.Status.GetErrorDetail())
	}
	t.searchRes = res.Result
	t.count = int(*res.MatchedCount)
	if res.Cursor != nil {
		t.searchCursor, t.more = res.Cursor, moreSearch
	} else {
		t.searchCursor, t.more = nil, nil
	}
	return nil
}

// SearchOptions are the options for searching an index. Passing a nil
// *SearchOptions is equivalent to using the default values.
type SearchOptions struct {
	// Limit is the maximum number of documents to return. The zero value
	// indicates no limit.
	Limit int

	// IDsOnly indicates that only document IDs should be returned for the search
	// operation; no document fields are populated.
	IDsOnly bool

	// Sort controls the ordering of search results.
	Sort *SortOptions

	// Fields specifies which document fields to include in the results. If omitted,
	// all document fields are returned. No more than 100 fields may be specified.
	Fields []string

	// Expressions specifies additional computed fields to add to each returned
	// document.
	Expressions []FieldExpression

	// TODO: cursor, offset, maybe others.
}

// FieldExpression defines a custom expression to evaluate for each result.
type FieldExpression struct {
	// Name is the name to use for the computed field.
	Name string

	// Expr is evaluated to provide a custom content snippet for each document.
	// See https://cloud.google.com/appengine/docs/go/search/options for
	// the supported expression syntax.
	Expr string
}

// SortOptions control the ordering and scoring of search results.
type SortOptions struct {
	// Expressions is a slice of expressions representing a multi-dimensional
	// sort.
	Expressions []SortExpression

	// Scorer, when specified, will cause the documents to be scored according to
	// search term frequency.
	Scorer Scorer

	// Limit is the maximum number of objects to score and/or sort. Limit cannot
	// be more than 10,000. The zero value indicates a default limit.
	Limit int
}

// SortExpression defines a single dimension for sorting a document.
type SortExpression struct {
	// Expr is evaluated to provide a sorting value for each document.
	// See https://cloud.google.com/appengine/docs/go/search/options for
	// the supported expression syntax.
	Expr string

	// Reverse causes the documents to be sorted in ascending order.
	Reverse bool

	// The default value to use when no field is present or the expresion
	// cannot be calculated for a document. For text sorts, Default must
	// be of type string; for numeric sorts, float64.
	Default interface{}
}

// A Scorer defines how a document is scored.
type Scorer interface {
	toProto(*pb.ScorerSpec)
}

type enumScorer struct {
	enum pb.ScorerSpec_Scorer
}

func (e enumScorer) toProto(spec *pb.ScorerSpec) {
	spec.Scorer = e.enum.Enum()
}

var (
	// MatchScorer assigns a score based on term frequency in a document.
	MatchScorer Scorer = enumScorer{pb.ScorerSpec_MATCH_SCORER}

	// RescoringMatchScorer assigns a score based on the quality of the query
	// match. It is similar to a MatchScorer but uses a more complex scoring
	// algorithm based on match term frequency and other factors like field type.
	// Please be aware that this algorithm is continually refined and can change
	// over time without notice. This means that the ordering of search results
	// that use this scorer can also change without notice.
	RescoringMatchScorer Scorer = enumScorer{pb.ScorerSpec_RESCORING_MATCH_SCORER}
)

func sortToProto(sort *SortOptions, params *pb.SearchParams) error {
	for _, e := range sort.Expressions {
		spec := &pb.SortSpec{
			SortExpression: proto.String(e.Expr),
		}
		if e.Reverse {
			spec.SortDescending = proto.Bool(false)
		}
		if e.Default != nil {
			switch d := e.Default.(type) {
			case float64:
				spec.DefaultValueNumeric = &d
			case string:
				spec.DefaultValueText = &d
			default:
				return fmt.Errorf("search: invalid Default type %T for expression %q", d, e.Expr)
			}
		}
		params.SortSpec = append(params.SortSpec, spec)
	}

	spec := &pb.ScorerSpec{}
	if sort.Limit > 0 {
		spec.Limit = proto.Int32(int32(sort.Limit))
		params.ScorerSpec = spec
	}
	if sort.Scorer != nil {
		sort.Scorer.toProto(spec)
		params.ScorerSpec = spec
	}

	return nil
}

// Iterator is the result of searching an index for a query or listing an
// index.
type Iterator struct {
	c     appengine.Context
	index *Index
	err   error

	listRes       []*pb.Document
	listStartID   string
	listInclusive bool

	searchRes    []*pb.SearchResult
	searchQuery  string
	searchCursor *string
	sort         *SortOptions

	fields []string
	exprs  []FieldExpression

	more func(*Iterator) error

	count   int
	limit   int // items left to return; -1 for unlimited.
	idsOnly bool
}

// Done is returned when a query iteration has completed.
var Done = errors.New("search: query has no more results")

// Count returns an approximation of the number of documents matched by the
// query. It is only valid to call for iterators returned by Search.
func (t *Iterator) Count() int { return t.count }

// Next returns the ID of the next result. When there are no more results,
// Done is returned as the error.
//
// dst must be a non-nil struct pointer, implement the FieldLoadSaver
// interface, or be a nil interface value. If a non-nil dst is provided, it
// will be filled with the indexed fields. dst is ignored if this iterator was
// created with an IDsOnly option.
func (t *Iterator) Next(dst interface{}) (string, error) {
	if t.err == nil && len(t.listRes)+len(t.searchRes) == 0 && t.more != nil {
		t.err = t.more(t)
	}
	if t.err != nil {
		return "", t.err
	}

	var doc *pb.Document
	var exprs []*pb.Field
	switch {
	case len(t.listRes) != 0:
		doc = t.listRes[0]
		t.listRes = t.listRes[1:]
	case len(t.searchRes) != 0:
		doc = t.searchRes[0].Document
		exprs = t.searchRes[0].Expression
		t.searchRes = t.searchRes[1:]
	default:
		return "", Done
	}
	if doc == nil {
		return "", errors.New("search: internal error: no document returned")
	}
	if !t.idsOnly && dst != nil {
		metadata := &DocumentMetadata{
			Rank: int(doc.GetOrderId()),
		}
		if err := loadDoc(dst, doc.Field, exprs, metadata); err != nil {
			return "", err
		}
	}
	if t.limit > 0 {
		t.limit--
		if t.limit == 0 {
			t.more = nil // prevent further fetches
		}
	}
	return doc.GetId(), nil
}

// saveDoc converts from a struct pointer or FieldLoadSaver to protobufs.
func saveDoc(src interface{}) ([]*pb.Field, *DocumentMetadata, error) {
	var err error
	var fields []Field
	var meta *DocumentMetadata
	switch x := src.(type) {
	case FieldLoadSaver:
		fields, meta, err = x.Save()
	default:
		fields, err = SaveStruct(src)
	}
	if err != nil {
		return nil, nil, err
	}
	f, err := fieldsToProto(fields)
	return f, meta, err
}

func fieldsToProto(src []Field) ([]*pb.Field, error) {
	// Maps to catch duplicate time or numeric fields.
	timeFields, numericFields := make(map[string]bool), make(map[string]bool)
	dst := make([]*pb.Field, 0, len(src))
	for _, f := range src {
		if !validFieldName(f.Name) {
			return nil, fmt.Errorf("search: invalid field name %q", f.Name)
		}
		fieldValue := &pb.FieldValue{}
		switch x := f.Value.(type) {
		case string:
			fieldValue.Type = pb.FieldValue_TEXT.Enum()
			fieldValue.StringValue = proto.String(x)
		case Atom:
			fieldValue.Type = pb.FieldValue_ATOM.Enum()
			fieldValue.StringValue = proto.String(string(x))
		case HTML:
			fieldValue.Type = pb.FieldValue_HTML.Enum()
			fieldValue.StringValue = proto.String(string(x))
		case time.Time:
			if timeFields[f.Name] {
				return nil, fmt.Errorf("search: duplicate time field %q", f.Name)
			}
			timeFields[f.Name] = true
			fieldValue.Type = pb.FieldValue_DATE.Enum()
			fieldValue.StringValue = proto.String(strconv.FormatInt(x.UnixNano()/1e6, 10))
		case float64:
			if numericFields[f.Name] {
				return nil, fmt.Errorf("search: duplicate numeric field %q", f.Name)
			}
			if !validFloat(x) {
				return nil, fmt.Errorf("search: numeric field %q with invalid value %f", f.Name, x)
			}
			numericFields[f.Name] = true
			fieldValue.Type = pb.FieldValue_NUMBER.Enum()
			fieldValue.StringValue = proto.String(strconv.FormatFloat(x, 'e', -1, 64))
		case appengine.GeoPoint:
			if !x.Valid() {
				return nil, fmt.Errorf(
					"search: GeoPoint field %q with invalid value %v",
					f.Name, x)
			}
			fieldValue.Type = pb.FieldValue_GEO.Enum()
			fieldValue.Geo = &pb.FieldValue_Geo{
				Lat: proto.Float64(x.Lat),
				Lng: proto.Float64(x.Lng),
			}
		default:
			return nil, fmt.Errorf("search: unsupported field type: %v", reflect.TypeOf(f.Value))
		}
		if f.Language != "" {
			switch f.Value.(type) {
			case string, HTML:
				if !validLanguage(f.Language) {
					return nil, fmt.Errorf("search: invalid language for field %q: %q", f.Name, f.Language)
				}
				fieldValue.Language = proto.String(f.Language)
			default:
				return nil, fmt.Errorf("search: setting language not supported for field %q of type %T", f.Name, f.Value)
			}
		}
		if p := fieldValue.StringValue; p != nil && !utf8.ValidString(*p) {
			return nil, fmt.Errorf("search: %q field is invalid UTF-8: %q", f.Name, *p)
		}
		dst = append(dst, &pb.Field{
			Name:  proto.String(f.Name),
			Value: fieldValue,
		})
	}
	return dst, nil
}

// loadDoc converts from protobufs and document metadata to a struct pointer or
// FieldLoadSaver/FieldMetadataLoadSaver. Two slices of fields may be provided:
// src represents the document's stored fields; exprs is the derived expressions
// requested by the developer. The latter may be empty.
func loadDoc(dst interface{}, src, exprs []*pb.Field, meta *DocumentMetadata) (err error) {
	fields, err := protoToFields(src)
	if err != nil {
		return err
	}
	if len(exprs) > 0 {
		exprFields, err := protoToFields(exprs)
		if err != nil {
			return err
		}
		// Mark each field as derived.
		for i := range exprFields {
			exprFields[i].Derived = true
		}
		fields = append(fields, exprFields...)
	}
	switch x := dst.(type) {
	case FieldLoadSaver:
		return x.Load(fields, meta)
	default:
		return LoadStruct(dst, fields)
	}
}

func protoToFields(fields []*pb.Field) ([]Field, error) {
	dst := make([]Field, 0, len(fields))
	for _, field := range fields {
		fieldValue := field.GetValue()
		f := Field{
			Name: field.GetName(),
		}
		switch fieldValue.GetType() {
		case pb.FieldValue_TEXT:
			f.Value = fieldValue.GetStringValue()
			f.Language = fieldValue.GetLanguage()
		case pb.FieldValue_ATOM:
			f.Value = Atom(fieldValue.GetStringValue())
		case pb.FieldValue_HTML:
			f.Value = HTML(fieldValue.GetStringValue())
			f.Language = fieldValue.GetLanguage()
		case pb.FieldValue_DATE:
			sv := fieldValue.GetStringValue()
			millis, err := strconv.ParseInt(sv, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("search: internal error: bad time.Time encoding %q: %v", sv, err)
			}
			f.Value = time.Unix(0, millis*1e6)
		case pb.FieldValue_NUMBER:
			sv := fieldValue.GetStringValue()
			x, err := strconv.ParseFloat(sv, 64)
			if err != nil {
				return nil, err
			}
			f.Value = x
		case pb.FieldValue_GEO:
			geoValue := fieldValue.GetGeo()
			geoPoint := appengine.GeoPoint{geoValue.GetLat(), geoValue.GetLng()}
			if !geoPoint.Valid() {
				return nil, fmt.Errorf("search: internal error: invalid GeoPoint encoding: %v", geoPoint)
			}
			f.Value = geoPoint
		default:
			return nil, fmt.Errorf("search: internal error: unknown data type %s", fieldValue.GetType())
		}
		dst = append(dst, f)
	}
	return dst, nil
}

func namespaceMod(m proto.Message, namespace string) {
	set := func(s **string) {
		if *s == nil {
			*s = &namespace
		}
	}
	switch m := m.(type) {
	case *pb.IndexDocumentRequest:
		set(&m.Params.IndexSpec.Namespace)
	case *pb.ListDocumentsRequest:
		set(&m.Params.IndexSpec.Namespace)
	case *pb.DeleteDocumentRequest:
		set(&m.Params.IndexSpec.Namespace)
	case *pb.SearchRequest:
		set(&m.Params.IndexSpec.Namespace)
	}
}

func init() {
	internal.RegisterErrorCodeMap("search", pb.SearchServiceError_ErrorCode_name)
	internal.NamespaceMods["search"] = namespaceMod
}
