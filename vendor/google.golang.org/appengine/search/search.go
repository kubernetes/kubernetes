// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package search // import "google.golang.org/appengine/search"

// TODO: let Put specify the document language: "en", "fr", etc. Also: order_id?? storage??
// TODO: Index.GetAll (or Iterator.GetAll)?
// TODO: struct <-> protobuf tests.
// TODO: enforce Python's MIN_NUMBER_VALUE and MIN_DATE (which would disallow a zero
// time.Time)? _MAXIMUM_STRING_LENGTH?

import (
	"errors"
	"fmt"
	"math"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"

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
	fieldNameRE = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_]*$`)
	languageRE  = regexp.MustCompile(`^[a-z]{2}$`)
)

// validFieldName is the Go equivalent of Python's _CheckFieldName. It checks
// the validity of both field and facet names.
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
func (x *Index) Put(c context.Context, id string, src interface{}) (string, error) {
	d, err := saveDoc(src)
	if err != nil {
		return "", err
	}
	if id != "" {
		if !validIndexNameOrDocID(id) {
			return "", fmt.Errorf("search: invalid ID %q", id)
		}
		d.Id = proto.String(id)
	}
	// spec is modified by Call when applying the current Namespace, so copy it to
	// avoid retaining the namespace beyond the scope of the Call.
	spec := x.spec
	req := &pb.IndexDocumentRequest{
		Params: &pb.IndexDocumentParams{
			Document:  []*pb.Document{d},
			IndexSpec: &spec,
		},
	}
	res := &pb.IndexDocumentResponse{}
	if err := internal.Call(c, "search", "IndexDocument", req, res); err != nil {
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
func (x *Index) Get(c context.Context, id string, dst interface{}) error {
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
	if err := internal.Call(c, "search", "ListDocuments", req, res); err != nil {
		return err
	}
	if res.Status == nil || res.Status.GetCode() != pb.SearchServiceError_OK {
		return fmt.Errorf("search: %s: %s", res.Status.GetCode(), res.Status.GetErrorDetail())
	}
	if len(res.Document) != 1 || res.Document[0].GetId() != id {
		return ErrNoSuchDocument
	}
	return loadDoc(dst, res.Document[0], nil)
}

// Delete deletes a document from the index.
func (x *Index) Delete(c context.Context, id string) error {
	req := &pb.DeleteDocumentRequest{
		Params: &pb.DeleteDocumentParams{
			DocId:     []string{id},
			IndexSpec: &x.spec,
		},
	}
	res := &pb.DeleteDocumentResponse{}
	if err := internal.Call(c, "search", "DeleteDocument", req, res); err != nil {
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
func (x *Index) List(c context.Context, opts *ListOptions) *Iterator {
	t := &Iterator{
		c:             c,
		index:         x,
		count:         -1,
		listInclusive: true,
		more:          moreList,
	}
	if opts != nil {
		t.listStartID = opts.StartID
		t.limit = opts.Limit
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
	if err := internal.Call(t.c, "search", "ListDocuments", req, res); err != nil {
		return err
	}
	if res.Status == nil || res.Status.GetCode() != pb.SearchServiceError_OK {
		return fmt.Errorf("search: %s: %s", res.Status.GetCode(), res.Status.GetErrorDetail())
	}
	t.listRes = res.Document
	t.listStartID, t.listInclusive, t.more = "", false, nil
	if len(res.Document) != 0 && t.limit <= 0 {
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
func (x *Index) Search(c context.Context, query string, opts *SearchOptions) *Iterator {
	t := &Iterator{
		c:           c,
		index:       x,
		searchQuery: query,
		more:        moreSearch,
	}
	if opts != nil {
		if opts.Cursor != "" {
			if opts.Offset != 0 {
				return errIter("at most one of Cursor and Offset may be specified")
			}
			t.searchCursor = proto.String(string(opts.Cursor))
		}
		t.limit = opts.Limit
		t.fields = opts.Fields
		t.idsOnly = opts.IDsOnly
		t.sort = opts.Sort
		t.exprs = opts.Expressions
		t.refinements = opts.Refinements
		t.facetOpts = opts.Facets
		t.searchOffset = opts.Offset
		t.countAccuracy = opts.CountAccuracy
	}
	return t
}

func moreSearch(t *Iterator) error {
	// We use per-result (rather than single/per-page) cursors since this
	// lets us return a Cursor for every iterator document. The two cursor
	// types are largely interchangeable: a page cursor is the same as the
	// last per-result cursor in a given search response.
	req := &pb.SearchRequest{
		Params: &pb.SearchParams{
			IndexSpec:  &t.index.spec,
			Query:      &t.searchQuery,
			Cursor:     t.searchCursor,
			CursorType: pb.SearchParams_PER_RESULT.Enum(),
			FieldSpec: &pb.FieldSpec{
				Name: t.fields,
			},
		},
	}
	if t.limit > 0 {
		req.Params.Limit = proto.Int32(int32(t.limit))
	}
	if t.searchOffset > 0 {
		req.Params.Offset = proto.Int32(int32(t.searchOffset))
		t.searchOffset = 0
	}
	if t.countAccuracy > 0 {
		req.Params.MatchedCountAccuracy = proto.Int32(int32(t.countAccuracy))
	}
	if t.idsOnly {
		req.Params.KeysOnly = &t.idsOnly
	}
	if t.sort != nil {
		if err := sortToProto(t.sort, req.Params); err != nil {
			return err
		}
	}
	if t.refinements != nil {
		if err := refinementsToProto(t.refinements, req.Params); err != nil {
			return err
		}
	}
	for _, e := range t.exprs {
		req.Params.FieldSpec.Expression = append(req.Params.FieldSpec.Expression, &pb.FieldSpec_Expression{
			Name:       proto.String(e.Name),
			Expression: proto.String(e.Expr),
		})
	}
	for _, f := range t.facetOpts {
		if err := f.setParams(req.Params); err != nil {
			return fmt.Errorf("bad FacetSearchOption: %v", err)
		}
	}
	// Don't repeat facet search.
	t.facetOpts = nil

	res := &pb.SearchResponse{}
	if err := internal.Call(t.c, "search", "Search", req, res); err != nil {
		return err
	}
	if res.Status == nil || res.Status.GetCode() != pb.SearchServiceError_OK {
		return fmt.Errorf("search: %s: %s", res.Status.GetCode(), res.Status.GetErrorDetail())
	}
	t.searchRes = res.Result
	if len(res.FacetResult) > 0 {
		t.facetRes = res.FacetResult
	}
	t.count = int(*res.MatchedCount)
	if t.limit > 0 {
		t.more = nil
	} else {
		t.more = moreSearch
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

	// Facets controls what facet information is returned for these search results.
	// If no options are specified, no facet results will be returned.
	Facets []FacetSearchOption

	// Refinements filters the returned documents by requiring them to contain facets
	// with specific values. Refinements are applied in conjunction for facets with
	// different names, and in disjunction otherwise.
	Refinements []Facet

	// Cursor causes the results to commence with the first document after
	// the document associated with the cursor.
	Cursor Cursor

	// Offset specifies the number of documents to skip over before returning results.
	// When specified, Cursor must be nil.
	Offset int

	// CountAccuracy specifies the maximum result count that can be expected to
	// be accurate. If zero, the count accuracy defaults to 20.
	CountAccuracy int
}

// Cursor represents an iterator's position.
//
// The string value of a cursor is web-safe. It can be saved and restored
// for later use.
type Cursor string

// FieldExpression defines a custom expression to evaluate for each result.
type FieldExpression struct {
	// Name is the name to use for the computed field.
	Name string

	// Expr is evaluated to provide a custom content snippet for each document.
	// See https://cloud.google.com/appengine/docs/go/search/options for
	// the supported expression syntax.
	Expr string
}

// FacetSearchOption controls what facet information is returned in search results.
type FacetSearchOption interface {
	setParams(*pb.SearchParams) error
}

// AutoFacetDiscovery returns a FacetSearchOption which enables automatic facet
// discovery for the search. Automatic facet discovery looks for the facets
// which appear the most often in the aggregate in the matched documents.
//
// The maximum number of facets returned is controlled by facetLimit, and the
// maximum number of values per facet by facetLimit. A limit of zero indicates
// a default limit should be used.
func AutoFacetDiscovery(facetLimit, valueLimit int) FacetSearchOption {
	return &autoFacetOpt{facetLimit, valueLimit}
}

type autoFacetOpt struct {
	facetLimit, valueLimit int
}

const defaultAutoFacetLimit = 10 // As per python runtime search.py.

func (o *autoFacetOpt) setParams(params *pb.SearchParams) error {
	lim := int32(o.facetLimit)
	if lim == 0 {
		lim = defaultAutoFacetLimit
	}
	params.AutoDiscoverFacetCount = &lim
	if o.valueLimit > 0 {
		params.FacetAutoDetectParam = &pb.FacetAutoDetectParam{
			ValueLimit: proto.Int32(int32(o.valueLimit)),
		}
	}
	return nil
}

// FacetDiscovery returns a FacetSearchOption which selects a facet to be
// returned with the search results. By default, the most frequently
// occurring values for that facet will be returned. However, you can also
// specify a list of particular Atoms or specific Ranges to return.
func FacetDiscovery(name string, value ...interface{}) FacetSearchOption {
	return &facetOpt{name, value}
}

type facetOpt struct {
	name   string
	values []interface{}
}

func (o *facetOpt) setParams(params *pb.SearchParams) error {
	req := &pb.FacetRequest{Name: &o.name}
	params.IncludeFacet = append(params.IncludeFacet, req)
	if len(o.values) == 0 {
		return nil
	}
	vtype := reflect.TypeOf(o.values[0])
	reqParam := &pb.FacetRequestParam{}
	for _, v := range o.values {
		if reflect.TypeOf(v) != vtype {
			return errors.New("values must all be Atom, or must all be Range")
		}
		switch v := v.(type) {
		case Atom:
			reqParam.ValueConstraint = append(reqParam.ValueConstraint, string(v))
		case Range:
			rng, err := rangeToProto(v)
			if err != nil {
				return fmt.Errorf("invalid range: %v", err)
			}
			reqParam.Range = append(reqParam.Range, rng)
		default:
			return fmt.Errorf("unsupported value type %T", v)
		}
	}
	req.Params = reqParam
	return nil
}

// FacetDocumentDepth returns a FacetSearchOption which controls the number of
// documents to be evaluated with preparing facet results.
func FacetDocumentDepth(depth int) FacetSearchOption {
	return facetDepthOpt(depth)
}

type facetDepthOpt int

func (o facetDepthOpt) setParams(params *pb.SearchParams) error {
	params.FacetDepth = proto.Int32(int32(o))
	return nil
}

// FacetResult represents the number of times a particular facet and value
// appeared in the documents matching a search request.
type FacetResult struct {
	Facet

	// Count is the number of times this specific facet and value appeared in the
	// matching documents.
	Count int
}

// Range represents a numeric range with inclusive start and exclusive end.
// Start may be specified as math.Inf(-1) to indicate there is no minimum
// value, and End may similarly be specified as math.Inf(1); at least one of
// Start or End must be a finite number.
type Range struct {
	Start, End float64
}

var (
	negInf = math.Inf(-1)
	posInf = math.Inf(1)
)

// AtLeast returns a Range matching any value greater than, or equal to, min.
func AtLeast(min float64) Range {
	return Range{Start: min, End: posInf}
}

// LessThan returns a Range matching any value less than max.
func LessThan(max float64) Range {
	return Range{Start: negInf, End: max}
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

func refinementsToProto(refinements []Facet, params *pb.SearchParams) error {
	for _, r := range refinements {
		ref := &pb.FacetRefinement{
			Name: proto.String(r.Name),
		}
		switch v := r.Value.(type) {
		case Atom:
			ref.Value = proto.String(string(v))
		case Range:
			rng, err := rangeToProto(v)
			if err != nil {
				return fmt.Errorf("search: refinement for facet %q: %v", r.Name, err)
			}
			// Unfortunately there are two identical messages for identify Facet ranges.
			ref.Range = &pb.FacetRefinement_Range{Start: rng.Start, End: rng.End}
		default:
			return fmt.Errorf("search: unsupported refinement for facet %q of type %T", r.Name, v)
		}
		params.FacetRefinement = append(params.FacetRefinement, ref)
	}
	return nil
}

func rangeToProto(r Range) (*pb.FacetRange, error) {
	rng := &pb.FacetRange{}
	if r.Start != negInf {
		if !validFloat(r.Start) {
			return nil, errors.New("invalid value for Start")
		}
		rng.Start = proto.String(strconv.FormatFloat(r.Start, 'e', -1, 64))
	} else if r.End == posInf {
		return nil, errors.New("either Start or End must be finite")
	}
	if r.End != posInf {
		if !validFloat(r.End) {
			return nil, errors.New("invalid value for End")
		}
		rng.End = proto.String(strconv.FormatFloat(r.End, 'e', -1, 64))
	}
	return rng, nil
}

func protoToRange(rng *pb.FacetRefinement_Range) Range {
	r := Range{Start: negInf, End: posInf}
	if x, err := strconv.ParseFloat(rng.GetStart(), 64); err != nil {
		r.Start = x
	}
	if x, err := strconv.ParseFloat(rng.GetEnd(), 64); err != nil {
		r.End = x
	}
	return r
}

// Iterator is the result of searching an index for a query or listing an
// index.
type Iterator struct {
	c     context.Context
	index *Index
	err   error

	listRes       []*pb.Document
	listStartID   string
	listInclusive bool

	searchRes    []*pb.SearchResult
	facetRes     []*pb.FacetResult
	searchQuery  string
	searchCursor *string
	searchOffset int
	sort         *SortOptions

	fields      []string
	exprs       []FieldExpression
	refinements []Facet
	facetOpts   []FacetSearchOption

	more func(*Iterator) error

	count         int
	countAccuracy int
	limit         int // items left to return; 0 for unlimited.
	idsOnly       bool
}

// errIter returns an iterator that only returns the given error.
func errIter(err string) *Iterator {
	return &Iterator{
		err: errors.New(err),
	}
}

// Done is returned when a query iteration has completed.
var Done = errors.New("search: query has no more results")

// Count returns an approximation of the number of documents matched by the
// query. It is only valid to call for iterators returned by Search.
func (t *Iterator) Count() int { return t.count }

// fetchMore retrieves more results, if there are no errors or pending results.
func (t *Iterator) fetchMore() {
	if t.err == nil && len(t.listRes)+len(t.searchRes) == 0 && t.more != nil {
		t.err = t.more(t)
	}
}

// Next returns the ID of the next result. When there are no more results,
// Done is returned as the error.
//
// dst must be a non-nil struct pointer, implement the FieldLoadSaver
// interface, or be a nil interface value. If a non-nil dst is provided, it
// will be filled with the indexed fields. dst is ignored if this iterator was
// created with an IDsOnly option.
func (t *Iterator) Next(dst interface{}) (string, error) {
	t.fetchMore()
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
		t.searchCursor = t.searchRes[0].Cursor
		t.searchRes = t.searchRes[1:]
	default:
		return "", Done
	}
	if doc == nil {
		return "", errors.New("search: internal error: no document returned")
	}
	if !t.idsOnly && dst != nil {
		if err := loadDoc(dst, doc, exprs); err != nil {
			return "", err
		}
	}
	return doc.GetId(), nil
}

// Cursor returns the cursor associated with the current document (that is,
// the document most recently returned by a call to Next).
//
// Passing this cursor in a future call to Search will cause those results
// to commence with the first document after the current document.
func (t *Iterator) Cursor() Cursor {
	if t.searchCursor == nil {
		return ""
	}
	return Cursor(*t.searchCursor)
}

// Facets returns the facets found within the search results, if any facets
// were requested in the SearchOptions.
func (t *Iterator) Facets() ([][]FacetResult, error) {
	t.fetchMore()
	if t.err != nil && t.err != Done {
		return nil, t.err
	}

	var facets [][]FacetResult
	for _, f := range t.facetRes {
		fres := make([]FacetResult, 0, len(f.Value))
		for _, v := range f.Value {
			ref := v.Refinement
			facet := FacetResult{
				Facet: Facet{Name: ref.GetName()},
				Count: int(v.GetCount()),
			}
			if ref.Value != nil {
				facet.Value = Atom(*ref.Value)
			} else {
				facet.Value = protoToRange(ref.Range)
			}
			fres = append(fres, facet)
		}
		facets = append(facets, fres)
	}
	return facets, nil
}

// saveDoc converts from a struct pointer or
// FieldLoadSaver/FieldMetadataLoadSaver to the Document protobuf.
func saveDoc(src interface{}) (*pb.Document, error) {
	var err error
	var fields []Field
	var meta *DocumentMetadata
	switch x := src.(type) {
	case FieldLoadSaver:
		fields, meta, err = x.Save()
	default:
		fields, meta, err = saveStructWithMeta(src)
	}
	if err != nil {
		return nil, err
	}

	fieldsProto, err := fieldsToProto(fields)
	if err != nil {
		return nil, err
	}
	d := &pb.Document{
		Field:   fieldsProto,
		OrderId: proto.Int32(int32(time.Since(orderIDEpoch).Seconds())),
	}
	if meta != nil {
		if meta.Rank != 0 {
			if !validDocRank(meta.Rank) {
				return nil, fmt.Errorf("search: invalid rank %d, must be [0, 2^31)", meta.Rank)
			}
			*d.OrderId = int32(meta.Rank)
		}
		if len(meta.Facets) > 0 {
			facets, err := facetsToProto(meta.Facets)
			if err != nil {
				return nil, err
			}
			d.Facet = facets
		}
	}
	return d, nil
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

func facetsToProto(src []Facet) ([]*pb.Facet, error) {
	dst := make([]*pb.Facet, 0, len(src))
	for _, f := range src {
		if !validFieldName(f.Name) {
			return nil, fmt.Errorf("search: invalid facet name %q", f.Name)
		}
		facetValue := &pb.FacetValue{}
		switch x := f.Value.(type) {
		case Atom:
			if !utf8.ValidString(string(x)) {
				return nil, fmt.Errorf("search: %q facet is invalid UTF-8: %q", f.Name, x)
			}
			facetValue.Type = pb.FacetValue_ATOM.Enum()
			facetValue.StringValue = proto.String(string(x))
		case float64:
			if !validFloat(x) {
				return nil, fmt.Errorf("search: numeric facet %q with invalid value %f", f.Name, x)
			}
			facetValue.Type = pb.FacetValue_NUMBER.Enum()
			facetValue.StringValue = proto.String(strconv.FormatFloat(x, 'e', -1, 64))
		default:
			return nil, fmt.Errorf("search: unsupported facet type: %v", reflect.TypeOf(f.Value))
		}
		dst = append(dst, &pb.Facet{
			Name:  proto.String(f.Name),
			Value: facetValue,
		})
	}
	return dst, nil
}

// loadDoc converts from protobufs to a struct pointer or
// FieldLoadSaver/FieldMetadataLoadSaver. The src param provides the document's
// stored fields and facets, and any document metadata.  An additional slice of
// fields, exprs, may optionally be provided to contain any derived expressions
// requested by the developer.
func loadDoc(dst interface{}, src *pb.Document, exprs []*pb.Field) (err error) {
	fields, err := protoToFields(src.Field)
	if err != nil {
		return err
	}
	facets, err := protoToFacets(src.Facet)
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
	meta := &DocumentMetadata{
		Rank:   int(src.GetOrderId()),
		Facets: facets,
	}
	switch x := dst.(type) {
	case FieldLoadSaver:
		return x.Load(fields, meta)
	default:
		return loadStructWithMeta(dst, fields, meta)
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

func protoToFacets(facets []*pb.Facet) ([]Facet, error) {
	if len(facets) == 0 {
		return nil, nil
	}
	dst := make([]Facet, 0, len(facets))
	for _, facet := range facets {
		facetValue := facet.GetValue()
		f := Facet{
			Name: facet.GetName(),
		}
		switch facetValue.GetType() {
		case pb.FacetValue_ATOM:
			f.Value = Atom(facetValue.GetStringValue())
		case pb.FacetValue_NUMBER:
			sv := facetValue.GetStringValue()
			x, err := strconv.ParseFloat(sv, 64)
			if err != nil {
				return nil, err
			}
			f.Value = x
		default:
			return nil, fmt.Errorf("search: internal error: unknown data type %s", facetValue.GetType())
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
