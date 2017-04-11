// Copyright 2012 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package search

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine"
	"google.golang.org/appengine/internal/aetesting"
	pb "google.golang.org/appengine/internal/search"
)

type TestDoc struct {
	String   string
	Atom     Atom
	HTML     HTML
	Float    float64
	Location appengine.GeoPoint
	Time     time.Time
}

type FieldListWithMeta struct {
	Fields FieldList
	Meta   *DocumentMetadata
}

func (f *FieldListWithMeta) Load(fields []Field, meta *DocumentMetadata) error {
	f.Meta = meta
	return f.Fields.Load(fields, nil)
}

func (f *FieldListWithMeta) Save() ([]Field, *DocumentMetadata, error) {
	fields, _, err := f.Fields.Save()
	return fields, f.Meta, err
}

// Assert that FieldListWithMeta satisfies FieldLoadSaver
var _ FieldLoadSaver = &FieldListWithMeta{}

var (
	float       = 3.14159
	floatOut    = "3.14159e+00"
	latitude    = 37.3894
	longitude   = 122.0819
	testGeo     = appengine.GeoPoint{latitude, longitude}
	testString  = "foo<b>bar"
	testTime    = time.Unix(1337324400, 0)
	testTimeOut = "1337324400000"
	searchMeta  = &DocumentMetadata{
		Rank: 42,
	}
	searchDoc = TestDoc{
		String:   testString,
		Atom:     Atom(testString),
		HTML:     HTML(testString),
		Float:    float,
		Location: testGeo,
		Time:     testTime,
	}
	searchFields = FieldList{
		Field{Name: "String", Value: testString},
		Field{Name: "Atom", Value: Atom(testString)},
		Field{Name: "HTML", Value: HTML(testString)},
		Field{Name: "Float", Value: float},
		Field{Name: "Location", Value: testGeo},
		Field{Name: "Time", Value: testTime},
	}
	// searchFieldsWithLang is a copy of the searchFields with the Language field
	// set on text/HTML Fields.
	searchFieldsWithLang = FieldList{}
	protoFields          = []*pb.Field{
		newStringValueField("String", testString, pb.FieldValue_TEXT),
		newStringValueField("Atom", testString, pb.FieldValue_ATOM),
		newStringValueField("HTML", testString, pb.FieldValue_HTML),
		newStringValueField("Float", floatOut, pb.FieldValue_NUMBER),
		{
			Name: proto.String("Location"),
			Value: &pb.FieldValue{
				Geo: &pb.FieldValue_Geo{
					Lat: proto.Float64(latitude),
					Lng: proto.Float64(longitude),
				},
				Type: pb.FieldValue_GEO.Enum(),
			},
		},
		newStringValueField("Time", testTimeOut, pb.FieldValue_DATE),
	}
)

func init() {
	for _, f := range searchFields {
		if f.Name == "String" || f.Name == "HTML" {
			f.Language = "en"
		}
		searchFieldsWithLang = append(searchFieldsWithLang, f)
	}
}

func newStringValueField(name, value string, valueType pb.FieldValue_ContentType) *pb.Field {
	return &pb.Field{
		Name: proto.String(name),
		Value: &pb.FieldValue{
			StringValue: proto.String(value),
			Type:        valueType.Enum(),
		},
	}
}

func newFacet(name, value string, valueType pb.FacetValue_ContentType) *pb.Facet {
	return &pb.Facet{
		Name: proto.String(name),
		Value: &pb.FacetValue{
			StringValue: proto.String(value),
			Type:        valueType.Enum(),
		},
	}
}

func TestValidIndexNameOrDocID(t *testing.T) {
	testCases := []struct {
		s    string
		want bool
	}{
		{"", true},
		{"!", false},
		{"$", true},
		{"!bad", false},
		{"good!", true},
		{"alsoGood", true},
		{"has spaces", false},
		{"is_inva\xffid_UTF-8", false},
		{"is_non-ASCïI", false},
		{"underscores_are_ok", true},
	}
	for _, tc := range testCases {
		if got := validIndexNameOrDocID(tc.s); got != tc.want {
			t.Errorf("%q: got %v, want %v", tc.s, got, tc.want)
		}
	}
}

func TestLoadDoc(t *testing.T) {
	got, want := TestDoc{}, searchDoc
	if err := loadDoc(&got, &pb.Document{Field: protoFields}, nil); err != nil {
		t.Fatalf("loadDoc: %v", err)
	}
	if got != want {
		t.Errorf("loadDoc: got %v, wanted %v", got, want)
	}
}

func TestSaveDoc(t *testing.T) {
	got, err := saveDoc(&searchDoc)
	if err != nil {
		t.Fatalf("saveDoc: %v", err)
	}
	want := protoFields
	if !reflect.DeepEqual(got.Field, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}
}

func TestLoadFieldList(t *testing.T) {
	var got FieldList
	want := searchFieldsWithLang
	if err := loadDoc(&got, &pb.Document{Field: protoFields}, nil); err != nil {
		t.Fatalf("loadDoc: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}
}

func TestLangFields(t *testing.T) {
	fl := &FieldList{
		{Name: "Foo", Value: "I am English", Language: "en"},
		{Name: "Bar", Value: "私は日本人だ", Language: "jp"},
	}
	var got FieldList
	doc, err := saveDoc(fl)
	if err != nil {
		t.Fatalf("saveDoc: %v", err)
	}
	if err := loadDoc(&got, doc, nil); err != nil {
		t.Fatalf("loadDoc: %v", err)
	}
	if want := fl; !reflect.DeepEqual(&got, want) {
		t.Errorf("got  %v\nwant %v", got, want)
	}
}

func TestSaveFieldList(t *testing.T) {
	got, err := saveDoc(&searchFields)
	if err != nil {
		t.Fatalf("saveDoc: %v", err)
	}
	want := protoFields
	if !reflect.DeepEqual(got.Field, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}
}

func TestLoadFieldAndExprList(t *testing.T) {
	var got, want FieldList
	for i, f := range searchFieldsWithLang {
		f.Derived = (i >= 2) // First 2 elements are "fields", next are "expressions".
		want = append(want, f)
	}
	doc, expr := &pb.Document{Field: protoFields[:2]}, protoFields[2:]
	if err := loadDoc(&got, doc, expr); err != nil {
		t.Fatalf("loadDoc: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got  %v\nwant %v", got, want)
	}
}

func TestLoadMeta(t *testing.T) {
	var got FieldListWithMeta
	want := FieldListWithMeta{
		Meta:   searchMeta,
		Fields: searchFieldsWithLang,
	}
	doc := &pb.Document{
		Field:   protoFields,
		OrderId: proto.Int32(42),
	}
	if err := loadDoc(&got, doc, nil); err != nil {
		t.Fatalf("loadDoc: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}
}

func TestSaveMeta(t *testing.T) {
	got, err := saveDoc(&FieldListWithMeta{
		Meta:   searchMeta,
		Fields: searchFields,
	})
	if err != nil {
		t.Fatalf("saveDoc: %v", err)
	}
	want := &pb.Document{
		Field:   protoFields,
		OrderId: proto.Int32(42),
	}
	if !proto.Equal(got, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}
}

func TestLoadSaveWithStruct(t *testing.T) {
	type gopher struct {
		Name string
		Info string  `search:"about"`
		Legs float64 `search:",facet"`
		Fuzz Atom    `search:"Fur,facet"`
	}

	doc := gopher{"Gopher", "Likes slide rules.", 4, Atom("furry")}
	pb := &pb.Document{
		Field: []*pb.Field{
			newStringValueField("Name", "Gopher", pb.FieldValue_TEXT),
			newStringValueField("about", "Likes slide rules.", pb.FieldValue_TEXT),
		},
		Facet: []*pb.Facet{
			newFacet("Legs", "4e+00", pb.FacetValue_NUMBER),
			newFacet("Fur", "furry", pb.FacetValue_ATOM),
		},
	}

	var gotDoc gopher
	if err := loadDoc(&gotDoc, pb, nil); err != nil {
		t.Fatalf("loadDoc: %v", err)
	}
	if !reflect.DeepEqual(gotDoc, doc) {
		t.Errorf("loading doc\ngot  %v\nwant %v", gotDoc, doc)
	}

	gotPB, err := saveDoc(&doc)
	if err != nil {
		t.Fatalf("saveDoc: %v", err)
	}
	gotPB.OrderId = nil // Don't test: it's time dependent.
	if !proto.Equal(gotPB, pb) {
		t.Errorf("saving doc\ngot  %v\nwant %v", gotPB, pb)
	}
}

func TestValidFieldNames(t *testing.T) {
	testCases := []struct {
		name  string
		valid bool
	}{
		{"Normal", true},
		{"Also_OK_123", true},
		{"Not so great", false},
		{"lower_case", true},
		{"Exclaim!", false},
		{"Hello세상아 안녕", false},
		{"", false},
		{"Hεllo", false},
		{strings.Repeat("A", 500), true},
		{strings.Repeat("A", 501), false},
	}

	for _, tc := range testCases {
		_, err := saveDoc(&FieldList{
			Field{Name: tc.name, Value: "val"},
		})
		if err != nil && !strings.Contains(err.Error(), "invalid field name") {
			t.Errorf("unexpected err %q for field name %q", err, tc.name)
		}
		if (err == nil) != tc.valid {
			t.Errorf("field %q: expected valid %t, received err %v", tc.name, tc.valid, err)
		}
	}
}

func TestValidLangs(t *testing.T) {
	testCases := []struct {
		field Field
		valid bool
	}{
		{Field{Name: "Foo", Value: "String", Language: ""}, true},
		{Field{Name: "Foo", Value: "String", Language: "en"}, true},
		{Field{Name: "Foo", Value: "String", Language: "aussie"}, false},
		{Field{Name: "Foo", Value: "String", Language: "12"}, false},
		{Field{Name: "Foo", Value: HTML("String"), Language: "en"}, true},
		{Field{Name: "Foo", Value: Atom("String"), Language: "en"}, false},
		{Field{Name: "Foo", Value: 42, Language: "en"}, false},
	}

	for _, tt := range testCases {
		_, err := saveDoc(&FieldList{tt.field})
		if err == nil != tt.valid {
			t.Errorf("Field %v, got error %v, wanted valid %t", tt.field, err, tt.valid)
		}
	}
}

func TestDuplicateFields(t *testing.T) {
	testCases := []struct {
		desc   string
		fields FieldList
		errMsg string // Non-empty if we expect an error
	}{
		{
			desc:   "multi string",
			fields: FieldList{{Name: "FieldA", Value: "val1"}, {Name: "FieldA", Value: "val2"}, {Name: "FieldA", Value: "val3"}},
		},
		{
			desc:   "multi atom",
			fields: FieldList{{Name: "FieldA", Value: Atom("val1")}, {Name: "FieldA", Value: Atom("val2")}, {Name: "FieldA", Value: Atom("val3")}},
		},
		{
			desc:   "mixed",
			fields: FieldList{{Name: "FieldA", Value: testString}, {Name: "FieldA", Value: testTime}, {Name: "FieldA", Value: float}},
		},
		{
			desc:   "multi time",
			fields: FieldList{{Name: "FieldA", Value: testTime}, {Name: "FieldA", Value: testTime}},
			errMsg: `duplicate time field "FieldA"`,
		},
		{
			desc:   "multi num",
			fields: FieldList{{Name: "FieldA", Value: float}, {Name: "FieldA", Value: float}},
			errMsg: `duplicate numeric field "FieldA"`,
		},
	}
	for _, tc := range testCases {
		_, err := saveDoc(&tc.fields)
		if (err == nil) != (tc.errMsg == "") || (err != nil && !strings.Contains(err.Error(), tc.errMsg)) {
			t.Errorf("%s: got err %v, wanted %q", tc.desc, err, tc.errMsg)
		}
	}
}

func TestLoadErrFieldMismatch(t *testing.T) {
	testCases := []struct {
		desc string
		dst  interface{}
		src  []*pb.Field
		err  error
	}{
		{
			desc: "missing",
			dst:  &struct{ One string }{},
			src:  []*pb.Field{newStringValueField("Two", "woop!", pb.FieldValue_TEXT)},
			err: &ErrFieldMismatch{
				FieldName: "Two",
				Reason:    "no such struct field",
			},
		},
		{
			desc: "wrong type",
			dst:  &struct{ Num float64 }{},
			src:  []*pb.Field{newStringValueField("Num", "woop!", pb.FieldValue_TEXT)},
			err: &ErrFieldMismatch{
				FieldName: "Num",
				Reason:    "type mismatch: float64 for string data",
			},
		},
		{
			desc: "unsettable",
			dst:  &struct{ lower string }{},
			src:  []*pb.Field{newStringValueField("lower", "woop!", pb.FieldValue_TEXT)},
			err: &ErrFieldMismatch{
				FieldName: "lower",
				Reason:    "cannot set struct field",
			},
		},
	}
	for _, tc := range testCases {
		err := loadDoc(tc.dst, &pb.Document{Field: tc.src}, nil)
		if !reflect.DeepEqual(err, tc.err) {
			t.Errorf("%s, got err %v, wanted %v", tc.desc, err, tc.err)
		}
	}
}

func TestLimit(t *testing.T) {
	index, err := Open("Doc")
	if err != nil {
		t.Fatalf("err from Open: %v", err)
	}
	c := aetesting.FakeSingleContext(t, "search", "Search", func(req *pb.SearchRequest, res *pb.SearchResponse) error {
		limit := 20 // Default per page.
		if req.Params.Limit != nil {
			limit = int(*req.Params.Limit)
		}
		res.Status = &pb.RequestStatus{Code: pb.SearchServiceError_OK.Enum()}
		res.MatchedCount = proto.Int64(int64(limit))
		for i := 0; i < limit; i++ {
			res.Result = append(res.Result, &pb.SearchResult{Document: &pb.Document{}})
			res.Cursor = proto.String("moreresults")
		}
		return nil
	})

	const maxDocs = 500 // Limit maximum number of docs.
	testCases := []struct {
		limit, want int
	}{
		{limit: 0, want: maxDocs},
		{limit: 42, want: 42},
		{limit: 100, want: 100},
		{limit: 1000, want: maxDocs},
	}

	for _, tt := range testCases {
		it := index.Search(c, "gopher", &SearchOptions{Limit: tt.limit, IDsOnly: true})
		count := 0
		for ; count < maxDocs; count++ {
			_, err := it.Next(nil)
			if err == Done {
				break
			}
			if err != nil {
				t.Fatalf("err after %d: %v", count, err)
			}
		}
		if count != tt.want {
			t.Errorf("got %d results, expected %d", count, tt.want)
		}
	}
}

func TestPut(t *testing.T) {
	index, err := Open("Doc")
	if err != nil {
		t.Fatalf("err from Open: %v", err)
	}

	c := aetesting.FakeSingleContext(t, "search", "IndexDocument", func(in *pb.IndexDocumentRequest, out *pb.IndexDocumentResponse) error {
		expectedIn := &pb.IndexDocumentRequest{
			Params: &pb.IndexDocumentParams{
				Document: []*pb.Document{
					{Field: protoFields, OrderId: proto.Int32(42)},
				},
				IndexSpec: &pb.IndexSpec{
					Name: proto.String("Doc"),
				},
			},
		}
		if !proto.Equal(in, expectedIn) {
			return fmt.Errorf("unsupported argument:\ngot  %v\nwant %v", in, expectedIn)
		}
		*out = pb.IndexDocumentResponse{
			Status: []*pb.RequestStatus{
				{Code: pb.SearchServiceError_OK.Enum()},
			},
			DocId: []string{
				"doc_id",
			},
		}
		return nil
	})

	id, err := index.Put(c, "", &FieldListWithMeta{
		Meta:   searchMeta,
		Fields: searchFields,
	})
	if err != nil {
		t.Fatal(err)
	}
	if want := "doc_id"; id != want {
		t.Errorf("Got doc ID %q, want %q", id, want)
	}
}

func TestPutAutoOrderID(t *testing.T) {
	index, err := Open("Doc")
	if err != nil {
		t.Fatalf("err from Open: %v", err)
	}

	c := aetesting.FakeSingleContext(t, "search", "IndexDocument", func(in *pb.IndexDocumentRequest, out *pb.IndexDocumentResponse) error {
		if len(in.Params.GetDocument()) < 1 {
			return fmt.Errorf("expected at least one Document, got %v", in)
		}
		got, want := in.Params.Document[0].GetOrderId(), int32(time.Since(orderIDEpoch).Seconds())
		if d := got - want; -5 > d || d > 5 {
			return fmt.Errorf("got OrderId %d, want near %d", got, want)
		}
		*out = pb.IndexDocumentResponse{
			Status: []*pb.RequestStatus{
				{Code: pb.SearchServiceError_OK.Enum()},
			},
			DocId: []string{
				"doc_id",
			},
		}
		return nil
	})

	if _, err := index.Put(c, "", &searchFields); err != nil {
		t.Fatal(err)
	}
}

func TestPutBadStatus(t *testing.T) {
	index, err := Open("Doc")
	if err != nil {
		t.Fatalf("err from Open: %v", err)
	}

	c := aetesting.FakeSingleContext(t, "search", "IndexDocument", func(_ *pb.IndexDocumentRequest, out *pb.IndexDocumentResponse) error {
		*out = pb.IndexDocumentResponse{
			Status: []*pb.RequestStatus{
				{
					Code:        pb.SearchServiceError_INVALID_REQUEST.Enum(),
					ErrorDetail: proto.String("insufficient gophers"),
				},
			},
		}
		return nil
	})

	wantErr := "search: INVALID_REQUEST: insufficient gophers"
	if _, err := index.Put(c, "", &searchFields); err == nil || err.Error() != wantErr {
		t.Fatalf("Put: got %v error, want %q", err, wantErr)
	}
}

func TestSortOptions(t *testing.T) {
	index, err := Open("Doc")
	if err != nil {
		t.Fatalf("err from Open: %v", err)
	}

	noErr := errors.New("") // Sentinel err to return to prevent sending request.

	testCases := []struct {
		desc       string
		sort       *SortOptions
		wantSort   []*pb.SortSpec
		wantScorer *pb.ScorerSpec
		wantErr    string
	}{
		{
			desc: "No SortOptions",
		},
		{
			desc: "Basic",
			sort: &SortOptions{
				Expressions: []SortExpression{
					{Expr: "dog"},
					{Expr: "cat", Reverse: true},
					{Expr: "gopher", Default: "blue"},
					{Expr: "fish", Default: 2.0},
				},
				Limit:  42,
				Scorer: MatchScorer,
			},
			wantSort: []*pb.SortSpec{
				{SortExpression: proto.String("dog")},
				{SortExpression: proto.String("cat"), SortDescending: proto.Bool(false)},
				{SortExpression: proto.String("gopher"), DefaultValueText: proto.String("blue")},
				{SortExpression: proto.String("fish"), DefaultValueNumeric: proto.Float64(2)},
			},
			wantScorer: &pb.ScorerSpec{
				Limit:  proto.Int32(42),
				Scorer: pb.ScorerSpec_MATCH_SCORER.Enum(),
			},
		},
		{
			desc: "Bad expression default",
			sort: &SortOptions{
				Expressions: []SortExpression{
					{Expr: "dog", Default: true},
				},
			},
			wantErr: `search: invalid Default type bool for expression "dog"`,
		},
		{
			desc:       "RescoringMatchScorer",
			sort:       &SortOptions{Scorer: RescoringMatchScorer},
			wantScorer: &pb.ScorerSpec{Scorer: pb.ScorerSpec_RESCORING_MATCH_SCORER.Enum()},
		},
	}

	for _, tt := range testCases {
		c := aetesting.FakeSingleContext(t, "search", "Search", func(req *pb.SearchRequest, _ *pb.SearchResponse) error {
			params := req.Params
			if !reflect.DeepEqual(params.SortSpec, tt.wantSort) {
				t.Errorf("%s: params.SortSpec=%v; want %v", tt.desc, params.SortSpec, tt.wantSort)
			}
			if !reflect.DeepEqual(params.ScorerSpec, tt.wantScorer) {
				t.Errorf("%s: params.ScorerSpec=%v; want %v", tt.desc, params.ScorerSpec, tt.wantScorer)
			}
			return noErr // Always return some error to prevent response parsing.
		})

		it := index.Search(c, "gopher", &SearchOptions{Sort: tt.sort})
		_, err := it.Next(nil)
		if err == nil {
			t.Fatalf("%s: err==nil; should not happen", tt.desc)
		}
		if err.Error() != tt.wantErr {
			t.Errorf("%s: got error %q, want %q", tt.desc, err, tt.wantErr)
		}
	}
}

func TestFieldSpec(t *testing.T) {
	index, err := Open("Doc")
	if err != nil {
		t.Fatalf("err from Open: %v", err)
	}

	errFoo := errors.New("foo") // sentinel error when there isn't one.

	testCases := []struct {
		desc string
		opts *SearchOptions
		want *pb.FieldSpec
	}{
		{
			desc: "No options",
			want: &pb.FieldSpec{},
		},
		{
			desc: "Fields",
			opts: &SearchOptions{
				Fields: []string{"one", "two"},
			},
			want: &pb.FieldSpec{
				Name: []string{"one", "two"},
			},
		},
		{
			desc: "Expressions",
			opts: &SearchOptions{
				Expressions: []FieldExpression{
					{Name: "one", Expr: "price * quantity"},
					{Name: "two", Expr: "min(daily_use, 10) * rate"},
				},
			},
			want: &pb.FieldSpec{
				Expression: []*pb.FieldSpec_Expression{
					{Name: proto.String("one"), Expression: proto.String("price * quantity")},
					{Name: proto.String("two"), Expression: proto.String("min(daily_use, 10) * rate")},
				},
			},
		},
	}

	for _, tt := range testCases {
		c := aetesting.FakeSingleContext(t, "search", "Search", func(req *pb.SearchRequest, _ *pb.SearchResponse) error {
			params := req.Params
			if !reflect.DeepEqual(params.FieldSpec, tt.want) {
				t.Errorf("%s: params.FieldSpec=%v; want %v", tt.desc, params.FieldSpec, tt.want)
			}
			return errFoo // Always return some error to prevent response parsing.
		})

		it := index.Search(c, "gopher", tt.opts)
		if _, err := it.Next(nil); err != errFoo {
			t.Fatalf("%s: got error %v; want %v", tt.desc, err, errFoo)
		}
	}
}

func TestBasicSearchOpts(t *testing.T) {
	index, err := Open("Doc")
	if err != nil {
		t.Fatalf("err from Open: %v", err)
	}

	noErr := errors.New("") // Sentinel err to return to prevent sending request.

	testCases := []struct {
		desc          string
		facetOpts     []FacetSearchOption
		cursor        Cursor
		offset        int
		countAccuracy int
		want          *pb.SearchParams
		wantErr       string
	}{
		{
			desc: "No options",
			want: &pb.SearchParams{},
		},
		{
			desc: "Default auto discovery",
			facetOpts: []FacetSearchOption{
				AutoFacetDiscovery(0, 0),
			},
			want: &pb.SearchParams{
				AutoDiscoverFacetCount: proto.Int32(10),
			},
		},
		{
			desc: "Auto discovery",
			facetOpts: []FacetSearchOption{
				AutoFacetDiscovery(7, 12),
			},
			want: &pb.SearchParams{
				AutoDiscoverFacetCount: proto.Int32(7),
				FacetAutoDetectParam: &pb.FacetAutoDetectParam{
					ValueLimit: proto.Int32(12),
				},
			},
		},
		{
			desc: "Param Depth",
			facetOpts: []FacetSearchOption{
				AutoFacetDiscovery(7, 12),
			},
			want: &pb.SearchParams{
				AutoDiscoverFacetCount: proto.Int32(7),
				FacetAutoDetectParam: &pb.FacetAutoDetectParam{
					ValueLimit: proto.Int32(12),
				},
			},
		},
		{
			desc: "Doc depth",
			facetOpts: []FacetSearchOption{
				FacetDocumentDepth(123),
			},
			want: &pb.SearchParams{
				FacetDepth: proto.Int32(123),
			},
		},
		{
			desc: "Facet discovery",
			facetOpts: []FacetSearchOption{
				FacetDiscovery("colour"),
				FacetDiscovery("size", Atom("M"), Atom("L")),
				FacetDiscovery("price", LessThan(7), Range{7, 14}, AtLeast(14)),
			},
			want: &pb.SearchParams{
				IncludeFacet: []*pb.FacetRequest{
					{Name: proto.String("colour")},
					{Name: proto.String("size"), Params: &pb.FacetRequestParam{
						ValueConstraint: []string{"M", "L"},
					}},
					{Name: proto.String("price"), Params: &pb.FacetRequestParam{
						Range: []*pb.FacetRange{
							{End: proto.String("7e+00")},
							{Start: proto.String("7e+00"), End: proto.String("1.4e+01")},
							{Start: proto.String("1.4e+01")},
						},
					}},
				},
			},
		},
		{
			desc: "Facet discovery - bad value",
			facetOpts: []FacetSearchOption{
				FacetDiscovery("colour", true),
			},
			wantErr: "bad FacetSearchOption: unsupported value type bool",
		},
		{
			desc: "Facet discovery - mix value types",
			facetOpts: []FacetSearchOption{
				FacetDiscovery("colour", Atom("blue"), AtLeast(7)),
			},
			wantErr: "bad FacetSearchOption: values must all be Atom, or must all be Range",
		},
		{
			desc: "Facet discovery - invalid range",
			facetOpts: []FacetSearchOption{
				FacetDiscovery("colour", Range{negInf, posInf}),
			},
			wantErr: "bad FacetSearchOption: invalid range: either Start or End must be finite",
		},
		{
			desc:   "Cursor",
			cursor: Cursor("mycursor"),
			want: &pb.SearchParams{
				Cursor: proto.String("mycursor"),
			},
		},
		{
			desc:   "Offset",
			offset: 121,
			want: &pb.SearchParams{
				Offset: proto.Int32(121),
			},
		},
		{
			desc:    "Cursor and Offset set",
			cursor:  Cursor("mycursor"),
			offset:  121,
			wantErr: "at most one of Cursor and Offset may be specified",
		},
		{
			desc:          "Count accuracy",
			countAccuracy: 100,
			want: &pb.SearchParams{
				MatchedCountAccuracy: proto.Int32(100),
			},
		},
	}

	for _, tt := range testCases {
		c := aetesting.FakeSingleContext(t, "search", "Search", func(req *pb.SearchRequest, _ *pb.SearchResponse) error {
			if tt.want == nil {
				t.Errorf("%s: expected call to fail", tt.desc)
				return nil
			}
			// Set default fields.
			tt.want.Query = proto.String("gopher")
			tt.want.IndexSpec = &pb.IndexSpec{Name: proto.String("Doc")}
			tt.want.CursorType = pb.SearchParams_PER_RESULT.Enum()
			tt.want.FieldSpec = &pb.FieldSpec{}
			if got := req.Params; !reflect.DeepEqual(got, tt.want) {
				t.Errorf("%s: params=%v; want %v", tt.desc, got, tt.want)
			}
			return noErr // Always return some error to prevent response parsing.
		})

		it := index.Search(c, "gopher", &SearchOptions{
			Facets:        tt.facetOpts,
			Cursor:        tt.cursor,
			Offset:        tt.offset,
			CountAccuracy: tt.countAccuracy,
		})
		_, err := it.Next(nil)
		if err == nil {
			t.Fatalf("%s: err==nil; should not happen", tt.desc)
		}
		if err.Error() != tt.wantErr {
			t.Errorf("%s: got error %q, want %q", tt.desc, err, tt.wantErr)
		}
	}
}

func TestFacetRefinements(t *testing.T) {
	index, err := Open("Doc")
	if err != nil {
		t.Fatalf("err from Open: %v", err)
	}

	noErr := errors.New("") // Sentinel err to return to prevent sending request.

	testCases := []struct {
		desc    string
		refine  []Facet
		want    []*pb.FacetRefinement
		wantErr string
	}{
		{
			desc: "No refinements",
		},
		{
			desc: "Basic",
			refine: []Facet{
				{Name: "fur", Value: Atom("fluffy")},
				{Name: "age", Value: LessThan(123)},
				{Name: "age", Value: AtLeast(0)},
				{Name: "legs", Value: Range{Start: 3, End: 5}},
			},
			want: []*pb.FacetRefinement{
				{Name: proto.String("fur"), Value: proto.String("fluffy")},
				{Name: proto.String("age"), Range: &pb.FacetRefinement_Range{End: proto.String("1.23e+02")}},
				{Name: proto.String("age"), Range: &pb.FacetRefinement_Range{Start: proto.String("0e+00")}},
				{Name: proto.String("legs"), Range: &pb.FacetRefinement_Range{Start: proto.String("3e+00"), End: proto.String("5e+00")}},
			},
		},
		{
			desc: "Infinite range",
			refine: []Facet{
				{Name: "age", Value: Range{Start: negInf, End: posInf}},
			},
			wantErr: `search: refinement for facet "age": either Start or End must be finite`,
		},
		{
			desc: "Bad End value in range",
			refine: []Facet{
				{Name: "age", Value: LessThan(2147483648)},
			},
			wantErr: `search: refinement for facet "age": invalid value for End`,
		},
		{
			desc: "Bad Start value in range",
			refine: []Facet{
				{Name: "age", Value: AtLeast(-2147483649)},
			},
			wantErr: `search: refinement for facet "age": invalid value for Start`,
		},
		{
			desc: "Unknown value type",
			refine: []Facet{
				{Name: "age", Value: "you can't use strings!"},
			},
			wantErr: `search: unsupported refinement for facet "age" of type string`,
		},
	}

	for _, tt := range testCases {
		c := aetesting.FakeSingleContext(t, "search", "Search", func(req *pb.SearchRequest, _ *pb.SearchResponse) error {
			if got := req.Params.FacetRefinement; !reflect.DeepEqual(got, tt.want) {
				t.Errorf("%s: params.FacetRefinement=%v; want %v", tt.desc, got, tt.want)
			}
			return noErr // Always return some error to prevent response parsing.
		})

		it := index.Search(c, "gopher", &SearchOptions{Refinements: tt.refine})
		_, err := it.Next(nil)
		if err == nil {
			t.Fatalf("%s: err==nil; should not happen", tt.desc)
		}
		if err.Error() != tt.wantErr {
			t.Errorf("%s: got error %q, want %q", tt.desc, err, tt.wantErr)
		}
	}
}

func TestNamespaceResetting(t *testing.T) {
	namec := make(chan *string, 1)
	c0 := aetesting.FakeSingleContext(t, "search", "IndexDocument", func(req *pb.IndexDocumentRequest, res *pb.IndexDocumentResponse) error {
		namec <- req.Params.IndexSpec.Namespace
		return fmt.Errorf("RPC error")
	})

	// Check that wrapping c0 in a namespace twice works correctly.
	c1, err := appengine.Namespace(c0, "A")
	if err != nil {
		t.Fatalf("appengine.Namespace: %v", err)
	}
	c2, err := appengine.Namespace(c1, "") // should act as the original context
	if err != nil {
		t.Fatalf("appengine.Namespace: %v", err)
	}

	i := (&Index{})

	i.Put(c0, "something", &searchDoc)
	if ns := <-namec; ns != nil {
		t.Errorf(`Put with c0: ns = %q, want nil`, *ns)
	}

	i.Put(c1, "something", &searchDoc)
	if ns := <-namec; ns == nil {
		t.Error(`Put with c1: ns = nil, want "A"`)
	} else if *ns != "A" {
		t.Errorf(`Put with c1: ns = %q, want "A"`, *ns)
	}

	i.Put(c2, "something", &searchDoc)
	if ns := <-namec; ns != nil {
		t.Errorf(`Put with c2: ns = %q, want nil`, *ns)
	}
}
