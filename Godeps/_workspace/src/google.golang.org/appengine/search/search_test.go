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
	if err := loadDoc(&got, protoFields, nil, nil); err != nil {
		t.Fatalf("loadDoc: %v", err)
	}
	if got != want {
		t.Errorf("loadDoc: got %v, wanted %v", got, want)
	}
}

func TestSaveDoc(t *testing.T) {
	got, _, err := saveDoc(&searchDoc)
	if err != nil {
		t.Fatalf("saveDoc: %v", err)
	}
	want := protoFields
	if !reflect.DeepEqual(got, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}
}

func TestLoadFieldList(t *testing.T) {
	var got FieldList
	want := searchFieldsWithLang
	if err := loadDoc(&got, protoFields, nil, nil); err != nil {
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
	protoFields, _, err := saveDoc(fl)
	if err != nil {
		t.Fatalf("saveDoc: %v", err)
	}
	if err := loadDoc(&got, protoFields, nil, nil); err != nil {
		t.Fatalf("loadDoc: %v", err)
	}
	if want := fl; !reflect.DeepEqual(&got, want) {
		t.Errorf("got  %v\nwant %v", got, want)
	}
}

func TestSaveFieldList(t *testing.T) {
	got, _, err := saveDoc(&searchFields)
	if err != nil {
		t.Fatalf("saveDoc: %v", err)
	}
	want := protoFields
	if !reflect.DeepEqual(got, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}
}

func TestLoadFieldAndExprList(t *testing.T) {
	var got, want FieldList
	for i, f := range searchFieldsWithLang {
		f.Derived = (i >= 2) // First 2 elements are "fields", next are "expressions".
		want = append(want, f)
	}
	if err := loadDoc(&got, protoFields[:2], protoFields[2:], nil); err != nil {
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
	if err := loadDoc(&got, protoFields, nil, searchMeta); err != nil {
		t.Fatalf("loadDoc: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("got  %v\nwant %v", got, want)
	}
}

func TestSaveMeta(t *testing.T) {
	got, gotMeta, err := saveDoc(&FieldListWithMeta{
		Meta:   searchMeta,
		Fields: searchFields,
	})
	if err != nil {
		t.Fatalf("saveDoc: %v", err)
	}
	if want := protoFields; !reflect.DeepEqual(got, want) {
		t.Errorf("\ngot  %v\nwant %v", got, want)
	}
	if want := searchMeta; !reflect.DeepEqual(gotMeta, want) {
		t.Errorf("\ngot  %v\nwant %v", gotMeta, want)
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
		{"lower_case", false},
		{"Exclaim!", false},
		{"Hello세상아 안녕", false},
		{"", false},
		{"Hεllo", false},
		{strings.Repeat("A", 500), true},
		{strings.Repeat("A", 501), false},
	}

	for _, tc := range testCases {
		_, _, err := saveDoc(&FieldList{
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
		_, _, err := saveDoc(&FieldList{tt.field})
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
		_, _, err := saveDoc(&tc.fields)
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
		err := loadDoc(tc.dst, tc.src, nil, nil)
		if !reflect.DeepEqual(err, tc.err) {
			t.Errorf("%s, got err %v, wanted %v", tc.desc, err, tc.err)
		}
	}
}

func TestLimit(t *testing.T) {
	more := func(it *Iterator) error {
		if it.limit == 0 {
			return errors.New("Iterator.limit should not be zero in next")
		}
		// Page up to 20 items at once.
		ret := 20
		if it.limit > 0 && it.limit < ret {
			ret = it.limit
		}
		it.listRes = make([]*pb.Document, ret)
		for i := range it.listRes {
			it.listRes[i] = &pb.Document{}
		}
		return nil
	}

	it := &Iterator{
		more:  more,
		limit: 42,
	}

	count := 0
	for {
		_, err := it.Next(nil)
		if err == Done {
			break
		}
		if err != nil {
			t.Fatalf("err after %d: %v", count, err)
		}
		count++
	}
	if count != 42 {
		t.Errorf("got %d results, expected 42", count)
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

	noErr := errors.New("") // sentinel error when there isn't one…

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
