// Copyright 2011 Google Inc. All Rights Reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine/internal"
	"google.golang.org/appengine/internal/aetesting"
	pb "google.golang.org/appengine/internal/datastore"
)

var (
	path1 = &pb.Path{
		Element: []*pb.Path_Element{
			{
				Type: proto.String("Gopher"),
				Id:   proto.Int64(6),
			},
		},
	}
	path2 = &pb.Path{
		Element: []*pb.Path_Element{
			{
				Type: proto.String("Gopher"),
				Id:   proto.Int64(6),
			},
			{
				Type: proto.String("Gopher"),
				Id:   proto.Int64(8),
			},
		},
	}
)

func fakeRunQuery(in *pb.Query, out *pb.QueryResult) error {
	expectedIn := &pb.Query{
		App:     proto.String("dev~fake-app"),
		Kind:    proto.String("Gopher"),
		Compile: proto.Bool(true),
	}
	if !proto.Equal(in, expectedIn) {
		return fmt.Errorf("unsupported argument: got %v want %v", in, expectedIn)
	}
	*out = pb.QueryResult{
		Result: []*pb.EntityProto{
			{
				Key: &pb.Reference{
					App:  proto.String("s~test-app"),
					Path: path1,
				},
				EntityGroup: path1,
				Property: []*pb.Property{
					{
						Meaning: pb.Property_TEXT.Enum(),
						Name:    proto.String("Name"),
						Value: &pb.PropertyValue{
							StringValue: proto.String("George"),
						},
					},
					{
						Name: proto.String("Height"),
						Value: &pb.PropertyValue{
							Int64Value: proto.Int64(32),
						},
					},
				},
			},
			{
				Key: &pb.Reference{
					App:  proto.String("s~test-app"),
					Path: path2,
				},
				EntityGroup: path1, // ancestor is George
				Property: []*pb.Property{
					{
						Meaning: pb.Property_TEXT.Enum(),
						Name:    proto.String("Name"),
						Value: &pb.PropertyValue{
							StringValue: proto.String("Rufus"),
						},
					},
					// No height for Rufus.
				},
			},
		},
		MoreResults: proto.Bool(false),
	}
	return nil
}

type StructThatImplementsPLS struct{}

func (StructThatImplementsPLS) Load(p []Property) error   { return nil }
func (StructThatImplementsPLS) Save() ([]Property, error) { return nil, nil }

var _ PropertyLoadSaver = StructThatImplementsPLS{}

type StructPtrThatImplementsPLS struct{}

func (*StructPtrThatImplementsPLS) Load(p []Property) error   { return nil }
func (*StructPtrThatImplementsPLS) Save() ([]Property, error) { return nil, nil }

var _ PropertyLoadSaver = &StructPtrThatImplementsPLS{}

type PropertyMap map[string]Property

func (m PropertyMap) Load(props []Property) error {
	for _, p := range props {
		if p.Multiple {
			return errors.New("PropertyMap does not support multiple properties")
		}
		m[p.Name] = p
	}
	return nil
}

func (m PropertyMap) Save() ([]Property, error) {
	props := make([]Property, 0, len(m))
	for _, p := range m {
		if p.Multiple {
			return nil, errors.New("PropertyMap does not support multiple properties")
		}
		props = append(props, p)
	}
	return props, nil
}

var _ PropertyLoadSaver = PropertyMap{}

type Gopher struct {
	Name   string
	Height int
}

// typeOfEmptyInterface is the type of interface{}, but we can't use
// reflect.TypeOf((interface{})(nil)) directly because TypeOf takes an
// interface{}.
var typeOfEmptyInterface = reflect.TypeOf((*interface{})(nil)).Elem()

func TestCheckMultiArg(t *testing.T) {
	testCases := []struct {
		v        interface{}
		mat      multiArgType
		elemType reflect.Type
	}{
		// Invalid cases.
		{nil, multiArgTypeInvalid, nil},
		{Gopher{}, multiArgTypeInvalid, nil},
		{&Gopher{}, multiArgTypeInvalid, nil},
		{PropertyList{}, multiArgTypeInvalid, nil}, // This is a special case.
		{PropertyMap{}, multiArgTypeInvalid, nil},
		{[]*PropertyList(nil), multiArgTypeInvalid, nil},
		{[]*PropertyMap(nil), multiArgTypeInvalid, nil},
		{[]**Gopher(nil), multiArgTypeInvalid, nil},
		{[]*interface{}(nil), multiArgTypeInvalid, nil},
		// Valid cases.
		{
			[]PropertyList(nil),
			multiArgTypePropertyLoadSaver,
			reflect.TypeOf(PropertyList{}),
		},
		{
			[]PropertyMap(nil),
			multiArgTypePropertyLoadSaver,
			reflect.TypeOf(PropertyMap{}),
		},
		{
			[]StructThatImplementsPLS(nil),
			multiArgTypePropertyLoadSaver,
			reflect.TypeOf(StructThatImplementsPLS{}),
		},
		{
			[]StructPtrThatImplementsPLS(nil),
			multiArgTypePropertyLoadSaver,
			reflect.TypeOf(StructPtrThatImplementsPLS{}),
		},
		{
			[]Gopher(nil),
			multiArgTypeStruct,
			reflect.TypeOf(Gopher{}),
		},
		{
			[]*Gopher(nil),
			multiArgTypeStructPtr,
			reflect.TypeOf(Gopher{}),
		},
		{
			[]interface{}(nil),
			multiArgTypeInterface,
			typeOfEmptyInterface,
		},
	}
	for _, tc := range testCases {
		mat, elemType := checkMultiArg(reflect.ValueOf(tc.v))
		if mat != tc.mat || elemType != tc.elemType {
			t.Errorf("checkMultiArg(%T): got %v, %v want %v, %v",
				tc.v, mat, elemType, tc.mat, tc.elemType)
		}
	}
}

func TestSimpleQuery(t *testing.T) {
	struct1 := Gopher{Name: "George", Height: 32}
	struct2 := Gopher{Name: "Rufus"}
	pList1 := PropertyList{
		{
			Name:  "Name",
			Value: "George",
		},
		{
			Name:  "Height",
			Value: int64(32),
		},
	}
	pList2 := PropertyList{
		{
			Name:  "Name",
			Value: "Rufus",
		},
	}
	pMap1 := PropertyMap{
		"Name": Property{
			Name:  "Name",
			Value: "George",
		},
		"Height": Property{
			Name:  "Height",
			Value: int64(32),
		},
	}
	pMap2 := PropertyMap{
		"Name": Property{
			Name:  "Name",
			Value: "Rufus",
		},
	}

	testCases := []struct {
		dst  interface{}
		want interface{}
	}{
		// The destination must have type *[]P, *[]S or *[]*S, for some non-interface
		// type P such that *P implements PropertyLoadSaver, or for some struct type S.
		{new([]Gopher), &[]Gopher{struct1, struct2}},
		{new([]*Gopher), &[]*Gopher{&struct1, &struct2}},
		{new([]PropertyList), &[]PropertyList{pList1, pList2}},
		{new([]PropertyMap), &[]PropertyMap{pMap1, pMap2}},

		// Any other destination type is invalid.
		{0, nil},
		{Gopher{}, nil},
		{PropertyList{}, nil},
		{PropertyMap{}, nil},
		{[]int{}, nil},
		{[]Gopher{}, nil},
		{[]PropertyList{}, nil},
		{new(int), nil},
		{new(Gopher), nil},
		{new(PropertyList), nil}, // This is a special case.
		{new(PropertyMap), nil},
		{new([]int), nil},
		{new([]map[int]int), nil},
		{new([]map[string]Property), nil},
		{new([]map[string]interface{}), nil},
		{new([]*int), nil},
		{new([]*map[int]int), nil},
		{new([]*map[string]Property), nil},
		{new([]*map[string]interface{}), nil},
		{new([]**Gopher), nil},
		{new([]*PropertyList), nil},
		{new([]*PropertyMap), nil},
	}
	for _, tc := range testCases {
		nCall := 0
		c := aetesting.FakeSingleContext(t, "datastore_v3", "RunQuery", func(in *pb.Query, out *pb.QueryResult) error {
			nCall++
			return fakeRunQuery(in, out)
		})
		c = internal.WithAppIDOverride(c, "dev~fake-app")

		var (
			expectedErr   error
			expectedNCall int
		)
		if tc.want == nil {
			expectedErr = ErrInvalidEntityType
		} else {
			expectedNCall = 1
		}
		keys, err := NewQuery("Gopher").GetAll(c, tc.dst)
		if err != expectedErr {
			t.Errorf("dst type %T: got error [%v], want [%v]", tc.dst, err, expectedErr)
			continue
		}
		if nCall != expectedNCall {
			t.Errorf("dst type %T: Context.Call was called an incorrect number of times: got %d want %d", tc.dst, nCall, expectedNCall)
			continue
		}
		if err != nil {
			continue
		}

		key1 := NewKey(c, "Gopher", "", 6, nil)
		expectedKeys := []*Key{
			key1,
			NewKey(c, "Gopher", "", 8, key1),
		}
		if l1, l2 := len(keys), len(expectedKeys); l1 != l2 {
			t.Errorf("dst type %T: got %d keys, want %d keys", tc.dst, l1, l2)
			continue
		}
		for i, key := range keys {
			if key.AppID() != "s~test-app" {
				t.Errorf(`dst type %T: Key #%d's AppID = %q, want "s~test-app"`, tc.dst, i, key.AppID())
				continue
			}
			if !keysEqual(key, expectedKeys[i]) {
				t.Errorf("dst type %T: got key #%d %v, want %v", tc.dst, i, key, expectedKeys[i])
				continue
			}
		}

		if !reflect.DeepEqual(tc.dst, tc.want) {
			t.Errorf("dst type %T: Entities got %+v, want %+v", tc.dst, tc.dst, tc.want)
			continue
		}
	}
}

// keysEqual is like (*Key).Equal, but ignores the App ID.
func keysEqual(a, b *Key) bool {
	for a != nil && b != nil {
		if a.Kind() != b.Kind() || a.StringID() != b.StringID() || a.IntID() != b.IntID() {
			return false
		}
		a, b = a.Parent(), b.Parent()
	}
	return a == b
}

func TestQueriesAreImmutable(t *testing.T) {
	// Test that deriving q2 from q1 does not modify q1.
	q0 := NewQuery("foo")
	q1 := NewQuery("foo")
	q2 := q1.Offset(2)
	if !reflect.DeepEqual(q0, q1) {
		t.Errorf("q0 and q1 were not equal")
	}
	if reflect.DeepEqual(q1, q2) {
		t.Errorf("q1 and q2 were equal")
	}

	// Test that deriving from q4 twice does not conflict, even though
	// q4 has a long list of order clauses. This tests that the arrays
	// backed by a query's slice of orders are not shared.
	f := func() *Query {
		q := NewQuery("bar")
		// 47 is an ugly number that is unlikely to be near a re-allocation
		// point in repeated append calls. For example, it's not near a power
		// of 2 or a multiple of 10.
		for i := 0; i < 47; i++ {
			q = q.Order(fmt.Sprintf("x%d", i))
		}
		return q
	}
	q3 := f().Order("y")
	q4 := f()
	q5 := q4.Order("y")
	q6 := q4.Order("z")
	if !reflect.DeepEqual(q3, q5) {
		t.Errorf("q3 and q5 were not equal")
	}
	if reflect.DeepEqual(q5, q6) {
		t.Errorf("q5 and q6 were equal")
	}
}

func TestFilterParser(t *testing.T) {
	testCases := []struct {
		filterStr     string
		wantOK        bool
		wantFieldName string
		wantOp        operator
	}{
		// Supported ops.
		{"x<", true, "x", lessThan},
		{"x <", true, "x", lessThan},
		{"x  <", true, "x", lessThan},
		{"   x   <  ", true, "x", lessThan},
		{"x <=", true, "x", lessEq},
		{"x =", true, "x", equal},
		{"x >=", true, "x", greaterEq},
		{"x >", true, "x", greaterThan},
		{"in >", true, "in", greaterThan},
		{"in>", true, "in", greaterThan},
		// Valid but (currently) unsupported ops.
		{"x!=", false, "", 0},
		{"x !=", false, "", 0},
		{" x  !=  ", false, "", 0},
		{"x IN", false, "", 0},
		{"x in", false, "", 0},
		// Invalid ops.
		{"x EQ", false, "", 0},
		{"x lt", false, "", 0},
		{"x <>", false, "", 0},
		{"x >>", false, "", 0},
		{"x ==", false, "", 0},
		{"x =<", false, "", 0},
		{"x =>", false, "", 0},
		{"x !", false, "", 0},
		{"x ", false, "", 0},
		{"x", false, "", 0},
	}
	for _, tc := range testCases {
		q := NewQuery("foo").Filter(tc.filterStr, 42)
		if ok := q.err == nil; ok != tc.wantOK {
			t.Errorf("%q: ok=%t, want %t", tc.filterStr, ok, tc.wantOK)
			continue
		}
		if !tc.wantOK {
			continue
		}
		if len(q.filter) != 1 {
			t.Errorf("%q: len=%d, want %d", tc.filterStr, len(q.filter), 1)
			continue
		}
		got, want := q.filter[0], filter{tc.wantFieldName, tc.wantOp, 42}
		if got != want {
			t.Errorf("%q: got %v, want %v", tc.filterStr, got, want)
			continue
		}
	}
}

func TestQueryToProto(t *testing.T) {
	// The context is required to make Keys for the test cases.
	var got *pb.Query
	NoErr := errors.New("No error")
	c := aetesting.FakeSingleContext(t, "datastore_v3", "RunQuery", func(in *pb.Query, out *pb.QueryResult) error {
		got = in
		return NoErr // return a non-nil error so Run doesn't keep going.
	})
	c = internal.WithAppIDOverride(c, "dev~fake-app")

	testCases := []struct {
		desc  string
		query *Query
		want  *pb.Query
		err   string
	}{
		{
			desc:  "empty",
			query: NewQuery(""),
			want:  &pb.Query{},
		},
		{
			desc:  "standard query",
			query: NewQuery("kind").Order("-I").Filter("I >", 17).Filter("U =", "Dave").Limit(7).Offset(42),
			want: &pb.Query{
				Kind: proto.String("kind"),
				Filter: []*pb.Query_Filter{
					{
						Op: pb.Query_Filter_GREATER_THAN.Enum(),
						Property: []*pb.Property{
							{
								Name:     proto.String("I"),
								Value:    &pb.PropertyValue{Int64Value: proto.Int64(17)},
								Multiple: proto.Bool(false),
							},
						},
					},
					{
						Op: pb.Query_Filter_EQUAL.Enum(),
						Property: []*pb.Property{
							{
								Name:     proto.String("U"),
								Value:    &pb.PropertyValue{StringValue: proto.String("Dave")},
								Multiple: proto.Bool(false),
							},
						},
					},
				},
				Order: []*pb.Query_Order{
					{
						Property:  proto.String("I"),
						Direction: pb.Query_Order_DESCENDING.Enum(),
					},
				},
				Limit:  proto.Int32(7),
				Offset: proto.Int32(42),
			},
		},
		{
			desc:  "ancestor",
			query: NewQuery("").Ancestor(NewKey(c, "kind", "Mummy", 0, nil)),
			want: &pb.Query{
				Ancestor: &pb.Reference{
					App: proto.String("dev~fake-app"),
					Path: &pb.Path{
						Element: []*pb.Path_Element{{Type: proto.String("kind"), Name: proto.String("Mummy")}},
					},
				},
			},
		},
		{
			desc:  "projection",
			query: NewQuery("").Project("A", "B"),
			want: &pb.Query{
				PropertyName: []string{"A", "B"},
			},
		},
		{
			desc:  "projection with distinct",
			query: NewQuery("").Project("A", "B").Distinct(),
			want: &pb.Query{
				PropertyName:        []string{"A", "B"},
				GroupByPropertyName: []string{"A", "B"},
			},
		},
		{
			desc:  "keys only",
			query: NewQuery("").KeysOnly(),
			want: &pb.Query{
				KeysOnly:           proto.Bool(true),
				RequirePerfectPlan: proto.Bool(true),
			},
		},
		{
			desc:  "empty filter",
			query: NewQuery("kind").Filter("=", 17),
			err:   "empty query filter field nam",
		},
		{
			desc:  "bad filter type",
			query: NewQuery("kind").Filter("M =", map[string]bool{}),
			err:   "bad query filter value type",
		},
		{
			desc:  "bad filter operator",
			query: NewQuery("kind").Filter("I <<=", 17),
			err:   `invalid operator "<<=" in filter "I <<="`,
		},
		{
			desc:  "empty order",
			query: NewQuery("kind").Order(""),
			err:   "empty order",
		},
		{
			desc:  "bad order direction",
			query: NewQuery("kind").Order("+I"),
			err:   `invalid order: "+I`,
		},
	}

	for _, tt := range testCases {
		got = nil
		if _, err := tt.query.Run(c).Next(nil); err != NoErr {
			if tt.err == "" || !strings.Contains(err.Error(), tt.err) {
				t.Errorf("%s: error %v, want %q", tt.desc, err, tt.err)
			}
			continue
		}
		if tt.err != "" {
			t.Errorf("%s: no error, want %q", tt.desc, tt.err)
			continue
		}
		// Fields that are common to all protos.
		tt.want.App = proto.String("dev~fake-app")
		tt.want.Compile = proto.Bool(true)
		if !proto.Equal(got, tt.want) {
			t.Errorf("%s:\ngot  %v\nwant %v", tt.desc, got, tt.want)
		}
	}
}
