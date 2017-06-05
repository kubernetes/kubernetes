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
	"errors"
	"fmt"
	"reflect"
	"sort"
	"testing"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	pb "google.golang.org/genproto/googleapis/datastore/v1"
	"google.golang.org/grpc"
)

var (
	key1 = &pb.Key{
		Path: []*pb.Key_PathElement{
			{
				Kind:   "Gopher",
				IdType: &pb.Key_PathElement_Id{6},
			},
		},
	}
	key2 = &pb.Key{
		Path: []*pb.Key_PathElement{
			{
				Kind:   "Gopher",
				IdType: &pb.Key_PathElement_Id{6},
			},
			{
				Kind:   "Gopher",
				IdType: &pb.Key_PathElement_Id{8},
			},
		},
	}
)

type fakeClient struct {
	pb.DatastoreClient
	queryFn  func(*pb.RunQueryRequest) (*pb.RunQueryResponse, error)
	commitFn func(*pb.CommitRequest) (*pb.CommitResponse, error)
}

func (c *fakeClient) RunQuery(_ context.Context, req *pb.RunQueryRequest, _ ...grpc.CallOption) (*pb.RunQueryResponse, error) {
	return c.queryFn(req)
}

func (c *fakeClient) Commit(_ context.Context, req *pb.CommitRequest, _ ...grpc.CallOption) (*pb.CommitResponse, error) {
	return c.commitFn(req)
}

func fakeRunQuery(in *pb.RunQueryRequest) (*pb.RunQueryResponse, error) {
	expectedIn := &pb.RunQueryRequest{
		QueryType: &pb.RunQueryRequest_Query{&pb.Query{
			Kind: []*pb.KindExpression{{Name: "Gopher"}},
		}},
	}
	if !proto.Equal(in, expectedIn) {
		return nil, fmt.Errorf("unsupported argument: got %v want %v", in, expectedIn)
	}
	return &pb.RunQueryResponse{
		Batch: &pb.QueryResultBatch{
			MoreResults:      pb.QueryResultBatch_NO_MORE_RESULTS,
			EntityResultType: pb.EntityResult_FULL,
			EntityResults: []*pb.EntityResult{
				{
					Entity: &pb.Entity{
						Key: key1,
						Properties: map[string]*pb.Value{
							"Name":   {ValueType: &pb.Value_StringValue{"George"}},
							"Height": {ValueType: &pb.Value_IntegerValue{32}},
						},
					},
				},
				{
					Entity: &pb.Entity{
						Key: key2,
						Properties: map[string]*pb.Value{
							"Name": {ValueType: &pb.Value_StringValue{"Rufus"}},
							// No height for Rufus.
						},
					},
				},
			},
		},
	}, nil
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
		m[p.Name] = p
	}
	return nil
}

func (m PropertyMap) Save() ([]Property, error) {
	props := make([]Property, 0, len(m))
	for _, p := range m {
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
			Name:  "Height",
			Value: int64(32),
		},
		{
			Name:  "Name",
			Value: "George",
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
		client := &Client{
			client: &fakeClient{
				queryFn: func(req *pb.RunQueryRequest) (*pb.RunQueryResponse, error) {
					nCall++
					return fakeRunQuery(req)
				},
			},
		}
		ctx := context.Background()

		var (
			expectedErr   error
			expectedNCall int
		)
		if tc.want == nil {
			expectedErr = ErrInvalidEntityType
		} else {
			expectedNCall = 1
		}
		keys, err := client.GetAll(ctx, NewQuery("Gopher"), tc.dst)
		if err != expectedErr {
			t.Errorf("dst type %T: got error %v, want %v", tc.dst, err, expectedErr)
			continue
		}
		if nCall != expectedNCall {
			t.Errorf("dst type %T: Context.Call was called an incorrect number of times: got %d want %d", tc.dst, nCall, expectedNCall)
			continue
		}
		if err != nil {
			continue
		}

		key1 := NewKey(ctx, "Gopher", "", 6, nil)
		expectedKeys := []*Key{
			key1,
			NewKey(ctx, "Gopher", "", 8, key1),
		}
		if l1, l2 := len(keys), len(expectedKeys); l1 != l2 {
			t.Errorf("dst type %T: got %d keys, want %d keys", tc.dst, l1, l2)
			continue
		}
		for i, key := range keys {
			if !keysEqual(key, expectedKeys[i]) {
				t.Errorf("dst type %T: got key #%d %v, want %v", tc.dst, i, key, expectedKeys[i])
				continue
			}
		}

		// Make sure we sort any PropertyList items (the order is not deterministic).
		if pLists, ok := tc.dst.(*[]PropertyList); ok {
			for _, p := range *pLists {
				sort.Sort(byName(p))
			}
		}

		if !reflect.DeepEqual(tc.dst, tc.want) {
			t.Errorf("dst type %T: Entities\ngot  %+v\nwant %+v", tc.dst, tc.dst, tc.want)
			continue
		}
	}
}

// keysEqual is like (*Key).Equal, but ignores the App ID.
func keysEqual(a, b *Key) bool {
	for a != nil && b != nil {
		if a.Kind() != b.Kind() || a.Name() != b.Name() || a.ID() != b.ID() {
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
		// Quoted and interesting field names.
		{"x > y =", true, "x > y", equal},
		{"` x ` =", true, " x ", equal},
		{`" x " =`, true, " x ", equal},
		{`" \"x " =`, true, ` "x `, equal},
		{`" x =`, false, "", 0},
		{`" x ="`, false, "", 0},
		{"` x \" =", false, "", 0},
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

func TestNamespaceQuery(t *testing.T) {
	gotNamespace := make(chan string, 1)
	ctx := context.Background()
	client := &Client{
		client: &fakeClient{
			queryFn: func(req *pb.RunQueryRequest) (*pb.RunQueryResponse, error) {
				if part := req.PartitionId; part != nil {
					gotNamespace <- part.NamespaceId
				} else {
					gotNamespace <- ""
				}
				return nil, errors.New("not implemented")
			},
		},
	}

	var gs []Gopher

	client.GetAll(ctx, NewQuery("gopher"), &gs)
	if got, want := <-gotNamespace, ""; got != want {
		t.Errorf("GetAll: got namespace %q, want %q", got, want)
	}
	client.Count(ctx, NewQuery("gopher"))
	if got, want := <-gotNamespace, ""; got != want {
		t.Errorf("Count: got namespace %q, want %q", got, want)
	}

	const ns = "not_default"
	ctx = WithNamespace(ctx, ns)

	client.GetAll(ctx, NewQuery("gopher"), &gs)
	if got, want := <-gotNamespace, ns; got != want {
		t.Errorf("GetAll: got namespace %q, want %q", got, want)
	}
	client.Count(ctx, NewQuery("gopher"))
	if got, want := <-gotNamespace, ns; got != want {
		t.Errorf("Count: got namespace %q, want %q", got, want)
	}
}

func TestReadOptions(t *testing.T) {
	tid := []byte{1}
	for _, test := range []struct {
		q    *Query
		want *pb.ReadOptions
	}{
		{
			q:    NewQuery(""),
			want: nil,
		},
		{
			q:    NewQuery("").Transaction(nil),
			want: nil,
		},
		{
			q:    NewQuery("").Transaction(&Transaction{id: tid}),
			want: &pb.ReadOptions{&pb.ReadOptions_Transaction{tid}},
		},
		{
			q:    NewQuery("").EventualConsistency(),
			want: &pb.ReadOptions{&pb.ReadOptions_ReadConsistency_{pb.ReadOptions_EVENTUAL}},
		},
	} {
		req := &pb.RunQueryRequest{}
		if err := test.q.toProto(req); err != nil {
			t.Fatalf("%+v: got %v, want no error", test.q, err)
		}
		if got := req.ReadOptions; !proto.Equal(got, test.want) {
			t.Errorf("%+v:\ngot  %+v\nwant %+v", test.q, got, test.want)
		}
	}
	// Test errors.
	for _, q := range []*Query{
		NewQuery("").Transaction(&Transaction{id: nil}),
		NewQuery("").Transaction(&Transaction{id: tid}).EventualConsistency(),
	} {
		req := &pb.RunQueryRequest{}
		if err := q.toProto(req); err == nil {
			t.Errorf("%+v: got nil, wanted error", q)
		}
	}
}
