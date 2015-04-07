// Copyright 2011 Google Inc. All Rights Reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"google.golang.org/appengine"
	pb "google.golang.org/appengine/internal/datastore"
)

const testAppID = "testApp"

type (
	myBlob   []byte
	myByte   byte
	myString string
)

func makeMyByteSlice(n int) []myByte {
	b := make([]myByte, n)
	for i := range b {
		b[i] = myByte(i)
	}
	return b
}

func makeInt8Slice(n int) []int8 {
	b := make([]int8, n)
	for i := range b {
		b[i] = int8(i)
	}
	return b
}

func makeUint8Slice(n int) []uint8 {
	b := make([]uint8, n)
	for i := range b {
		b[i] = uint8(i)
	}
	return b
}

func newKey(stringID string, parent *Key) *Key {
	return &Key{
		kind:     "kind",
		stringID: stringID,
		intID:    0,
		parent:   parent,
		appID:    testAppID,
	}
}

var (
	testKey0     = newKey("name0", nil)
	testKey1a    = newKey("name1", nil)
	testKey1b    = newKey("name1", nil)
	testKey2a    = newKey("name2", testKey0)
	testKey2b    = newKey("name2", testKey0)
	testGeoPt0   = appengine.GeoPoint{Lat: 1.2, Lng: 3.4}
	testGeoPt1   = appengine.GeoPoint{Lat: 5, Lng: 10}
	testBadGeoPt = appengine.GeoPoint{Lat: 1000, Lng: 34}
)

type B0 struct {
	B []byte
}

type B1 struct {
	B []int8
}

type B2 struct {
	B myBlob
}

type B3 struct {
	B []myByte
}

type B4 struct {
	B [][]byte
}

type B5 struct {
	B ByteString
}

type C0 struct {
	I int
	C chan int
}

type C1 struct {
	I int
	C *chan int
}

type C2 struct {
	I int
	C []chan int
}

type C3 struct {
	C string
}

type E struct{}

type G0 struct {
	G appengine.GeoPoint
}

type G1 struct {
	G []appengine.GeoPoint
}

type K0 struct {
	K *Key
}

type K1 struct {
	K []*Key
}

type N0 struct {
	X0
	Nonymous X0
	Ignore   string `datastore:"-"`
	Other    string
}

type N1 struct {
	X0
	Nonymous []X0
	Ignore   string `datastore:"-"`
	Other    string
}

type N2 struct {
	N1    `datastore:"red"`
	Green N1 `datastore:"green"`
	Blue  N1
	White N1 `datastore:"-"`
}

type O0 struct {
	I int64
}

type O1 struct {
	I int32
}

type U0 struct {
	U uint
}

type U1 struct {
	U string
}

type T struct {
	T time.Time
}

type X0 struct {
	S string
	I int
	i int
}

type X1 struct {
	S myString
	I int32
	J int64
}

type X2 struct {
	Z string
	i int
}

type X3 struct {
	S bool
	I int
}

type Y0 struct {
	B bool
	F []float64
	G []float64
}

type Y1 struct {
	B bool
	F float64
}

type Y2 struct {
	B bool
	F []int64
}

type Tagged struct {
	A int   `datastore:"a,noindex"`
	B []int `datastore:"b"`
	C int   `datastore:",noindex"`
	D int   `datastore:""`
	E int
	I int `datastore:"-"`
	J int `datastore:",noindex" json:"j"`

	Y0 `datastore:"-"`
	Z  chan int `datastore:"-,"`
}

type InvalidTagged1 struct {
	I int `datastore:"\t"`
}

type InvalidTagged2 struct {
	I int
	J int `datastore:"I"`
}

type Inner1 struct {
	W int32
	X string
}

type Inner2 struct {
	Y float64
}

type Inner3 struct {
	Z bool
}

type Outer struct {
	A int16
	I []Inner1
	J Inner2
	Inner3
}

type OuterEquivalent struct {
	A     int16
	IDotW []int32  `datastore:"I.W"`
	IDotX []string `datastore:"I.X"`
	JDotY float64  `datastore:"J.Y"`
	Z     bool
}

type Dotted struct {
	A DottedA `datastore:"A0.A1.A2"`
}

type DottedA struct {
	B DottedB `datastore:"B3"`
}

type DottedB struct {
	C int `datastore:"C4.C5"`
}

type SliceOfSlices struct {
	I int
	S []struct {
		J int
		F []float64
	}
}

type Recursive struct {
	I int
	R []Recursive
}

type MutuallyRecursive0 struct {
	I int
	R []MutuallyRecursive1
}

type MutuallyRecursive1 struct {
	I int
	R []MutuallyRecursive0
}

type Doubler struct {
	S string
	I int64
	B bool
}

func (d *Doubler) Load(props []Property) error {
	return LoadStruct(d, props)
}

func (d *Doubler) Save() ([]Property, error) {
	// Save the default Property slice to an in-memory buffer (a PropertyList).
	props, err := SaveStruct(d)
	if err != nil {
		return nil, err
	}
	var list PropertyList
	if err := list.Load(props); err != nil {
		return nil, err
	}

	// Edit that PropertyList, and send it on.
	for i := range list {
		switch v := list[i].Value.(type) {
		case string:
			// + means string concatenation.
			list[i].Value = v + v
		case int64:
			// + means integer addition.
			list[i].Value = v + v
		}
	}
	return list.Save()
}

var _ PropertyLoadSaver = (*Doubler)(nil)

type Deriver struct {
	S, Derived, Ignored string
}

func (e *Deriver) Load(props []Property) error {
	for _, p := range props {
		if p.Name != "S" {
			continue
		}
		e.S = p.Value.(string)
		e.Derived = "derived+" + e.S
	}
	return nil
}

func (e *Deriver) Save() ([]Property, error) {
	return []Property{
		{
			Name:  "S",
			Value: e.S,
		},
	}, nil
}

var _ PropertyLoadSaver = (*Deriver)(nil)

type BadMultiPropEntity struct{}

func (e *BadMultiPropEntity) Load(props []Property) error {
	return errors.New("unimplemented")
}

func (e *BadMultiPropEntity) Save() ([]Property, error) {
	// Write multiple properties with the same name "I", but Multiple is false.
	var props []Property
	for i := 0; i < 3; i++ {
		props = append(props, Property{
			Name:  "I",
			Value: int64(i),
		})
	}
	return props, nil
}

var _ PropertyLoadSaver = (*BadMultiPropEntity)(nil)

type BK struct {
	Key appengine.BlobKey
}

type testCase struct {
	desc   string
	src    interface{}
	want   interface{}
	putErr string
	getErr string
}

var testCases = []testCase{
	{
		"chan save fails",
		&C0{I: -1},
		&E{},
		"unsupported struct field",
		"",
	},
	{
		"*chan save fails",
		&C1{I: -1},
		&E{},
		"unsupported struct field",
		"",
	},
	{
		"[]chan save fails",
		&C2{I: -1, C: make([]chan int, 8)},
		&E{},
		"unsupported struct field",
		"",
	},
	{
		"chan load fails",
		&C3{C: "not a chan"},
		&C0{},
		"",
		"type mismatch",
	},
	{
		"*chan load fails",
		&C3{C: "not a *chan"},
		&C1{},
		"",
		"type mismatch",
	},
	{
		"[]chan load fails",
		&C3{C: "not a []chan"},
		&C2{},
		"",
		"type mismatch",
	},
	{
		"empty struct",
		&E{},
		&E{},
		"",
		"",
	},
	{
		"geopoint",
		&G0{G: testGeoPt0},
		&G0{G: testGeoPt0},
		"",
		"",
	},
	{
		"geopoint invalid",
		&G0{G: testBadGeoPt},
		&G0{},
		"invalid GeoPoint value",
		"",
	},
	{
		"geopoint as props",
		&G0{G: testGeoPt0},
		&PropertyList{
			Property{Name: "G", Value: testGeoPt0, NoIndex: false, Multiple: false},
		},
		"",
		"",
	},
	{
		"geopoint slice",
		&G1{G: []appengine.GeoPoint{testGeoPt0, testGeoPt1}},
		&G1{G: []appengine.GeoPoint{testGeoPt0, testGeoPt1}},
		"",
		"",
	},
	{
		"key",
		&K0{K: testKey1a},
		&K0{K: testKey1b},
		"",
		"",
	},
	{
		"key with parent",
		&K0{K: testKey2a},
		&K0{K: testKey2b},
		"",
		"",
	},
	{
		"nil key",
		&K0{},
		&K0{},
		"",
		"",
	},
	{
		"all nil keys in slice",
		&K1{[]*Key{nil, nil}},
		&K1{[]*Key{nil, nil}},
		"",
		"",
	},
	{
		"some nil keys in slice",
		&K1{[]*Key{testKey1a, nil, testKey2a}},
		&K1{[]*Key{testKey1b, nil, testKey2b}},
		"",
		"",
	},
	{
		"overflow",
		&O0{I: 1 << 48},
		&O1{},
		"",
		"overflow",
	},
	{
		"time",
		&T{T: time.Unix(1e9, 0)},
		&T{T: time.Unix(1e9, 0)},
		"",
		"",
	},
	{
		"time as props",
		&T{T: time.Unix(1e9, 0)},
		&PropertyList{
			Property{Name: "T", Value: time.Unix(1e9, 0), NoIndex: false, Multiple: false},
		},
		"",
		"",
	},
	{
		"uint save",
		&U0{U: 1},
		&U0{},
		"unsupported struct field",
		"",
	},
	{
		"uint load",
		&U1{U: "not a uint"},
		&U0{},
		"",
		"type mismatch",
	},
	{
		"zero",
		&X0{},
		&X0{},
		"",
		"",
	},
	{
		"basic",
		&X0{S: "one", I: 2, i: 3},
		&X0{S: "one", I: 2},
		"",
		"",
	},
	{
		"save string/int load myString/int32",
		&X0{S: "one", I: 2, i: 3},
		&X1{S: "one", I: 2},
		"",
		"",
	},
	{
		"missing fields",
		&X0{S: "one", I: 2, i: 3},
		&X2{},
		"",
		"no such struct field",
	},
	{
		"save string load bool",
		&X0{S: "one", I: 2, i: 3},
		&X3{I: 2},
		"",
		"type mismatch",
	},
	{
		"basic slice",
		&Y0{B: true, F: []float64{7, 8, 9}},
		&Y0{B: true, F: []float64{7, 8, 9}},
		"",
		"",
	},
	{
		"save []float64 load float64",
		&Y0{B: true, F: []float64{7, 8, 9}},
		&Y1{B: true},
		"",
		"requires a slice",
	},
	{
		"save []float64 load []int64",
		&Y0{B: true, F: []float64{7, 8, 9}},
		&Y2{B: true},
		"",
		"type mismatch",
	},
	{
		"single slice is too long",
		&Y0{F: make([]float64, maxIndexedProperties+1)},
		&Y0{},
		"too many indexed properties",
		"",
	},
	{
		"two slices are too long",
		&Y0{F: make([]float64, maxIndexedProperties), G: make([]float64, maxIndexedProperties)},
		&Y0{},
		"too many indexed properties",
		"",
	},
	{
		"one slice and one scalar are too long",
		&Y0{F: make([]float64, maxIndexedProperties), B: true},
		&Y0{},
		"too many indexed properties",
		"",
	},
	{
		"long blob",
		&B0{B: makeUint8Slice(maxIndexedProperties + 1)},
		&B0{B: makeUint8Slice(maxIndexedProperties + 1)},
		"",
		"",
	},
	{
		"long []int8 is too long",
		&B1{B: makeInt8Slice(maxIndexedProperties + 1)},
		&B1{},
		"too many indexed properties",
		"",
	},
	{
		"short []int8",
		&B1{B: makeInt8Slice(3)},
		&B1{B: makeInt8Slice(3)},
		"",
		"",
	},
	{
		"long myBlob",
		&B2{B: makeUint8Slice(maxIndexedProperties + 1)},
		&B2{B: makeUint8Slice(maxIndexedProperties + 1)},
		"",
		"",
	},
	{
		"short myBlob",
		&B2{B: makeUint8Slice(3)},
		&B2{B: makeUint8Slice(3)},
		"",
		"",
	},
	{
		"long []myByte",
		&B3{B: makeMyByteSlice(maxIndexedProperties + 1)},
		&B3{B: makeMyByteSlice(maxIndexedProperties + 1)},
		"",
		"",
	},
	{
		"short []myByte",
		&B3{B: makeMyByteSlice(3)},
		&B3{B: makeMyByteSlice(3)},
		"",
		"",
	},
	{
		"slice of blobs",
		&B4{B: [][]byte{
			makeUint8Slice(3),
			makeUint8Slice(4),
			makeUint8Slice(5),
		}},
		&B4{B: [][]byte{
			makeUint8Slice(3),
			makeUint8Slice(4),
			makeUint8Slice(5),
		}},
		"",
		"",
	},
	{
		"short ByteString",
		&B5{B: ByteString(makeUint8Slice(3))},
		&B5{B: ByteString(makeUint8Slice(3))},
		"",
		"",
	},
	{
		"short ByteString as props",
		&B5{B: ByteString(makeUint8Slice(3))},
		&PropertyList{
			Property{Name: "B", Value: ByteString(makeUint8Slice(3)), NoIndex: false, Multiple: false},
		},
		"",
		"",
	},
	{
		"short ByteString into string",
		&B5{B: ByteString("legacy")},
		&struct{ B string }{"legacy"},
		"",
		"",
	},
	{
		"[]byte must be noindex",
		&PropertyList{
			Property{Name: "B", Value: makeUint8Slice(3), NoIndex: false},
		},
		nil,
		"cannot index a []byte valued Property",
		"",
	},
	{
		"save tagged load props",
		&Tagged{A: 1, B: []int{21, 22, 23}, C: 3, D: 4, E: 5, I: 6, J: 7},
		&PropertyList{
			// A and B are renamed to a and b; A and C are noindex, I is ignored.
			// Indexed properties are loaded before raw properties. Thus, the
			// result is: b, b, b, D, E, a, c.
			Property{Name: "b", Value: int64(21), NoIndex: false, Multiple: true},
			Property{Name: "b", Value: int64(22), NoIndex: false, Multiple: true},
			Property{Name: "b", Value: int64(23), NoIndex: false, Multiple: true},
			Property{Name: "D", Value: int64(4), NoIndex: false, Multiple: false},
			Property{Name: "E", Value: int64(5), NoIndex: false, Multiple: false},
			Property{Name: "a", Value: int64(1), NoIndex: true, Multiple: false},
			Property{Name: "C", Value: int64(3), NoIndex: true, Multiple: false},
			Property{Name: "J", Value: int64(7), NoIndex: true, Multiple: false},
		},
		"",
		"",
	},
	{
		"save tagged load tagged",
		&Tagged{A: 1, B: []int{21, 22, 23}, C: 3, D: 4, E: 5, I: 6, J: 7},
		&Tagged{A: 1, B: []int{21, 22, 23}, C: 3, D: 4, E: 5, J: 7},
		"",
		"",
	},
	{
		"save props load tagged",
		&PropertyList{
			Property{Name: "A", Value: int64(11), NoIndex: true, Multiple: false},
			Property{Name: "a", Value: int64(12), NoIndex: true, Multiple: false},
		},
		&Tagged{A: 12},
		"",
		`cannot load field "A"`,
	},
	{
		"invalid tagged1",
		&InvalidTagged1{I: 1},
		&InvalidTagged1{},
		"struct tag has invalid property name",
		"",
	},
	{
		"invalid tagged2",
		&InvalidTagged2{I: 1, J: 2},
		&InvalidTagged2{},
		"struct tag has repeated property name",
		"",
	},
	{
		"doubler",
		&Doubler{S: "s", I: 1, B: true},
		&Doubler{S: "ss", I: 2, B: true},
		"",
		"",
	},
	{
		"save struct load props",
		&X0{S: "s", I: 1},
		&PropertyList{
			Property{Name: "S", Value: "s", NoIndex: false, Multiple: false},
			Property{Name: "I", Value: int64(1), NoIndex: false, Multiple: false},
		},
		"",
		"",
	},
	{
		"save props load struct",
		&PropertyList{
			Property{Name: "S", Value: "s", NoIndex: false, Multiple: false},
			Property{Name: "I", Value: int64(1), NoIndex: false, Multiple: false},
		},
		&X0{S: "s", I: 1},
		"",
		"",
	},
	{
		"nil-value props",
		&PropertyList{
			Property{Name: "I", Value: nil, NoIndex: false, Multiple: false},
			Property{Name: "B", Value: nil, NoIndex: false, Multiple: false},
			Property{Name: "S", Value: nil, NoIndex: false, Multiple: false},
			Property{Name: "F", Value: nil, NoIndex: false, Multiple: false},
			Property{Name: "K", Value: nil, NoIndex: false, Multiple: false},
			Property{Name: "T", Value: nil, NoIndex: false, Multiple: false},
			Property{Name: "J", Value: nil, NoIndex: false, Multiple: true},
			Property{Name: "J", Value: int64(7), NoIndex: false, Multiple: true},
			Property{Name: "J", Value: nil, NoIndex: false, Multiple: true},
		},
		&struct {
			I int64
			B bool
			S string
			F float64
			K *Key
			T time.Time
			J []int64
		}{
			J: []int64{0, 7, 0},
		},
		"",
		"",
	},
	{
		"save outer load props",
		&Outer{
			A: 1,
			I: []Inner1{
				{10, "ten"},
				{20, "twenty"},
				{30, "thirty"},
			},
			J: Inner2{
				Y: 3.14,
			},
			Inner3: Inner3{
				Z: true,
			},
		},
		&PropertyList{
			Property{Name: "A", Value: int64(1), NoIndex: false, Multiple: false},
			Property{Name: "I.W", Value: int64(10), NoIndex: false, Multiple: true},
			Property{Name: "I.X", Value: "ten", NoIndex: false, Multiple: true},
			Property{Name: "I.W", Value: int64(20), NoIndex: false, Multiple: true},
			Property{Name: "I.X", Value: "twenty", NoIndex: false, Multiple: true},
			Property{Name: "I.W", Value: int64(30), NoIndex: false, Multiple: true},
			Property{Name: "I.X", Value: "thirty", NoIndex: false, Multiple: true},
			Property{Name: "J.Y", Value: float64(3.14), NoIndex: false, Multiple: false},
			Property{Name: "Z", Value: true, NoIndex: false, Multiple: false},
		},
		"",
		"",
	},
	{
		"save props load outer-equivalent",
		&PropertyList{
			Property{Name: "A", Value: int64(1), NoIndex: false, Multiple: false},
			Property{Name: "I.W", Value: int64(10), NoIndex: false, Multiple: true},
			Property{Name: "I.X", Value: "ten", NoIndex: false, Multiple: true},
			Property{Name: "I.W", Value: int64(20), NoIndex: false, Multiple: true},
			Property{Name: "I.X", Value: "twenty", NoIndex: false, Multiple: true},
			Property{Name: "I.W", Value: int64(30), NoIndex: false, Multiple: true},
			Property{Name: "I.X", Value: "thirty", NoIndex: false, Multiple: true},
			Property{Name: "J.Y", Value: float64(3.14), NoIndex: false, Multiple: false},
			Property{Name: "Z", Value: true, NoIndex: false, Multiple: false},
		},
		&OuterEquivalent{
			A:     1,
			IDotW: []int32{10, 20, 30},
			IDotX: []string{"ten", "twenty", "thirty"},
			JDotY: 3.14,
			Z:     true,
		},
		"",
		"",
	},
	{
		"save outer-equivalent load outer",
		&OuterEquivalent{
			A:     1,
			IDotW: []int32{10, 20, 30},
			IDotX: []string{"ten", "twenty", "thirty"},
			JDotY: 3.14,
			Z:     true,
		},
		&Outer{
			A: 1,
			I: []Inner1{
				{10, "ten"},
				{20, "twenty"},
				{30, "thirty"},
			},
			J: Inner2{
				Y: 3.14,
			},
			Inner3: Inner3{
				Z: true,
			},
		},
		"",
		"",
	},
	{
		"dotted names save",
		&Dotted{A: DottedA{B: DottedB{C: 88}}},
		&PropertyList{
			Property{Name: "A0.A1.A2.B3.C4.C5", Value: int64(88), NoIndex: false, Multiple: false},
		},
		"",
		"",
	},
	{
		"dotted names load",
		&PropertyList{
			Property{Name: "A0.A1.A2.B3.C4.C5", Value: int64(99), NoIndex: false, Multiple: false},
		},
		&Dotted{A: DottedA{B: DottedB{C: 99}}},
		"",
		"",
	},
	{
		"save struct load deriver",
		&X0{S: "s", I: 1},
		&Deriver{S: "s", Derived: "derived+s"},
		"",
		"",
	},
	{
		"save deriver load struct",
		&Deriver{S: "s", Derived: "derived+s", Ignored: "ignored"},
		&X0{S: "s"},
		"",
		"",
	},
	{
		"bad multi-prop entity",
		&BadMultiPropEntity{},
		&BadMultiPropEntity{},
		"Multiple is false",
		"",
	},
	// Regression: CL 25062824 broke handling of appengine.BlobKey fields.
	{
		"appengine.BlobKey",
		&BK{Key: "blah"},
		&BK{Key: "blah"},
		"",
		"",
	},
	{
		"zero time.Time",
		&T{T: time.Time{}},
		&T{T: time.Time{}},
		"",
		"",
	},
	{
		"time.Time near Unix zero time",
		&T{T: time.Unix(0, 4e3)},
		&T{T: time.Unix(0, 4e3)},
		"",
		"",
	},
	{
		"time.Time, far in the future",
		&T{T: time.Date(99999, 1, 1, 0, 0, 0, 0, time.UTC)},
		&T{T: time.Date(99999, 1, 1, 0, 0, 0, 0, time.UTC)},
		"",
		"",
	},
	{
		"time.Time, very far in the past",
		&T{T: time.Date(-300000, 1, 1, 0, 0, 0, 0, time.UTC)},
		&T{},
		"time value out of range",
		"",
	},
	{
		"time.Time, very far in the future",
		&T{T: time.Date(294248, 1, 1, 0, 0, 0, 0, time.UTC)},
		&T{},
		"time value out of range",
		"",
	},
	{
		"structs",
		&N0{
			X0:       X0{S: "one", I: 2, i: 3},
			Nonymous: X0{S: "four", I: 5, i: 6},
			Ignore:   "ignore",
			Other:    "other",
		},
		&N0{
			X0:       X0{S: "one", I: 2},
			Nonymous: X0{S: "four", I: 5},
			Other:    "other",
		},
		"",
		"",
	},
	{
		"slice of structs",
		&N1{
			X0: X0{S: "one", I: 2, i: 3},
			Nonymous: []X0{
				{S: "four", I: 5, i: 6},
				{S: "seven", I: 8, i: 9},
				{S: "ten", I: 11, i: 12},
				{S: "thirteen", I: 14, i: 15},
			},
			Ignore: "ignore",
			Other:  "other",
		},
		&N1{
			X0: X0{S: "one", I: 2},
			Nonymous: []X0{
				{S: "four", I: 5},
				{S: "seven", I: 8},
				{S: "ten", I: 11},
				{S: "thirteen", I: 14},
			},
			Other: "other",
		},
		"",
		"",
	},
	{
		"structs with slices of structs",
		&N2{
			N1: N1{
				X0: X0{S: "rouge"},
				Nonymous: []X0{
					{S: "rosso0"},
					{S: "rosso1"},
				},
			},
			Green: N1{
				X0: X0{S: "vert"},
				Nonymous: []X0{
					{S: "verde0"},
					{S: "verde1"},
					{S: "verde2"},
				},
			},
			Blue: N1{
				X0: X0{S: "bleu"},
				Nonymous: []X0{
					{S: "blu0"},
					{S: "blu1"},
					{S: "blu2"},
					{S: "blu3"},
				},
			},
		},
		&N2{
			N1: N1{
				X0: X0{S: "rouge"},
				Nonymous: []X0{
					{S: "rosso0"},
					{S: "rosso1"},
				},
			},
			Green: N1{
				X0: X0{S: "vert"},
				Nonymous: []X0{
					{S: "verde0"},
					{S: "verde1"},
					{S: "verde2"},
				},
			},
			Blue: N1{
				X0: X0{S: "bleu"},
				Nonymous: []X0{
					{S: "blu0"},
					{S: "blu1"},
					{S: "blu2"},
					{S: "blu3"},
				},
			},
		},
		"",
		"",
	},
	{
		"save structs load props",
		&N2{
			N1: N1{
				X0: X0{S: "rouge"},
				Nonymous: []X0{
					{S: "rosso0"},
					{S: "rosso1"},
				},
			},
			Green: N1{
				X0: X0{S: "vert"},
				Nonymous: []X0{
					{S: "verde0"},
					{S: "verde1"},
					{S: "verde2"},
				},
			},
			Blue: N1{
				X0: X0{S: "bleu"},
				Nonymous: []X0{
					{S: "blu0"},
					{S: "blu1"},
					{S: "blu2"},
					{S: "blu3"},
				},
			},
		},
		&PropertyList{
			Property{Name: "red.S", Value: "rouge", NoIndex: false, Multiple: false},
			Property{Name: "red.I", Value: int64(0), NoIndex: false, Multiple: false},
			Property{Name: "red.Nonymous.S", Value: "rosso0", NoIndex: false, Multiple: true},
			Property{Name: "red.Nonymous.I", Value: int64(0), NoIndex: false, Multiple: true},
			Property{Name: "red.Nonymous.S", Value: "rosso1", NoIndex: false, Multiple: true},
			Property{Name: "red.Nonymous.I", Value: int64(0), NoIndex: false, Multiple: true},
			Property{Name: "red.Other", Value: "", NoIndex: false, Multiple: false},
			Property{Name: "green.S", Value: "vert", NoIndex: false, Multiple: false},
			Property{Name: "green.I", Value: int64(0), NoIndex: false, Multiple: false},
			Property{Name: "green.Nonymous.S", Value: "verde0", NoIndex: false, Multiple: true},
			Property{Name: "green.Nonymous.I", Value: int64(0), NoIndex: false, Multiple: true},
			Property{Name: "green.Nonymous.S", Value: "verde1", NoIndex: false, Multiple: true},
			Property{Name: "green.Nonymous.I", Value: int64(0), NoIndex: false, Multiple: true},
			Property{Name: "green.Nonymous.S", Value: "verde2", NoIndex: false, Multiple: true},
			Property{Name: "green.Nonymous.I", Value: int64(0), NoIndex: false, Multiple: true},
			Property{Name: "green.Other", Value: "", NoIndex: false, Multiple: false},
			Property{Name: "Blue.S", Value: "bleu", NoIndex: false, Multiple: false},
			Property{Name: "Blue.I", Value: int64(0), NoIndex: false, Multiple: false},
			Property{Name: "Blue.Nonymous.S", Value: "blu0", NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.I", Value: int64(0), NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.S", Value: "blu1", NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.I", Value: int64(0), NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.S", Value: "blu2", NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.I", Value: int64(0), NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.S", Value: "blu3", NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.I", Value: int64(0), NoIndex: false, Multiple: true},
			Property{Name: "Blue.Other", Value: "", NoIndex: false, Multiple: false},
		},
		"",
		"",
	},
	{
		"save props load structs with ragged fields",
		&PropertyList{
			Property{Name: "red.S", Value: "rot", NoIndex: false, Multiple: false},
			Property{Name: "green.Nonymous.I", Value: int64(10), NoIndex: false, Multiple: true},
			Property{Name: "green.Nonymous.I", Value: int64(11), NoIndex: false, Multiple: true},
			Property{Name: "green.Nonymous.I", Value: int64(12), NoIndex: false, Multiple: true},
			Property{Name: "green.Nonymous.I", Value: int64(13), NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.S", Value: "blau0", NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.I", Value: int64(20), NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.S", Value: "blau1", NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.I", Value: int64(21), NoIndex: false, Multiple: true},
			Property{Name: "Blue.Nonymous.S", Value: "blau2", NoIndex: false, Multiple: true},
		},
		&N2{
			N1: N1{
				X0: X0{S: "rot"},
			},
			Green: N1{
				Nonymous: []X0{
					{I: 10},
					{I: 11},
					{I: 12},
					{I: 13},
				},
			},
			Blue: N1{
				Nonymous: []X0{
					{S: "blau0", I: 20},
					{S: "blau1", I: 21},
					{S: "blau2"},
				},
			},
		},
		"",
		"",
	},
	{
		"save structs with noindex tags",
		&struct {
			A struct {
				X string `datastore:",noindex"`
				Y string
			} `datastore:",noindex"`
			B struct {
				X string `datastore:",noindex"`
				Y string
			}
		}{},
		&PropertyList{
			Property{Name: "B.Y", Value: "", NoIndex: false, Multiple: false},
			Property{Name: "A.X", Value: "", NoIndex: true, Multiple: false},
			Property{Name: "A.Y", Value: "", NoIndex: true, Multiple: false},
			Property{Name: "B.X", Value: "", NoIndex: true, Multiple: false},
		},
		"",
		"",
	},
	{
		"embedded struct with name override",
		&struct {
			Inner1 `datastore:"foo"`
		}{},
		&PropertyList{
			Property{Name: "foo.W", Value: int64(0), NoIndex: false, Multiple: false},
			Property{Name: "foo.X", Value: "", NoIndex: false, Multiple: false},
		},
		"",
		"",
	},
	{
		"slice of slices",
		&SliceOfSlices{},
		nil,
		"flattening nested structs leads to a slice of slices",
		"",
	},
	{
		"recursive struct",
		&Recursive{},
		nil,
		"recursive struct",
		"",
	},
	{
		"mutually recursive struct",
		&MutuallyRecursive0{},
		nil,
		"recursive struct",
		"",
	},
	{
		"non-exported struct fields",
		&struct {
			i, J int64
		}{i: 1, J: 2},
		&PropertyList{
			Property{Name: "J", Value: int64(2), NoIndex: false, Multiple: false},
		},
		"",
		"",
	},
	{
		"json.RawMessage",
		&struct {
			J json.RawMessage
		}{
			J: json.RawMessage("rawr"),
		},
		&PropertyList{
			Property{Name: "J", Value: []byte("rawr"), NoIndex: true, Multiple: false},
		},
		"",
		"",
	},
	{
		"json.RawMessage to myBlob",
		&struct {
			B json.RawMessage
		}{
			B: json.RawMessage("rawr"),
		},
		&B2{B: myBlob("rawr")},
		"",
		"",
	},
}

// checkErr returns the empty string if either both want and err are zero,
// or if want is a non-empty substring of err's string representation.
func checkErr(want string, err error) string {
	if err != nil {
		got := err.Error()
		if want == "" || strings.Index(got, want) == -1 {
			return got
		}
	} else if want != "" {
		return fmt.Sprintf("want error %q", want)
	}
	return ""
}

func TestRoundTrip(t *testing.T) {
	for _, tc := range testCases {
		p, err := saveEntity(testAppID, testKey0, tc.src)
		if s := checkErr(tc.putErr, err); s != "" {
			t.Errorf("%s: save: %s", tc.desc, s)
			continue
		}
		if p == nil {
			continue
		}
		var got interface{}
		if _, ok := tc.want.(*PropertyList); ok {
			got = new(PropertyList)
		} else {
			got = reflect.New(reflect.TypeOf(tc.want).Elem()).Interface()
		}
		err = loadEntity(got, p)
		if s := checkErr(tc.getErr, err); s != "" {
			t.Errorf("%s: load: %s", tc.desc, s)
			continue
		}
		equal := false
		if gotT, ok := got.(*T); ok {
			// Round tripping a time.Time can result in a different time.Location: Local instead of UTC.
			// We therefore test equality explicitly, instead of relying on reflect.DeepEqual.
			equal = gotT.T.Equal(tc.want.(*T).T)
		} else {
			equal = reflect.DeepEqual(got, tc.want)
		}
		if !equal {
			t.Errorf("%s: compare: got %v want %v", tc.desc, got, tc.want)
			continue
		}
	}
}

func TestQueryConstruction(t *testing.T) {
	tests := []struct {
		q, exp *Query
		err    string
	}{
		{
			q: NewQuery("Foo"),
			exp: &Query{
				kind:  "Foo",
				limit: -1,
			},
		},
		{
			// Regular filtered query with standard spacing.
			q: NewQuery("Foo").Filter("foo >", 7),
			exp: &Query{
				kind: "Foo",
				filter: []filter{
					{
						FieldName: "foo",
						Op:        greaterThan,
						Value:     7,
					},
				},
				limit: -1,
			},
		},
		{
			// Filtered query with no spacing.
			q: NewQuery("Foo").Filter("foo=", 6),
			exp: &Query{
				kind: "Foo",
				filter: []filter{
					{
						FieldName: "foo",
						Op:        equal,
						Value:     6,
					},
				},
				limit: -1,
			},
		},
		{
			// Filtered query with funky spacing.
			q: NewQuery("Foo").Filter(" foo< ", 8),
			exp: &Query{
				kind: "Foo",
				filter: []filter{
					{
						FieldName: "foo",
						Op:        lessThan,
						Value:     8,
					},
				},
				limit: -1,
			},
		},
		{
			// Filtered query with multicharacter op.
			q: NewQuery("Foo").Filter("foo >=", 9),
			exp: &Query{
				kind: "Foo",
				filter: []filter{
					{
						FieldName: "foo",
						Op:        greaterEq,
						Value:     9,
					},
				},
				limit: -1,
			},
		},
		{
			// Query with ordering.
			q: NewQuery("Foo").Order("bar"),
			exp: &Query{
				kind: "Foo",
				order: []order{
					{
						FieldName: "bar",
						Direction: ascending,
					},
				},
				limit: -1,
			},
		},
		{
			// Query with reverse ordering, and funky spacing.
			q: NewQuery("Foo").Order(" - bar"),
			exp: &Query{
				kind: "Foo",
				order: []order{
					{
						FieldName: "bar",
						Direction: descending,
					},
				},
				limit: -1,
			},
		},
		{
			// Query with an empty ordering.
			q:   NewQuery("Foo").Order(""),
			err: "empty order",
		},
		{
			// Query with a + ordering.
			q:   NewQuery("Foo").Order("+bar"),
			err: "invalid order",
		},
	}
	for i, test := range tests {
		if test.q.err != nil {
			got := test.q.err.Error()
			if !strings.Contains(got, test.err) {
				t.Errorf("%d: error mismatch: got %q want something containing %q", i, got, test.err)
			}
			continue
		}
		if !reflect.DeepEqual(test.q, test.exp) {
			t.Errorf("%d: mismatch: got %v want %v", i, test.q, test.exp)
		}
	}
}

func TestStringMeaning(t *testing.T) {
	var xx [4]interface{}
	xx[0] = &struct {
		X string
	}{"xx0"}
	xx[1] = &struct {
		X string `datastore:",noindex"`
	}{"xx1"}
	xx[2] = &struct {
		X []byte
	}{[]byte("xx2")}
	xx[3] = &struct {
		X []byte `datastore:",noindex"`
	}{[]byte("xx3")}

	indexed := [4]bool{
		true,
		false,
		false, // A []byte is always no-index.
		false,
	}
	want := [4]pb.Property_Meaning{
		pb.Property_NO_MEANING,
		pb.Property_TEXT,
		pb.Property_BLOB,
		pb.Property_BLOB,
	}

	for i, x := range xx {
		props, err := SaveStruct(x)
		if err != nil {
			t.Errorf("i=%d: SaveStruct: %v", i, err)
			continue
		}
		e, err := propertiesToProto("appID", testKey0, props)
		if err != nil {
			t.Errorf("i=%d: propertiesToProto: %v", i, err)
			continue
		}
		var p *pb.Property
		switch {
		case indexed[i] && len(e.Property) == 1:
			p = e.Property[0]
		case !indexed[i] && len(e.RawProperty) == 1:
			p = e.RawProperty[0]
		default:
			t.Errorf("i=%d: EntityProto did not have expected property slice", i)
			continue
		}
		if got := p.GetMeaning(); got != want[i] {
			t.Errorf("i=%d: meaning: got %v, want %v", i, got, want[i])
			continue
		}
	}
}
