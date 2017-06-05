// Copyright 2011 Google Inc. All Rights Reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

package datastore

import (
	"reflect"
	"testing"
	"time"

	"google.golang.org/appengine"
)

func TestValidPropertyName(t *testing.T) {
	testCases := []struct {
		name string
		want bool
	}{
		// Invalid names.
		{"", false},
		{"'", false},
		{".", false},
		{"..", false},
		{".foo", false},
		{"0", false},
		{"00", false},
		{"X.X.4.X.X", false},
		{"\n", false},
		{"\x00", false},
		{"abc\xffz", false},
		{"foo.", false},
		{"foo..", false},
		{"foo..bar", false},
		{"☃", false},
		{`"`, false},
		// Valid names.
		{"AB", true},
		{"Abc", true},
		{"X.X.X.X.X", true},
		{"_", true},
		{"_0", true},
		{"a", true},
		{"a_B", true},
		{"f00", true},
		{"f0o", true},
		{"fo0", true},
		{"foo", true},
		{"foo.bar", true},
		{"foo.bar.baz", true},
		{"世界", true},
	}
	for _, tc := range testCases {
		got := validPropertyName(tc.name)
		if got != tc.want {
			t.Errorf("%q: got %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestStructCodec(t *testing.T) {
	type oStruct struct {
		O int
	}
	type pStruct struct {
		P int
		Q int
	}
	type rStruct struct {
		R int
		S pStruct
		T oStruct
		oStruct
	}
	type uStruct struct {
		U int
		v int
	}
	type vStruct struct {
		V string `datastore:",noindex"`
	}
	oStructCodec := &structCodec{
		byIndex: []structTag{
			{name: "O"},
		},
		byName: map[string]fieldCodec{
			"O": {index: 0},
		},
		complete: true,
	}
	pStructCodec := &structCodec{
		byIndex: []structTag{
			{name: "P"},
			{name: "Q"},
		},
		byName: map[string]fieldCodec{
			"P": {index: 0},
			"Q": {index: 1},
		},
		complete: true,
	}
	rStructCodec := &structCodec{
		byIndex: []structTag{
			{name: "R"},
			{name: "S."},
			{name: "T."},
			{name: ""},
		},
		byName: map[string]fieldCodec{
			"R":   {index: 0},
			"S.P": {index: 1, substructCodec: pStructCodec},
			"S.Q": {index: 1, substructCodec: pStructCodec},
			"T.O": {index: 2, substructCodec: oStructCodec},
			"O":   {index: 3, substructCodec: oStructCodec},
		},
		complete: true,
	}
	uStructCodec := &structCodec{
		byIndex: []structTag{
			{name: "U"},
			{name: "v"},
		},
		byName: map[string]fieldCodec{
			"U": {index: 0},
			"v": {index: 1},
		},
		complete: true,
	}
	vStructCodec := &structCodec{
		byIndex: []structTag{
			{name: "V", noIndex: true},
		},
		byName: map[string]fieldCodec{
			"V": {index: 0},
		},
		complete: true,
	}

	testCases := []struct {
		desc        string
		structValue interface{}
		want        *structCodec
	}{
		{
			"oStruct",
			oStruct{},
			oStructCodec,
		},
		{
			"pStruct",
			pStruct{},
			pStructCodec,
		},
		{
			"rStruct",
			rStruct{},
			rStructCodec,
		},
		{
			"uStruct",
			uStruct{},
			uStructCodec,
		},
		{
			"non-basic fields",
			struct {
				B appengine.BlobKey
				K *Key
				T time.Time
			}{},
			&structCodec{
				byIndex: []structTag{
					{name: "B"},
					{name: "K"},
					{name: "T"},
				},
				byName: map[string]fieldCodec{
					"B": {index: 0},
					"K": {index: 1},
					"T": {index: 2},
				},
				complete: true,
			},
		},
		{
			"struct tags with ignored embed",
			struct {
				A       int `datastore:"a,noindex"`
				B       int `datastore:"b"`
				C       int `datastore:",noindex"`
				D       int `datastore:""`
				E       int
				I       int `datastore:"-"`
				J       int `datastore:",noindex" json:"j"`
				oStruct `datastore:"-"`
			}{},
			&structCodec{
				byIndex: []structTag{
					{name: "a", noIndex: true},
					{name: "b", noIndex: false},
					{name: "C", noIndex: true},
					{name: "D", noIndex: false},
					{name: "E", noIndex: false},
					{name: "-", noIndex: false},
					{name: "J", noIndex: true},
					{name: "-", noIndex: false},
				},
				byName: map[string]fieldCodec{
					"a": {index: 0},
					"b": {index: 1},
					"C": {index: 2},
					"D": {index: 3},
					"E": {index: 4},
					"J": {index: 6},
				},
				complete: true,
			},
		},
		{
			"unexported fields",
			struct {
				A int
				b int
				C int `datastore:"x"`
				d int `datastore:"Y"`
			}{},
			&structCodec{
				byIndex: []structTag{
					{name: "A"},
					{name: "b"},
					{name: "x"},
					{name: "Y"},
				},
				byName: map[string]fieldCodec{
					"A": {index: 0},
					"b": {index: 1},
					"x": {index: 2},
					"Y": {index: 3},
				},
				complete: true,
			},
		},
		{
			"nested and embedded structs",
			struct {
				A   int
				B   int
				CC  oStruct
				DDD rStruct
				oStruct
			}{},
			&structCodec{
				byIndex: []structTag{
					{name: "A"},
					{name: "B"},
					{name: "CC."},
					{name: "DDD."},
					{name: ""},
				},
				byName: map[string]fieldCodec{
					"A":       {index: 0},
					"B":       {index: 1},
					"CC.O":    {index: 2, substructCodec: oStructCodec},
					"DDD.R":   {index: 3, substructCodec: rStructCodec},
					"DDD.S.P": {index: 3, substructCodec: rStructCodec},
					"DDD.S.Q": {index: 3, substructCodec: rStructCodec},
					"DDD.T.O": {index: 3, substructCodec: rStructCodec},
					"DDD.O":   {index: 3, substructCodec: rStructCodec},
					"O":       {index: 4, substructCodec: oStructCodec},
				},
				complete: true,
			},
		},
		{
			"struct tags with nested and embedded structs",
			struct {
				A       int     `datastore:"-"`
				B       int     `datastore:"w"`
				C       oStruct `datastore:"xx"`
				D       rStruct `datastore:"y"`
				oStruct `datastore:"z"`
			}{},
			&structCodec{
				byIndex: []structTag{
					{name: "-"},
					{name: "w"},
					{name: "xx."},
					{name: "y."},
					{name: "z."},
				},
				byName: map[string]fieldCodec{
					"w":     {index: 1},
					"xx.O":  {index: 2, substructCodec: oStructCodec},
					"y.R":   {index: 3, substructCodec: rStructCodec},
					"y.S.P": {index: 3, substructCodec: rStructCodec},
					"y.S.Q": {index: 3, substructCodec: rStructCodec},
					"y.T.O": {index: 3, substructCodec: rStructCodec},
					"y.O":   {index: 3, substructCodec: rStructCodec},
					"z.O":   {index: 4, substructCodec: oStructCodec},
				},
				complete: true,
			},
		},
		{
			"unexported nested and embedded structs",
			struct {
				a int
				B int
				c uStruct
				D uStruct
				uStruct
			}{},
			&structCodec{
				byIndex: []structTag{
					{name: "a"},
					{name: "B"},
					{name: "c."},
					{name: "D."},
					{name: ""},
				},
				byName: map[string]fieldCodec{
					"a":   {index: 0},
					"B":   {index: 1},
					"c.U": {index: 2, substructCodec: uStructCodec},
					"c.v": {index: 2, substructCodec: uStructCodec},
					"D.U": {index: 3, substructCodec: uStructCodec},
					"D.v": {index: 3, substructCodec: uStructCodec},
					"U":   {index: 4, substructCodec: uStructCodec},
					"v":   {index: 4, substructCodec: uStructCodec},
				},
				complete: true,
			},
		},
		{
			"noindex nested struct",
			struct {
				A oStruct `datastore:",noindex"`
			}{},
			&structCodec{
				byIndex: []structTag{
					{name: "A.", noIndex: true},
				},
				byName: map[string]fieldCodec{
					"A.O": {index: 0, substructCodec: oStructCodec},
				},
				complete: true,
			},
		},
		{
			"noindex slice",
			struct {
				A []string `datastore:",noindex"`
			}{},
			&structCodec{
				byIndex: []structTag{
					{name: "A", noIndex: true},
				},
				byName: map[string]fieldCodec{
					"A": {index: 0},
				},
				hasSlice: true,
				complete: true,
			},
		},
		{
			"noindex embedded struct slice",
			struct {
				// vStruct has a single field, V, also with noindex.
				A []vStruct `datastore:",noindex"`
			}{},
			&structCodec{
				byIndex: []structTag{
					{name: "A.", noIndex: true},
				},
				byName: map[string]fieldCodec{
					"A.V": {index: 0, substructCodec: vStructCodec},
				},
				hasSlice: true,
				complete: true,
			},
		},
	}

	for _, tc := range testCases {
		got, err := getStructCodec(reflect.TypeOf(tc.structValue))
		if err != nil {
			t.Errorf("%s: getStructCodec: %v", tc.desc, err)
			continue
		}
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("%s\ngot  %+v\nwant %+v\n", tc.desc, got, tc.want)
			continue
		}
	}
}

func TestRepeatedPropertyName(t *testing.T) {
	good := []interface{}{
		struct {
			A int `datastore:"-"`
		}{},
		struct {
			A int `datastore:"b"`
			B int
		}{},
		struct {
			A int
			B int `datastore:"B"`
		}{},
		struct {
			A int `datastore:"B"`
			B int `datastore:"-"`
		}{},
		struct {
			A int `datastore:"-"`
			B int `datastore:"A"`
		}{},
		struct {
			A int `datastore:"B"`
			B int `datastore:"A"`
		}{},
		struct {
			A int `datastore:"B"`
			B int `datastore:"C"`
			C int `datastore:"A"`
		}{},
		struct {
			A int `datastore:"B"`
			B int `datastore:"C"`
			C int `datastore:"D"`
		}{},
	}
	bad := []interface{}{
		struct {
			A int `datastore:"B"`
			B int
		}{},
		struct {
			A int
			B int `datastore:"A"`
		}{},
		struct {
			A int `datastore:"C"`
			B int `datastore:"C"`
		}{},
		struct {
			A int `datastore:"B"`
			B int `datastore:"C"`
			C int `datastore:"B"`
		}{},
	}
	testGetStructCodec(t, good, bad)
}

func TestFlatteningNestedStructs(t *testing.T) {
	type deepGood struct {
		A struct {
			B []struct {
				C struct {
					D int
				}
			}
		}
	}
	type deepBad struct {
		A struct {
			B []struct {
				C struct {
					D []int
				}
			}
		}
	}
	type iSay struct {
		Tomato int
	}
	type youSay struct {
		Tomato int
	}
	type tweedledee struct {
		Dee int `datastore:"D"`
	}
	type tweedledum struct {
		Dum int `datastore:"D"`
	}

	good := []interface{}{
		struct {
			X []struct {
				Y string
			}
		}{},
		struct {
			X []struct {
				Y []byte
			}
		}{},
		struct {
			P []int
			X struct {
				Y []int
			}
		}{},
		struct {
			X struct {
				Y []int
			}
			Q []int
		}{},
		struct {
			P []int
			X struct {
				Y []int
			}
			Q []int
		}{},
		struct {
			deepGood
		}{},
		struct {
			DG deepGood
		}{},
		struct {
			Foo struct {
				Z int `datastore:"X"`
			} `datastore:"A"`
			Bar struct {
				Z int `datastore:"Y"`
			} `datastore:"A"`
		}{},
	}
	bad := []interface{}{
		struct {
			X []struct {
				Y []string
			}
		}{},
		struct {
			X []struct {
				Y []int
			}
		}{},
		struct {
			deepBad
		}{},
		struct {
			DB deepBad
		}{},
		struct {
			iSay
			youSay
		}{},
		struct {
			tweedledee
			tweedledum
		}{},
		struct {
			Foo struct {
				Z int
			} `datastore:"A"`
			Bar struct {
				Z int
			} `datastore:"A"`
		}{},
	}
	testGetStructCodec(t, good, bad)
}

func testGetStructCodec(t *testing.T, good []interface{}, bad []interface{}) {
	for _, x := range good {
		if _, err := getStructCodec(reflect.TypeOf(x)); err != nil {
			t.Errorf("type %T: got non-nil error (%s), want nil", x, err)
		}
	}
	for _, x := range bad {
		if _, err := getStructCodec(reflect.TypeOf(x)); err == nil {
			t.Errorf("type %T: got nil error, want non-nil", x)
		}
	}
}

func TestNilKeyIsStored(t *testing.T) {
	x := struct {
		K *Key
		I int
	}{}
	p := PropertyList{}
	// Save x as properties.
	p1, _ := SaveStruct(&x)
	p.Load(p1)
	// Set x's fields to non-zero.
	x.K = &Key{}
	x.I = 2
	// Load x from properties.
	p2, _ := p.Save()
	LoadStruct(&x, p2)
	// Check that x's fields were set to zero.
	if x.K != nil {
		t.Errorf("K field was not zero")
	}
	if x.I != 0 {
		t.Errorf("I field was not zero")
	}
}
