// // +build testing

// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

// This file contains values used by tests and benchmarks.
// JSON/BSON do not like maps with keys that are not strings,
// so we only use maps with string keys here.

import (
	"math"
	"time"
)

var testStrucTime = time.Date(2012, 2, 2, 2, 2, 2, 2000, time.UTC).UTC()

type AnonInTestStruc struct {
	AS        string
	AI64      int64
	AI16      int16
	AUi64     uint64
	ASslice   []string
	AI64slice []int64
	AF64slice []float64
	// AMI32U32  map[int32]uint32
	// AMU32F64 map[uint32]float64 // json/bson do not like it
	AMSU16 map[string]uint16
}

type AnonInTestStrucIntf struct {
	Islice []interface{}
	Ms     map[string]interface{}
	Nintf  interface{} //don't set this, so we can test for nil
	T      time.Time
}

type TestStruc struct {
	_struct struct{} `codec:",omitempty"` //set omitempty for every field

	S    string
	I64  int64
	I16  int16
	Ui64 uint64
	Ui8  uint8
	B    bool
	By   uint8 // byte: msgp doesn't like byte

	Sslice    []string
	I64slice  []int64
	I16slice  []int16
	Ui64slice []uint64
	Ui8slice  []uint8
	Bslice    []bool
	Byslice   []byte

	Iptrslice []*int64

	AnonInTestStruc

	//M map[interface{}]interface{}  `json:"-",bson:"-"`
	Msi64 map[string]int64

	// make this a ptr, so that it could be set or not.
	// for comparison (e.g. with msgp), give it a struct tag (so it is not inlined),
	// make this one omitempty (so it is included if nil).
	*AnonInTestStrucIntf `codec:",omitempty"`

	Nmap       map[string]bool //don't set this, so we can test for nil
	Nslice     []byte          //don't set this, so we can test for nil
	Nint64     *int64          //don't set this, so we can test for nil
	Mtsptr     map[string]*TestStruc
	Mts        map[string]TestStruc
	Its        []*TestStruc
	Nteststruc *TestStruc
}

// small struct for testing that codecgen works for unexported types
type tLowerFirstLetter struct {
	I int
	u uint64
	S string
	b []byte
}

func newTestStruc(depth int, bench bool, useInterface, useStringKeyOnly bool) (ts *TestStruc) {
	var i64a, i64b, i64c, i64d int64 = 64, 6464, 646464, 64646464

	ts = &TestStruc{
		S:    "some string",
		I64:  math.MaxInt64 * 2 / 3, // 64,
		I16:  1616,
		Ui64: uint64(int64(math.MaxInt64 * 2 / 3)), // 64, //don't use MaxUint64, as bson can't write it
		Ui8:  160,
		B:    true,
		By:   5,

		Sslice:    []string{"one", "two", "three"},
		I64slice:  []int64{1111, 2222, 3333},
		I16slice:  []int16{44, 55, 66},
		Ui64slice: []uint64{12121212, 34343434, 56565656},
		Ui8slice:  []uint8{210, 211, 212},
		Bslice:    []bool{true, false, true, false},
		Byslice:   []byte{13, 14, 15},

		Msi64: map[string]int64{
			"one": 1,
			"two": 2,
		},
		AnonInTestStruc: AnonInTestStruc{
			// There's more leeway in altering this.
			AS:    "A-String",
			AI64:  -64646464,
			AI16:  1616,
			AUi64: 64646464,
			// (U+1D11E)G-clef character may be represented in json as "\uD834\uDD1E".
			// single reverse solidus character may be represented in json as "\u005C".
			// include these in ASslice below.
			ASslice: []string{"Aone", "Atwo", "Athree",
				"Afour.reverse_solidus.\u005c", "Afive.Gclef.\U0001d11E"},
			AI64slice: []int64{1, -22, 333, -4444, 55555, -666666},
			AMSU16:    map[string]uint16{"1": 1, "22": 2, "333": 3, "4444": 4},
			AF64slice: []float64{11.11e-11, 22.22E+22, 33.33E-33, 44.44e+44, 555.55E-6, 666.66E6},
		},
	}
	if useInterface {
		ts.AnonInTestStrucIntf = &AnonInTestStrucIntf{
			Islice: []interface{}{"true", true, "no", false, uint64(288), float64(0.4)},
			Ms: map[string]interface{}{
				"true":     "true",
				"int64(9)": false,
			},
			T: testStrucTime,
		}
	}

	//For benchmarks, some things will not work.
	if !bench {
		//json and bson require string keys in maps
		//ts.M = map[interface{}]interface{}{
		//	true: "true",
		//	int8(9): false,
		//}
		//gob cannot encode nil in element in array (encodeArray: nil element)
		ts.Iptrslice = []*int64{nil, &i64a, nil, &i64b, nil, &i64c, nil, &i64d, nil}
		// ts.Iptrslice = nil
	}
	if !useStringKeyOnly {
		// ts.AnonInTestStruc.AMU32F64 = map[uint32]float64{1: 1, 2: 2, 3: 3} // Json/Bson barf
	}
	if depth > 0 {
		depth--
		if ts.Mtsptr == nil {
			ts.Mtsptr = make(map[string]*TestStruc)
		}
		if ts.Mts == nil {
			ts.Mts = make(map[string]TestStruc)
		}
		ts.Mtsptr["0"] = newTestStruc(depth, bench, useInterface, useStringKeyOnly)
		ts.Mts["0"] = *(ts.Mtsptr["0"])
		ts.Its = append(ts.Its, ts.Mtsptr["0"])
	}
	return
}

// Some other types

type Sstring string
type Bbool bool
type Sstructsmall struct {
	A int
}

type Sstructbig struct {
	A int
	B bool
	c string
	// Sval Sstruct
	Ssmallptr *Sstructsmall
	Ssmall    *Sstructsmall
	Sptr      *Sstructbig
}

type SstructbigMapBySlice struct {
	_struct struct{} `codec:",toarray"`
	A       int
	B       bool
	c       string
	// Sval Sstruct
	Ssmallptr *Sstructsmall
	Ssmall    *Sstructsmall
	Sptr      *Sstructbig
}

type Sinterface interface {
	Noop()
}
