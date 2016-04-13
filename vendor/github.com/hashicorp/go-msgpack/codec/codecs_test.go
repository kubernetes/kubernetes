// Copyright (c) 2012, 2013 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

// Test works by using a slice of interfaces.
// It can test for encoding/decoding into/from a nil interface{}
// or passing the object to encode/decode into.
//
// There are basically 2 main tests here.
// First test internally encodes and decodes things and verifies that
// the artifact was as expected.
// Second test will use python msgpack to create a bunch of golden files,
// read those files, and compare them to what it should be. It then
// writes those files back out and compares the byte streams.
//
// Taken together, the tests are pretty extensive.

import (
	"bytes"
	"encoding/gob"
	"flag"
	"fmt"
	"io/ioutil"
	"math"
	"net"
	"net/rpc"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"runtime"
	"strconv"
	"sync/atomic"
	"testing"
	"time"
)

type testVerifyArg int

const (
	testVerifyMapTypeSame testVerifyArg = iota
	testVerifyMapTypeStrIntf
	testVerifyMapTypeIntfIntf
	// testVerifySliceIntf
	testVerifyForPython
)

var (
	testInitDebug      bool
	testUseIoEncDec    bool
	testStructToArray  bool
	testWriteNoSymbols bool

	_                         = fmt.Printf
	skipVerifyVal interface{} = &(struct{}{})

	// For Go Time, do not use a descriptive timezone.
	// It's unnecessary, and makes it harder to do a reflect.DeepEqual.
	// The Offset already tells what the offset should be, if not on UTC and unknown zone name.
	timeLoc        = time.FixedZone("", -8*60*60) // UTC-08:00 //time.UTC-8
	timeToCompare1 = time.Date(2012, 2, 2, 2, 2, 2, 2000, timeLoc)
	timeToCompare2 = time.Date(1900, 2, 2, 2, 2, 2, 2000, timeLoc)
	timeToCompare3 = time.Unix(0, 0).UTC()
	timeToCompare4 = time.Time{}.UTC()

	table              []interface{} // main items we encode
	tableVerify        []interface{} // we verify encoded things against this after decode
	tableTestNilVerify []interface{} // for nil interface, use this to verify (rules are different)
	tablePythonVerify  []interface{} // for verifying for python, since Python sometimes
	// will encode a float32 as float64, or large int as uint
	testRpcInt   = new(TestRpcInt)
	testMsgpackH = &MsgpackHandle{}
	testBincH    = &BincHandle{}
	testSimpleH  = &SimpleHandle{}
)

func testInitFlags() {
	// delete(testDecOpts.ExtFuncs, timeTyp)
	flag.BoolVar(&testInitDebug, "tg", false, "Test Debug")
	flag.BoolVar(&testUseIoEncDec, "ti", false, "Use IO Reader/Writer for Marshal/Unmarshal")
	flag.BoolVar(&testStructToArray, "ts", false, "Set StructToArray option")
	flag.BoolVar(&testWriteNoSymbols, "tn", false, "Set NoSymbols option")
}

type AnonInTestStruc struct {
	AS        string
	AI64      int64
	AI16      int16
	AUi64     uint64
	ASslice   []string
	AI64slice []int64
}

type TestStruc struct {
	S    string
	I64  int64
	I16  int16
	Ui64 uint64
	Ui8  uint8
	B    bool
	By   byte

	Sslice    []string
	I64slice  []int64
	I16slice  []int16
	Ui64slice []uint64
	Ui8slice  []uint8
	Bslice    []bool
	Byslice   []byte

	Islice    []interface{}
	Iptrslice []*int64

	AnonInTestStruc

	//M map[interface{}]interface{}  `json:"-",bson:"-"`
	Ms    map[string]interface{}
	Msi64 map[string]int64

	Nintf      interface{} //don't set this, so we can test for nil
	T          time.Time
	Nmap       map[string]bool //don't set this, so we can test for nil
	Nslice     []byte          //don't set this, so we can test for nil
	Nint64     *int64          //don't set this, so we can test for nil
	Mtsptr     map[string]*TestStruc
	Mts        map[string]TestStruc
	Its        []*TestStruc
	Nteststruc *TestStruc
}

type TestABC struct {
	A, B, C string
}

type TestRpcInt struct {
	i int
}

func (r *TestRpcInt) Update(n int, res *int) error      { r.i = n; *res = r.i; return nil }
func (r *TestRpcInt) Square(ignore int, res *int) error { *res = r.i * r.i; return nil }
func (r *TestRpcInt) Mult(n int, res *int) error        { *res = r.i * n; return nil }
func (r *TestRpcInt) EchoStruct(arg TestABC, res *string) error {
	*res = fmt.Sprintf("%#v", arg)
	return nil
}
func (r *TestRpcInt) Echo123(args []string, res *string) error {
	*res = fmt.Sprintf("%#v", args)
	return nil
}

func testVerifyVal(v interface{}, arg testVerifyArg) (v2 interface{}) {
	//for python msgpack,
	//  - all positive integers are unsigned 64-bit ints
	//  - all floats are float64
	switch iv := v.(type) {
	case int8:
		if iv > 0 {
			v2 = uint64(iv)
		} else {
			v2 = int64(iv)
		}
	case int16:
		if iv > 0 {
			v2 = uint64(iv)
		} else {
			v2 = int64(iv)
		}
	case int32:
		if iv > 0 {
			v2 = uint64(iv)
		} else {
			v2 = int64(iv)
		}
	case int64:
		if iv > 0 {
			v2 = uint64(iv)
		} else {
			v2 = int64(iv)
		}
	case uint8:
		v2 = uint64(iv)
	case uint16:
		v2 = uint64(iv)
	case uint32:
		v2 = uint64(iv)
	case uint64:
		v2 = uint64(iv)
	case float32:
		v2 = float64(iv)
	case float64:
		v2 = float64(iv)
	case []interface{}:
		m2 := make([]interface{}, len(iv))
		for j, vj := range iv {
			m2[j] = testVerifyVal(vj, arg)
		}
		v2 = m2
	case map[string]bool:
		switch arg {
		case testVerifyMapTypeSame:
			m2 := make(map[string]bool)
			for kj, kv := range iv {
				m2[kj] = kv
			}
			v2 = m2
		case testVerifyMapTypeStrIntf, testVerifyForPython:
			m2 := make(map[string]interface{})
			for kj, kv := range iv {
				m2[kj] = kv
			}
			v2 = m2
		case testVerifyMapTypeIntfIntf:
			m2 := make(map[interface{}]interface{})
			for kj, kv := range iv {
				m2[kj] = kv
			}
			v2 = m2
		}
	case map[string]interface{}:
		switch arg {
		case testVerifyMapTypeSame:
			m2 := make(map[string]interface{})
			for kj, kv := range iv {
				m2[kj] = testVerifyVal(kv, arg)
			}
			v2 = m2
		case testVerifyMapTypeStrIntf, testVerifyForPython:
			m2 := make(map[string]interface{})
			for kj, kv := range iv {
				m2[kj] = testVerifyVal(kv, arg)
			}
			v2 = m2
		case testVerifyMapTypeIntfIntf:
			m2 := make(map[interface{}]interface{})
			for kj, kv := range iv {
				m2[kj] = testVerifyVal(kv, arg)
			}
			v2 = m2
		}
	case map[interface{}]interface{}:
		m2 := make(map[interface{}]interface{})
		for kj, kv := range iv {
			m2[testVerifyVal(kj, arg)] = testVerifyVal(kv, arg)
		}
		v2 = m2
	case time.Time:
		switch arg {
		case testVerifyForPython:
			if iv2 := iv.UnixNano(); iv2 > 0 {
				v2 = uint64(iv2)
			} else {
				v2 = int64(iv2)
			}
		default:
			v2 = v
		}
	default:
		v2 = v
	}
	return
}

func testInit() {
	gob.Register(new(TestStruc))
	if testInitDebug {
		ts0 := newTestStruc(2, false)
		fmt.Printf("====> depth: %v, ts: %#v\n", 2, ts0)
	}

	testBincH.StructToArray = testStructToArray
	if testWriteNoSymbols {
		testBincH.AsSymbols = AsSymbolNone
	} else {
		testBincH.AsSymbols = AsSymbolAll
	}
	testMsgpackH.StructToArray = testStructToArray
	testMsgpackH.RawToString = true
	// testMsgpackH.AddExt(byteSliceTyp, 0, testMsgpackH.BinaryEncodeExt, testMsgpackH.BinaryDecodeExt)
	// testMsgpackH.AddExt(timeTyp, 1, testMsgpackH.TimeEncodeExt, testMsgpackH.TimeDecodeExt)
	timeEncExt := func(rv reflect.Value) ([]byte, error) {
		return encodeTime(rv.Interface().(time.Time)), nil
	}
	timeDecExt := func(rv reflect.Value, bs []byte) error {
		tt, err := decodeTime(bs)
		if err == nil {
			rv.Set(reflect.ValueOf(tt))
		}
		return err
	}

	// add extensions for msgpack, simple for time.Time, so we can encode/decode same way.
	testMsgpackH.AddExt(timeTyp, 1, timeEncExt, timeDecExt)
	testSimpleH.AddExt(timeTyp, 1, timeEncExt, timeDecExt)

	primitives := []interface{}{
		int8(-8),
		int16(-1616),
		int32(-32323232),
		int64(-6464646464646464),
		uint8(192),
		uint16(1616),
		uint32(32323232),
		uint64(6464646464646464),
		byte(192),
		float32(-3232.0),
		float64(-6464646464.0),
		float32(3232.0),
		float64(6464646464.0),
		false,
		true,
		nil,
		"someday",
		"",
		"bytestring",
		timeToCompare1,
		timeToCompare2,
		timeToCompare3,
		timeToCompare4,
	}
	mapsAndStrucs := []interface{}{
		map[string]bool{
			"true":  true,
			"false": false,
		},
		map[string]interface{}{
			"true":         "True",
			"false":        false,
			"uint16(1616)": uint16(1616),
		},
		//add a complex combo map in here. (map has list which has map)
		//note that after the first thing, everything else should be generic.
		map[string]interface{}{
			"list": []interface{}{
				int16(1616),
				int32(32323232),
				true,
				float32(-3232.0),
				map[string]interface{}{
					"TRUE":  true,
					"FALSE": false,
				},
				[]interface{}{true, false},
			},
			"int32":        int32(32323232),
			"bool":         true,
			"LONG STRING":  "123456789012345678901234567890123456789012345678901234567890",
			"SHORT STRING": "1234567890",
		},
		map[interface{}]interface{}{
			true:       "true",
			uint8(138): false,
			"false":    uint8(200),
		},
		newTestStruc(0, false),
	}

	table = []interface{}{}
	table = append(table, primitives...)    //0-19 are primitives
	table = append(table, primitives)       //20 is a list of primitives
	table = append(table, mapsAndStrucs...) //21-24 are maps. 25 is a *struct

	tableVerify = make([]interface{}, len(table))
	tableTestNilVerify = make([]interface{}, len(table))
	tablePythonVerify = make([]interface{}, len(table))

	lp := len(primitives)
	av := tableVerify
	for i, v := range table {
		if i == lp+3 {
			av[i] = skipVerifyVal
			continue
		}
		//av[i] = testVerifyVal(v, testVerifyMapTypeSame)
		switch v.(type) {
		case []interface{}:
			av[i] = testVerifyVal(v, testVerifyMapTypeSame)
		case map[string]interface{}:
			av[i] = testVerifyVal(v, testVerifyMapTypeSame)
		case map[interface{}]interface{}:
			av[i] = testVerifyVal(v, testVerifyMapTypeSame)
		default:
			av[i] = v
		}
	}

	av = tableTestNilVerify
	for i, v := range table {
		if i > lp+3 {
			av[i] = skipVerifyVal
			continue
		}
		av[i] = testVerifyVal(v, testVerifyMapTypeStrIntf)
	}

	av = tablePythonVerify
	for i, v := range table {
		if i > lp+3 {
			av[i] = skipVerifyVal
			continue
		}
		av[i] = testVerifyVal(v, testVerifyForPython)
	}

	tablePythonVerify = tablePythonVerify[:24]
}

func testUnmarshal(v interface{}, data []byte, h Handle) error {
	if testUseIoEncDec {
		return NewDecoder(bytes.NewBuffer(data), h).Decode(v)
	}
	return NewDecoderBytes(data, h).Decode(v)
}

func testMarshal(v interface{}, h Handle) (bs []byte, err error) {
	if testUseIoEncDec {
		var buf bytes.Buffer
		err = NewEncoder(&buf, h).Encode(v)
		bs = buf.Bytes()
		return
	}
	err = NewEncoderBytes(&bs, h).Encode(v)
	return
}

func testMarshalErr(v interface{}, h Handle, t *testing.T, name string) (bs []byte, err error) {
	if bs, err = testMarshal(v, h); err != nil {
		logT(t, "Error encoding %s: %v, Err: %v", name, v, err)
		t.FailNow()
	}
	return
}

func testUnmarshalErr(v interface{}, data []byte, h Handle, t *testing.T, name string) (err error) {
	if err = testUnmarshal(v, data, h); err != nil {
		logT(t, "Error Decoding into %s: %v, Err: %v", name, v, err)
		t.FailNow()
	}
	return
}

func newTestStruc(depth int, bench bool) (ts *TestStruc) {
	var i64a, i64b, i64c, i64d int64 = 64, 6464, 646464, 64646464

	ts = &TestStruc{
		S:    "some string",
		I64:  math.MaxInt64 * 2 / 3, // 64,
		I16:  16,
		Ui64: uint64(int64(math.MaxInt64 * 2 / 3)), // 64, //don't use MaxUint64, as bson can't write it
		Ui8:  160,
		B:    true,
		By:   5,

		Sslice:    []string{"one", "two", "three"},
		I64slice:  []int64{1, 2, 3},
		I16slice:  []int16{4, 5, 6},
		Ui64slice: []uint64{137, 138, 139},
		Ui8slice:  []uint8{210, 211, 212},
		Bslice:    []bool{true, false, true, false},
		Byslice:   []byte{13, 14, 15},

		Islice: []interface{}{"true", true, "no", false, uint64(288), float64(0.4)},

		Ms: map[string]interface{}{
			"true":     "true",
			"int64(9)": false,
		},
		Msi64: map[string]int64{
			"one": 1,
			"two": 2,
		},
		T: timeToCompare1,
		AnonInTestStruc: AnonInTestStruc{
			AS:        "A-String",
			AI64:      64,
			AI16:      16,
			AUi64:     64,
			ASslice:   []string{"Aone", "Atwo", "Athree"},
			AI64slice: []int64{1, 2, 3},
		},
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
	if depth > 0 {
		depth--
		if ts.Mtsptr == nil {
			ts.Mtsptr = make(map[string]*TestStruc)
		}
		if ts.Mts == nil {
			ts.Mts = make(map[string]TestStruc)
		}
		ts.Mtsptr["0"] = newTestStruc(depth, bench)
		ts.Mts["0"] = *(ts.Mtsptr["0"])
		ts.Its = append(ts.Its, ts.Mtsptr["0"])
	}
	return
}

// doTestCodecTableOne allows us test for different variations based on arguments passed.
func doTestCodecTableOne(t *testing.T, testNil bool, h Handle,
	vs []interface{}, vsVerify []interface{}) {
	//if testNil, then just test for when a pointer to a nil interface{} is passed. It should work.
	//Current setup allows us test (at least manually) the nil interface or typed interface.
	logT(t, "================ TestNil: %v ================\n", testNil)
	for i, v0 := range vs {
		logT(t, "..............................................")
		logT(t, "         Testing: #%d:, %T, %#v\n", i, v0, v0)
		b0, err := testMarshalErr(v0, h, t, "v0")
		if err != nil {
			continue
		}
		logT(t, "         Encoded bytes: len: %v, %v\n", len(b0), b0)

		var v1 interface{}

		if testNil {
			err = testUnmarshal(&v1, b0, h)
		} else {
			if v0 != nil {
				v0rt := reflect.TypeOf(v0) // ptr
				rv1 := reflect.New(v0rt)
				err = testUnmarshal(rv1.Interface(), b0, h)
				v1 = rv1.Elem().Interface()
				// v1 = reflect.Indirect(reflect.ValueOf(v1)).Interface()
			}
		}

		logT(t, "         v1 returned: %T, %#v", v1, v1)
		// if v1 != nil {
		//	logT(t, "         v1 returned: %T, %#v", v1, v1)
		//	//we always indirect, because ptr to typed value may be passed (if not testNil)
		//	v1 = reflect.Indirect(reflect.ValueOf(v1)).Interface()
		// }
		if err != nil {
			logT(t, "-------- Error: %v. Partial return: %v", err, v1)
			failT(t)
			continue
		}
		v0check := vsVerify[i]
		if v0check == skipVerifyVal {
			logT(t, "        Nil Check skipped: Decoded: %T, %#v\n", v1, v1)
			continue
		}

		if err = deepEqual(v0check, v1); err == nil {
			logT(t, "++++++++ Before and After marshal matched\n")
		} else {
			logT(t, "-------- Before and After marshal do not match: Error: %v"+
				" ====> GOLDEN: (%T) %#v, DECODED: (%T) %#v\n", err, v0check, v0check, v1, v1)
			failT(t)
		}
	}
}

func testCodecTableOne(t *testing.T, h Handle) {
	// func TestMsgpackAllExperimental(t *testing.T) {
	// dopts := testDecOpts(nil, nil, false, true, true),

	switch v := h.(type) {
	case *MsgpackHandle:
		var oldWriteExt, oldRawToString bool
		oldWriteExt, v.WriteExt = v.WriteExt, true
		oldRawToString, v.RawToString = v.RawToString, true
		doTestCodecTableOne(t, false, h, table, tableVerify)
		v.WriteExt, v.RawToString = oldWriteExt, oldRawToString
	default:
		doTestCodecTableOne(t, false, h, table, tableVerify)
	}
	// func TestMsgpackAll(t *testing.T) {
	idxTime, numPrim, numMap := 19, 23, 4

	//skip []interface{} containing time.Time
	doTestCodecTableOne(t, false, h, table[:numPrim], tableVerify[:numPrim])
	doTestCodecTableOne(t, false, h, table[numPrim+1:], tableVerify[numPrim+1:])
	// func TestMsgpackNilStringMap(t *testing.T) {
	var oldMapType reflect.Type
	v := h.getBasicHandle()
	oldMapType, v.MapType = v.MapType, mapStrIntfTyp

	//skip time.Time, []interface{} containing time.Time, last map, and newStruc
	doTestCodecTableOne(t, true, h, table[:idxTime], tableTestNilVerify[:idxTime])
	doTestCodecTableOne(t, true, h, table[numPrim+1:numPrim+numMap], tableTestNilVerify[numPrim+1:numPrim+numMap])

	v.MapType = oldMapType

	// func TestMsgpackNilIntf(t *testing.T) {

	//do newTestStruc and last element of map
	doTestCodecTableOne(t, true, h, table[numPrim+numMap:], tableTestNilVerify[numPrim+numMap:])
	//TODO? What is this one?
	//doTestCodecTableOne(t, true, h, table[17:18], tableTestNilVerify[17:18])
}

func testCodecMiscOne(t *testing.T, h Handle) {
	b, err := testMarshalErr(32, h, t, "32")
	// Cannot do this nil one, because faster type assertion decoding will panic
	// var i *int32
	// if err = testUnmarshal(b, i, nil); err == nil {
	// 	logT(t, "------- Expecting error because we cannot unmarshal to int32 nil ptr")
	// 	t.FailNow()
	// }
	var i2 int32 = 0
	err = testUnmarshalErr(&i2, b, h, t, "int32-ptr")
	if i2 != int32(32) {
		logT(t, "------- didn't unmarshal to 32: Received: %d", i2)
		t.FailNow()
	}

	// func TestMsgpackDecodePtr(t *testing.T) {
	ts := newTestStruc(0, false)
	b, err = testMarshalErr(ts, h, t, "pointer-to-struct")
	if len(b) < 40 {
		logT(t, "------- Size must be > 40. Size: %d", len(b))
		t.FailNow()
	}
	logT(t, "------- b: %v", b)
	ts2 := new(TestStruc)
	err = testUnmarshalErr(ts2, b, h, t, "pointer-to-struct")
	if ts2.I64 != math.MaxInt64*2/3 {
		logT(t, "------- Unmarshal wrong. Expect I64 = 64. Got: %v", ts2.I64)
		t.FailNow()
	}

	// func TestMsgpackIntfDecode(t *testing.T) {
	m := map[string]int{"A": 2, "B": 3}
	p := []interface{}{m}
	bs, err := testMarshalErr(p, h, t, "p")

	m2 := map[string]int{}
	p2 := []interface{}{m2}
	err = testUnmarshalErr(&p2, bs, h, t, "&p2")

	if m2["A"] != 2 || m2["B"] != 3 {
		logT(t, "m2 not as expected: expecting: %v, got: %v", m, m2)
		t.FailNow()
	}
	// log("m: %v, m2: %v, p: %v, p2: %v", m, m2, p, p2)
	checkEqualT(t, p, p2, "p=p2")
	checkEqualT(t, m, m2, "m=m2")
	if err = deepEqual(p, p2); err == nil {
		logT(t, "p and p2 match")
	} else {
		logT(t, "Not Equal: %v. p: %v, p2: %v", err, p, p2)
		t.FailNow()
	}
	if err = deepEqual(m, m2); err == nil {
		logT(t, "m and m2 match")
	} else {
		logT(t, "Not Equal: %v. m: %v, m2: %v", err, m, m2)
		t.FailNow()
	}

	// func TestMsgpackDecodeStructSubset(t *testing.T) {
	// test that we can decode a subset of the stream
	mm := map[string]interface{}{"A": 5, "B": 99, "C": 333}
	bs, err = testMarshalErr(mm, h, t, "mm")
	type ttt struct {
		A uint8
		C int32
	}
	var t2 ttt
	testUnmarshalErr(&t2, bs, h, t, "t2")
	t3 := ttt{5, 333}
	checkEqualT(t, t2, t3, "t2=t3")

	// println(">>>>>")
	// test simple arrays, non-addressable arrays, slices
	type tarr struct {
		A int64
		B [3]int64
		C []byte
		D [3]byte
	}
	var tarr0 = tarr{1, [3]int64{2, 3, 4}, []byte{4, 5, 6}, [3]byte{7, 8, 9}}
	// test both pointer and non-pointer (value)
	for _, tarr1 := range []interface{}{tarr0, &tarr0} {
		bs, err = testMarshalErr(tarr1, h, t, "tarr1")
		var tarr2 tarr
		testUnmarshalErr(&tarr2, bs, h, t, "tarr2")
		checkEqualT(t, tarr0, tarr2, "tarr0=tarr2")
		// fmt.Printf(">>>> err: %v. tarr1: %v, tarr2: %v\n", err, tarr0, tarr2)
	}

	// test byte array, even if empty (msgpack only)
	if h == testMsgpackH {
		type ystruct struct {
			Anarray []byte
		}
		var ya = ystruct{}
		testUnmarshalErr(&ya, []byte{0x91, 0x90}, h, t, "ya")
	}
}

func testCodecEmbeddedPointer(t *testing.T, h Handle) {
	type Z int
	type A struct {
		AnInt int
	}
	type B struct {
		*Z
		*A
		MoreInt int
	}
	var z Z = 4
	x1 := &B{&z, &A{5}, 6}
	bs, err := testMarshalErr(x1, h, t, "x1")
	// fmt.Printf("buf: len(%v): %x\n", buf.Len(), buf.Bytes())
	var x2 = new(B)
	err = testUnmarshalErr(x2, bs, h, t, "x2")
	err = checkEqualT(t, x1, x2, "x1=x2")
	_ = err
}

func doTestRpcOne(t *testing.T, rr Rpc, h Handle, doRequest bool, exitSleepMs time.Duration,
) (port int) {
	// rpc needs EOF, which is sent via a panic, and so must be recovered.
	if !recoverPanicToErr {
		logT(t, "EXPECTED. set recoverPanicToErr=true, since rpc needs EOF")
		t.FailNow()
	}
	srv := rpc.NewServer()
	srv.Register(testRpcInt)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	// log("listener: %v", ln.Addr())
	checkErrT(t, err)
	port = (ln.Addr().(*net.TCPAddr)).Port
	// var opts *DecoderOptions
	// opts := testDecOpts
	// opts.MapType = mapStrIntfTyp
	// opts.RawToString = false
	serverExitChan := make(chan bool, 1)
	var serverExitFlag uint64 = 0
	serverFn := func() {
		for {
			conn1, err1 := ln.Accept()
			// if err1 != nil {
			// 	//fmt.Printf("accept err1: %v\n", err1)
			// 	continue
			// }
			if atomic.LoadUint64(&serverExitFlag) == 1 {
				serverExitChan <- true
				conn1.Close()
				return // exit serverFn goroutine
			}
			if err1 == nil {
				var sc rpc.ServerCodec = rr.ServerCodec(conn1, h)
				srv.ServeCodec(sc)
			}
		}
	}

	clientFn := func(cc rpc.ClientCodec) {
		cl := rpc.NewClientWithCodec(cc)
		defer cl.Close()
		var up, sq, mult int
		var rstr string
		// log("Calling client")
		checkErrT(t, cl.Call("TestRpcInt.Update", 5, &up))
		// log("Called TestRpcInt.Update")
		checkEqualT(t, testRpcInt.i, 5, "testRpcInt.i=5")
		checkEqualT(t, up, 5, "up=5")
		checkErrT(t, cl.Call("TestRpcInt.Square", 1, &sq))
		checkEqualT(t, sq, 25, "sq=25")
		checkErrT(t, cl.Call("TestRpcInt.Mult", 20, &mult))
		checkEqualT(t, mult, 100, "mult=100")
		checkErrT(t, cl.Call("TestRpcInt.EchoStruct", TestABC{"Aa", "Bb", "Cc"}, &rstr))
		checkEqualT(t, rstr, fmt.Sprintf("%#v", TestABC{"Aa", "Bb", "Cc"}), "rstr=")
		checkErrT(t, cl.Call("TestRpcInt.Echo123", []string{"A1", "B2", "C3"}, &rstr))
		checkEqualT(t, rstr, fmt.Sprintf("%#v", []string{"A1", "B2", "C3"}), "rstr=")
	}

	connFn := func() (bs net.Conn) {
		// log("calling f1")
		bs, err2 := net.Dial(ln.Addr().Network(), ln.Addr().String())
		//fmt.Printf("f1. bs: %v, err2: %v\n", bs, err2)
		checkErrT(t, err2)
		return
	}

	exitFn := func() {
		atomic.StoreUint64(&serverExitFlag, 1)
		bs := connFn()
		<-serverExitChan
		bs.Close()
		// serverExitChan <- true
	}

	go serverFn()
	runtime.Gosched()
	//time.Sleep(100 * time.Millisecond)
	if exitSleepMs == 0 {
		defer ln.Close()
		defer exitFn()
	}
	if doRequest {
		bs := connFn()
		cc := rr.ClientCodec(bs, h)
		clientFn(cc)
	}
	if exitSleepMs != 0 {
		go func() {
			defer ln.Close()
			time.Sleep(exitSleepMs)
			exitFn()
		}()
	}
	return
}

// Comprehensive testing that generates data encoded from python msgpack,
// and validates that our code can read and write it out accordingly.
// We keep this unexported here, and put actual test in ext_dep_test.go.
// This way, it can be excluded by excluding file completely.
func doTestMsgpackPythonGenStreams(t *testing.T) {
	logT(t, "TestPythonGenStreams")
	tmpdir, err := ioutil.TempDir("", "golang-msgpack-test")
	if err != nil {
		logT(t, "-------- Unable to create temp directory\n")
		t.FailNow()
	}
	defer os.RemoveAll(tmpdir)
	logT(t, "tmpdir: %v", tmpdir)
	cmd := exec.Command("python", "msgpack_test.py", "testdata", tmpdir)
	//cmd.Stdin = strings.NewReader("some input")
	//cmd.Stdout = &out
	var cmdout []byte
	if cmdout, err = cmd.CombinedOutput(); err != nil {
		logT(t, "-------- Error running msgpack_test.py testdata. Err: %v", err)
		logT(t, "         %v", string(cmdout))
		t.FailNow()
	}

	oldMapType := testMsgpackH.MapType
	for i, v := range tablePythonVerify {
		testMsgpackH.MapType = oldMapType
		//load up the golden file based on number
		//decode it
		//compare to in-mem object
		//encode it again
		//compare to output stream
		logT(t, "..............................................")
		logT(t, "         Testing: #%d: %T, %#v\n", i, v, v)
		var bss []byte
		bss, err = ioutil.ReadFile(filepath.Join(tmpdir, strconv.Itoa(i)+".golden"))
		if err != nil {
			logT(t, "-------- Error reading golden file: %d. Err: %v", i, err)
			failT(t)
			continue
		}
		testMsgpackH.MapType = mapStrIntfTyp

		var v1 interface{}
		if err = testUnmarshal(&v1, bss, testMsgpackH); err != nil {
			logT(t, "-------- Error decoding stream: %d: Err: %v", i, err)
			failT(t)
			continue
		}
		if v == skipVerifyVal {
			continue
		}
		//no need to indirect, because we pass a nil ptr, so we already have the value
		//if v1 != nil { v1 = reflect.Indirect(reflect.ValueOf(v1)).Interface() }
		if err = deepEqual(v, v1); err == nil {
			logT(t, "++++++++ Objects match")
		} else {
			logT(t, "-------- Objects do not match: %v. Source: %T. Decoded: %T", err, v, v1)
			logT(t, "--------   AGAINST: %#v", v)
			logT(t, "--------   DECODED: %#v <====> %#v", v1, reflect.Indirect(reflect.ValueOf(v1)).Interface())
			failT(t)
		}
		bsb, err := testMarshal(v1, testMsgpackH)
		if err != nil {
			logT(t, "Error encoding to stream: %d: Err: %v", i, err)
			failT(t)
			continue
		}
		if err = deepEqual(bsb, bss); err == nil {
			logT(t, "++++++++ Bytes match")
		} else {
			logT(t, "???????? Bytes do not match. %v.", err)
			xs := "--------"
			if reflect.ValueOf(v).Kind() == reflect.Map {
				xs = "        "
				logT(t, "%s It's a map. Ok that they don't match (dependent on ordering).", xs)
			} else {
				logT(t, "%s It's not a map. They should match.", xs)
				failT(t)
			}
			logT(t, "%s   FROM_FILE: %4d] %v", xs, len(bss), bss)
			logT(t, "%s     ENCODED: %4d] %v", xs, len(bsb), bsb)
		}
	}
	testMsgpackH.MapType = oldMapType
}

// To test MsgpackSpecRpc, we test 3 scenarios:
//    - Go Client to Go RPC Service (contained within TestMsgpackRpcSpec)
//    - Go client to Python RPC Service (contained within doTestMsgpackRpcSpecGoClientToPythonSvc)
//    - Python Client to Go RPC Service (contained within doTestMsgpackRpcSpecPythonClientToGoSvc)
//
// This allows us test the different calling conventions
//    - Go Service requires only one argument
//    - Python Service allows multiple arguments

func doTestMsgpackRpcSpecGoClientToPythonSvc(t *testing.T) {
	openPort := "6789"
	cmd := exec.Command("python", "msgpack_test.py", "rpc-server", openPort, "2")
	checkErrT(t, cmd.Start())
	time.Sleep(100 * time.Millisecond) // time for python rpc server to start
	bs, err2 := net.Dial("tcp", ":"+openPort)
	checkErrT(t, err2)
	cc := MsgpackSpecRpc.ClientCodec(bs, testMsgpackH)
	cl := rpc.NewClientWithCodec(cc)
	defer cl.Close()
	var rstr string
	checkErrT(t, cl.Call("EchoStruct", TestABC{"Aa", "Bb", "Cc"}, &rstr))
	//checkEqualT(t, rstr, "{'A': 'Aa', 'B': 'Bb', 'C': 'Cc'}")
	var mArgs MsgpackSpecRpcMultiArgs = []interface{}{"A1", "B2", "C3"}
	checkErrT(t, cl.Call("Echo123", mArgs, &rstr))
	checkEqualT(t, rstr, "1:A1 2:B2 3:C3", "rstr=")
}

func doTestMsgpackRpcSpecPythonClientToGoSvc(t *testing.T) {
	port := doTestRpcOne(t, MsgpackSpecRpc, testMsgpackH, false, 1*time.Second)
	//time.Sleep(1000 * time.Millisecond)
	cmd := exec.Command("python", "msgpack_test.py", "rpc-client-go-service", strconv.Itoa(port))
	var cmdout []byte
	var err error
	if cmdout, err = cmd.CombinedOutput(); err != nil {
		logT(t, "-------- Error running msgpack_test.py rpc-client-go-service. Err: %v", err)
		logT(t, "         %v", string(cmdout))
		t.FailNow()
	}
	checkEqualT(t, string(cmdout),
		fmt.Sprintf("%#v\n%#v\n", []string{"A1", "B2", "C3"}, TestABC{"Aa", "Bb", "Cc"}), "cmdout=")
}

func TestBincCodecsTable(t *testing.T) {
	testCodecTableOne(t, testBincH)
}

func TestBincCodecsMisc(t *testing.T) {
	testCodecMiscOne(t, testBincH)
}

func TestBincCodecsEmbeddedPointer(t *testing.T) {
	testCodecEmbeddedPointer(t, testBincH)
}

func TestSimpleCodecsTable(t *testing.T) {
	testCodecTableOne(t, testSimpleH)
}

func TestSimpleCodecsMisc(t *testing.T) {
	testCodecMiscOne(t, testSimpleH)
}

func TestSimpleCodecsEmbeddedPointer(t *testing.T) {
	testCodecEmbeddedPointer(t, testSimpleH)
}

func TestMsgpackCodecsTable(t *testing.T) {
	testCodecTableOne(t, testMsgpackH)
}

func TestMsgpackCodecsMisc(t *testing.T) {
	testCodecMiscOne(t, testMsgpackH)
}

func TestMsgpackCodecsEmbeddedPointer(t *testing.T) {
	testCodecEmbeddedPointer(t, testMsgpackH)
}

func TestBincRpcGo(t *testing.T) {
	doTestRpcOne(t, GoRpc, testBincH, true, 0)
}

func _TestSimpleRpcGo(t *testing.T) {
	doTestRpcOne(t, GoRpc, testSimpleH, true, 0)
}

func TestMsgpackRpcGo(t *testing.T) {
	doTestRpcOne(t, GoRpc, testMsgpackH, true, 0)
}

func TestMsgpackRpcSpec(t *testing.T) {
	doTestRpcOne(t, MsgpackSpecRpc, testMsgpackH, true, 0)
}

// TODO:
//   Add Tests for:
//   - decoding empty list/map in stream into a nil slice/map
//   - binary(M|Unm)arsher support for time.Time
