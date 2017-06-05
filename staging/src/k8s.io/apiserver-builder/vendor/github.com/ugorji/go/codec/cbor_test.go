// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import (
	"bufio"
	"bytes"
	"encoding/hex"
	"math"
	"os"
	"regexp"
	"strings"
	"testing"
)

func TestCborIndefiniteLength(t *testing.T) {
	oldMapType := testCborH.MapType
	defer func() {
		testCborH.MapType = oldMapType
	}()
	testCborH.MapType = testMapStrIntfTyp
	// var (
	// 	M1 map[string][]byte
	// 	M2 map[uint64]bool
	// 	L1 []interface{}
	// 	S1 []string
	// 	B1 []byte
	// )
	var v, vv interface{}
	// define it (v), encode it using indefinite lengths, decode it (vv), compare v to vv
	v = map[string]interface{}{
		"one-byte-key":   []byte{1, 2, 3, 4, 5, 6},
		"two-string-key": "two-value",
		"three-list-key": []interface{}{true, false, uint64(1), int64(-1)},
	}
	var buf bytes.Buffer
	// buf.Reset()
	e := NewEncoder(&buf, testCborH)
	buf.WriteByte(cborBdIndefiniteMap)
	//----
	buf.WriteByte(cborBdIndefiniteString)
	e.MustEncode("one-")
	e.MustEncode("byte-")
	e.MustEncode("key")
	buf.WriteByte(cborBdBreak)

	buf.WriteByte(cborBdIndefiniteBytes)
	e.MustEncode([]byte{1, 2, 3})
	e.MustEncode([]byte{4, 5, 6})
	buf.WriteByte(cborBdBreak)

	//----
	buf.WriteByte(cborBdIndefiniteString)
	e.MustEncode("two-")
	e.MustEncode("string-")
	e.MustEncode("key")
	buf.WriteByte(cborBdBreak)

	buf.WriteByte(cborBdIndefiniteString)
	e.MustEncode([]byte("two-")) // encode as bytes, to check robustness of code
	e.MustEncode([]byte("value"))
	buf.WriteByte(cborBdBreak)

	//----
	buf.WriteByte(cborBdIndefiniteString)
	e.MustEncode("three-")
	e.MustEncode("list-")
	e.MustEncode("key")
	buf.WriteByte(cborBdBreak)

	buf.WriteByte(cborBdIndefiniteArray)
	e.MustEncode(true)
	e.MustEncode(false)
	e.MustEncode(uint64(1))
	e.MustEncode(int64(-1))
	buf.WriteByte(cborBdBreak)

	buf.WriteByte(cborBdBreak) // close map

	NewDecoderBytes(buf.Bytes(), testCborH).MustDecode(&vv)
	if err := deepEqual(v, vv); err != nil {
		logT(t, "-------- Before and After marshal do not match: Error: %v", err)
		logT(t, "    ....... GOLDEN:  (%T) %#v", v, v)
		logT(t, "    ....... DECODED: (%T) %#v", vv, vv)
		failT(t)
	}
}

type testCborGolden struct {
	Base64     string      `codec:"cbor"`
	Hex        string      `codec:"hex"`
	Roundtrip  bool        `codec:"roundtrip"`
	Decoded    interface{} `codec:"decoded"`
	Diagnostic string      `codec:"diagnostic"`
	Skip       bool        `codec:"skip"`
}

// Some tests are skipped because they include numbers outside the range of int64/uint64
func doTestCborGoldens(t *testing.T) {
	oldMapType := testCborH.MapType
	defer func() {
		testCborH.MapType = oldMapType
	}()
	testCborH.MapType = testMapStrIntfTyp
	// decode test-cbor-goldens.json into a list of []*testCborGolden
	// for each one,
	// - decode hex into []byte bs
	// - decode bs into interface{} v
	// - compare both using deepequal
	// - for any miss, record it
	var gs []*testCborGolden
	f, err := os.Open("test-cbor-goldens.json")
	if err != nil {
		logT(t, "error opening test-cbor-goldens.json: %v", err)
		failT(t)
	}
	defer f.Close()
	jh := new(JsonHandle)
	jh.MapType = testMapStrIntfTyp
	// d := NewDecoder(f, jh)
	d := NewDecoder(bufio.NewReader(f), jh)
	// err = d.Decode(&gs)
	d.MustDecode(&gs)
	if err != nil {
		logT(t, "error json decoding test-cbor-goldens.json: %v", err)
		failT(t)
	}

	tagregex := regexp.MustCompile(`[\d]+\(.+?\)`)
	hexregex := regexp.MustCompile(`h'([0-9a-fA-F]*)'`)
	for i, g := range gs {
		// fmt.Printf("%v, skip: %v, isTag: %v, %s\n", i, g.Skip, tagregex.MatchString(g.Diagnostic), g.Diagnostic)
		// skip tags or simple or those with prefix, as we can't verify them.
		if g.Skip || strings.HasPrefix(g.Diagnostic, "simple(") || tagregex.MatchString(g.Diagnostic) {
			// fmt.Printf("%v: skipped\n", i)
			logT(t, "[%v] skipping because skip=true OR unsupported simple value or Tag Value", i)
			continue
		}
		// println("++++++++++++", i, "g.Diagnostic", g.Diagnostic)
		if hexregex.MatchString(g.Diagnostic) {
			// println(i, "g.Diagnostic matched hex")
			if s2 := g.Diagnostic[2 : len(g.Diagnostic)-1]; s2 == "" {
				g.Decoded = zeroByteSlice
			} else if bs2, err2 := hex.DecodeString(s2); err2 == nil {
				g.Decoded = bs2
			}
			// fmt.Printf("%v: hex: %v\n", i, g.Decoded)
		}
		bs, err := hex.DecodeString(g.Hex)
		if err != nil {
			logT(t, "[%v] error hex decoding %s [%v]: %v", i, g.Hex, err)
			failT(t)
		}
		var v interface{}
		NewDecoderBytes(bs, testCborH).MustDecode(&v)
		if _, ok := v.(RawExt); ok {
			continue
		}
		// check the diagnostics to compare
		switch g.Diagnostic {
		case "Infinity":
			b := math.IsInf(v.(float64), 1)
			testCborError(t, i, math.Inf(1), v, nil, &b)
		case "-Infinity":
			b := math.IsInf(v.(float64), -1)
			testCborError(t, i, math.Inf(-1), v, nil, &b)
		case "NaN":
			// println(i, "checking NaN")
			b := math.IsNaN(v.(float64))
			testCborError(t, i, math.NaN(), v, nil, &b)
		case "undefined":
			b := v == nil
			testCborError(t, i, nil, v, nil, &b)
		default:
			v0 := g.Decoded
			// testCborCoerceJsonNumber(reflect.ValueOf(&v0))
			testCborError(t, i, v0, v, deepEqual(v0, v), nil)
		}
	}
}

func testCborError(t *testing.T, i int, v0, v1 interface{}, err error, equal *bool) {
	if err == nil && equal == nil {
		// fmt.Printf("%v testCborError passed (err and equal nil)\n", i)
		return
	}
	if err != nil {
		logT(t, "[%v] deepEqual error: %v", i, err)
		logT(t, "    ....... GOLDEN:  (%T) %#v", v0, v0)
		logT(t, "    ....... DECODED: (%T) %#v", v1, v1)
		failT(t)
	}
	if equal != nil && !*equal {
		logT(t, "[%v] values not equal", i)
		logT(t, "    ....... GOLDEN:  (%T) %#v", v0, v0)
		logT(t, "    ....... DECODED: (%T) %#v", v1, v1)
		failT(t)
	}
	// fmt.Printf("%v testCborError passed (checks passed)\n", i)
}

func TestCborGoldens(t *testing.T) {
	doTestCborGoldens(t)
}
