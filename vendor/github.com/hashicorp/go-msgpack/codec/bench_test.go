// Copyright (c) 2012, 2013 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"flag"
	"fmt"
	"reflect"
	"runtime"
	"testing"
	"time"
)

// Sample way to run:
// go test -bi -bv -bd=1 -benchmem -bench=.

var (
	_       = fmt.Printf
	benchTs *TestStruc

	approxSize int

	benchDoInitBench     bool
	benchVerify          bool
	benchUnscientificRes bool = false
	//depth of 0 maps to ~400bytes json-encoded string, 1 maps to ~1400 bytes, etc
	//For depth>1, we likely trigger stack growth for encoders, making benchmarking unreliable.
	benchDepth     int
	benchInitDebug bool
	benchCheckers  []benchChecker
)

type benchEncFn func(interface{}) ([]byte, error)
type benchDecFn func([]byte, interface{}) error
type benchIntfFn func() interface{}

type benchChecker struct {
	name     string
	encodefn benchEncFn
	decodefn benchDecFn
}

func benchInitFlags() {
	flag.BoolVar(&benchInitDebug, "bg", false, "Bench Debug")
	flag.IntVar(&benchDepth, "bd", 1, "Bench Depth: If >1, potential unreliable results due to stack growth")
	flag.BoolVar(&benchDoInitBench, "bi", false, "Run Bench Init")
	flag.BoolVar(&benchVerify, "bv", false, "Verify Decoded Value during Benchmark")
	flag.BoolVar(&benchUnscientificRes, "bu", false, "Show Unscientific Results during Benchmark")
}

func benchInit() {
	benchTs = newTestStruc(benchDepth, true)
	approxSize = approxDataSize(reflect.ValueOf(benchTs))
	bytesLen := 1024 * 4 * (benchDepth + 1) * (benchDepth + 1)
	if bytesLen < approxSize {
		bytesLen = approxSize
	}

	benchCheckers = append(benchCheckers,
		benchChecker{"msgpack", fnMsgpackEncodeFn, fnMsgpackDecodeFn},
		benchChecker{"binc-nosym", fnBincNoSymEncodeFn, fnBincNoSymDecodeFn},
		benchChecker{"binc-sym", fnBincSymEncodeFn, fnBincSymDecodeFn},
		benchChecker{"simple", fnSimpleEncodeFn, fnSimpleDecodeFn},
		benchChecker{"gob", fnGobEncodeFn, fnGobDecodeFn},
		benchChecker{"json", fnJsonEncodeFn, fnJsonDecodeFn},
	)
	if benchDoInitBench {
		runBenchInit()
	}
}

func runBenchInit() {
	logT(nil, "..............................................")
	logT(nil, "BENCHMARK INIT: %v", time.Now())
	logT(nil, "To run full benchmark comparing encodings (MsgPack, Binc, Simple, JSON, GOB, etc), "+
		"use: \"go test -bench=.\"")
	logT(nil, "Benchmark: ")
	logT(nil, "\tStruct recursive Depth:             %d", benchDepth)
	if approxSize > 0 {
		logT(nil, "\tApproxDeepSize Of benchmark Struct: %d bytes", approxSize)
	}
	if benchUnscientificRes {
		logT(nil, "Benchmark One-Pass Run (with Unscientific Encode/Decode times): ")
	} else {
		logT(nil, "Benchmark One-Pass Run:")
	}
	for _, bc := range benchCheckers {
		doBenchCheck(bc.name, bc.encodefn, bc.decodefn)
	}
	logT(nil, "..............................................")
	if benchInitDebug {
		logT(nil, "<<<<====>>>> depth: %v, ts: %#v\n", benchDepth, benchTs)
	}
}

func fnBenchNewTs() interface{} {
	return new(TestStruc)
}

func doBenchCheck(name string, encfn benchEncFn, decfn benchDecFn) {
	runtime.GC()
	tnow := time.Now()
	buf, err := encfn(benchTs)
	if err != nil {
		logT(nil, "\t%10s: **** Error encoding benchTs: %v", name, err)
	}
	encDur := time.Now().Sub(tnow)
	encLen := len(buf)
	runtime.GC()
	if !benchUnscientificRes {
		logT(nil, "\t%10s: len: %d bytes\n", name, encLen)
		return
	}
	tnow = time.Now()
	if err = decfn(buf, new(TestStruc)); err != nil {
		logT(nil, "\t%10s: **** Error decoding into new TestStruc: %v", name, err)
	}
	decDur := time.Now().Sub(tnow)
	logT(nil, "\t%10s: len: %d bytes, encode: %v, decode: %v\n", name, encLen, encDur, decDur)
}

func fnBenchmarkEncode(b *testing.B, encName string, ts interface{}, encfn benchEncFn) {
	runtime.GC()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := encfn(ts)
		if err != nil {
			logT(b, "Error encoding benchTs: %s: %v", encName, err)
			b.FailNow()
		}
	}
}

func fnBenchmarkDecode(b *testing.B, encName string, ts interface{},
	encfn benchEncFn, decfn benchDecFn, newfn benchIntfFn,
) {
	buf, err := encfn(ts)
	if err != nil {
		logT(b, "Error encoding benchTs: %s: %v", encName, err)
		b.FailNow()
	}
	runtime.GC()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ts = newfn()
		if err = decfn(buf, ts); err != nil {
			logT(b, "Error decoding into new TestStruc: %s: %v", encName, err)
			b.FailNow()
		}
		if benchVerify {
			if vts, vok := ts.(*TestStruc); vok {
				verifyTsTree(b, vts)
			}
		}
	}
}

func verifyTsTree(b *testing.B, ts *TestStruc) {
	var ts0, ts1m, ts2m, ts1s, ts2s *TestStruc
	ts0 = ts

	if benchDepth > 0 {
		ts1m, ts1s = verifyCheckAndGet(b, ts0)
	}

	if benchDepth > 1 {
		ts2m, ts2s = verifyCheckAndGet(b, ts1m)
	}
	for _, tsx := range []*TestStruc{ts0, ts1m, ts2m, ts1s, ts2s} {
		if tsx != nil {
			verifyOneOne(b, tsx)
		}
	}
}

func verifyCheckAndGet(b *testing.B, ts0 *TestStruc) (ts1m *TestStruc, ts1s *TestStruc) {
	// if len(ts1m.Ms) <= 2 {
	// 	logT(b, "Error: ts1m.Ms len should be > 2. Got: %v", len(ts1m.Ms))
	// 	b.FailNow()
	// }
	if len(ts0.Its) == 0 {
		logT(b, "Error: ts0.Islice len should be > 0. Got: %v", len(ts0.Its))
		b.FailNow()
	}
	ts1m = ts0.Mtsptr["0"]
	ts1s = ts0.Its[0]
	if ts1m == nil || ts1s == nil {
		logT(b, "Error: At benchDepth 1, No *TestStruc found")
		b.FailNow()
	}
	return
}

func verifyOneOne(b *testing.B, ts *TestStruc) {
	if ts.I64slice[2] != int64(3) {
		logT(b, "Error: Decode failed by checking values")
		b.FailNow()
	}
}

func fnMsgpackEncodeFn(ts interface{}) (bs []byte, err error) {
	err = NewEncoderBytes(&bs, testMsgpackH).Encode(ts)
	return
}

func fnMsgpackDecodeFn(buf []byte, ts interface{}) error {
	return NewDecoderBytes(buf, testMsgpackH).Decode(ts)
}

func fnBincEncodeFn(ts interface{}, sym AsSymbolFlag) (bs []byte, err error) {
	tSym := testBincH.AsSymbols
	testBincH.AsSymbols = sym
	err = NewEncoderBytes(&bs, testBincH).Encode(ts)
	testBincH.AsSymbols = tSym
	return
}

func fnBincDecodeFn(buf []byte, ts interface{}, sym AsSymbolFlag) (err error) {
	tSym := testBincH.AsSymbols
	testBincH.AsSymbols = sym
	err = NewDecoderBytes(buf, testBincH).Decode(ts)
	testBincH.AsSymbols = tSym
	return
}

func fnBincNoSymEncodeFn(ts interface{}) (bs []byte, err error) {
	return fnBincEncodeFn(ts, AsSymbolNone)
}

func fnBincNoSymDecodeFn(buf []byte, ts interface{}) error {
	return fnBincDecodeFn(buf, ts, AsSymbolNone)
}

func fnBincSymEncodeFn(ts interface{}) (bs []byte, err error) {
	return fnBincEncodeFn(ts, AsSymbolAll)
}

func fnBincSymDecodeFn(buf []byte, ts interface{}) error {
	return fnBincDecodeFn(buf, ts, AsSymbolAll)
}

func fnSimpleEncodeFn(ts interface{}) (bs []byte, err error) {
	err = NewEncoderBytes(&bs, testSimpleH).Encode(ts)
	return
}

func fnSimpleDecodeFn(buf []byte, ts interface{}) error {
	return NewDecoderBytes(buf, testSimpleH).Decode(ts)
}

func fnGobEncodeFn(ts interface{}) ([]byte, error) {
	bbuf := new(bytes.Buffer)
	err := gob.NewEncoder(bbuf).Encode(ts)
	return bbuf.Bytes(), err
}

func fnGobDecodeFn(buf []byte, ts interface{}) error {
	return gob.NewDecoder(bytes.NewBuffer(buf)).Decode(ts)
}

func fnJsonEncodeFn(ts interface{}) ([]byte, error) {
	return json.Marshal(ts)
}

func fnJsonDecodeFn(buf []byte, ts interface{}) error {
	return json.Unmarshal(buf, ts)
}

func Benchmark__Msgpack____Encode(b *testing.B) {
	fnBenchmarkEncode(b, "msgpack", benchTs, fnMsgpackEncodeFn)
}

func Benchmark__Msgpack____Decode(b *testing.B) {
	fnBenchmarkDecode(b, "msgpack", benchTs, fnMsgpackEncodeFn, fnMsgpackDecodeFn, fnBenchNewTs)
}

func Benchmark__Binc_NoSym_Encode(b *testing.B) {
	fnBenchmarkEncode(b, "binc", benchTs, fnBincNoSymEncodeFn)
}

func Benchmark__Binc_NoSym_Decode(b *testing.B) {
	fnBenchmarkDecode(b, "binc", benchTs, fnBincNoSymEncodeFn, fnBincNoSymDecodeFn, fnBenchNewTs)
}

func Benchmark__Binc_Sym___Encode(b *testing.B) {
	fnBenchmarkEncode(b, "binc", benchTs, fnBincSymEncodeFn)
}

func Benchmark__Binc_Sym___Decode(b *testing.B) {
	fnBenchmarkDecode(b, "binc", benchTs, fnBincSymEncodeFn, fnBincSymDecodeFn, fnBenchNewTs)
}

func Benchmark__Simple____Encode(b *testing.B) {
	fnBenchmarkEncode(b, "simple", benchTs, fnSimpleEncodeFn)
}

func Benchmark__Simple____Decode(b *testing.B) {
	fnBenchmarkDecode(b, "simple", benchTs, fnSimpleEncodeFn, fnSimpleDecodeFn, fnBenchNewTs)
}

func Benchmark__Gob________Encode(b *testing.B) {
	fnBenchmarkEncode(b, "gob", benchTs, fnGobEncodeFn)
}

func Benchmark__Gob________Decode(b *testing.B) {
	fnBenchmarkDecode(b, "gob", benchTs, fnGobEncodeFn, fnGobDecodeFn, fnBenchNewTs)
}

func Benchmark__Json_______Encode(b *testing.B) {
	fnBenchmarkEncode(b, "json", benchTs, fnJsonEncodeFn)
}

func Benchmark__Json_______Decode(b *testing.B) {
	fnBenchmarkDecode(b, "json", benchTs, fnJsonEncodeFn, fnJsonDecodeFn, fnBenchNewTs)
}
