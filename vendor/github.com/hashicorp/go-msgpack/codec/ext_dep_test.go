// //+build ignore

// Copyright (c) 2012, 2013 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a BSD-style license found in the LICENSE file.

package codec

// This file includes benchmarks which have dependencies on 3rdparty
// packages (bson and vmihailenco/msgpack) which must be installed locally.
//
// To run the benchmarks including these 3rdparty packages, first
//   - Uncomment first line in this file (put // // in front of it)
//   - Get those packages:
//       go get github.com/vmihailenco/msgpack
//       go get labix.org/v2/mgo/bson
//   - Run:
//       go test -bi -bench=.

import (
	"testing"

	vmsgpack "gopkg.in/vmihailenco/msgpack.v2"
	"labix.org/v2/mgo/bson"
)

func init() {
	benchCheckers = append(benchCheckers,
		benchChecker{"v-msgpack", fnVMsgpackEncodeFn, fnVMsgpackDecodeFn},
		benchChecker{"bson", fnBsonEncodeFn, fnBsonDecodeFn},
	)
}

func fnVMsgpackEncodeFn(ts interface{}) ([]byte, error) {
	return vmsgpack.Marshal(ts)
}

func fnVMsgpackDecodeFn(buf []byte, ts interface{}) error {
	return vmsgpack.Unmarshal(buf, ts)
}

func fnBsonEncodeFn(ts interface{}) ([]byte, error) {
	return bson.Marshal(ts)
}

func fnBsonDecodeFn(buf []byte, ts interface{}) error {
	return bson.Unmarshal(buf, ts)
}

func Benchmark__Bson_______Encode(b *testing.B) {
	fnBenchmarkEncode(b, "bson", benchTs, fnBsonEncodeFn)
}

func Benchmark__Bson_______Decode(b *testing.B) {
	fnBenchmarkDecode(b, "bson", benchTs, fnBsonEncodeFn, fnBsonDecodeFn, fnBenchNewTs)
}

func Benchmark__VMsgpack___Encode(b *testing.B) {
	fnBenchmarkEncode(b, "v-msgpack", benchTs, fnVMsgpackEncodeFn)
}

func Benchmark__VMsgpack___Decode(b *testing.B) {
	fnBenchmarkDecode(b, "v-msgpack", benchTs, fnVMsgpackEncodeFn, fnVMsgpackDecodeFn, fnBenchNewTs)
}

func TestMsgpackPythonGenStreams(t *testing.T) {
	doTestMsgpackPythonGenStreams(t)
}

func TestMsgpackRpcSpecGoClientToPythonSvc(t *testing.T) {
	doTestMsgpackRpcSpecGoClientToPythonSvc(t)
}

func TestMsgpackRpcSpecPythonClientToGoSvc(t *testing.T) {
	doTestMsgpackRpcSpecPythonClientToGoSvc(t)
}
