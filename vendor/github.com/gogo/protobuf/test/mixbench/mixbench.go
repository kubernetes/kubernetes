// Protocol Buffers for Go with Gadgets
//
// Copyright (c) 2013, The GoGo Authors. All rights reserved.
// http://github.com/gogo/protobuf
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"strings"
)

type MixMatch struct {
	Old []string
	New []string
}

func (this *MixMatch) Regenerate() {
	fmt.Printf("mixbench\n")
	uuidData, err := ioutil.ReadFile("../uuid.go")
	if err != nil {
		panic(err)
	}
	if err = ioutil.WriteFile("./testdata/uuid.go", uuidData, 0666); err != nil {
		panic(err)
	}
	data, err := ioutil.ReadFile("../thetest.proto")
	if err != nil {
		panic(err)
	}
	content := string(data)
	for i, old := range this.Old {
		content = strings.Replace(content, old, this.New[i], -1)
	}
	if err = ioutil.WriteFile("./testdata/thetest.proto", []byte(content), 0666); err != nil {
		panic(err)
	}
	var regenerate = exec.Command("protoc", "--gogo_out=.", "-I=../../:../../protobuf/:../../../../../:.", "./testdata/thetest.proto")
	fmt.Printf("regenerating\n")
	out, err := regenerate.CombinedOutput()
	fmt.Printf("regenerate output: %v\n", string(out))
	if err != nil {
		panic(err)
	}
}

func (this *MixMatch) Bench(rgx string, outFileName string) {
	if err := os.MkdirAll("./testdata", 0777); err != nil {
		panic(err)
	}
	this.Regenerate()
	var test = exec.Command("go", "test", "-test.timeout=20m", "-test.v", "-test.run=XXX", "-test.bench="+rgx, "./testdata/")
	fmt.Printf("benching\n")
	out, err := test.CombinedOutput()
	fmt.Printf("bench output: %v\n", string(out))
	if err != nil {
		panic(err)
	}
	if err := ioutil.WriteFile(outFileName, out, 0666); err != nil {
		panic(err)
	}
	if err := os.RemoveAll("./testdata"); err != nil {
		panic(err)
	}
}

func NewMixMatch(marshaler, unmarshaler, unsafe_marshaler, unsafe_unmarshaler bool) *MixMatch {
	mm := &MixMatch{}
	if marshaler {
		mm.Old = append(mm.Old, "option (gogoproto.marshaler_all) = false;")
		mm.New = append(mm.New, "option (gogoproto.marshaler_all) = true;")
	} else {
		mm.Old = append(mm.Old, "option (gogoproto.marshaler_all) = true;")
		mm.New = append(mm.New, "option (gogoproto.marshaler_all) = false;")
	}
	if unmarshaler {
		mm.Old = append(mm.Old, "option (gogoproto.unmarshaler_all) = false;")
		mm.New = append(mm.New, "option (gogoproto.unmarshaler_all) = true;")
	} else {
		mm.Old = append(mm.Old, "option (gogoproto.unmarshaler_all) = true;")
		mm.New = append(mm.New, "option (gogoproto.unmarshaler_all) = false;")
	}
	if unsafe_marshaler {
		mm.Old = append(mm.Old, "option (gogoproto.unsafe_marshaler_all) = false;")
		mm.New = append(mm.New, "option (gogoproto.unsafe_marshaler_all) = true;")
	} else {
		mm.Old = append(mm.Old, "option (gogoproto.unsafe_marshaler_all) = true;")
		mm.New = append(mm.New, "option (gogoproto.unsafe_marshaler_all) = false;")
	}
	if unsafe_unmarshaler {
		mm.Old = append(mm.Old, "option (gogoproto.unsafe_unmarshaler_all) = false;")
		mm.New = append(mm.New, "option (gogoproto.unsafe_unmarshaler_all) = true;")
	} else {
		mm.Old = append(mm.Old, "option (gogoproto.unsafe_unmarshaler_all) = true;")
		mm.New = append(mm.New, "option (gogoproto.unsafe_unmarshaler_all) = false;")
	}
	return mm
}

func main() {
	NewMixMatch(true, true, false, false).Bench("ProtoMarshal", "marshaler.txt")
	NewMixMatch(false, false, false, false).Bench("ProtoMarshal", "marshal.txt")
	NewMixMatch(false, false, true, true).Bench("ProtoMarshal", "unsafe_marshaler.txt")
	NewMixMatch(true, true, false, false).Bench("ProtoUnmarshal", "unmarshaler.txt")
	NewMixMatch(false, false, false, false).Bench("ProtoUnmarshal", "unmarshal.txt")
	NewMixMatch(false, false, true, true).Bench("ProtoUnmarshal", "unsafe_unmarshaler.txt")
	fmt.Println("Running benchcmp will show the performance difference between using reflect and generated code for marshalling and unmarshalling of protocol buffers")
	fmt.Println("$GOROOT/misc/benchcmp marshal.txt marshaler.txt")
	fmt.Println("$GOROOT/misc/benchcmp unmarshal.txt unmarshaler.txt")
	fmt.Println("$GOROOT/misc/benchcmp marshal.txt unsafe_marshaler.txt")
	fmt.Println("$GOROOT/misc/benchcmp unmarshal.txt unsafe_unmarshaler.txt")
}
