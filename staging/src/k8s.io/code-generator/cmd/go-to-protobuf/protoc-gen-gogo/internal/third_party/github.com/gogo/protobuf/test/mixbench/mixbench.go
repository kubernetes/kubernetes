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
	"flag"
	"fmt"
	"io/ioutil"
	"os/exec"
	"sort"
	"strings"
)

func bench(folder, rgx, outFileName string) {
	var test = exec.Command("go", "test", "-test.timeout=20m", "-test.v", "-test.run=XXX", "-test.bench="+rgx, folder)
	fmt.Printf("benching %v - %v - %v\n", folder, rgx, outFileName)
	out, err := test.CombinedOutput()
	fmt.Printf("bench output: %v\n", string(out))
	if err != nil {
		panic(err)
	}
	if err := ioutil.WriteFile(outFileName, out, 0666); err != nil {
		panic(err)
	}
}

func main() {
	flag.Parse()
	fmt.Printf("Running benches: %v\n", benchList)
	for _, bench := range strings.Split(benchList, " ") {
		b, ok := benches[bench]
		if !ok {
			fmt.Printf("No benchmark with name: %v\n", bench)
			continue
		}
		b()
	}
	if strings.Contains(benchList, "all") {
		fmt.Println("Running benchcmp will show the performance difference between using reflect and generated code for marshalling and unmarshalling of protocol buffers")
		fmt.Println("benchcmp ./test/mixbench/marshal.txt ./test/mixbench/marshaler.txt")
		fmt.Println("benchcmp ./test/mixbench/unmarshal.txt ./test/mixbench/unmarshaler.txt")
	}
}

var benches = make(map[string]func())

var benchList string

func init() {
	benches["marshaler"] = func() { bench("./test/combos/both/", "ProtoMarshal", "./test/mixbench/marshaler.txt") }
	benches["marshal"] = func() { bench("./test/", "ProtoMarshal", "./test/mixbench/marshal.txt") }
	benches["unmarshaler"] = func() { bench("./test/combos/both/", "ProtoUnmarshal", "./test/mixbench/unmarshaler.txt") }
	benches["unmarshal"] = func() { bench("./test/", "ProtoUnmarshal", "./test/mixbench/unmarshal.txt") }
	var ops []string
	for k := range benches {
		ops = append(ops, k)
	}
	sort.Strings(ops)
	benches["all"] = benchall(ops)
	ops = append(ops, "all")
	flag.StringVar(&benchList, "benchlist", "all", fmt.Sprintf("List of benchmarks to run. Options: %v", ops))
}

func benchall(ops []string) func() {
	return func() {
		for _, o := range ops {
			benches[o]()
		}
	}
}
