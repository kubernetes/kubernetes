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
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/gogo/protobuf/version"
)

type MixMatch struct {
	Old      []string
	Filename string
	Args     []string
}

func (this MixMatch) Gen(folder string, news []string) {
	if err := os.MkdirAll(folder, 0777); err != nil {
		panic(err)
	}
	data, err := ioutil.ReadFile(this.Filename)
	if err != nil {
		panic(err)
	}
	content := string(data)
	for i, old := range this.Old {
		if !strings.Contains(content, old) {
			panic(fmt.Errorf("could not find string {%s} to replace with {%s}", old, news[i]))
		}
		content = strings.Replace(content, old, news[i], 1)
		if strings.Contains(content, old) && old != news[i] {
			panic(fmt.Errorf("found another string {%s} after it was replaced with {%s}", old, news[i]))
		}
	}
	if err = ioutil.WriteFile(filepath.Join(folder, this.Filename), []byte(content), 0666); err != nil {
		panic(err)
	}
	args := append(this.Args, filepath.Join(folder, this.Filename))
	var regenerate = exec.Command("protoc", args...)
	out, err := regenerate.CombinedOutput()

	failed := false
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		text := scanner.Text()
		fmt.Println("protoc-gen-combo: ", text)
		if !strings.Contains(text, "WARNING") {
			failed = true
		}
	}

	if err != nil {
		fmt.Print("protoc-gen-combo: error: ", err)
		failed = true
	}

	if failed {
		os.Exit(1)
	}
}

func filter(ss []string, flag string) ([]string, string) {
	s := make([]string, 0, len(ss))
	var v string
	for i := range ss {
		if strings.Contains(ss[i], flag) {
			vs := strings.Split(ss[i], "=")
			v = vs[1]
			continue
		}
		s = append(s, ss[i])
	}
	return s, v
}

func filterArgs(ss []string) ([]string, []string) {
	var args []string
	var flags []string
	for i := range ss {
		if strings.Contains(ss[i], "=") {
			flags = append(flags, ss[i])
			continue
		}
		args = append(args, ss[i])
	}
	return flags, args
}

func main() {
	flag.String("version", "2.3.0", "minimum protoc version")
	flag.Bool("default", true, "generate the case where everything is false")
	flags, args := filterArgs(os.Args[1:])
	var min string
	flags, min = filter(flags, "-version")
	if len(min) == 0 {
		min = "2.3.1"
	}
	if !version.AtLeast(min) {
		fmt.Printf("protoc version not high enough to parse this proto file\n")
		return
	}
	if len(args) != 1 {
		fmt.Printf("protoc-gen-combo expects a filename\n")
		os.Exit(1)
	}
	filename := args[0]
	var def string
	flags, def = filter(flags, "-default")
	if _, err := exec.LookPath("protoc"); err != nil {
		panic("cannot find protoc in PATH")
	}
	m := MixMatch{
		Old: []string{
			"option (gogoproto.unmarshaler_all) = false;",
			"option (gogoproto.marshaler_all) = false;",
		},
		Filename: filename,
		Args:     flags,
	}
	if def != "false" {
		m.Gen("./combos/neither/", []string{
			"option (gogoproto.unmarshaler_all) = false;",
			"option (gogoproto.marshaler_all) = false;",
		})
	}
	m.Gen("./combos/marshaler/", []string{
		"option (gogoproto.unmarshaler_all) = false;",
		"option (gogoproto.marshaler_all) = true;",
	})
	m.Gen("./combos/unmarshaler/", []string{
		"option (gogoproto.unmarshaler_all) = true;",
		"option (gogoproto.marshaler_all) = false;",
	})
	m.Gen("./combos/both/", []string{
		"option (gogoproto.unmarshaler_all) = true;",
		"option (gogoproto.marshaler_all) = true;",
	})
}
