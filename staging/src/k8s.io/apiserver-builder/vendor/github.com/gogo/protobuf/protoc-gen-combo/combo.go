// Copyright (c) 2013, Vastech SA (PTY) LTD. All rights reserved.
// http://github.com/gogo/protobuf/gogoproto
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
	Plugins  string
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
	args := append([]string{"--gogo_out=" + this.Plugins + "."}, this.Args...)
	args = append(args, filepath.Join(folder, this.Filename))
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

var min = flag.String("version", "2.3.0", "minimum protoc version")
var proto_path = flag.String("proto_path", ".", "")
var def = flag.Bool("default", true, "generate the case where everything is false")
var plugins = flag.String("plugins", "", "--gogo_out=plugins=<plugins>:.")

func main() {
	flag.Parse()
	if !version.AtLeast(*min) {
		fmt.Printf("protoc version not high enough to parse this proto file\n")
		return
	}
	args := flag.Args()
	filename := args[0]
	args = append([]string{"--proto_path=" + *proto_path})
	if _, err := exec.LookPath("protoc"); err != nil {
		panic("cannot find protoc in PATH")
	}
	pluginStr := ""
	if len(*plugins) > 0 {
		pluginStr = "plugins=" + *plugins + ":"
	}
	m := MixMatch{
		Old: []string{
			"option (gogoproto.unmarshaler_all) = false;",
			"option (gogoproto.marshaler_all) = false;",
			"option (gogoproto.unsafe_unmarshaler_all) = false;",
			"option (gogoproto.unsafe_marshaler_all) = false;",
		},
		Filename: filename,
		Args:     args,
		Plugins:  pluginStr,
	}
	if *def {
		m.Gen("./combos/neither/", []string{
			"option (gogoproto.unmarshaler_all) = false;",
			"option (gogoproto.marshaler_all) = false;",
			"option (gogoproto.unsafe_unmarshaler_all) = false;",
			"option (gogoproto.unsafe_marshaler_all) = false;",
		})
	}
	m.Gen("./combos/marshaler/", []string{
		"option (gogoproto.unmarshaler_all) = false;",
		"option (gogoproto.marshaler_all) = true;",
		"option (gogoproto.unsafe_unmarshaler_all) = false;",
		"option (gogoproto.unsafe_marshaler_all) = false;",
	})
	m.Gen("./combos/unmarshaler/", []string{
		"option (gogoproto.unmarshaler_all) = true;",
		"option (gogoproto.marshaler_all) = false;",
		"option (gogoproto.unsafe_unmarshaler_all) = false;",
		"option (gogoproto.unsafe_marshaler_all) = false;",
	})
	m.Gen("./combos/both/", []string{
		"option (gogoproto.unmarshaler_all) = true;",
		"option (gogoproto.marshaler_all) = true;",
		"option (gogoproto.unsafe_unmarshaler_all) = false;",
		"option (gogoproto.unsafe_marshaler_all) = false;",
	})
	m.Gen("./combos/unsafemarshaler/", []string{
		"option (gogoproto.unmarshaler_all) = false;",
		"option (gogoproto.marshaler_all) = false;",
		"option (gogoproto.unsafe_unmarshaler_all) = false;",
		"option (gogoproto.unsafe_marshaler_all) = true;",
	})
	m.Gen("./combos/unsafeunmarshaler/", []string{
		"option (gogoproto.unmarshaler_all) = false;",
		"option (gogoproto.marshaler_all) = false;",
		"option (gogoproto.unsafe_unmarshaler_all) = true;",
		"option (gogoproto.unsafe_marshaler_all) = false;",
	})
	m.Gen("./combos/unsafeboth/", []string{
		"option (gogoproto.unmarshaler_all) = false;",
		"option (gogoproto.marshaler_all) = false;",
		"option (gogoproto.unsafe_unmarshaler_all) = true;",
		"option (gogoproto.unsafe_marshaler_all) = true;",
	})
}
