// Copyright 2018 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Filter and display messages produced by gnostic invocations.
package main

import (
	"flag"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/golang/protobuf/proto"
	"github.com/googleapis/gnostic/printer"

	plugins "github.com/googleapis/gnostic/plugins"
)

func readMessagesFromFileWithName(filename string) *plugins.Messages {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		fmt.Printf("File error: %v\n", err)
		os.Exit(1)
	}
	messages := &plugins.Messages{}
	err = proto.Unmarshal(data, messages)
	if err != nil {
		panic(err)
	}
	return messages
}

func printMessages(code *printer.Code, messages *plugins.Messages) {
	for _, message := range messages.Messages {
		line := fmt.Sprintf("%-7s %-14s %s %+v", 
				message.Level,
				message.Code,
				message.Text,
				message.Keys)
		code.Print(line)
        }
}

func main() {
	flag.Parse()
	args := flag.Args()

	if len(args) != 1 {
		fmt.Printf("Usage: report-messages <file.pb>\n")
		return
	}

	messages := readMessagesFromFileWithName(args[0])

	code := &printer.Code{}
	printMessages(code, messages)
	fmt.Printf("%s", code)
}
