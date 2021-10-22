// Copyright 2017 Google Inc. All Rights Reserved.
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

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"

	"github.com/golang/protobuf/proto"
)

func usage() string {
	return fmt.Sprintf(`
Usage: %s [OPTIONS]
Options:
  --v2
    Generate an OpenAPI v2 description.
  --v3
    Generate an OpenAPI v3 description.
`, path.Base(os.Args[0]))
}

func main() {
	openAPIv2 := false
	openAPIv3 := false

	for i, arg := range os.Args {
		if i == 0 {
			continue // skip the tool name
		}
		if arg == "--v2" {
			openAPIv2 = true
		} else if arg == "--v3" {
			openAPIv3 = true
		} else {
			fmt.Printf("Unknown option: %s.\n%s\n", arg, usage())
			os.Exit(-1)
		}
	}

	if !openAPIv2 && !openAPIv3 {
		openAPIv2 = true
	}

	if openAPIv2 {
		document := buildDocumentV2()
		bytes, err := proto.Marshal(document)
		if err != nil {
			panic(err)
		}
		err = ioutil.WriteFile("petstore-v2.pb", bytes, 0644)
		if err != nil {
			panic(err)
		}
	}

	if openAPIv3 {
		document := buildDocumentV3()
		bytes, err := proto.Marshal(document)
		if err != nil {
			panic(err)
		}
		err = ioutil.WriteFile("petstore-v3.pb", bytes, 0644)
		if err != nil {
			panic(err)
		}
	}
}
