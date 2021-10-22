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

// format-schema canonically formats a JSON schema.
package main

import (
	"fmt"
	"github.com/googleapis/gnostic/jsonschema"
	"os"
	"path"
)

func main() {
	if len(os.Args) != 2 {
		fmt.Printf("Usage: %s [filename]\n", path.Base(os.Args[0]))
		fmt.Printf("where [filename] is a path to a JSON schema to format.\n")
		os.Exit(0)
	}
	schema, err := jsonschema.NewSchemaFromFile(os.Args[1])
	if err != nil {
		panic(err)
	}
	output := schema.JSONString()
	fmt.Printf("%s\n", output)
}
