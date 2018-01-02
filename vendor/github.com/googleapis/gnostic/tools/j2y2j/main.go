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

// j2y2j converts JSON to YAML and YAML to JSON.
package main

import (
	"fmt"
	"github.com/googleapis/gnostic/jsonschema"
	"gopkg.in/yaml.v2"
	"io/ioutil"
	"os"
	"path"
)

func usage() {
	fmt.Printf("Usage: %s [filename] [--json] [--yaml]\n", path.Base(os.Args[0]))
	fmt.Printf("where [filename] is a path to a JSON or YAML file to convert\n")
	fmt.Printf("and --json or --yaml indicates conversion to the corresponding format.\n")
	os.Exit(0)
}

func main() {
	if len(os.Args) != 3 {
		usage()
	}

	filename := os.Args[1]
	file, err := ioutil.ReadFile(filename)
	if err != nil {
		panic(err)
	}
	var info yaml.MapSlice
	err = yaml.Unmarshal(file, &info)

	switch os.Args[2] {
	case "--json":
		result := jsonschema.Render(info)
		fmt.Printf("%s", result)
	case "--yaml":
		result, err := yaml.Marshal(info)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%s", string(result))
	default:
		usage()
	}
}
