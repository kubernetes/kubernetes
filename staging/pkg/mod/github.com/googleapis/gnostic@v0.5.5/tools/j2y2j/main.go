// Copyright 2017 Google LLC. All Rights Reserved.
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
	"io/ioutil"
	"os"
	"path"

	"github.com/googleapis/gnostic/jsonschema"
	"gopkg.in/yaml.v3"
)

func usage() {
	fmt.Printf("Usage: %s [filename] [--json] [--yaml]\n", path.Base(os.Args[0]))
	fmt.Printf("where [filename] is a path to a JSON or YAML file to convert\n")
	fmt.Printf("and --json or --yaml indicates conversion to the corresponding format.\n")
	os.Exit(0)
}

func dump(node *yaml.Node, indent string) {
	node.Style = 0
	fmt.Printf("%s%s: %+v\n", indent, node.Value, node)
	for _, c := range node.Content {
		dump(c, indent+"  ")
	}
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
	var node yaml.Node
	err = yaml.Unmarshal(file, &node)

	dump(&node, "")

	switch os.Args[2] {
	case "--json":
		result := jsonschema.Render(&node)
		fmt.Printf("%s", result)
	case "--yaml":
		result, err := yaml.Marshal(&node)
		if err != nil {
			panic(err)
		}
		fmt.Printf("%s", string(result))
	default:
		usage()
	}
}
