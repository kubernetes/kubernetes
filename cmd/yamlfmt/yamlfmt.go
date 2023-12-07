/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"flag"
	"fmt"
	"io"
	"os"

	"gopkg.in/yaml.v3"
)

func main() {
	indent := flag.Int("indent", 2, "default indent")
	flag.Parse()
	for _, path := range flag.Args() {
		sourceYaml, err := os.ReadFile(path)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			continue
		}
		rootNode, err := fetchYaml(sourceYaml)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			continue
		}
		writer, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0666)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			continue
		}
		err = streamYaml(writer, indent, rootNode)
		if err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			continue
		}
	}
}

func fetchYaml(sourceYaml []byte) (*yaml.Node, error) {
	rootNode := yaml.Node{}
	err := yaml.Unmarshal(sourceYaml, &rootNode)
	if err != nil {
		return nil, err
	}
	return &rootNode, nil
}

func streamYaml(writer io.Writer, indent *int, in *yaml.Node) error {
	encoder := yaml.NewEncoder(writer)
	encoder.SetIndent(*indent)
	err := encoder.Encode(in)
	if err != nil {
		return err
	}
	return encoder.Close()
}
