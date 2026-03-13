/*
Copyright 2015 The Kubernetes Authors.

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

// Package main defines the protoc-gen-gogo binary we use to generate our proto go files,
// as well as takes dependencies on the correct gogo/protobuf packages for godeps.
package main

import (
	"strings"

	"github.com/gogo/protobuf/vanity/command"

	// dependencies that are required for our packages
	_ "github.com/gogo/protobuf/gogoproto"
	_ "github.com/gogo/protobuf/proto"
	_ "github.com/gogo/protobuf/sortkeys"
)

func main() {
	// read input
	request := command.Read()

	// if we're given paths as inputs, generate .pb.go files based on those paths
	for _, file := range request.FileToGenerate {
		if strings.Contains(file, "/") {
			if request.Parameter != nil {
				*request.Parameter += ",paths=source_relative"
			} else {
				param := "paths=source_relative"
				request.Parameter = &param
			}
			break
		}
	}

	// generate
	command.Write(command.Generate(request))
}
