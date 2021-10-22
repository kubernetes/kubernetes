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

//go:generate ./COMPILE-PROTOS.sh

// Gnostic is a tool for building better REST APIs through knowledge.
//
// Gnostic reads declarative descriptions of REST APIs that conform
// to the OpenAPI Specification, reports errors, resolves internal
// dependencies, and puts the results in a binary form that can
// be used in any language that is supported by the Protocol Buffer
// tools.
//
// Gnostic models are validated and typed. This allows API tool
// developers to focus on their product and not worry about input
// validation and type checking.
//
// Gnostic calls plugins that implement a variety of API implementation
// and support features including generation of client and server
// support code.
package main

import (
	"fmt"
	"os"

	"github.com/googleapis/gnostic/lib"
)

func main() {
	// To simplify testing, Gnostic is implemented in an embeddable library.
	g := lib.NewGnostic(os.Args)
	err := g.Main()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s\n", err.Error())
		os.Exit(-1)
	}
}
