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

// go-to-protobuf generates a Protobuf IDL from a Go struct, respecting any
// existing IDL tags on the Go struct.
package main

import (
	"k8s.io/code-generator/cmd/go-to-protobuf/protobuf"

	flag "github.com/spf13/pflag"
)

var g = protobuf.New()

func init() {
	g.BindFlags(flag.CommandLine)
}

func main() {
	flag.Parse()
	protobuf.Run(g)
}
