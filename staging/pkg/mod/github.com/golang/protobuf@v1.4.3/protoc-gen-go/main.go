// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// protoc-gen-go is a plugin for the Google protocol buffer compiler to generate
// Go code. Install it by building this program and making it accessible within
// your PATH with the name:
//	protoc-gen-go
//
// The 'go' suffix becomes part of the argument for the protocol compiler,
// such that it can be invoked as:
//	protoc --go_out=paths=source_relative:. path/to/file.proto
//
// This generates Go bindings for the protocol buffer defined by file.proto.
// With that input, the output will be written to:
//	path/to/file.pb.go
//
// See the README and documentation for protocol buffers to learn more:
//	https://developers.google.com/protocol-buffers/
package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/golang/protobuf/internal/gengogrpc"
	gengo "google.golang.org/protobuf/cmd/protoc-gen-go/internal_gengo"
	"google.golang.org/protobuf/compiler/protogen"
)

func main() {
	var (
		flags        flag.FlagSet
		plugins      = flags.String("plugins", "", "list of plugins to enable (supported values: grpc)")
		importPrefix = flags.String("import_prefix", "", "prefix to prepend to import paths")
	)
	importRewriteFunc := func(importPath protogen.GoImportPath) protogen.GoImportPath {
		switch importPath {
		case "context", "fmt", "math":
			return importPath
		}
		if *importPrefix != "" {
			return protogen.GoImportPath(*importPrefix) + importPath
		}
		return importPath
	}
	protogen.Options{
		ParamFunc:         flags.Set,
		ImportRewriteFunc: importRewriteFunc,
	}.Run(func(gen *protogen.Plugin) error {
		grpc := false
		for _, plugin := range strings.Split(*plugins, ",") {
			switch plugin {
			case "grpc":
				grpc = true
			case "":
			default:
				return fmt.Errorf("protoc-gen-go: unknown plugin %q", plugin)
			}
		}
		for _, f := range gen.Files {
			if !f.Generate {
				continue
			}
			g := gengo.GenerateFile(gen, f)
			if grpc {
				gengogrpc.GenerateFileContent(gen, f, g)
			}
		}
		gen.SupportedFeatures = gengo.SupportedFeatures
		return nil
	})
}
