/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/parser"

	flag "github.com/spf13/pflag"
)

var (
	inputPackages        = flag.StringP("input-packages", "i" /*"k8s.io/kubernetes/pkg/api,k8s.io/kubernetes/pkg/apis/extensions"*/, "", "comma-separated list of directories to get input types from.")
	packages             = flag.StringP("packages", "p", "-k8s.io/kubernetes/pkg/util/intstr,-k8s.io/kubernetes/pkg/api/resource,-k8s.io/kubernetes/pkg/runtime,k8s.io/kubernetes/pkg/api/unversioned,k8s.io/kubernetes/pkg/api/v1,k8s.io/kubernetes/pkg/apis/extensions/v1beta1", "comma-separated list of directories to get input types from.")
	outputBase           = flag.StringP("output-base", "o", filepath.Join(os.Getenv("GOPATH"), "src"), "Output base; defaults to $GOPATH/src/")
	protoImport          = flag.String("proto-import", os.Getenv("PROTO_PATH"), "The search path for the core protobuf .protos, required.")
	onlyIDL              = flag.Bool("only-idl", false, "If true, only generate the IDL for each package.")
	skipGeneratedRewrite = flag.Bool("skip-generated-rewrite", false, "If true, skip fixing up the generated.pb.go file (debugging only).")
)

const (
	typesKindProtobuf = "Protobuf"
)

func main() {
	flag.Parse()

	b := parser.New()
	if len(*inputPackages) > 0 {
		for _, d := range strings.Split(*inputPackages, ",") {
			if err := b.AddDir(d); err != nil {
				log.Fatalf("Unable to add directory %q: %v", d, err)
			}
		}
	}

	pkgs := NewProtobufNamer()
	for _, d := range strings.Split(*packages, ",") {
		all := true
		if strings.HasPrefix(d, "-") {
			d = d[1:]
			all = false
		}
		name := protoSafePackage(d)
		parts := strings.SplitN(d, "=", 2)
		if len(parts) > 1 {
			d = parts[0]
			name = parts[1]
		}
		pkgs.Add(newProtobufPackage(d, name, all))
	}
	generatedPackages := pkgs.packages

	for _, p := range generatedPackages {
		if err := p.Clean(*outputBase); err != nil {
			log.Fatalf("Unable to clean package %s: %v", p.PackageName, err)
		}
	}

	for _, p := range generatedPackages {
		if err := b.AddDir(p.PackagePath); err != nil {
			log.Fatalf("Unable to add directory %q: %v", p.PackagePath, err)
		}
	}

	c, err := generator.NewContext(
		b,
		namer.NameSystems{
			"public": namer.NewPublicNamer(3),
			"proto":  pkgs,
		},
		"public",
	)
	c.FileTypes["protoidl"] = protoIDLFileType{}

	if err != nil {
		log.Fatalf("Failed making a context: %v", err)
	}

	if err := pkgs.AssignTypesToPackages(c); err != nil {
		log.Fatalf("Failed to identify common types: %v", err)
	}

	if err := c.ExecutePackages(*outputBase, pkgs.List()); err != nil {
		log.Fatalf("Failed executing generator: %v", err)
	}

	if *onlyIDL {
		return
	}

	if _, err := exec.LookPath("protoc"); err != nil {
		log.Fatalf("Unable to find 'protoc': %v", err)
	}

	searchArgs := []string{"-I", ".", "-I", *outputBase}
	if len(*protoImport) != 0 {
		searchArgs = append(searchArgs, "-I", *protoImport)
	}
	args := append(searchArgs, fmt.Sprintf("--gogo_out=%s", *outputBase))

	for _, p := range generatedPackages {
		path := filepath.Join(*outputBase, p.ImportPath())
		outputPath := filepath.Join(*outputBase, p.OutputPath())
		cmd := exec.Command("protoc", append(args, path)...)
		out, err := cmd.CombinedOutput()
		if len(out) > 0 {
			log.Printf(string(out))
		}
		if err != nil {
			log.Println(strings.Join(cmd.Args, " "))
			log.Fatalf("Unable to generate protoc on %s: %v", p.PackageName, err)
		}
		if !*skipGeneratedRewrite {
			if err := RewriteGeneratedGogoProtobufFile(outputPath, p.GoPackageName(), p.HasGoType); err != nil {
				log.Fatalf("Unable to rewrite generated %s: %v", outputPath, err)
			}
			cmd := exec.Command("goimports", "-w", outputPath)
			out, err := cmd.CombinedOutput()
			if len(out) > 0 {
				log.Printf(string(out))
			}
			if err != nil {
				log.Println(strings.Join(cmd.Args, " "))
				log.Fatalf("Unable to rewrite imports for %s: %v", p.PackageName, err)
			}
		}
	}
}
