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
	"bytes"
	"fmt"
	"log"
	"os/exec"
	"path/filepath"
	"strings"

	"k8s.io/kubernetes/cmd/libs/go2idl/args"
	"k8s.io/kubernetes/cmd/libs/go2idl/generator"
	"k8s.io/kubernetes/cmd/libs/go2idl/namer"
	"k8s.io/kubernetes/cmd/libs/go2idl/parser"
	"k8s.io/kubernetes/cmd/libs/go2idl/types"

	flag "github.com/spf13/pflag"
)

var (
	commonArgs           = args.Default()
	defaultOutputBase    = args.DefaultSourceTree()
	defaultProtoImport   = filepath.Join(defaultOutputBase, "k8s.io", "kubernetes", "Godeps", "_workspace", "src", "github.com", "gogo", "protobuf", "protobuf")
	packages             = flag.StringP("packages", "p", "+k8s.io/kubernetes/pkg/util/intstr,+k8s.io/kubernetes/pkg/api/resource,+k8s.io/kubernetes/pkg/runtime,k8s.io/kubernetes/pkg/api/unversioned,k8s.io/kubernetes/pkg/api/v1,k8s.io/kubernetes/pkg/apis/extensions/v1beta1", "comma-separated list of directories to get input types from. Directories prefixed with '-' are not generated, directories prefixed with '+' only create types with explicit IDL instructions.")
	outputBase           = flag.StringP("output-base", "o", defaultOutputBase, "Output base; defaults to $GOPATH/src/")
	protoImport          = flag.StringSlice("proto-import", []string{defaultProtoImport}, "The search path for the core protobuf .protos, required, defaults to GODEPS on path.")
	conditional          = flag.String("conditional", "", "An optional Golang build tag condition to add to the generated Go code")
	clean                = flag.Bool("clean", false, "If true, remove all generated files for the specified packages.")
	onlyIDL              = flag.Bool("only-idl", false, "If true, only generate the IDL for each package.")
	skipGeneratedRewrite = flag.Bool("skip-generated-rewrite", false, "If true, skip fixing up the generated.pb.go file (debugging only).")
	dropEmbeddedFields   = flag.String("drop-embedded-fields", "k8s.io/kubernetes/pkg/api/unversioned.TypeMeta", "Comma-delimited list of embedded Go types to omit from generated protobufs")
)

func init() {
	flag.StringVarP(&commonArgs.GoHeaderFilePath, "go-header-file", "h", commonArgs.GoHeaderFilePath, "File containing boilerplate header text. The string YEAR will be replaced with the current 4-digit year.")
	flag.BoolVar(&commonArgs.VerifyOnly, "verify-only", commonArgs.VerifyOnly, "If true, only verify existing output, do not write anything.")
}

const (
	typesKindProtobuf = "Protobuf"
)

func main() {
	flag.Parse()

	if commonArgs.VerifyOnly {
		*onlyIDL = true
		*clean = false
	}

	b := parser.New()
	b.AddBuildTags("proto")

	omitTypes := map[types.Name]struct{}{}
	for _, t := range strings.Split(*dropEmbeddedFields, ",") {
		name := types.Name{}
		if i := strings.LastIndex(t, "."); i != -1 {
			name.Package, name.Name = t[:i], t[i+1:]
		} else {
			name.Name = t
		}
		if len(name.Name) == 0 {
			log.Fatalf("--drop-embedded-types requires names in the form of [GOPACKAGE.]TYPENAME: %v", t)
		}
		omitTypes[name] = struct{}{}
	}

	boilerplate, err := commonArgs.LoadGoBoilerplate()
	if err != nil {
		log.Fatalf("Failed loading boilerplate: %v", err)
	}

	protobufNames := NewProtobufNamer()
	outputPackages := generator.Packages{}
	for _, d := range strings.Split(*packages, ",") {
		generateAllTypes, outputPackage := true, true
		switch {
		case strings.HasPrefix(d, "+"):
			d = d[1:]
			generateAllTypes = false
		case strings.HasPrefix(d, "-"):
			d = d[1:]
			outputPackage = false
		}
		name := protoSafePackage(d)
		parts := strings.SplitN(d, "=", 2)
		if len(parts) > 1 {
			d = parts[0]
			name = parts[1]
		}
		p := newProtobufPackage(d, name, generateAllTypes, omitTypes)
		header := append([]byte{}, boilerplate...)
		header = append(header, p.HeaderText...)
		p.HeaderText = header
		protobufNames.Add(p)
		if outputPackage {
			outputPackages = append(outputPackages, p)
		}
	}

	if !commonArgs.VerifyOnly {
		for _, p := range outputPackages {
			if err := p.(*protobufPackage).Clean(*outputBase); err != nil {
				log.Fatalf("Unable to clean package %s: %v", p.Name(), err)
			}
		}
	}

	if *clean {
		return
	}

	for _, p := range protobufNames.List() {
		if err := b.AddDir(p.Path()); err != nil {
			log.Fatalf("Unable to add directory %q: %v", p.Path(), err)
		}
	}

	c, err := generator.NewContext(
		b,
		namer.NameSystems{
			"public": namer.NewPublicNamer(3),
			"proto":  protobufNames,
		},
		"public",
	)
	c.Verify = commonArgs.VerifyOnly
	c.FileTypes["protoidl"] = protoIDLFileType{}

	if err != nil {
		log.Fatalf("Failed making a context: %v", err)
	}

	if err := protobufNames.AssignTypesToPackages(c); err != nil {
		log.Fatalf("Failed to identify common types: %v", err)
	}

	if err := c.ExecutePackages(*outputBase, outputPackages); err != nil {
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
		for _, s := range *protoImport {
			searchArgs = append(searchArgs, "-I", s)
		}
	}
	args := append(searchArgs, fmt.Sprintf("--gogo_out=%s", *outputBase))

	buf := &bytes.Buffer{}
	if len(*conditional) > 0 {
		fmt.Fprintf(buf, "// +build %s\n\n", *conditional)
	}
	buf.Write(boilerplate)

	for _, outputPackage := range outputPackages {
		p := outputPackage.(*protobufPackage)
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
			if err := RewriteGeneratedGogoProtobufFile(outputPath, p.GoPackageName(), p.HasGoType, buf.Bytes()); err != nil {
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

			cmd = exec.Command("gofmt", "-s", "-w", outputPath)
			out, err = cmd.CombinedOutput()
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
