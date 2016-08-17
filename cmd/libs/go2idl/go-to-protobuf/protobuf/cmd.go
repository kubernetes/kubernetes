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
package protobuf

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

type Generator struct {
	Common               args.GeneratorArgs
	Packages             string
	OutputBase           string
	ProtoImport          []string
	Conditional          string
	Clean                bool
	OnlyIDL              bool
	KeepGogoproto        bool
	SkipGeneratedRewrite bool
	DropEmbeddedFields   string
}

func New() *Generator {
	sourceTree := args.DefaultSourceTree()
	common := args.GeneratorArgs{
		OutputBase:       sourceTree,
		GoHeaderFilePath: filepath.Join(sourceTree, "k8s.io/kubernetes/hack/boilerplate/boilerplate.go.txt"),
	}
	defaultProtoImport := filepath.Join(sourceTree, "k8s.io", "kubernetes", "vendor", "github.com", "gogo", "protobuf", "protobuf")
	return &Generator{
		Common:      common,
		OutputBase:  sourceTree,
		ProtoImport: []string{defaultProtoImport},
		Packages: strings.Join([]string{
			`+k8s.io/kubernetes/pkg/util/intstr`,
			`+k8s.io/kubernetes/pkg/api/resource`,
			`+k8s.io/kubernetes/pkg/runtime`,
			`+k8s.io/kubernetes/pkg/watch/versioned`,
			`k8s.io/kubernetes/pkg/api/unversioned`,
			`k8s.io/kubernetes/pkg/api/v1`,
			`k8s.io/kubernetes/pkg/apis/policy/v1alpha1`,
			`k8s.io/kubernetes/pkg/apis/extensions/v1beta1`,
			`k8s.io/kubernetes/pkg/apis/autoscaling/v1`,
			`k8s.io/kubernetes/pkg/apis/authorization/v1beta1`,
			`k8s.io/kubernetes/pkg/apis/batch/v1`,
			`k8s.io/kubernetes/pkg/apis/batch/v2alpha1`,
			`k8s.io/kubernetes/pkg/apis/apps/v1alpha1`,
			`k8s.io/kubernetes/pkg/apis/authentication/v1beta1`,
			`k8s.io/kubernetes/pkg/apis/rbac/v1alpha1`,
			`k8s.io/kubernetes/federation/apis/federation/v1beta1`,
			`k8s.io/kubernetes/pkg/apis/certificates/v1alpha1`,
		}, ","),
		DropEmbeddedFields: "k8s.io/kubernetes/pkg/api/unversioned.TypeMeta",
	}
}

func (g *Generator) BindFlags(flag *flag.FlagSet) {
	flag.StringVarP(&g.Common.GoHeaderFilePath, "go-header-file", "h", g.Common.GoHeaderFilePath, "File containing boilerplate header text. The string YEAR will be replaced with the current 4-digit year.")
	flag.BoolVar(&g.Common.VerifyOnly, "verify-only", g.Common.VerifyOnly, "If true, only verify existing output, do not write anything.")
	flag.StringVarP(&g.Packages, "packages", "p", g.Packages, "comma-separated list of directories to get input types from. Directories prefixed with '-' are not generated, directories prefixed with '+' only create types with explicit IDL instructions.")
	flag.StringVarP(&g.OutputBase, "output-base", "o", g.OutputBase, "Output base; defaults to $GOPATH/src/")
	flag.StringSliceVar(&g.ProtoImport, "proto-import", g.ProtoImport, "The search path for the core protobuf .protos, required, defaults to GODEPS on path.")
	flag.StringVar(&g.Conditional, "conditional", g.Conditional, "An optional Golang build tag condition to add to the generated Go code")
	flag.BoolVar(&g.Clean, "clean", g.Clean, "If true, remove all generated files for the specified Packages.")
	flag.BoolVar(&g.OnlyIDL, "only-idl", g.OnlyIDL, "If true, only generate the IDL for each package.")
	flag.BoolVar(&g.KeepGogoproto, "keep-gogoproto", g.KeepGogoproto, "If true, the generated IDL will contain gogoprotobuf extensions which are normally removed")
	flag.BoolVar(&g.SkipGeneratedRewrite, "skip-generated-rewrite", g.SkipGeneratedRewrite, "If true, skip fixing up the generated.pb.go file (debugging only).")
	flag.StringVar(&g.DropEmbeddedFields, "drop-embedded-fields", g.DropEmbeddedFields, "Comma-delimited list of embedded Go types to omit from generated protobufs")
}

func Run(g *Generator) {
	if g.Common.VerifyOnly {
		g.OnlyIDL = true
		g.Clean = false
	}

	b := parser.New()
	b.AddBuildTags("proto")

	omitTypes := map[types.Name]struct{}{}
	for _, t := range strings.Split(g.DropEmbeddedFields, ",") {
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

	boilerplate, err := g.Common.LoadGoBoilerplate()
	if err != nil {
		log.Fatalf("Failed loading boilerplate: %v", err)
	}

	protobufNames := NewProtobufNamer()
	outputPackages := generator.Packages{}
	for _, d := range strings.Split(g.Packages, ",") {
		generateAllTypes, outputPackage := true, true
		switch {
		case strings.HasPrefix(d, "+"):
			d = d[1:]
			generateAllTypes = false
		case strings.HasPrefix(d, "-"):
			d = d[1:]
			outputPackage = false
		}
		if strings.Contains(d, "-") {
			log.Fatalf("Package names must be valid protobuf package identifiers, which allow only [a-z0-9_]: %s", d)
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

	if !g.Common.VerifyOnly {
		for _, p := range outputPackages {
			if err := p.(*protobufPackage).Clean(g.OutputBase); err != nil {
				log.Fatalf("Unable to clean package %s: %v", p.Name(), err)
			}
		}
	}

	if g.Clean {
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
	if err != nil {
		log.Fatalf("Failed making a context: %v", err)
	}

	c.Verify = g.Common.VerifyOnly
	c.FileTypes["protoidl"] = NewProtoFile()

	if err := protobufNames.AssignTypesToPackages(c); err != nil {
		log.Fatalf("Failed to identify Common types: %v", err)
	}

	if err := c.ExecutePackages(g.OutputBase, outputPackages); err != nil {
		log.Fatalf("Failed executing generator: %v", err)
	}

	if g.OnlyIDL {
		return
	}

	if _, err := exec.LookPath("protoc"); err != nil {
		log.Fatalf("Unable to find 'protoc': %v", err)
	}

	searchArgs := []string{"-I", ".", "-I", g.OutputBase}
	if len(g.ProtoImport) != 0 {
		for _, s := range g.ProtoImport {
			searchArgs = append(searchArgs, "-I", s)
		}
	}
	args := append(searchArgs, fmt.Sprintf("--gogo_out=%s", g.OutputBase))

	buf := &bytes.Buffer{}
	if len(g.Conditional) > 0 {
		fmt.Fprintf(buf, "// +build %s\n\n", g.Conditional)
	}
	buf.Write(boilerplate)

	for _, outputPackage := range outputPackages {
		p := outputPackage.(*protobufPackage)

		path := filepath.Join(g.OutputBase, p.ImportPath())
		outputPath := filepath.Join(g.OutputBase, p.OutputPath())

		// generate the gogoprotobuf protoc
		cmd := exec.Command("protoc", append(args, path)...)
		out, err := cmd.CombinedOutput()
		if len(out) > 0 {
			log.Printf(string(out))
		}
		if err != nil {
			log.Println(strings.Join(cmd.Args, " "))
			log.Fatalf("Unable to generate protoc on %s: %v", p.PackageName, err)
		}

		if g.SkipGeneratedRewrite {
			continue
		}

		// alter the generated protobuf file to remove the generated types (but leave the serializers) and rewrite the
		// package statement to match the desired package name
		if err := RewriteGeneratedGogoProtobufFile(outputPath, p.ExtractGeneratedType, p.OptionalTypeName, buf.Bytes()); err != nil {
			log.Fatalf("Unable to rewrite generated %s: %v", outputPath, err)
		}

		// sort imports
		cmd = exec.Command("goimports", "-w", outputPath)
		out, err = cmd.CombinedOutput()
		if len(out) > 0 {
			log.Printf(string(out))
		}
		if err != nil {
			log.Println(strings.Join(cmd.Args, " "))
			log.Fatalf("Unable to rewrite imports for %s: %v", p.PackageName, err)
		}

		// format and simplify the generated file
		cmd = exec.Command("gofmt", "-s", "-w", outputPath)
		out, err = cmd.CombinedOutput()
		if len(out) > 0 {
			log.Printf(string(out))
		}
		if err != nil {
			log.Println(strings.Join(cmd.Args, " "))
			log.Fatalf("Unable to apply gofmt for %s: %v", p.PackageName, err)
		}
	}

	if g.SkipGeneratedRewrite {
		return
	}

	if !g.KeepGogoproto {
		// generate, but do so without gogoprotobuf extensions
		for _, outputPackage := range outputPackages {
			p := outputPackage.(*protobufPackage)
			p.OmitGogo = true
		}
		if err := c.ExecutePackages(g.OutputBase, outputPackages); err != nil {
			log.Fatalf("Failed executing generator: %v", err)
		}
	}

	for _, outputPackage := range outputPackages {
		p := outputPackage.(*protobufPackage)

		if len(p.StructTags) == 0 {
			continue
		}

		pattern := filepath.Join(g.OutputBase, p.PackagePath, "*.go")
		files, err := filepath.Glob(pattern)
		if err != nil {
			log.Fatalf("Can't glob pattern %q: %v", pattern, err)
		}

		for _, s := range files {
			if strings.HasSuffix(s, "_test.go") {
				continue
			}
			if err := RewriteTypesWithProtobufStructTags(s, p.StructTags); err != nil {
				log.Fatalf("Unable to rewrite with struct tags %s: %v", s, err)
			}
		}
	}
}
