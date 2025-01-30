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
	"sort"
	"strings"

	flag "github.com/spf13/pflag"

	"k8s.io/gengo/v2"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/parser"
	"k8s.io/gengo/v2/types"
)

type Generator struct {
	GoHeaderFile         string
	APIMachineryPackages string
	Packages             string
	OutputDir            string
	ProtoImport          []string
	Conditional          string
	Clean                bool
	OnlyIDL              bool
	KeepGogoproto        bool
	SkipGeneratedRewrite bool
	DropEmbeddedFields   string
}

func New() *Generator {
	defaultSourceTree := "."
	return &Generator{
		OutputDir: defaultSourceTree,
		APIMachineryPackages: strings.Join([]string{
			`+k8s.io/apimachinery/pkg/util/intstr`,
			`+k8s.io/apimachinery/pkg/api/resource`,
			`+k8s.io/apimachinery/pkg/runtime/schema`,
			`+k8s.io/apimachinery/pkg/runtime`,
			`k8s.io/apimachinery/pkg/apis/meta/v1`,
			`k8s.io/apimachinery/pkg/apis/meta/v1beta1`,
			`k8s.io/apimachinery/pkg/apis/testapigroup/v1`,
		}, ","),
		Packages:           "",
		DropEmbeddedFields: "k8s.io/apimachinery/pkg/apis/meta/v1.TypeMeta",
	}
}

func (g *Generator) BindFlags(flag *flag.FlagSet) {
	flag.StringVarP(&g.GoHeaderFile, "go-header-file", "h", "", "File containing boilerplate header text. The string YEAR will be replaced with the current 4-digit year.")
	flag.StringVarP(&g.Packages, "packages", "p", g.Packages, "comma-separated list of directories to get input types from. Directories prefixed with '-' are not generated, directories prefixed with '+' only create types with explicit IDL instructions.")
	flag.StringVar(&g.APIMachineryPackages, "apimachinery-packages", g.APIMachineryPackages, "comma-separated list of directories to get apimachinery input types from which are needed by any API. Directories prefixed with '-' are not generated, directories prefixed with '+' only create types with explicit IDL instructions.")
	flag.StringVar(&g.OutputDir, "output-dir", g.OutputDir, "The base directory under which to generate results.")
	flag.StringSliceVar(&g.ProtoImport, "proto-import", g.ProtoImport, "A search path for imported protobufs (may be repeated).")
	flag.StringVar(&g.Conditional, "conditional", g.Conditional, "An optional Golang build tag condition to add to the generated Go code")
	flag.BoolVar(&g.Clean, "clean", g.Clean, "If true, remove all generated files for the specified Packages.")
	flag.BoolVar(&g.OnlyIDL, "only-idl", g.OnlyIDL, "If true, only generate the IDL for each package.")
	flag.BoolVar(&g.KeepGogoproto, "keep-gogoproto", g.KeepGogoproto, "If true, the generated IDL will contain gogoprotobuf extensions which are normally removed")
	flag.BoolVar(&g.SkipGeneratedRewrite, "skip-generated-rewrite", g.SkipGeneratedRewrite, "If true, skip fixing up the generated.pb.go file (debugging only).")
	flag.StringVar(&g.DropEmbeddedFields, "drop-embedded-fields", g.DropEmbeddedFields, "Comma-delimited list of embedded Go types to omit from generated protobufs")
}

// This roughly models gengo/v2.Execute.
func Run(g *Generator) {
	// Roughly models gengo/v2.newBuilder.

	p := parser.NewWithOptions(parser.Options{BuildTags: []string{"proto"}})

	var allInputs []string
	if len(g.APIMachineryPackages) != 0 {
		allInputs = append(allInputs, strings.Split(g.APIMachineryPackages, ",")...)
	}
	if len(g.Packages) != 0 {
		allInputs = append(allInputs, strings.Split(g.Packages, ",")...)
	}
	if len(allInputs) == 0 {
		log.Fatalf("Both apimachinery-packages and packages are empty. At least one package must be specified.")
	}

	// Build up a list of packages to load from all the inputs.  Track the
	// special modifiers for each.  NOTE: This does not support pkg/... syntax.
	type modifier struct {
		allTypes bool
		output   bool
		name     string
	}
	inputModifiers := map[string]modifier{}
	packages := make([]string, 0, len(allInputs))

	for _, d := range allInputs {
		modifier := modifier{allTypes: true, output: true}

		switch {
		case strings.HasPrefix(d, "+"):
			d = d[1:]
			modifier.allTypes = false
		case strings.HasPrefix(d, "-"):
			d = d[1:]
			modifier.output = false
		}
		name := protoSafePackage(d)
		parts := strings.SplitN(d, "=", 2)
		if len(parts) > 1 {
			d = parts[0]
			name = parts[1]
		}
		modifier.name = name

		packages = append(packages, d)
		inputModifiers[d] = modifier
	}

	// Load all the packages at once.
	if err := p.LoadPackages(packages...); err != nil {
		log.Fatalf("Unable to load packages: %v", err)
	}

	c, err := generator.NewContext(
		p,
		namer.NameSystems{
			"public": namer.NewPublicNamer(3),
		},
		"public",
	)
	if err != nil {
		log.Fatalf("Failed making a context: %v", err)
	}

	c.FileTypes["protoidl"] = NewProtoFile()

	// Roughly models gengo/v2.Execute calling the
	// tool-provided Targets() callback.

	boilerplate, err := gengo.GoBoilerplate(g.GoHeaderFile, "", "")
	if err != nil {
		log.Fatalf("Failed loading boilerplate (consider using the go-header-file flag): %v", err)
	}

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

	protobufNames := NewProtobufNamer()
	outputPackages := []generator.Target{}
	nonOutputPackages := map[string]struct{}{}

	for _, input := range c.Inputs {
		mod, found := inputModifiers[input]
		if !found {
			log.Fatalf("BUG: can't find input modifiers for %q", input)
		}
		pkg := c.Universe[input]
		protopkg := newProtobufPackage(pkg.Path, pkg.Dir, mod.name, mod.allTypes, omitTypes)
		header := append([]byte{}, boilerplate...)
		header = append(header, protopkg.HeaderComment...)
		protopkg.HeaderComment = header
		protobufNames.Add(protopkg)
		if mod.output {
			outputPackages = append(outputPackages, protopkg)
		} else {
			nonOutputPackages[mod.name] = struct{}{}
		}
	}
	c.Namers["proto"] = protobufNames

	for _, p := range outputPackages {
		if err := p.(*protobufPackage).Clean(); err != nil {
			log.Fatalf("Unable to clean package %s: %v", p.Name(), err)
		}
	}

	if g.Clean {
		return
	}

	// order package by imports, importees first
	deps := deps(c, protobufNames.packages)
	order, err := importOrder(deps)
	if err != nil {
		log.Fatalf("Failed to order packages by imports: %v", err)
	}
	topologicalPos := map[string]int{}
	for i, p := range order {
		topologicalPos[p] = i
	}
	sort.Sort(positionOrder{topologicalPos, protobufNames.packages})

	var localOutputPackages []generator.Target
	for _, p := range protobufNames.packages {
		if _, ok := nonOutputPackages[p.Name()]; ok {
			// if we're not outputting the package, don't include it in either package list
			continue
		}
		localOutputPackages = append(localOutputPackages, p)
	}

	if err := protobufNames.AssignTypesToPackages(c); err != nil {
		log.Fatalf("Failed to identify Common types: %v", err)
	}

	if err := c.ExecuteTargets(localOutputPackages); err != nil {
		log.Fatalf("Failed executing local generator: %v", err)
	}

	if g.OnlyIDL {
		return
	}

	if _, err := exec.LookPath("protoc"); err != nil {
		log.Fatalf("Unable to find 'protoc': %v", err)
	}

	searchArgs := []string{"-I", ".", "-I", g.OutputDir}
	if len(g.ProtoImport) != 0 {
		for _, s := range g.ProtoImport {
			searchArgs = append(searchArgs, "-I", s)
		}
	}
	// Despite docs saying that `--gogo_out=paths=source_relative:.` will
	// output the .pb.go file to the same directory as the .proto file, it
	// doesn't. Given example.com/foo/bar.proto (found in one of the -I paths
	// above), the output becomes
	// $output_base/example.com/foo/example.com/foo/bar.pb.go - basically
	// useless.  Users should set the output-dir to a single dir under which
	// all the packages in question live (e.g. staging/src in kubernetes).
	// Alternately, we could generate into a temp path and then move the
	// resulting file back to the input dir, but that seems brittle in other
	// ways.
	args := searchArgs
	args = append(args, fmt.Sprintf("--gogo_out=%s", g.OutputDir))

	buf := &bytes.Buffer{}
	if len(g.Conditional) > 0 {
		fmt.Fprintf(buf, "// +build %s\n\n", g.Conditional)
	}
	buf.Write(boilerplate)

	for _, outputPackage := range outputPackages {
		p := outputPackage.(*protobufPackage)

		path := filepath.Join(g.OutputDir, p.ImportPath())
		outputPath := filepath.Join(g.OutputDir, p.OutputPath())

		// generate the gogoprotobuf protoc
		cmd := exec.Command("protoc", append(args, path)...)
		out, err := cmd.CombinedOutput()
		if err != nil {
			log.Println(strings.Join(cmd.Args, " "))
			log.Println(string(out))
			log.Fatalf("Unable to run protoc on %s: %v", p.Name(), err)
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
			log.Print(string(out))
		}
		if err != nil {
			log.Println(strings.Join(cmd.Args, " "))
			log.Fatalf("Unable to rewrite imports for %s: %v", p.Name(), err)
		}

		// format and simplify the generated file
		cmd = exec.Command("gofmt", "-s", "-w", outputPath)
		out, err = cmd.CombinedOutput()
		if len(out) > 0 {
			log.Print(string(out))
		}
		if err != nil {
			log.Println(strings.Join(cmd.Args, " "))
			log.Fatalf("Unable to apply gofmt for %s: %v", p.Name(), err)
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
		if err := c.ExecuteTargets(localOutputPackages); err != nil {
			log.Fatalf("Failed executing local generator: %v", err)
		}
	}

	for _, outputPackage := range outputPackages {
		p := outputPackage.(*protobufPackage)

		if len(p.StructTags) == 0 {
			continue
		}

		pattern := filepath.Join(g.OutputDir, p.Path(), "*.go")
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

func deps(c *generator.Context, pkgs []*protobufPackage) map[string][]string {
	ret := map[string][]string{}
	for _, p := range pkgs {
		pkg, ok := c.Universe[p.Path()]
		if !ok {
			log.Fatalf("Unrecognized package: %s", p.Path())
		}

		for _, d := range pkg.Imports {
			ret[p.Path()] = append(ret[p.Path()], d.Path)
		}
	}
	return ret
}

// given a set of pkg->[]deps, return the order that ensures all deps are processed before the things that depend on them
func importOrder(deps map[string][]string) ([]string, error) {
	// add all nodes and edges
	var remainingNodes = map[string]struct{}{}
	var graph = map[edge]struct{}{}
	for to, froms := range deps {
		remainingNodes[to] = struct{}{}
		for _, from := range froms {
			remainingNodes[from] = struct{}{}
			graph[edge{from: from, to: to}] = struct{}{}
		}
	}

	// find initial nodes without any dependencies
	sorted := findAndRemoveNodesWithoutDependencies(remainingNodes, graph)
	for i := 0; i < len(sorted); i++ {
		node := sorted[i]
		removeEdgesFrom(node, graph)
		sorted = append(sorted, findAndRemoveNodesWithoutDependencies(remainingNodes, graph)...)
	}
	if len(remainingNodes) > 0 {
		return nil, fmt.Errorf("cycle: remaining nodes: %#v, remaining edges: %#v", remainingNodes, graph)
	}
	// for _, n := range sorted {
	// 	 fmt.Println("topological order", n)
	// }
	return sorted, nil
}

// edge describes a from->to relationship in a graph
type edge struct {
	from string
	to   string
}

// findAndRemoveNodesWithoutDependencies finds nodes in the given set which are not pointed to by any edges in the graph,
// removes them from the set of nodes, and returns them in sorted order
func findAndRemoveNodesWithoutDependencies(nodes map[string]struct{}, graph map[edge]struct{}) []string {
	roots := []string{}
	// iterate over all nodes as potential "to" nodes
	for node := range nodes {
		incoming := false
		// iterate over all remaining edges
		for edge := range graph {
			// if there's any edge to the node we care about, it's not a root
			if edge.to == node {
				incoming = true
				break
			}
		}
		// if there are no incoming edges, remove from the set of remaining nodes and add to our results
		if !incoming {
			delete(nodes, node)
			roots = append(roots, node)
		}
	}
	sort.Strings(roots)
	return roots
}

// removeEdgesFrom removes any edges from the graph where edge.from == node
func removeEdgesFrom(node string, graph map[edge]struct{}) {
	for edge := range graph {
		if edge.from == node {
			delete(graph, edge)
		}
	}
}

type positionOrder struct {
	pos      map[string]int
	elements []*protobufPackage
}

func (o positionOrder) Len() int {
	return len(o.elements)
}

func (o positionOrder) Less(i, j int) bool {
	return o.pos[o.elements[i].Path()] < o.pos[o.elements[j].Path()]
}

func (o positionOrder) Swap(i, j int) {
	o.elements[i], o.elements[j] = o.elements[j], o.elements[i]
}
