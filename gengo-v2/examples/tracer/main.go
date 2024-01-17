/*
Copyright 2023 The Kubernetes Authors.

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

// tracer is a trivial gengo/v2 program which prints the various hooks as they
// are called.
package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"k8s.io/gengo/v2/args"
	"k8s.io/gengo/v2/generator"
	"k8s.io/gengo/v2/namer"
	"k8s.io/gengo/v2/types"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	arguments := args.Default()

	// Gengo apps start with arguments.
	if err := arguments.Execute(getNameSystems(), getDefaultNameSystem(), getTargets); err != nil {
		klog.ErrorS(err, "fatal error")
		os.Exit(1)
	}
	klog.V(2).InfoS("completed successfully")
}

func trace(format string, args ...any) {
	if !strings.HasSuffix(format, "\n") {
		format += "\n"
	}
	fmt.Printf("DBG: "+format, args...)
}

// getNameSystems returns the name system used by the generators in this package.
func getNameSystems() namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer("", nil),
	}
}

// getDefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func getDefaultNameSystem() string {
	return "raw"
}

// getTargets is called after the inputs have been loaded.  It is expected to
// examine the provided context and return a list of Packages which will be
// executed further.
func getTargets(c *generator.Context, arguments *args.GeneratorArgs) []generator.Target {
	trace("getTargets")

	// Make sure we don't actually write a file.
	c.FileTypes = map[string]generator.FileType{
		"null": nullFile{},
	}

	targets := []generator.Target{}
	for _, input := range c.Inputs {
		klog.V(2).InfoS("processing", "pkg", input)

		pkg := c.Universe[input]
		if pkg == nil { // e.g. the input had no Go files
			continue
		}

		targets = append(targets, &generator.SimpleTarget{
			PkgName: pkg.Name,
			PkgPath: pkg.Path,
			PkgDir:  pkg.SourcePath,

			// FilterFunc returns true if this Package cares about this type.
			// Each Generator has its own Filter method which will be checked
			// subsequently.  This will be called for every type in every
			// loaded package, not just things in our inputs.
			FilterFunc: func(c *generator.Context, t *types.Type) bool {
				trace("FilterFunc{%s}: %s", pkg.Path, t.String())
				// Only handle types that are under our input dirs.
				for _, input := range c.Inputs {
					if input == t.Name.Package {
						return true
					}
				}
				return false
			},

			// GeneratorsFunc returns a list of Generators, each of which is
			// responsible for a single output file (though multiple generators
			// may write to the same one).
			GeneratorsFunc: func(c *generator.Context) (generators []generator.Generator) {
				trace("GeneratorsFunc{%s}", pkg.Path)
				return []generator.Generator{
					&tracerGenerator{myPackage: pkg},
				}
			},
		})
	}

	return targets
}

// Our custom Generator type.
type tracerGenerator struct {
	myPackage *types.Package
}

var _ generator.Generator = &tracerGenerator{}

func (g *tracerGenerator) Name() string     { return "gengo tracer" }
func (g *tracerGenerator) Filename() string { return "never_written" }
func (g *tracerGenerator) FileType() string { return "null" }

// Filter returns true if this Generator cares about this type.
// This will be called for every type which made it through this Package's
// Filter method.
func (g *tracerGenerator) Filter(_ *generator.Context, t *types.Type) bool {
	trace("tracerGenerator{%s}.Filter: %s", g.myPackage.Path, t.String())
	// Only keep types in this package.
	return t.Name.Package == g.myPackage.Path
}

// Namers returns a set of NameSystems which will be merged with the namers
// provided when executing this package. In case of a name collision, the
// values produced here will win.
func (g *tracerGenerator) Namers(*generator.Context) namer.NameSystems {
	trace("tracerGenerator{%s}.Namers", g.myPackage.Path)
	return nil
}

// PackageVars should return an array of variable lines, suitable to be written
// inside a `var ( ... )` block.
func (g *tracerGenerator) PackageVars(*generator.Context) []string {
	trace("tracerGenerator{%s}.PackageVars", g.myPackage.Path)
	return nil
}

// PackageVars should return an array of const lines, suitable to be written
// inside a `const ( ... )` block.
func (g *tracerGenerator) PackageConsts(*generator.Context) []string {
	trace("tracerGenerator{%s}.PackageConsts", g.myPackage.Path)
	return nil
}

// Init should emit any per-generator code (init functions, etc.)  Per-type
// code can be emitted in GenerateType.
func (g *tracerGenerator) Init(*generator.Context, io.Writer) error {
	trace("tracerGenerator{%s}.Init", g.myPackage.Path)
	return nil
}

// GenerateType should emit code for the specified type.  This will be called
// for every type which made it through this Generator's Filter method.
func (g *tracerGenerator) GenerateType(_ *generator.Context, t *types.Type, _ io.Writer) error {
	trace("tracerGenerator{%s}.GenerateType: %s", g.myPackage.Path, t.String())
	return nil
}

// Imports should return an array of import lines, suitable to be written
// inside an `import ( ... )` block.
func (g *tracerGenerator) Imports(*generator.Context) []string {
	trace("tracerGenerator{%s}.Imports", g.myPackage.Path)
	return nil
}

// Finalize should emit any final per-generator code.
func (g *tracerGenerator) Finalize(*generator.Context, io.Writer) error {
	trace("tracerGenerator{%s}.Finalize", g.myPackage.Path)
	return nil
}

// nullFile represents a file that does not exist and should not be written to.
type nullFile struct{}

var _ generator.FileType = nullFile{}

func (nullFile) AssembleFile(*generator.File, string) error { return nil }
