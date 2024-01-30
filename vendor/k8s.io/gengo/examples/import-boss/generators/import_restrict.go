/*
Copyright 2016 The Kubernetes Authors.

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

// Package generators has the generators for the import-boss utility.
package generators

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"k8s.io/gengo/args"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"
	"sigs.k8s.io/yaml"

	"k8s.io/klog/v2"
)

const (
	goModFile          = "go.mod"
	importBossFileType = "import-boss"
)

// NameSystems returns the name system used by the generators in this package.
func NameSystems() namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer("", nil),
	}
}

// DefaultNameSystem returns the default name system for ordering the types to be
// processed by the generators in this package.
func DefaultNameSystem() string {
	return "raw"
}

// Packages makes the import-boss package definition.
func Packages(c *generator.Context, arguments *args.GeneratorArgs) generator.Packages {
	pkgs := generator.Packages{}
	c.FileTypes = map[string]generator.FileType{
		importBossFileType: importRuleFile{c},
	}

	for _, p := range c.Universe {
		if !inputIncludes(arguments.InputDirs, p) {
			// Don't run on e.g. third party dependencies.
			continue
		}
		savedPackage := p
		pkgs = append(pkgs, &generator.DefaultPackage{
			PackageName: p.Name,
			PackagePath: p.Path,
			Source:      p.SourcePath,
			// GeneratorFunc returns a list of generators. Each generator makes a
			// single file.
			GeneratorFunc: func(c *generator.Context) (generators []generator.Generator) {
				return []generator.Generator{&importRules{
					myPackage: savedPackage,
				}}
			},
			FilterFunc: func(c *generator.Context, t *types.Type) bool {
				return false
			},
		})
	}

	return pkgs
}

// inputIncludes returns true if the given package is a (sub) package of one of
// the InputDirs.
func inputIncludes(inputs []string, p *types.Package) bool {
	// TODO: This does not handle conversion of local paths (./foo) into
	// canonical packages (github.com/example/project/foo).
	for _, input := range inputs {
		// Normalize paths
		input := strings.TrimSuffix(input, "/")
		input = strings.TrimPrefix(input, "./vendor/")
		seek := strings.TrimSuffix(p.Path, "/")

		if input == seek {
			return true
		}
		if strings.HasSuffix(input, "...") {
			input = strings.TrimSuffix(input, "...")
			if strings.HasPrefix(seek+"/", input) {
				return true
			}
		}
	}
	return false
}

// A single import restriction rule.
type Rule struct {
	// All import paths that match this regexp...
	SelectorRegexp string
	// ... must have one of these prefixes ...
	AllowedPrefixes []string
	// ... and must not have one of these prefixes.
	ForbiddenPrefixes []string
}

type InverseRule struct {
	Rule
	// True if the rule is to be applied to transitive imports.
	Transitive bool
}

type fileFormat struct {
	CurrentImports []string

	Rules        []Rule
	InverseRules []InverseRule

	path string
}

func readFile(path string) (*fileFormat, error) {
	currentBytes, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("couldn't read %v: %v", path, err)
	}

	var current fileFormat
	err = yaml.Unmarshal(currentBytes, &current)
	if err != nil {
		return nil, fmt.Errorf("couldn't unmarshal %v: %v", path, err)
	}
	current.path = path
	return &current, nil
}

func writeFile(path string, ff *fileFormat) error {
	raw, err := json.MarshalIndent(ff, "", "\t")
	if err != nil {
		return fmt.Errorf("couldn't format data for file %v.\n%#v", path, ff)
	}
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("couldn't open %v for writing: %v", path, err)
	}
	defer f.Close()
	_, err = f.Write(raw)
	return err
}

// This does the actual checking, since it knows the literal destination file.
type importRuleFile struct {
	context *generator.Context
}

func (irf importRuleFile) AssembleFile(f *generator.File, path string) error {
	return irf.VerifyFile(f, path)
}

// TODO: make a flag to enable this, or expose this information in some other way.
func (importRuleFile) listEntireImportTree(f *generator.File, path string) error {
	// If the file exists, populate its current imports. This is mostly to help
	// humans figure out what they need to fix.
	if _, err := os.Stat(path); err != nil {
		// Ignore packages which haven't opted in by adding an .import-restrictions file.
		return nil
	}

	current, err := readFile(path)
	if err != nil {
		return err
	}

	current.CurrentImports = []string{}
	for v := range f.Imports {
		current.CurrentImports = append(current.CurrentImports, v)
	}
	sort.Strings(current.CurrentImports)

	return writeFile(path, current)
}

// removeLastDir removes the last directory, but leaves the file name
// unchanged. It returns the new path and the removed directory. So:
// "a/b/c/file" -> ("a/b/file", "c")
func removeLastDir(path string) (newPath, removedDir string) {
	dir, file := filepath.Split(path)
	dir = strings.TrimSuffix(dir, string(filepath.Separator))
	return filepath.Join(filepath.Dir(dir), file), filepath.Base(dir)
}

// isGoModRoot checks if a directory is the root directory for a package
// by checking for the existence of a 'go.mod' file in that directory.
func isGoModRoot(path string) bool {
	_, err := os.Stat(filepath.Join(filepath.Dir(path), goModFile))
	return err == nil
}

// recursiveRead collects all '.import-restriction' files, between the current directory,
// and the package root when Go modules are enabled, or $GOPATH/src when they are not.
func recursiveRead(path string) ([]*fileFormat, error) {
	restrictionFiles := make([]*fileFormat, 0)

	for {
		if _, err := os.Stat(path); err == nil {
			rules, err := readFile(path)
			if err != nil {
				return nil, err
			}

			restrictionFiles = append(restrictionFiles, rules)
		}

		nextPath, removedDir := removeLastDir(path)
		if nextPath == path || isGoModRoot(path) || removedDir == "src" {
			break
		}

		path = nextPath
	}

	return restrictionFiles, nil
}

func (irf importRuleFile) VerifyFile(f *generator.File, path string) error {
	restrictionFiles, err := recursiveRead(filepath.Join(f.PackageSourcePath, f.Name))
	if err != nil {
		return fmt.Errorf("error finding rules file: %v", err)
	}

	if err := irf.verifyRules(restrictionFiles, f); err != nil {
		return err
	}

	return irf.verifyInverseRules(restrictionFiles, f)
}

func (irf importRuleFile) verifyRules(restrictionFiles []*fileFormat, f *generator.File) error {
	selectors := make([][]*regexp.Regexp, len(restrictionFiles))
	for i, restrictionFile := range restrictionFiles {
		for _, r := range restrictionFile.Rules {
			re, err := regexp.Compile(r.SelectorRegexp)
			if err != nil {
				return fmt.Errorf("regexp `%s` in file %q doesn't compile: %v", r.SelectorRegexp, restrictionFile.path, err)
			}

			selectors[i] = append(selectors[i], re)
		}
	}

	forbiddenImports := map[string]string{}
	allowedMismatchedImports := []string{}

	for v := range f.Imports {
		explicitlyAllowed := false

	NextRestrictionFiles:
		for i, rules := range restrictionFiles {
			for j, r := range rules.Rules {
				matching := selectors[i][j].MatchString(v)
				klog.V(5).Infof("Checking %v matches %v: %v\n", r.SelectorRegexp, v, matching)
				if !matching {
					continue
				}
				for _, forbidden := range r.ForbiddenPrefixes {
					klog.V(4).Infof("Checking %v against %v\n", v, forbidden)
					if strings.HasPrefix(v, forbidden) {
						forbiddenImports[v] = forbidden
					}
				}
				for _, allowed := range r.AllowedPrefixes {
					klog.V(4).Infof("Checking %v against %v\n", v, allowed)
					if strings.HasPrefix(v, allowed) {
						explicitlyAllowed = true
						break
					}
				}

				if !explicitlyAllowed {
					allowedMismatchedImports = append(allowedMismatchedImports, v)
				} else {
					klog.V(2).Infof("%v importing %v allowed by %v\n", f.PackagePath, v, restrictionFiles[i].path)
					break NextRestrictionFiles
				}
			}
		}
	}

	if len(forbiddenImports) > 0 || len(allowedMismatchedImports) > 0 {
		var errorBuilder strings.Builder
		for i, f := range forbiddenImports {
			fmt.Fprintf(&errorBuilder, "import %v has forbidden prefix %v\n", i, f)
		}
		if len(allowedMismatchedImports) > 0 {
			sort.Sort(sort.StringSlice(allowedMismatchedImports))
			fmt.Fprintf(&errorBuilder, "the following imports did not match any allowed prefix:\n")
			for _, i := range allowedMismatchedImports {
				fmt.Fprintf(&errorBuilder, "  %v\n", i)
			}
		}
		return errors.New(errorBuilder.String())
	}

	return nil
}

// verifyInverseRules checks that all packages that import a package are allowed to import it.
func (irf importRuleFile) verifyInverseRules(restrictionFiles []*fileFormat, f *generator.File) error {
	// compile all Selector regex in all restriction files
	selectors := make([][]*regexp.Regexp, len(restrictionFiles))
	for i, restrictionFile := range restrictionFiles {
		for _, r := range restrictionFile.InverseRules {
			re, err := regexp.Compile(r.SelectorRegexp)
			if err != nil {
				return fmt.Errorf("regexp `%s` in file %q doesn't compile: %v", r.SelectorRegexp, restrictionFile.path, err)
			}

			selectors[i] = append(selectors[i], re)
		}
	}

	directImport := map[string]bool{}
	for _, imp := range irf.context.IncomingImports()[f.PackagePath] {
		directImport[imp] = true
	}

	forbiddenImports := map[string]string{}
	allowedMismatchedImports := []string{}

	for _, v := range irf.context.TransitiveIncomingImports()[f.PackagePath] {
		explicitlyAllowed := false

	NextRestrictionFiles:
		for i, rules := range restrictionFiles {
			for j, r := range rules.InverseRules {
				if !r.Transitive && !directImport[v] {
					continue
				}

				re := selectors[i][j]
				matching := re.MatchString(v)
				klog.V(4).Infof("Checking %v matches %v (importing %v: %v\n", r.SelectorRegexp, v, f.PackagePath, matching)
				if !matching {
					continue
				}
				for _, forbidden := range r.ForbiddenPrefixes {
					klog.V(4).Infof("Checking %v against %v\n", v, forbidden)
					if strings.HasPrefix(v, forbidden) {
						forbiddenImports[v] = forbidden
					}
				}
				for _, allowed := range r.AllowedPrefixes {
					klog.V(4).Infof("Checking %v against %v\n", v, allowed)
					if strings.HasPrefix(v, allowed) {
						explicitlyAllowed = true
						break
					}
				}
				if !explicitlyAllowed {
					allowedMismatchedImports = append(allowedMismatchedImports, v)
				} else {
					klog.V(2).Infof("%v importing %v allowed by %v\n", v, f.PackagePath, restrictionFiles[i].path)
					break NextRestrictionFiles
				}
			}
		}
	}

	if len(forbiddenImports) > 0 || len(allowedMismatchedImports) > 0 {
		var errorBuilder strings.Builder
		for i, f := range forbiddenImports {
			fmt.Fprintf(&errorBuilder, "(inverse): import %v has forbidden prefix %v\n", i, f)
		}
		if len(allowedMismatchedImports) > 0 {
			sort.Sort(sort.StringSlice(allowedMismatchedImports))
			fmt.Fprintf(&errorBuilder, "(inverse): the following imports did not match any allowed prefix:\n")
			for _, i := range allowedMismatchedImports {
				fmt.Fprintf(&errorBuilder, "  %v\n", i)
			}
		}
		return errors.New(errorBuilder.String())
	}

	return nil
}

// importRules produces a file with a set for a single type.
type importRules struct {
	myPackage *types.Package
	imports   namer.ImportTracker
}

var (
	_ = generator.Generator(&importRules{})
	_ = generator.FileType(importRuleFile{})
)

func (r *importRules) Name() string                                                  { return "import rules" }
func (r *importRules) Filter(*generator.Context, *types.Type) bool                   { return false }
func (r *importRules) Namers(*generator.Context) namer.NameSystems                   { return nil }
func (r *importRules) PackageVars(*generator.Context) []string                       { return []string{} }
func (r *importRules) PackageConsts(*generator.Context) []string                     { return []string{} }
func (r *importRules) GenerateType(*generator.Context, *types.Type, io.Writer) error { return nil }
func (r *importRules) Filename() string                                              { return ".import-restrictions" }
func (r *importRules) FileType() string                                              { return importBossFileType }
func (r *importRules) Init(c *generator.Context, w io.Writer) error                  { return nil }
func (r *importRules) Finalize(*generator.Context, io.Writer) error                  { return nil }

func dfsImports(dest *[]string, seen map[string]bool, p *types.Package) {
	for _, p2 := range p.Imports {
		if seen[p2.Path] {
			continue
		}
		seen[p2.Path] = true
		dfsImports(dest, seen, p2)
		*dest = append(*dest, p2.Path)
	}
}

func (r *importRules) Imports(*generator.Context) []string {
	all := []string{}
	dfsImports(&all, map[string]bool{}, r.myPackage)
	return all
}
