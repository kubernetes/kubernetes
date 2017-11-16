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

	"github.com/golang/glog"
)

const (
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
		importBossFileType: importRuleFile{},
	}

	for _, p := range c.Universe {
		if !arguments.InputIncludes(p) {
			// Don't run on e.g. third party dependencies.
			continue
		}
		savedPackage := p
		pkgs = append(pkgs, &generator.DefaultPackage{
			PackageName: p.Name,
			PackagePath: p.Path,
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

// A single import restriction rule.
type Rule struct {
	// All import paths that match this regexp...
	SelectorRegexp string
	// ... must have one of these prefixes ...
	AllowedPrefixes []string
	// ... and must not have one of these prefixes.
	ForbiddenPrefixes []string
}

type fileFormat struct {
	CurrentImports []string

	Rules []Rule
}

func readFile(path string) (*fileFormat, error) {
	currentBytes, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("couldn't read %v: %v", path, err)
	}

	var current fileFormat
	err = json.Unmarshal(currentBytes, &current)
	if err != nil {
		return nil, fmt.Errorf("couldn't unmarshal %v: %v", path, err)
	}
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
type importRuleFile struct{}

func (importRuleFile) AssembleFile(f *generator.File, path string) error {
	return nil
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

// Keep going up a directory until we find an .import-restrictions file.
func recursiveRead(path string) (*fileFormat, string, error) {
	for {
		if _, err := os.Stat(path); err == nil {
			ff, err := readFile(path)
			return ff, path, err
		}

		nextPath, removedDir := removeLastDir(path)
		if nextPath == path || removedDir == "src" {
			break
		}
		path = nextPath
	}
	return nil, "", nil
}

func (importRuleFile) VerifyFile(f *generator.File, path string) error {
	rules, actualPath, err := recursiveRead(path)
	if err != nil {
		return fmt.Errorf("error finding rules file: %v", err)
	}

	if rules == nil {
		// No restrictions on this directory.
		return nil
	}

	for _, r := range rules.Rules {
		re, err := regexp.Compile(r.SelectorRegexp)
		if err != nil {
			return fmt.Errorf("regexp `%s` in file %q doesn't compile: %v", r.SelectorRegexp, actualPath, err)
		}
		for v := range f.Imports {
			glog.V(4).Infof("Checking %v matches %v: %v\n", r.SelectorRegexp, v, re.MatchString(v))
			if !re.MatchString(v) {
				continue
			}
			for _, forbidden := range r.ForbiddenPrefixes {
				glog.V(4).Infof("Checking %v against %v\n", v, forbidden)
				if strings.HasPrefix(v, forbidden) {
					return fmt.Errorf("import %v has forbidden prefix %v", v, forbidden)
				}
			}
			found := false
			for _, allowed := range r.AllowedPrefixes {
				glog.V(4).Infof("Checking %v against %v\n", v, allowed)
				if strings.HasPrefix(v, allowed) {
					found = true
					break
				}
			}
			if !found {
				return fmt.Errorf("import %v did not match any allowed prefix", v)
			}
		}
	}
	if len(rules.Rules) > 0 {
		glog.V(2).Infof("%v passes rules found in %v\n", path, actualPath)
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
