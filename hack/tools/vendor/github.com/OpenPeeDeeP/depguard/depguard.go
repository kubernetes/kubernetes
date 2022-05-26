package depguard

import (
	"go/build"
	"go/token"
	"io/ioutil"
	"path"
	"sort"
	"strings"

	"github.com/gobwas/glob"
	"golang.org/x/tools/go/loader"
)

// ListType states what kind of list is passed in.
type ListType int

const (
	// LTBlacklist states the list given is a blacklist. (default)
	LTBlacklist ListType = iota
	// LTWhitelist states the list given is a whitelist.
	LTWhitelist
)

// StringToListType makes it easier to turn a string into a ListType.
// It assumes that the string representation is lower case.
var StringToListType = map[string]ListType{
	"allowlist": LTWhitelist,
	"denylist":  LTBlacklist,
	"whitelist": LTWhitelist,
	"blacklist": LTBlacklist,
}

// Issue with the package with PackageName at the Position.
type Issue struct {
	PackageName string
	Position    token.Position
}

// Wrapper for glob patterns that allows for custom negation
type negatableGlob struct {
	g      glob.Glob
	negate bool
}

// Depguard checks imports to make sure they follow the given list and constraints.
type Depguard struct {
	ListType      ListType
	IncludeGoRoot bool

	Packages       []string
	prefixPackages []string
	globPackages   []glob.Glob

	TestPackages       []string
	prefixTestPackages []string
	globTestPackages   []glob.Glob

	IgnoreFileRules       []string
	prefixIgnoreFileRules []string
	globIgnoreFileRules   []negatableGlob

	prefixRoot []string
}

// Run checks for dependencies given the program and validates them against
// Packages.
func (dg *Depguard) Run(config *loader.Config, prog *loader.Program) ([]*Issue, error) {
	// Shortcut execution on an empty blacklist as that means every package is allowed
	if dg.ListType == LTBlacklist && len(dg.Packages) == 0 {
		return nil, nil
	}

	if err := dg.initialize(config, prog); err != nil {
		return nil, err
	}
	directImports, err := dg.createImportMap(prog)
	if err != nil {
		return nil, err
	}
	var issues []*Issue
	for pkg, positions := range directImports {
		for _, pos := range positions {
			if ignoreFile(pos.Filename, dg.prefixIgnoreFileRules, dg.globIgnoreFileRules) {
				continue
			}

			prefixList, globList := dg.prefixPackages, dg.globPackages
			if len(dg.TestPackages) > 0 && strings.Index(pos.Filename, "_test.go") != -1 {
				prefixList, globList = dg.prefixTestPackages, dg.globTestPackages
			}

			if dg.flagIt(pkg, prefixList, globList) {
				issues = append(issues, &Issue{
					PackageName: pkg,
					Position:    pos,
				})
			}
		}
	}
	return issues, nil
}

func (dg *Depguard) initialize(config *loader.Config, prog *loader.Program) error {
	// parse ordinary guarded packages
	for _, pkg := range dg.Packages {
		if strings.ContainsAny(pkg, "!?*[]{}") {
			g, err := glob.Compile(pkg, '/')
			if err != nil {
				return err
			}
			dg.globPackages = append(dg.globPackages, g)
		} else {
			dg.prefixPackages = append(dg.prefixPackages, pkg)
		}
	}

	// Sort the packages so we can have a faster search in the array
	sort.Strings(dg.prefixPackages)

	// parse guarded tests packages
	for _, pkg := range dg.TestPackages {
		if strings.ContainsAny(pkg, "!?*[]{}") {
			g, err := glob.Compile(pkg, '/')
			if err != nil {
				return err
			}
			dg.globTestPackages = append(dg.globTestPackages, g)
		} else {
			dg.prefixTestPackages = append(dg.prefixTestPackages, pkg)
		}
	}

	// Sort the test packages so we can have a faster search in the array
	sort.Strings(dg.prefixTestPackages)

	// parse ignore file rules
	for _, rule := range dg.IgnoreFileRules {
		if strings.ContainsAny(rule, "!?*[]{}") {
			ng := negatableGlob{}
			if strings.HasPrefix(rule, "!") {
				ng.negate = true
				rule = rule[1:] // Strip out the leading '!'
			} else {
				ng.negate = false
			}

			g, err := glob.Compile(rule, '/')
			if err != nil {
				return err
			}
			ng.g = g

			dg.globIgnoreFileRules = append(dg.globIgnoreFileRules, ng)
		} else {
			dg.prefixIgnoreFileRules = append(dg.prefixIgnoreFileRules, rule)
		}
	}

	// Sort the rules so we can have a faster search in the array
	sort.Strings(dg.prefixIgnoreFileRules)

	if !dg.IncludeGoRoot {
		var err error
		dg.prefixRoot, err = listRootPrefixs(config.Build)
		if err != nil {
			return err
		}
	}

	return nil
}

func (dg *Depguard) createImportMap(prog *loader.Program) (map[string][]token.Position, error) {
	importMap := make(map[string][]token.Position)
	// For the directly imported packages
	for _, imported := range prog.InitialPackages() {
		// Go through their files
		for _, file := range imported.Files {
			// And populate a map of all direct imports and their positions
			// This will filter out GoRoot depending on the Depguard.IncludeGoRoot
			for _, fileImport := range file.Imports {
				fileImportPath := cleanBasicLitString(fileImport.Path.Value)
				if !dg.IncludeGoRoot && dg.isRoot(fileImportPath) {
					continue
				}
				position := prog.Fset.Position(fileImport.Pos())
				positions, found := importMap[fileImportPath]
				if !found {
					importMap[fileImportPath] = []token.Position{
						position,
					}
					continue
				}
				importMap[fileImportPath] = append(positions, position)
			}
		}
	}
	return importMap, nil
}

func ignoreFile(filename string, prefixList []string, negatableGlobList []negatableGlob) bool {
	if strInPrefixList(filename, prefixList) {
		return true
	}
	return strInNegatableGlobList(filename, negatableGlobList)
}

func pkgInList(pkg string, prefixList []string, globList []glob.Glob) bool {
	if strInPrefixList(pkg, prefixList) {
		return true
	}
	return strInGlobList(pkg, globList)
}

func strInPrefixList(str string, prefixList []string) bool {
	// Idx represents where in the prefix slice the passed in string would go
	// when sorted. -1 Just means that it would be at the very front of the slice.
	idx := sort.Search(len(prefixList), func(i int) bool {
		return prefixList[i] > str
	}) - 1
	// This means that the string passed in has no way to be prefixed by anything
	// in the prefix list as it is already smaller then everything
	if idx == -1 {
		return false
	}
	return strings.HasPrefix(str, prefixList[idx])
}

func strInGlobList(str string, globList []glob.Glob) bool {
	for _, g := range globList {
		if g.Match(str) {
			return true
		}
	}
	return false
}

func strInNegatableGlobList(str string, negatableGlobList []negatableGlob) bool {
	for _, ng := range negatableGlobList {
		// Return true when:
		//  - Match is true and negate is off
		//  - Match is false and negate is on
		if ng.g.Match(str) != ng.negate {
			return true
		}
	}
	return false
}

// InList | WhiteList | BlackList
//   y   |           |     x
//   n   |     x     |
func (dg *Depguard) flagIt(pkg string, prefixList []string, globList []glob.Glob) bool {
	return pkgInList(pkg, prefixList, globList) == (dg.ListType == LTBlacklist)
}

func cleanBasicLitString(value string) string {
	return strings.Trim(value, "\"\\")
}

// We can do this as all imports that are not root are either prefixed with a domain
// or prefixed with `./` or `/` to dictate it is a local file reference
func listRootPrefixs(buildCtx *build.Context) ([]string, error) {
	if buildCtx == nil {
		buildCtx = &build.Default
	}
	root := path.Join(buildCtx.GOROOT, "src")
	fs, err := ioutil.ReadDir(root)
	if err != nil {
		return nil, err
	}
	var pkgPrefix []string
	for _, f := range fs {
		if !f.IsDir() {
			continue
		}
		pkgPrefix = append(pkgPrefix, f.Name())
	}
	return pkgPrefix, nil
}

func (dg *Depguard) isRoot(importPath string) bool {
	// Idx represents where in the package slice the passed in package would go
	// when sorted. -1 Just means that it would be at the very front of the slice.
	idx := sort.Search(len(dg.prefixRoot), func(i int) bool {
		return dg.prefixRoot[i] > importPath
	}) - 1
	// This means that the package passed in has no way to be prefixed by anything
	// in the package list as it is already smaller then everything
	if idx == -1 {
		return false
	}
	// if it is prefixed by a root prefix we need to check if it is an exact match
	// or prefix with `/` as this could return false posative if the domain was
	// `archive.com` for example as `archive` is a go root package.
	if strings.HasPrefix(importPath, dg.prefixRoot[idx]) {
		return strings.HasPrefix(importPath, dg.prefixRoot[idx]+"/") || importPath == dg.prefixRoot[idx]
	}
	return false
}
