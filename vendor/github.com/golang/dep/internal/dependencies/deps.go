/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dependencies

import (
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
)

// TODO(yhodique): this file is horrendous, clean it up !

// DepsBuilder provides a way to compute direct dependencies of a package
type DepsBuilder struct {
	Root          string
	Package       string
	LocalPackages []string
	SkipSubdirs   []string
	BlackList     []string
	internal      *internalBuilder
}

type internalBuilder struct {
	Root          string
	Package       string
	BlackList     []string
	LocalPackages map[string]interface{}
	SkipSubdirs   map[string]interface{}
}

func (b *DepsBuilder) compile() *internalBuilder {
	if b.internal != nil {
		return b.internal
	}

	loc := make(map[string]interface{})
	for _, p := range b.LocalPackages {
		loc[p] = nil
	}

	skip := make(map[string]interface{})
	for _, d := range b.SkipSubdirs {
		skip[d] = nil
	}

	return &internalBuilder{
		Root:          b.Root,
		Package:       b.Package,
		BlackList:     b.BlackList,
		LocalPackages: loc,
		SkipSubdirs:   skip,
	}
}

// GetPackageDependencies gives back the list of direct dependencies for the current context
func (b *DepsBuilder) GetPackageDependencies() ([]string, error) {
	return b.compile().getPackageDependencies()
}

type Node struct {
	Imports     []string
	TestImports []string
}

type Graph map[string]Node

func (g Graph) TransitiveClosure(node string) []string {
	visited := make(map[string]bool)
	stack := g[node].Imports
	// consider test dependencies at top-level only
	stack = append(stack, g[node].TestImports...)

	for _, n := range stack {
		visited[n] = true
	}

	for len(stack) > 0 {
		item := stack[0]
		stack = stack[1:]

		next := g[item]
		for _, n := range next.Imports {
			if visited[n] {
				continue
			}

			stack = append(stack, n)
			visited[n] = true
		}
	}

	res := make([]string, 0)
	for n, _ := range visited {
		res = append(res, n)
	}

	return res
}

func (g Graph) RecursiveTransitiveClosure(node string) []string {
	closure := make(map[string]interface{})

	for n, _ := range g {
		if n == node || strings.HasPrefix(n, node+"/") {
			for _, item := range g.TransitiveClosure(n) {
				closure[item] = nil
			}
		}
	}

	res := make([]string, 0)
	for n, _ := range closure {
		if !strings.HasPrefix(n, node+"/") {
			res = append(res, n)
		}
	}
	return res
}

func (b *DepsBuilder) GetFullDependencyGraph() (Graph, error) {
	return b.compile().getFullDependencyGraph()
}

func (b *internalBuilder) getFullDependencyGraph() (Graph, error) {
	dirs := make([]string, 0)
	filepath.Walk(b.Root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info == nil || !info.IsDir() {
			return nil
		}

		base := filepath.Base(path)
		if base == "testdata" || strings.HasPrefix(base, ".") || strings.HasPrefix(base, "_") {
			return filepath.SkipDir
		}

		r, _ := filepath.Rel(b.Root, path)
		r = strings.TrimPrefix(r, "./")
		dirs = append(dirs, r)
		return nil
	})

	m := make(map[string]string)
MAINLOOP:
	for _, d := range dirs {
		if strings.HasPrefix(d, "vendor/") {
			m[d] = strings.TrimPrefix(d, "vendor/")
			continue
		}

		for k, _ := range b.SkipSubdirs {
			srcPfx := filepath.Join(k, "src") + "/"
			if strings.HasPrefix(d, srcPfx) {
				m[d] = strings.TrimPrefix(d, srcPfx)
				continue MAINLOOP
			}

			if strings.HasPrefix(d, k+"/") {
				continue MAINLOOP
			}
		}

		m[d] = filepath.Join(b.Package, d)
	}

	g := make(map[string]Node)
	for k, v := range m {
		subdeps, err := b.packageAllDeps(filepath.Join(b.Root, k))

		if err != nil {
			continue
		}

		g[v] = subdeps
	}

	return Graph(g), nil
}

func (b *internalBuilder) getPackageDependencies() ([]string, error) {
	deps := make(map[string]interface{})

	err := filepath.Walk(b.Root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if info == nil || !info.IsDir() {
			return nil
		}

		base := filepath.Base(path)
		if base == "vendor" || base == "testdata" || strings.HasPrefix(base, ".") || strings.HasPrefix(base, "_") {
			return filepath.SkipDir
		}

		if _, ok := b.SkipSubdirs[path]; ok {
			return filepath.SkipDir
		}

		subdeps, err := b.packageDeps(path)
		if err != nil {
			return err
		}

		for k := range subdeps {
			deps[k] = nil
		}

		return nil
	})

	if err != nil {
		return nil, err
	}

	res := make([]string, 0)
	for k := range deps {
		res = append(res, k)
	}
	return res, nil
}

func (b *internalBuilder) packageDeps(pack string) (map[string]interface{}, error) {
	depsMap := make(map[string]interface{})

	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, pack, nil, parser.ImportsOnly)
	if err != nil {
		return nil, err
	}

	for _, p := range pkgs {
		for _, f := range p.Files {
			for _, i := range f.Imports {
				d := i.Path.Value
				d = d[1 : len(d)-1] // remove quotes
				if b.isExternalDependency(d) {
					depsMap[d] = nil
				}
			}
		}
	}

	return depsMap, nil
}

func (b *internalBuilder) packageAllDeps(pack string) (Node, error) {
	depsMap := make(map[string]interface{})
	testDepsMap := make(map[string]interface{})

	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, pack, nil, parser.ImportsOnly)
	if err != nil {
		return Node{}, err
	}

	if len(pkgs) == 0 {
		return Node{}, fmt.Errorf("not a go package")
	}

	for _, p := range pkgs {
		// TODO(yhodique): that's a gross hack that kinda works only because
		// we're interested only in importable stuff for now.
		if p.Name == "main" && len(pkgs) > 1 {
			continue
		}
		for fname, f := range p.Files {
			for _, i := range f.Imports {
				d := i.Path.Value
				d = d[1 : len(d)-1] // remove quotes
				if b.isBlacklisted(d) {
					continue
				}
				if !b.isStandardDependency(d) {
					if strings.HasSuffix(fname, "_test.go") {
						testDepsMap[d] = nil
					} else {
						depsMap[d] = nil
					}
				}
			}
		}
	}

	res := make([]string, 0)
	for k, _ := range depsMap {
		res = append(res, k)
	}

	testRes := make([]string, 0)
	for k, _ := range testDepsMap {
		if _, ok := depsMap[k]; ok {
			continue
		}
		testRes = append(testRes, k)
	}

	return Node{
		Imports:     res,
		TestImports: testRes,
	}, nil
}

func (b *internalBuilder) isStandardDependency(pack string) bool {
	cpts := strings.Split(pack, "/")
	lead := cpts[0]
	return !strings.Contains(lead, ".")
}

func (b *internalBuilder) isExternalDependency(pack string) bool {
	if strings.HasPrefix(pack, b.Package) {
		return false
	}

	for lp := range b.LocalPackages {
		if strings.HasPrefix(pack, lp) {
			return false
		}
	}

	return !b.isStandardDependency(pack)
}

func (b *internalBuilder) isBlacklisted(pack string) bool {
	for _, p := range b.BlackList {
		if p == pack {
			return true
		}

		if strings.HasPrefix(pack, p+"/") {
			return true
		}

	}
	return false
}
