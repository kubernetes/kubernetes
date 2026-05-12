// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"context"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/internal/gopathwalk"
)

// ProcessEnvSource implements the [Source] interface using the legacy
// [ProcessEnv] abstraction.
type ProcessEnvSource struct {
	env      *ProcessEnv
	srcDir   string
	filename string
	pkgName  string
}

// NewProcessEnvSource returns a [ProcessEnvSource] wrapping the given
// env, to be used for fixing imports in the file with name filename in package
// named pkgName.
func NewProcessEnvSource(env *ProcessEnv, filename, pkgName string) (*ProcessEnvSource, error) {
	abs, err := filepath.Abs(filename)
	if err != nil {
		return nil, err
	}
	srcDir := filepath.Dir(abs)
	return &ProcessEnvSource{
		env:      env,
		srcDir:   srcDir,
		filename: filename,
		pkgName:  pkgName,
	}, nil
}

func (s *ProcessEnvSource) LoadPackageNames(ctx context.Context, srcDir string, unknown []string) (map[string]string, error) {
	r, err := s.env.GetResolver()
	if err != nil {
		return nil, err
	}
	return r.loadPackageNames(unknown, srcDir)
}

func (s *ProcessEnvSource) ResolveReferences(ctx context.Context, filename string, refs map[string]map[string]bool) ([]*Result, error) {
	var mu sync.Mutex
	found := make(map[string][]pkgDistance)
	callback := &scanCallback{
		rootFound: func(gopathwalk.Root) bool {
			return true // We want everything.
		},
		dirFound: func(pkg *pkg) bool {
			return pkgIsCandidate(filename, refs, pkg)
		},
		packageNameLoaded: func(pkg *pkg) bool {
			if _, want := refs[pkg.packageName]; !want {
				return false
			}
			if pkg.dir == s.srcDir && s.pkgName == pkg.packageName {
				// The candidate is in the same directory and has the
				// same package name. Don't try to import ourselves.
				return false
			}
			if !CanUse(filename, pkg.dir) {
				return false
			}
			mu.Lock()
			defer mu.Unlock()
			found[pkg.packageName] = append(found[pkg.packageName], pkgDistance{pkg, distance(s.srcDir, pkg.dir)})
			return false // We'll do our own loading after we sort.
		},
	}
	resolver, err := s.env.GetResolver()
	if err != nil {
		return nil, err
	}
	if err := resolver.scan(ctx, callback); err != nil {
		return nil, err
	}

	g, ctx := errgroup.WithContext(ctx)

	searcher := symbolSearcher{
		logf:        s.env.logf,
		srcDir:      s.srcDir,
		xtest:       strings.HasSuffix(s.pkgName, "_test"),
		loadExports: resolver.loadExports,
	}

	var resultMu sync.Mutex
	results := make(map[string]*Result, len(refs))
	for pkgName, symbols := range refs {
		g.Go(func() error {
			found, err := searcher.search(ctx, found[pkgName], pkgName, symbols)
			if err != nil {
				return err
			}
			if found == nil {
				return nil // No matching package.
			}

			imp := &ImportInfo{
				ImportPath: found.importPathShort,
			}
			pkg := &PackageInfo{
				Name:    pkgName,
				Exports: symbols,
			}
			resultMu.Lock()
			results[pkgName] = &Result{Import: imp, Package: pkg}
			resultMu.Unlock()
			return nil
		})
	}
	if err := g.Wait(); err != nil {
		return nil, err
	}
	var ans []*Result
	for _, x := range results {
		ans = append(ans, x)
	}
	return ans, nil
}
