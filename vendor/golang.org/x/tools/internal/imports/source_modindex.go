// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package imports

import (
	"context"
	"sync"
	"time"

	"golang.org/x/tools/internal/modindex"
)

// This code is here rather than in the modindex package
// to avoid import loops

// TODO(adonovan): this code is only used by a test in this package.
// Can we delete it? Or is there a plan to call NewIndexSource from
// cmd/goimports?

// implements Source using modindex, so only for module cache.
//
// this is perhaps over-engineered. A new Index is read at first use.
// And then Update is called after every 15 minutes, and a new Index
// is read if the index changed. It is not clear the Mutex is needed.
type IndexSource struct {
	modcachedir string
	mu          sync.Mutex
	index       *modindex.Index // (access via getIndex)
	expires     time.Time
}

// create a new Source. Called from NewView in cache/session.go.
func NewIndexSource(cachedir string) *IndexSource {
	return &IndexSource{modcachedir: cachedir}
}

func (s *IndexSource) LoadPackageNames(ctx context.Context, srcDir string, paths []ImportPath) (map[ImportPath]PackageName, error) {
	/// This is used by goimports to resolve the package names of imports of the
	// current package, which is irrelevant for the module cache.
	return nil, nil
}

func (s *IndexSource) ResolveReferences(ctx context.Context, filename string, missing References) ([]*Result, error) {
	index, err := s.getIndex()
	if err != nil {
		return nil, err
	}
	var cs []modindex.Candidate
	for pkg, nms := range missing {
		for nm := range nms {
			x := index.Lookup(pkg, nm, false)
			cs = append(cs, x...)
		}
	}
	found := make(map[string]*Result)
	for _, c := range cs {
		var x *Result
		if x = found[c.ImportPath]; x == nil {
			x = &Result{
				Import: &ImportInfo{
					ImportPath: c.ImportPath,
					Name:       "",
				},
				Package: &PackageInfo{
					Name:    c.PkgName,
					Exports: make(map[string]bool),
				},
			}
			found[c.ImportPath] = x
		}
		x.Package.Exports[c.Name] = true
	}
	var ans []*Result
	for _, x := range found {
		ans = append(ans, x)
	}
	return ans, nil
}

func (s *IndexSource) getIndex() (*modindex.Index, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	// (s.index = nil => s.expires is zero,
	// so the first condition is strictly redundant.
	// But it makes the postcondition very clear.)
	if s.index == nil || time.Now().After(s.expires) {
		index, err := modindex.Update(s.modcachedir)
		if err != nil {
			return nil, err
		}
		s.index = index
		s.expires = index.ValidAt.Add(15 * time.Minute) // (refresh period)
	}
	// Inv: s.index != nil

	return s.index, nil
}
