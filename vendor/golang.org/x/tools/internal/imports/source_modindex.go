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

// implements Source using modindex, so only for module cache.
//
// this is perhaps over-engineered. A new Index is read at first use.
// And then Update is called after every 15 minutes, and a new Index
// is read if the index changed. It is not clear the Mutex is needed.
type IndexSource struct {
	modcachedir string
	mutex       sync.Mutex
	ix          *modindex.Index
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
	if err := s.maybeReadIndex(); err != nil {
		return nil, err
	}
	var cs []modindex.Candidate
	for pkg, nms := range missing {
		for nm := range nms {
			x := s.ix.Lookup(pkg, nm, false)
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

func (s *IndexSource) maybeReadIndex() error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	var readIndex bool
	if time.Now().After(s.expires) {
		ok, err := modindex.Update(s.modcachedir)
		if err != nil {
			return err
		}
		if ok {
			readIndex = true
		}
	}

	if readIndex || s.ix == nil {
		ix, err := modindex.ReadIndex(s.modcachedir)
		if err != nil {
			return err
		}
		s.ix = ix
		// for now refresh every 15 minutes
		s.expires = time.Now().Add(time.Minute * 15)
	}

	return nil
}
