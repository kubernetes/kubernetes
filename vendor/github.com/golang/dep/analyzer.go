// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dep

import (
	"os"
	"path/filepath"

	"github.com/golang/dep/gps"
	"github.com/golang/dep/internal/fs"
)

// Analyzer implements gps.ProjectAnalyzer.
type Analyzer struct{}

// HasDepMetadata determines if a dep manifest exists at the specified path.
func (a Analyzer) HasDepMetadata(path string) bool {
	mf := filepath.Join(path, ManifestName)
	fileOK, err := fs.IsRegular(mf)
	return err == nil && fileOK
}

// DeriveManifestAndLock reads and returns the manifest at path/ManifestName or nil if one is not found.
// The Lock is always nil for now.
func (a Analyzer) DeriveManifestAndLock(path string, n gps.ProjectRoot) (gps.Manifest, gps.Lock, error) {
	if !a.HasDepMetadata(path) {
		return nil, nil, nil
	}

	f, err := os.Open(filepath.Join(path, ManifestName))
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()

	// Ignore warnings irrelevant to user.
	m, _, err := readManifest(f)
	if err != nil {
		return nil, nil, err
	}

	return m, nil, nil
}

// Info returns Analyzer's name and version info.
func (a Analyzer) Info() gps.ProjectAnalyzerInfo {
	return gps.ProjectAnalyzerInfo{
		Name:    "dep",
		Version: 1,
	}
}
