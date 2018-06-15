// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package govendor

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/golang/dep"
	"github.com/golang/dep/gps"
	"github.com/golang/dep/internal/importers/base"
	"github.com/pkg/errors"
)

const govendorDir = "vendor"
const govendorName = "vendor.json"

// Importer imports govendor configuration into the dep configuration format.
type Importer struct {
	*base.Importer

	file govendorFile
}

// NewImporter for govendor.
func NewImporter(logger *log.Logger, verbose bool, sm gps.SourceManager) *Importer {
	return &Importer{Importer: base.NewImporter(logger, verbose, sm)}
}

// File is the structure of the vendor file.
type govendorFile struct {
	RootPath string // Import path of vendor folder
	Ignore   string
	Package  []*govendorPackage
}

// Package represents each package.
type govendorPackage struct {
	// See the vendor spec for definitions.
	Origin   string
	Path     string
	Revision string
	Version  string
}

// Name of the importer.
func (g *Importer) Name() string {
	return "govendor"
}

// HasDepMetadata checks if a directory contains config that the importer can handle.
func (g *Importer) HasDepMetadata(dir string) bool {
	y := filepath.Join(dir, govendorDir, govendorName)
	if _, err := os.Stat(y); err != nil {
		return false
	}
	return true
}

// Import the config found in the directory.
func (g *Importer) Import(dir string, pr gps.ProjectRoot) (*dep.Manifest, *dep.Lock, error) {
	err := g.load(dir)
	if err != nil {
		return nil, nil, err
	}

	m, l := g.convert(pr)
	return m, l, nil
}

func (g *Importer) load(projectDir string) error {
	g.Logger.Println("Detected govendor configuration file...")
	v := filepath.Join(projectDir, govendorDir, govendorName)
	if g.Verbose {
		g.Logger.Printf("  Loading %s", v)
	}
	vb, err := ioutil.ReadFile(v)
	if err != nil {
		return errors.Wrapf(err, "unable to read %s", v)
	}
	err = json.Unmarshal(vb, &g.file)
	if err != nil {
		return errors.Wrapf(err, "unable to parse %s", v)
	}
	return nil
}

func (g *Importer) convert(pr gps.ProjectRoot) (*dep.Manifest, *dep.Lock) {
	g.Logger.Println("Converting from vendor.json...")

	packages := make([]base.ImportedPackage, 0, len(g.file.Package))
	for _, pkg := range g.file.Package {
		// Path must not be empty
		if pkg.Path == "" {
			g.Logger.Println(
				"  Warning: Skipping project. Invalid govendor configuration, Path is required",
			)
			continue
		}

		// There are valid govendor configs in the wild that don't have a revision set
		// so we are not requiring it to be set during import

		ip := base.ImportedPackage{
			Name:     pkg.Path,
			Source:   pkg.Origin,
			LockHint: pkg.Revision,
		}
		packages = append(packages, ip)
	}

	g.ImportPackages(packages, true)

	if len(g.file.Ignore) > 0 {
		// Govendor has three use cases here
		// 1. 'test' - special case for ignoring test files
		// 2. build tags - any string without a slash (/) in it
		// 3. path and path prefix - any string with a slash (/) in it.
		//   The path case could be a full path or just a prefix.
		// Dep doesn't support build tags right now: https://github.com/golang/dep/issues/120
		for _, i := range strings.Split(g.file.Ignore, " ") {
			if !strings.Contains(i, "/") {
				g.Logger.Printf("  Govendor was configured to ignore the %s build tag, but that isn't supported by dep yet, and will be ignored. See https://github.com/golang/dep/issues/291.", i)
				continue
			}

			var ignorePattern string
			_, err := g.SourceManager.DeduceProjectRoot(i)
			if err == nil { // external package
				ignorePattern = i
			} else { // relative package path in the current project
				ignorePattern = path.Join(string(pr), i)
			}

			// Convert to a a wildcard ignore
			ignorePattern = strings.TrimRight(ignorePattern, "/")
			ignorePattern += "*"

			g.Manifest.Ignored = append(g.Manifest.Ignored, ignorePattern)
		}
	}

	return g.Manifest, g.Lock
}
