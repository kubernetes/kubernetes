// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gvt

import (
	"encoding/json"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/golang/dep"
	"github.com/golang/dep/gps"
	"github.com/golang/dep/internal/importers/base"
	"github.com/pkg/errors"
)

const gvtPath = "vendor" + string(os.PathSeparator) + "manifest"

// Importer imports gvt configuration into the dep configuration format.
type Importer struct {
	*base.Importer
	gvtConfig gvtManifest
}

// NewImporter for gvt. It handles gb (gb-vendor) too as they share a common manifest file & format
func NewImporter(logger *log.Logger, verbose bool, sm gps.SourceManager) *Importer {
	return &Importer{Importer: base.NewImporter(logger, verbose, sm)}
}

type gvtManifest struct {
	Deps []gvtPkg `json:"dependencies"`
}

type gvtPkg struct {
	ImportPath string
	Repository string
	Revision   string
	Branch     string
}

// Name of the importer.
func (g *Importer) Name() string {
	return "gvt"
}

// HasDepMetadata checks if a directory contains config that the importer can handle.
func (g *Importer) HasDepMetadata(dir string) bool {
	y := filepath.Join(dir, gvtPath)
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
	g.Logger.Println("Detected gb/gvt configuration files...")
	j := filepath.Join(projectDir, gvtPath)
	if g.Verbose {
		g.Logger.Printf("  Loading %s", j)
	}
	jb, err := ioutil.ReadFile(j)
	if err != nil {
		return errors.Wrapf(err, "unable to read %s", j)
	}
	err = json.Unmarshal(jb, &g.gvtConfig)
	if err != nil {
		return errors.Wrapf(err, "unable to parse %s", j)
	}

	return nil
}

func (g *Importer) convert(pr gps.ProjectRoot) (*dep.Manifest, *dep.Lock) {
	g.Logger.Println("Converting from vendor/manifest ...")

	packages := make([]base.ImportedPackage, 0, len(g.gvtConfig.Deps))
	for _, pkg := range g.gvtConfig.Deps {
		// Validate
		if pkg.ImportPath == "" {
			g.Logger.Println(
				"  Warning: Skipping project. Invalid gvt configuration, ImportPath is required",
			)
			continue
		}

		if pkg.Revision == "" {
			g.Logger.Printf(
				"  Warning: Invalid gvt configuration, Revision not found for ImportPath %q\n",
				pkg.ImportPath,
			)
		}

		var contstraintHint = ""
		if pkg.Branch == "HEAD" {
			// gb-vendor sets "branch" to "HEAD", if the package was feteched via -tag or -revision,
			// we pass the revision as the constraint hint
			contstraintHint = pkg.Revision
		} else if pkg.Branch != "master" {
			// both gvt & gb-vendor set "branch" to "master" unless a different branch was requested.
			// so it's not really a constraint unless it's a different branch
			contstraintHint = pkg.Branch
		}

		ip := base.ImportedPackage{
			Name:           pkg.ImportPath,
			Source:         pkg.Repository,
			LockHint:       pkg.Revision,
			ConstraintHint: contstraintHint,
		}
		packages = append(packages, ip)
	}

	g.ImportPackages(packages, true)
	return g.Manifest, g.Lock
}
