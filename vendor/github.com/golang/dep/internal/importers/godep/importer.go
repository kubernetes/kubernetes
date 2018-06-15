// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package godep

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

const godepPath = "Godeps" + string(os.PathSeparator) + "Godeps.json"

// Importer imports godep configuration into the dep configuration format.
type Importer struct {
	*base.Importer
	json godepJSON
}

// NewImporter for godep.
func NewImporter(logger *log.Logger, verbose bool, sm gps.SourceManager) *Importer {
	return &Importer{Importer: base.NewImporter(logger, verbose, sm)}
}

type godepJSON struct {
	Imports []godepPackage `json:"Deps"`
}

type godepPackage struct {
	ImportPath string `json:"ImportPath"`
	Rev        string `json:"Rev"`
	Comment    string `json:"Comment"`
}

// Name of the importer.
func (g *Importer) Name() string {
	return "godep"
}

// HasDepMetadata checks if a directory contains config that the importer can handle.
func (g *Importer) HasDepMetadata(dir string) bool {
	y := filepath.Join(dir, godepPath)
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
	g.Logger.Println("Detected godep configuration files...")
	j := filepath.Join(projectDir, godepPath)
	if g.Verbose {
		g.Logger.Printf("  Loading %s", j)
	}
	jb, err := ioutil.ReadFile(j)
	if err != nil {
		return errors.Wrapf(err, "unable to read %s", j)
	}
	err = json.Unmarshal(jb, &g.json)
	if err != nil {
		return errors.Wrapf(err, "unable to parse %s", j)
	}

	return nil
}

func (g *Importer) convert(pr gps.ProjectRoot) (*dep.Manifest, *dep.Lock) {
	g.Logger.Println("Converting from Godeps.json ...")

	packages := make([]base.ImportedPackage, 0, len(g.json.Imports))
	for _, pkg := range g.json.Imports {
		// Validate
		if pkg.ImportPath == "" {
			g.Logger.Println(
				"  Warning: Skipping project. Invalid godep configuration, ImportPath is required",
			)
			continue
		}

		if pkg.Rev == "" {
			g.Logger.Printf(
				"  Warning: Invalid godep configuration, Rev not found for ImportPath %q\n",
				pkg.ImportPath,
			)
		}

		ip := base.ImportedPackage{
			Name:           pkg.ImportPath,
			LockHint:       pkg.Rev,
			ConstraintHint: pkg.Comment,
		}
		packages = append(packages, ip)
	}

	g.ImportPackages(packages, true)
	return g.Manifest, g.Lock
}
