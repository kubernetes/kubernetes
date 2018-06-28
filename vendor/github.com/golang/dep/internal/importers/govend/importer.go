// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package govend

import (
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/go-yaml/yaml"
	"github.com/golang/dep"
	"github.com/golang/dep/gps"
	"github.com/golang/dep/internal/importers/base"
	"github.com/pkg/errors"
)

// ToDo: govend supports json and xml formats as well and we will add support for other formats in next PR - @RaviTezu
// govend don't have a separate lock file.
const govendYAMLName = "vendor.yml"

// Importer imports govend configuration in to the dep configuration format.
type Importer struct {
	*base.Importer
	yaml govendYAML
}

// NewImporter for govend.
func NewImporter(logger *log.Logger, verbose bool, sm gps.SourceManager) *Importer {
	return &Importer{Importer: base.NewImporter(logger, verbose, sm)}
}

type govendYAML struct {
	Imports []govendPackage `yaml:"vendors"`
}

type govendPackage struct {
	Path     string `yaml:"path"`
	Revision string `yaml:"rev"`
}

// Name of the importer.
func (g *Importer) Name() string {
	return "govend"
}

// HasDepMetadata checks if a directory contains config that the importer can handle.
func (g *Importer) HasDepMetadata(dir string) bool {
	y := filepath.Join(dir, govendYAMLName)
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

// load the govend configuration files.
func (g *Importer) load(projectDir string) error {
	g.Logger.Println("Detected govend configuration files...")
	y := filepath.Join(projectDir, govendYAMLName)
	if g.Verbose {
		g.Logger.Printf("  Loading %s", y)
	}
	yb, err := ioutil.ReadFile(y)
	if err != nil {
		return errors.Wrapf(err, "unable to read %s", y)
	}
	err = yaml.Unmarshal(yb, &g.yaml)
	if err != nil {
		return errors.Wrapf(err, "unable to parse %s", y)
	}
	return nil
}

// convert the govend configuration files into dep configuration files.
func (g *Importer) convert(pr gps.ProjectRoot) (*dep.Manifest, *dep.Lock) {
	g.Logger.Println("Converting from vendor.yaml...")

	packages := make([]base.ImportedPackage, 0, len(g.yaml.Imports))
	for _, pkg := range g.yaml.Imports {
		// Path must not be empty
		if pkg.Path == "" {
			g.Logger.Println(
				"  Warning: Skipping project. Invalid govend configuration, path is required",
			)
			continue
		}

		if pkg.Revision == "" {
			// Do not add 'empty constraints' to the manifest. Solve will add to lock if required.
			g.Logger.Printf(
				"  Warning: Skipping import with empty constraints. "+
					"The solve step will add the dependency to the lock if needed: %q\n",
				pkg.Path,
			)
			continue
		}

		ip := base.ImportedPackage{
			Name:     pkg.Path,
			LockHint: pkg.Revision,
		}
		packages = append(packages, ip)
	}

	g.ImportPackages(packages, true)
	return g.Manifest, g.Lock
}
