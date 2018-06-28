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

package kdep

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"

	"github.com/golang/dep"
	"github.com/golang/dep/gps"
	"github.com/golang/dep/gps/pkgtree"
	toml "github.com/pelletier/go-toml"
)

// Metadata holds kdep metadata information
type Metadata struct {
	IsKdepRoot          bool     `toml:"kdep-project,omitempty"`
	IsKdepChild         bool     `toml:"kdep-child,omitempty"`
	GodepCompat         bool     `toml:"kdep-godep-compat,omitempty"`
	LocalGopaths        []string `toml:"kdep-local-gopaths"`
	LocalDeps           []string `toml:"kdep-local-deps"`
	UninterestingTags   []string `toml:"kdep-uninteresting-tags"`
	BlackListedPackages []string `toml:"kdep-blacklisted-packages"`
}

type manifest struct {
	Meta Metadata `toml:"metadata,omitempty"`
}

// Manifest wraps dep.Manifest to support kdep projects
type Manifest struct {
	*dep.Manifest
	Meta         *Metadata
	SubManifests map[string]*dep.Manifest
	ImportRoot   string
	Dependencies []string
}

// WrapManifest generates a kdep Manifest
func WrapManifest(m *dep.Manifest) *Manifest {
	return &Manifest{m, nil, nil, "", nil}
}

func manifestFromProject(p *dep.Project) *Manifest {
	fname := filepath.Join(p.AbsRoot, dep.ManifestName)

	meta := extractMetadata(fname)
	m := &Manifest{p.Manifest, meta, nil, string(p.ImportRoot), nil}
	return m
}

func extractMetadata(fname string) *Metadata {
	mf, _ := os.Open(fname)
	defer mf.Close()

	buf := &bytes.Buffer{}
	// dep has read it already, we're good
	_, _ = buf.ReadFrom(mf)
	m := manifest{}

	err := toml.Unmarshal(buf.Bytes(), &m)
	if err != nil {
		return &Metadata{}
	}

	return &m.Meta
}

// DependencyConstraints computes the aggregate set of dependency constraints for a kdep project
func (m *Manifest) DependencyConstraints() gps.ProjectConstraints {
	constraints := m.Manifest.DependencyConstraints()

	for _, sub := range m.SubManifests {
		extra := sub.DependencyConstraints()
		for root, props := range extra {
			p, ok := constraints[root]
			if ok {
				p.Constraint = p.Constraint.Intersect(props.Constraint)
			} else {
				constraints[root] = props
			}
		}
	}

	return constraints
}

// Overrides computes the aggregate set of overrides for a kdep project
func (m *Manifest) Overrides() gps.ProjectConstraints {
	constraints := m.Manifest.Overrides()

	for _, sub := range m.SubManifests {
		extra := sub.Overrides()
		for root, props := range extra {
			p, ok := constraints[root]
			if ok {
				p.Constraint = p.Constraint.Intersect(props.Constraint)
			} else {
				constraints[root] = props
			}
		}
	}
	return constraints
}

// IgnoredPackages computes the aggregate set of ignored packages for a kdep project
func (m *Manifest) IgnoredPackages() *pkgtree.IgnoredRuleset {
	ignored := make([]string, len(m.SubManifests)+1)
	i := 0
	for k := range m.SubManifests {
		ignored[i] = fmt.Sprintf("%s/*", k)
		i++
	}
	ignored[i] = m.ImportRoot
	return pkgtree.NewIgnoredRuleset(append(ignored, m.Manifest.IgnoredPackages().ToSlice()...))
}

// RequiredPackages computes the aggregate set of required packages for a kdep project
func (m *Manifest) RequiredPackages() map[string]bool {
	required := m.Manifest.RequiredPackages()
	for _, sub := range m.SubManifests {
		for k, v := range sub.RequiredPackages() {
			required[k] = v
		}
	}

	// since we ignore all the root packages, let's require all their
	// dependencies
	for _, v := range m.Dependencies {
		required[v] = true
	}
	return required
}

var _ gps.RootManifest = (*Manifest)(nil)
