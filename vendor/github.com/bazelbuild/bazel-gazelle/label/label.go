/* Copyright 2016 The Bazel Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package label provides utilities for parsing and manipulating
// Bazel labels. See
// https://docs.bazel.build/versions/master/build-ref.html#labels
// for more information.
package label

import (
	"fmt"
	"log"
	"path"
	"regexp"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/pathtools"
)

// A Label represents a label of a build target in Bazel. Labels have three
// parts: a repository name, a package name, and a target name, formatted
// as @repo//pkg:target.
type Label struct {
	// Repo is the repository name. If omitted, the label refers to a target
	// in the current repository.
	Repo string

	// Pkg is the package name, which is usually the directory that contains
	// the target. If both Repo and Pkg are omitted, the label is relative.
	Pkg string

	// Name is the name of the target the label refers to. If omitted, Name
	// is assumed to be the same as Pkg.
	Name string

	// Relative indicates whether the label refers to a target in the current
	// package. Relative is true if and only if Repo and Pkg are both omitted.
	Relative bool
}

// New constructs a new label from components.
func New(repo, pkg, name string) Label {
	return Label{Repo: repo, Pkg: pkg, Name: name}
}

// NoLabel is the zero  value of Label. It is not a valid label and may be
// returned when an error occurs.
var NoLabel = Label{}

var (
	labelRepoRegexp = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_]*$`)
	labelPkgRegexp  = regexp.MustCompile(`^[A-Za-z0-9/._-]*$`)
	labelNameRegexp = regexp.MustCompile(`^[A-Za-z0-9_/.+=,@~-]*$`)
)

// Parse reads a label from a string.
// See https://docs.bazel.build/versions/master/build-ref.html#lexi.
func Parse(s string) (Label, error) {
	origStr := s

	relative := true
	var repo string
	if strings.HasPrefix(s, "@") {
		relative = false
		endRepo := strings.Index(s, "//")
		if endRepo < 0 {
			return NoLabel, fmt.Errorf("label parse error: repository does not end with '//': %q", origStr)
		}
		repo = s[len("@"):endRepo]
		if !labelRepoRegexp.MatchString(repo) {
			return NoLabel, fmt.Errorf("label parse error: repository has invalid characters: %q", origStr)
		}
		s = s[endRepo:]
	}

	var pkg string
	if strings.HasPrefix(s, "//") {
		relative = false
		endPkg := strings.Index(s, ":")
		if endPkg < 0 {
			pkg = s[len("//"):]
			s = ""
		} else {
			pkg = s[len("//"):endPkg]
			s = s[endPkg:]
		}
		if !labelPkgRegexp.MatchString(pkg) {
			return NoLabel, fmt.Errorf("label parse error: package has invalid characters: %q", origStr)
		}
	}

	if s == ":" {
		return NoLabel, fmt.Errorf("label parse error: empty name: %q", origStr)
	}
	name := strings.TrimPrefix(s, ":")
	if !labelNameRegexp.MatchString(name) {
		return NoLabel, fmt.Errorf("label parse error: name has invalid characters: %q", origStr)
	}

	if pkg == "" && name == "" {
		return NoLabel, fmt.Errorf("label parse error: empty package and name: %q", origStr)
	}
	if name == "" {
		name = path.Base(pkg)
	}

	return Label{
		Repo:     repo,
		Pkg:      pkg,
		Name:     name,
		Relative: relative,
	}, nil
}

func (l Label) String() string {
	if l.Relative {
		return fmt.Sprintf(":%s", l.Name)
	}

	var repo string
	if l.Repo != "" {
		repo = fmt.Sprintf("@%s", l.Repo)
	}

	if path.Base(l.Pkg) == l.Name {
		return fmt.Sprintf("%s//%s", repo, l.Pkg)
	}
	return fmt.Sprintf("%s//%s:%s", repo, l.Pkg, l.Name)
}

// Abs computes an absolute label (one with a repository and package name)
// from this label. If this label is already absolute, it is returned
// unchanged.
func (l Label) Abs(repo, pkg string) Label {
	if !l.Relative {
		return l
	}
	return Label{Repo: repo, Pkg: pkg, Name: l.Name}
}

// Rel attempts to compute a relative label from this label. If this label
// is already relative or is in a different package, this label may be
// returned unchanged.
func (l Label) Rel(repo, pkg string) Label {
	if l.Relative || l.Repo != repo {
		return l
	}
	if l.Pkg == pkg {
		return Label{Name: l.Name, Relative: true}
	}
	return Label{Pkg: l.Pkg, Name: l.Name}
}

// Equal returns whether two labels are exactly the same. It does not return
// true for different labels that refer to the same target.
func (l Label) Equal(other Label) bool {
	return l.Repo == other.Repo &&
		l.Pkg == other.Pkg &&
		l.Name == other.Name &&
		l.Relative == other.Relative
}

// Contains returns whether other is contained by the package of l or a
// sub-package. Neither label may be relative.
func (l Label) Contains(other Label) bool {
	if l.Relative {
		log.Panicf("l must not be relative: %s", l)
	}
	if other.Relative {
		log.Panicf("other must not be relative: %s", other)
	}
	result := l.Repo == other.Repo && pathtools.HasPrefix(other.Pkg, l.Pkg)
	return result
}

// ImportPathToBazelRepoName converts a Go import path into a bazel repo name
// following the guidelines in http://bazel.io/docs/be/functions.html#workspace
func ImportPathToBazelRepoName(importpath string) string {
	importpath = strings.ToLower(importpath)
	components := strings.Split(importpath, "/")
	labels := strings.Split(components[0], ".")
	var reversed []string
	for i := range labels {
		l := labels[len(labels)-i-1]
		reversed = append(reversed, l)
	}
	repo := strings.Join(append(reversed, components[1:]...), "_")
	return strings.NewReplacer("-", "_", ".", "_").Replace(repo)
}
