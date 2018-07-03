/* Copyright 2017 The Bazel Authors. All rights reserved.

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

package repos

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/internal/generator"
	bf "github.com/bazelbuild/buildtools/build"
)

// Repo describes an external repository rule declared in a Bazel
// WORKSPACE file.
type Repo struct {
	// Name is the value of the "name" attribute of the repository rule.
	Name string

	// GoPrefix is the portion of the Go import path for the root of this
	// repository. Usually the same as Remote.
	GoPrefix string

	// Commit is the revision at which a repository is checked out (for example,
	// a Git commit id).
	Commit string

	// Tag is the name of the version at which a repository is checked out.
	Tag string

	// Remote is the URL the repository can be cloned or checked out from.
	Remote string

	// VCS is the version control system used to check out the repository.
	// May also be "http" for HTTP archives.
	VCS string
}

type byName []Repo

func (s byName) Len() int           { return len(s) }
func (s byName) Less(i, j int) bool { return s[i].Name < s[j].Name }
func (s byName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

type lockFileFormat int

const (
	unknownFormat lockFileFormat = iota
	depFormat
)

var lockFileParsers = map[lockFileFormat]func(string) ([]Repo, error){
	depFormat: importRepoRulesDep,
}

// ImportRepoRules reads the lock file of a vendoring tool and returns
// a list of equivalent repository rules that can be merged into a WORKSPACE
// file. The format of the file is inferred from its basename. Currently,
// only Gopkg.lock is supported.
func ImportRepoRules(filename string) ([]bf.Expr, error) {
	format := getLockFileFormat(filename)
	if format == unknownFormat {
		return nil, fmt.Errorf(`%s: unrecognized lock file format. Expected "Gopkg.lock"`, filename)
	}
	parser := lockFileParsers[format]
	repos, err := parser(filename)
	if err != nil {
		return nil, fmt.Errorf("error parsing %q: %v", filename, err)
	}
	sort.Stable(byName(repos))

	rules := make([]bf.Expr, 0, len(repos))
	for _, repo := range repos {
		rules = append(rules, GenerateRule(repo))
	}
	return rules, nil
}

func getLockFileFormat(filename string) lockFileFormat {
	switch filepath.Base(filename) {
	case "Gopkg.lock":
		return depFormat
	default:
		return unknownFormat
	}
}

// GenerateRule returns a repository rule for the given repository that can
// be written in a WORKSPACE file.
func GenerateRule(repo Repo) bf.Expr {
	attrs := []generator.KeyValue{
		{Key: "name", Value: repo.Name},
		{Key: "commit", Value: repo.Commit},
		{Key: "importpath", Value: repo.GoPrefix},
	}
	if repo.Remote != "" {
		attrs = append(attrs, generator.KeyValue{Key: "remote", Value: repo.Remote})
	}
	if repo.VCS != "" {
		attrs = append(attrs, generator.KeyValue{Key: "vcs", Value: repo.VCS})
	}
	return generator.NewRule("go_repository", attrs)
}

// FindExternalRepo attempts to locate the directory where Bazel has fetched
// the external repository with the given name. An error is returned if the
// repository directory cannot be located.
func FindExternalRepo(repoRoot, name string) (string, error) {
	// See https://docs.bazel.build/versions/master/output_directories.html
	// for documentation on Bazel directory layout.
	// We expect the bazel-out symlink in the workspace root directory to point to
	// <output-base>/execroot/<workspace-name>/bazel-out
	// We expect the external repository to be checked out at
	// <output-base>/external/<name>
	// Note that users can change the prefix for most of the Bazel symlinks with
	// --symlink_prefix, but this does not include bazel-out.
	externalPath := strings.Join([]string{repoRoot, "bazel-out", "..", "..", "..", "external", name}, string(os.PathSeparator))
	cleanPath, err := filepath.EvalSymlinks(externalPath)
	if err != nil {
		return "", err
	}
	st, err := os.Stat(cleanPath)
	if err != nil {
		return "", err
	}
	if !st.IsDir() {
		return "", fmt.Errorf("%s: not a directory", externalPath)
	}
	return cleanPath, nil
}

// ListRepositories extracts metadata about repositories declared in a
// WORKSPACE file.
//
// The set of repositories returned is necessarily incomplete, since we don't
// evaluate the file, and repositories may be declared in macros in other files.
func ListRepositories(workspace *bf.File) []Repo {
	var repos []Repo
	for _, e := range workspace.Stmt {
		call, ok := e.(*bf.CallExpr)
		if !ok {
			continue
		}
		r := bf.Rule{Call: call}
		name := r.Name()
		if name == "" {
			continue
		}
		var repo Repo
		switch r.Kind() {
		case "go_repository":
			// TODO(jayconrod): extract other fields needed by go_repository.
			// Currently, we don't use the result of this function to produce new
			// go_repository rules, so it doesn't matter.
			goPrefix := r.AttrString("importpath")
			revision := r.AttrString("commit")
			remote := r.AttrString("remote")
			vcs := r.AttrString("vcs")
			if goPrefix == "" {
				continue
			}
			repo = Repo{
				Name:     name,
				GoPrefix: goPrefix,
				Commit:   revision,
				Remote:   remote,
				VCS:      vcs,
			}

			// TODO(jayconrod): infer from {new_,}git_repository, {new_,}http_archive,
			// local_repository.

		default:
			continue
		}
		repos = append(repos, repo)
	}

	// TODO(jayconrod): look for directives that describe repositories that
	// aren't declared in the top-level of WORKSPACE (e.g., behind a macro).

	return repos
}
