/* Copyright 2019 The Bazel Authors. All rights reserved.

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

package language

import (
	"github.com/bazelbuild/bazel-gazelle/config"
	"github.com/bazelbuild/bazel-gazelle/repo"
	"github.com/bazelbuild/bazel-gazelle/rule"
)

// RepoUpdater may be implemented by languages that support updating
// repository rules that provide named libraries.
//
// EXPERIMENTAL: this may change or be removed.
type RepoUpdater interface {
	UpdateRepos(args UpdateReposArgs) UpdateReposResult
}

// UpdateReposArgs contains arguments for RepoUpdater.UpdateRepos.
// Arguments are passed in a struct value so that new fields may be added
// in the future without breaking existing implementations.
//
// EXPERIMENTAL: this may change or be removed.
type UpdateReposArgs struct {
	// Config is the configuration for the main workspace.
	Config *config.Config

	// Imports is a list of libraries to update. UpdateRepos should return
	// repository rules that provide these libraries. It may also return
	// repository rules providing transitive dependencies.
	Imports []string

	// Cache stores information fetched from the network and ensures that
	// the same request isn't made multiple times.
	Cache *repo.RemoteCache
}

// UpdateReposResult contains return values for RepoUpdater.UpdateRepos.
// Results are returned through a struct so that new (optional) fields may be
// added without breaking existing implementations.
//
// EXPERIMENTAL: this may change or be removed.
type UpdateReposResult struct {
	// Gen is a list of repository rules that provide libraries named by
	// UpdateImportArgs.Imports. These will be merged with existing rules or
	// added to WORKSPACE. This list may be shorter or longer than the list
	// of imports, since a single repository may provide multiple imports,
	// and additional repositories may be needed for transitive dependencies.
	Gen []*rule.Rule

	// Error is any fatal error that occurred. Non-fatal errors should be logged.
	Error error
}

// RepoImporter may be implemented by languages that support importing
// repository rules from another build system.
//
// EXPERIMENTAL: this may change or be removed.
type RepoImporter interface {
	// CanImport returns whether a given configuration file may be imported
	// with this extension. Only one extension may import any given file.
	// ImportRepos will not be called unless this returns true.
	CanImport(path string) bool

	// ImportRepos generates a list of repository rules by reading a
	// configuration file from another build system.
	ImportRepos(args ImportReposArgs) ImportReposResult
}

// ImportReposArgs contains arguments for RepoImporter.ImportRepos.
// Arguments are passed in a struct value so that new fields may be added
// in the future without breaking existing implementations.
//
// EXPERIMENTAL: this may change or be removed.
type ImportReposArgs struct {
	// Config is the configuration for the main workspace.
	Config *config.Config

	// Path is the name of the configuration file to import.
	Path string

	// Prune indicates whether repository rules that are no longer needed
	// should be deleted. This means the Empty list in the result should be
	// filled in.
	Prune bool

	// Cache stores information fetched from the network and ensures that
	// the same request isn't made multiple times.
	Cache *repo.RemoteCache
}

// ImportReposResult contains return values for RepoImporter.ImportRepos.
// Results are returned through a struct so that new (optional) fields may
// be added without breaking existing implementations.
//
// EXPERIMENTAL: this may change or be removed.
type ImportReposResult struct {
	// Gen is a list of imported repository rules.
	Gen []*rule.Rule

	// Empty is a list of repository rules that may be deleted. This should only
	// be set if ImportReposArgs.Prune is true.
	Empty []*rule.Rule

	// Error is any fatal error that occurred. Non-fatal errors should be logged.
	Error error
}
