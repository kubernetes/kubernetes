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

package config

const (
	// RulesGoRepoName is the canonical name of the rules_go repository. It must
	// match the workspace name in WORKSPACE.
	// TODO(jayconrod): move to language/go.
	RulesGoRepoName = "io_bazel_rules_go"

	// GazelleImportsKey is an internal attribute that lists imported packages
	// on generated rules. It is replaced with "deps" during import resolution.
	GazelleImportsKey = "_gazelle_imports"
)
