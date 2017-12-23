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

package resolve

import (
	"github.com/bazelbuild/bazel-gazelle/internal/label"
	"github.com/bazelbuild/bazel-gazelle/internal/pathtools"
	"github.com/bazelbuild/bazel-gazelle/internal/repos"
)

// externalResolver resolves import paths to external repositories. It uses
// vcs to determine the prefix of the import path that corresponds to the root
// of the repository (this will perform a network fetch for unqualified paths).
// The prefix is converted to a Bazel external name repo according to the
// guidelines in http://bazel.io/docs/be/functions.html#workspace. The remaining
// portion of the import path is treated as the package name.
type externalResolver struct {
	l  *label.Labeler
	rc *repos.RemoteCache
}

var _ nonlocalResolver = (*externalResolver)(nil)

func newExternalResolver(l *label.Labeler, rc *repos.RemoteCache) *externalResolver {
	return &externalResolver{l: l, rc: rc}
}

// Resolve resolves "importPath" into a label, assuming that it is a label in an
// external repository. It also assumes that the external repository follows the
// recommended reverse-DNS form of workspace name as described in
// http://bazel.io/docs/be/functions.html#workspace.
func (r *externalResolver) resolve(importPath string) (label.Label, error) {
	prefix, repo, err := r.rc.Root(importPath)
	if err != nil {
		return label.NoLabel, err
	}

	var pkg string
	if importPath != prefix {
		pkg = pathtools.TrimPrefix(importPath, prefix)
	}

	l := r.l.LibraryLabel(pkg)
	l.Repo = repo
	return l, nil
}
