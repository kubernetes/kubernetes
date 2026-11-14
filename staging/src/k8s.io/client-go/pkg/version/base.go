/*
Copyright 2014 The Kubernetes Authors.

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

package version

// Base version information.
//
// The version itself is built from the constants below and the VCS information
// recorded by the Go toolchain. The variables are optional overrides which
// take precedence over that information. They are set with ldflags -X, but
// only when building with KUBE_GIT_VERSION_FILE, i.e. when the build has to
// use externally provided version information. gitCommit and gitRefs also get
// replaced by "git archive", which works for GitHub tar downloads. buildDate
// is always set with ldflags -X.
//
// TODO: reconsider whether KUBE_GIT_VERSION_FILE is needed. Removing it would
// simplify the source code because ldflags injection would no longer be
// necessary for anything except buildDate.
var (
	gitVersion   = ""            // complete semantic version, replaces the one built from the constants
	gitCommit    = "$Format:%H$" // sha1 from git, ignored while it is still the unexpanded placeholder
	gitRefs      = "$Format:%D$" // refs of the archived commit, e.g. "HEAD -> master, tag: v1.38.0"
	gitTreeState = ""            // "clean" or "dirty"

	buildDate = "1970-01-01T00:00:00Z" // build date in ISO8601 format, output of $(date -u +'%Y-%m-%dT%H:%M:%SZ')
)

// These must be kept in sync with k8s.io/component-base/version.
const (
	defaultKubeBinaryVersion = "1.38"
	kubePatchVersion         = 0
	kubePreReleaseVersion    = "beta.0"
)
