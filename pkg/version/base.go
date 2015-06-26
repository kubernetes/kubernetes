/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
// This is the fallback data used when version information from git is not
// provided via go ldflags. It provides an approximation of the Kubernetes
// version for ad-hoc builds (e.g. `go build`) that cannot get the version
// information from git.
//
// The "-dev" suffix in the version info indicates that fact, and it means the
// current build is from a version greater that version. For example, v0.7-dev
// means version > 0.7 and < 0.8. (There's exceptions to this rule, see
// docs/releasing.md for more details.)
//
// When releasing a new Kubernetes version, this file should be updated to
// reflect the new version, and then a git annotated tag (using format vX.Y
// where X == Major version and Y == Minor version) should be created to point
// to the commit that updates pkg/version/base.go

var (
	// TODO: Deprecate gitMajor and gitMinor, use only gitVersion instead.
	gitMajor     string = "0"              // major version, always numeric
	gitMinor     string = "20.0+"          // minor version, numeric possibly followed by "+"
	gitVersion   string = "v0.20.0-dev"    // version from git, output of $(git describe)
	gitCommit    string = ""               // sha1 from git, output of $(git rev-parse HEAD)
	gitTreeState string = "not a git tree" // state of git tree, either "clean" or "dirty"
)
