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

import (
	"fmt"
	"runtime"
	"strconv"

	utilversion "k8s.io/apimachinery/pkg/util/version"
	apimachineryversion "k8s.io/apimachinery/pkg/version"
)

// Get returns the overall codebase version. It's for detecting
// what code a binary was built from.
//
// Major, Minor and the base of GitVersion come from constants in this package,
// i.e. they describe the Kubernetes source code that this package comes from.
// GitCommit, GitTreeState and the build metadata in GitVersion come from the
// VCS information that the Go toolchain embeds in the main module of a binary.
// BuildDate is set by the Kubernetes build with ldflags.
//
// Minor has a trailing "+" if and only if the source is for a pre-release,
// i.e. KubePreReleaseVersion is not empty. It does not depend on tags or
// on the commit distance to them.
//
// Behavior in different scenarios:
//
//   - "go build" in a git checkout: VCS information is embedded, so
//     GitVersion is v<major>.<minor>.<patch>[-<pre>]+<commit>[-dirty].
//     The "+<commit>" is omitted if the commit is tagged with exactly
//     the version from the constants.
//   - "go build -buildvcs=false", or a tree without .git: GitCommit and
//     GitTreeState are empty and GitVersion has no build metadata.
//   - "git archive" export or GitHub source tarball: the commit and refs
//     recorded in base.go are used. GitTreeState is "archive" and GitVersion
//     has "+<commit>", unless the refs contain a tag which is exactly the
//     version from the constants.
//   - "go run" and "go test": the Go toolchain does not embed VCS information
//     for these, so the result is the same as with -buildvcs=false (or the
//     archive case for an archive).
//   - "make build" on the host: same as "go build". BuildDate is the time of
//     the build instead of the placeholder.
//   - "make build" in a container: same, provided that git works in the
//     container. The build runs as the owner of the source tree, which
//     avoids git's "dubious ownership" error.
//   - Build with KUBE_GIT_VERSION_FILE: GitVersion, GitCommit and GitTreeState
//     come from that file instead.
//   - Library of a downstream application: the constants still describe the
//     Kubernetes version of this package, but the VCS information describes
//     the application, not Kubernetes. A tagged application also does not
//     get a clean GitVersion because its tag does not match the constants.
//     Applications should not rely on these fields for anything except the
//     Kubernetes base version.
func Get() apimachineryversion.Info {
	binaryVersion := utilversion.MustParseMajorMinor(defaultKubeBinaryVersion)
	vcs := vcsInfo()
	minor := strconv.FormatUint(uint64(binaryVersion.Minor()), 10)
	if kubePreReleaseVersion != "" {
		minor += "+"
	}
	return apimachineryversion.Info{
		Major:        strconv.FormatUint(uint64(binaryVersion.Major()), 10),
		Minor:        minor,
		GitVersion:   buildVersion(),
		GitCommit:    vcs.commit,
		GitTreeState: vcs.treeState,
		BuildDate:    buildDate,
		GoVersion:    runtime.Version(),
		Compiler:     runtime.Compiler,
		Platform:     fmt.Sprintf("%s/%s", runtime.GOOS, runtime.GOARCH),
	}
}
