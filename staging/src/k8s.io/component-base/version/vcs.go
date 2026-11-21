/*
Copyright 2026 The Kubernetes Authors.

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
	"runtime/debug"
	"strings"
	"sync"
)

type vcsData struct {
	commit      string
	treeState   string
	mainVersion string
}

// vcsInfo returns the commit, tree state and main module version recorded by
// the Go toolchain. Without VCS stamping, e.g. in a "git archive" export, it
// falls back to the values that "git archive" put into base.go. The tree
// state is then "archive" and a tag in the refs is treated like the main
// module version of a tagged build. gitCommit and gitTreeState from base.go
// take precedence if they are set.
var vcsInfo = sync.OnceValue(func() vcsData {
	var d vcsData
	if info, ok := debug.ReadBuildInfo(); ok {
		// A clean tag for a tagged commit, otherwise a pseudo-version or "(devel)".
		d.mainVersion = info.Main.Version
		for _, s := range info.Settings {
			switch s.Key {
			case "vcs.revision":
				d.commit = s.Value
			case "vcs.modified":
				if s.Value == "true" {
					d.treeState = "dirty"
				} else {
					d.treeState = "clean"
				}
			}
		}
	}
	if d.commit == "" {
		d = archiveInfo(gitCommit, gitRefs)
	}
	// An unexpanded "git archive" placeholder is not a commit.
	if gitCommit != "" && !strings.HasPrefix(gitCommit, "$") {
		d.commit = gitCommit
	}
	if gitTreeState != "" {
		d.treeState = gitTreeState
	}
	return d
})

// archiveInfo interprets the values that "git archive" put into base.go.
// An unexpanded placeholder starts with "$".
func archiveInfo(commit, refs string) (d vcsData) {
	if commit == "" || strings.HasPrefix(commit, "$") {
		return
	}
	d.commit = commit
	d.treeState = "archive"
	// refs is a comma-separated list like "HEAD -> main, tag: v1.38.0".
	v := baseVersion()
	for _, ref := range strings.Split(refs, ", ") {
		if ref == "tag: "+v {
			d.mainVersion = v
		}
	}
	return
}

// buildVersion is the semantic version of this build, derived from
// DefaultKubeBinaryVersion, KubePatchVersion, KubePreReleaseVersion and the
// VCS information, e.g. v1.38.0-beta.0+69a8e1ff315563-dirty.
var buildVersion = sync.OnceValue(func() string {
	d := vcsInfo()
	return formatVersion(gitVersion, d.commit, d.treeState, d.mainVersion)
})

// baseVersion is the version from the constants, without build metadata.
func baseVersion() string {
	v := fmt.Sprintf("v%s.%d", DefaultKubeBinaryVersion, KubePatchVersion)
	if KubePreReleaseVersion != "" {
		v += "-" + KubePreReleaseVersion
	}
	return v
}

func formatVersion(override, commit, treeState, mainVersion string) string {
	if override != "" {
		return override
	}
	v := baseVersion()
	// The Go toolchain reports a clean tag as main module version, so there
	// is no need for the commit.
	if mainVersion == v {
		return v
	}
	hasCommit := len(commit) >= 14
	if hasCommit {
		v += "+" + commit[:14]
	}
	if treeState == "dirty" {
		if hasCommit {
			v += "-dirty"
		} else {
			v += "+dirty"
		}
	}
	return v
}
