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
	"runtime"
	"strings"
	"testing"

	utilversion "k8s.io/apimachinery/pkg/util/version"
)

func TestGet(t *testing.T) {
	info := Get()

	major, minor, _ := strings.Cut(DefaultKubeBinaryVersion, ".")
	if info.Major != major {
		t.Errorf("Major: expected %q, got %q", major, info.Major)
	}
	expectMinor := minor
	if KubePreReleaseVersion != "" {
		expectMinor += "+"
	}
	if info.Minor != expectMinor {
		t.Errorf("Minor: expected %q, got %q", expectMinor, info.Minor)
	}

	v, err := utilversion.ParseSemantic(info.GitVersion)
	if err != nil {
		t.Fatalf("GitVersion %q is not a semantic version: %v", info.GitVersion, err)
	}
	expectPrefix := fmt.Sprintf("v%s.%d", DefaultKubeBinaryVersion, KubePatchVersion)
	if !strings.HasPrefix(info.GitVersion, expectPrefix) {
		t.Errorf("GitVersion: expected prefix %q, got %q (parsed as %v)", expectPrefix, info.GitVersion, v)
	}

	if info.GoVersion != runtime.Version() {
		t.Errorf("GoVersion: expected %q, got %q", runtime.Version(), info.GoVersion)
	}
	if info.Compiler != runtime.Compiler {
		t.Errorf("Compiler: expected %q, got %q", runtime.Compiler, info.Compiler)
	}
	if expect := runtime.GOOS + "/" + runtime.GOARCH; info.Platform != expect {
		t.Errorf("Platform: expected %q, got %q", expect, info.Platform)
	}
}

func TestFormatVersion(t *testing.T) {
	base := fmt.Sprintf("v%s.%d", DefaultKubeBinaryVersion, KubePatchVersion)
	if KubePreReleaseVersion != "" {
		base += "-" + KubePreReleaseVersion
	}
	const hash = "69a8e1ff31556386089beebee70216693c229c10"

	for name, tc := range map[string]struct {
		commit, treeState, mainVersion string
		expect                         string
	}{
		"clean":               {commit: hash, treeState: "clean", expect: base + "+69a8e1ff315563"},
		"dirty":               {commit: hash, treeState: "dirty", expect: base + "+69a8e1ff315563-dirty"},
		"no-tree-state":       {commit: hash, expect: base + "+69a8e1ff315563"},
		"unexpanded-archive":  {commit: "$Format:%H$", expect: base},
		"no-commit":           {expect: base},
		"no-commit-and-dirty": {treeState: "dirty", expect: base + "+dirty"},
		"tagged-clean":        {commit: hash, treeState: "clean", mainVersion: base, expect: base},
		"tagged-dirty":        {commit: hash, treeState: "dirty", mainVersion: base + "+dirty", expect: base + "+69a8e1ff315563-dirty"},
		"untagged-pseudo":     {commit: hash, treeState: "clean", mainVersion: "v1.37.1-0.20261112000000-69a8e1ff3155", expect: base + "+69a8e1ff315563"},
		"other-tag":           {commit: hash, treeState: "clean", mainVersion: "v0.0.1", expect: base + "+69a8e1ff315563"},
		"short-commit":        {commit: "abc", treeState: "clean", expect: base},
	} {
		t.Run(name, func(t *testing.T) {
			actual := formatVersion(tc.commit, tc.treeState, tc.mainVersion)
			if actual != tc.expect {
				t.Errorf("expected %q, got %q", tc.expect, actual)
			}
			if _, err := utilversion.ParseSemantic(actual); err != nil {
				t.Errorf("%q is not a semantic version: %v", actual, err)
			}
		})
	}
}

func TestArchiveInfo(t *testing.T) {
	const hash = "69a8e1ff31556386089beebee70216693c229c10"
	base := baseVersion()

	for name, tc := range map[string]struct {
		commit, refs string
		expect       vcsData
	}{
		"unexpanded":     {commit: "$Format:%H$", refs: "$Format:%D$"},
		"no-commit":      {refs: "tag: " + base},
		"untagged":       {commit: hash, refs: "HEAD -> master, origin/master", expect: vcsData{commit: hash, treeState: "archive"}},
		"tagged":         {commit: hash, refs: "HEAD -> release-1.38, tag: " + base + ", origin/release-1.38", expect: vcsData{commit: hash, treeState: "archive", mainVersion: base}},
		"other-tag":      {commit: hash, refs: "tag: v0.0.1", expect: vcsData{commit: hash, treeState: "archive"}},
		"tag-with-extra": {commit: hash, refs: "tag: " + base + "-x", expect: vcsData{commit: hash, treeState: "archive"}},
	} {
		t.Run(name, func(t *testing.T) {
			if actual := archiveInfo(tc.commit, tc.refs); actual != tc.expect {
				t.Errorf("expected %+v, got %+v", tc.expect, actual)
			}
		})
	}
}
