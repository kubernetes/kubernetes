/*
Copyright 2019 The Kubernetes Authors.

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

package metrics_test

import (
	"fmt"
	"strings"
	"testing"

	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/testutil"
	"k8s.io/component-base/version"
)

func TestRegisterBuildInfo(t *testing.T) {
	registry := metrics.NewKubeRegistry()

	metrics.RegisterBuildInfo(registry)

	info := version.Get()

	want := fmt.Sprintf(`
		# HELP kubernetes_build_info [BETA] A metric with a constant '1' value labeled by major, minor, git version, git commit, git tree state, build date, Go version, and compiler from which Kubernetes was built, and platform on which it is running.
		# TYPE kubernetes_build_info gauge
		kubernetes_build_info{build_date=%q,compiler=%q,git_commit=%q,git_tree_state=%q,git_version=%q,go_version=%q,major=%q,minor=%q,platform=%q} 1
	`,
		info.BuildDate,
		info.Compiler,
		info.GitCommit,
		info.GitTreeState,
		info.GitVersion,
		info.GoVersion,
		info.Major,
		info.Minor,
		info.Platform,
	)

	if err := testutil.GatherAndCompare(registry, strings.NewReader(want), "kubernetes_build_info"); err != nil {
		t.Fatal(err)
	}
}
