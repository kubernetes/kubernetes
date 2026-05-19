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

package version

import (
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/version"
)

var (
	buildInfo = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Name:           "kubernetes_build_info",
			Help:           "A metric with a constant '1' value labeled by major, minor, git version, git commit, git tree state, build date, Go version, and compiler from which Kubernetes was built, and platform on which it is running.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"major", "minor", "git_version", "git_commit", "git_tree_state", "build_date", "go_version", "compiler", "platform"},
	)
)

// RegisterBuildInfo registers the build and version info in a metadata metric in prometheus
func init() {
	info := version.Get()
	legacyregistry.MustRegister(buildInfo)
	buildInfo.WithLabelValues(info.Major, info.Minor, info.GitVersion, info.GitCommit, info.GitTreeState, info.BuildDate, info.GoVersion, info.Compiler, info.Platform).Set(1)
}
