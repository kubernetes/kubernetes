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

// Package tools is used to track binary dependencies with go modules
// https://github.com/golang/go/wiki/Modules#how-can-i-track-tool-dependencies-for-a-module
package tools

import (
	// linting tools
	_ "github.com/client9/misspell/cmd/misspell"
	_ "golang.org/x/lint/golint"
	_ "honnef.co/go/tools/cmd/staticcheck"

	// benchmarking tools
	_ "github.com/cespare/prettybench"
	_ "gotest.tools"
	_ "gotest.tools/gotestsum"

	// bazel-related tools
	_ "github.com/bazelbuild/bazel-gazelle/cmd/gazelle"
	_ "github.com/bazelbuild/buildtools/buildozer"
	_ "k8s.io/repo-infra/cmd/kazel"
)
