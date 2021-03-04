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

package kustomize

import (
	"io"

	"k8s.io/cli-runtime/pkg/kustomize/k8sdeps"
	"sigs.k8s.io/kustomize/pkg/commands/build"
	"sigs.k8s.io/kustomize/pkg/fs"
)

// RunKustomizeBuild runs kustomize build given a filesystem and a path
func RunKustomizeBuild(out io.Writer, fSys fs.FileSystem, path string) error {
	f := k8sdeps.NewFactory()
	o := build.NewOptions(path, "")
	return o.RunBuild(out, fSys, f.ResmapF, f.TransformerF)
}
