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

	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/api/krusty"
)

// RunKustomizeBuild runs kustomize build given a filesystem and a path
func RunKustomizeBuild(out io.Writer, fSys filesys.FileSystem, path string) error {
	o := NewOptions(path, "")
	o.outOrder = legacy
	k := krusty.MakeKustomizer(fSys, o.makeOptions())
	m, err := k.Run(o.kustomizationPath)
	if err != nil {
		return err
	}
	return o.emitResources(out, fSys, m)
}
