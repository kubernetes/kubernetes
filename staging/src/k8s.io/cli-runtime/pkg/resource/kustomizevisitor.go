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

package resource

import (
	"bytes"

	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/api/krusty"
)

// KustomizeVisitor handles kustomization.yaml files.
type KustomizeVisitor struct {
	mapper *mapper
	schema ContentValidator
	// Directory expected to contain a kustomization file.
	dirPath string
	// File system containing dirPath.
	fSys filesys.FileSystem
	// Holds result of kustomize build, retained for tests.
	yml []byte
}

// Visit passes the result of a kustomize build to a StreamVisitor.
func (v *KustomizeVisitor) Visit(fn VisitorFunc) error {
	kOpts := krusty.MakeDefaultOptions()
	kOpts.DoLegacyResourceSort = true
	k := krusty.MakeKustomizer(kOpts)
	m, err := k.Run(v.fSys, v.dirPath)
	if err != nil {
		return err
	}
	v.yml, err = m.AsYaml()
	if err != nil {
		return err
	}
	sv := NewStreamVisitor(
		bytes.NewReader(v.yml), v.mapper, v.dirPath, v.schema)
	return sv.Visit(fn)
}
