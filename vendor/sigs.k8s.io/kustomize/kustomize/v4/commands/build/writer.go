// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package build

import (
	"path/filepath"
	"strings"

	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/kyaml/filesys"
	"sigs.k8s.io/yaml"
)

type Writer struct {
	fSys filesys.FileSystem
}

func MakeWriter(fSys filesys.FileSystem) *Writer {
	return &Writer{
		fSys: fSys,
	}
}

func (w Writer) WriteIndividualFiles(dirPath string, m resmap.ResMap) error {
	byNamespace := m.GroupedByCurrentNamespace()
	for namespace, resList := range byNamespace {
		for _, res := range resList {
			fName := fileName(res)
			if len(byNamespace) > 1 {
				fName = strings.ToLower(namespace) + "_" + fName
			}
			if err := w.write(dirPath, fName, res); err != nil {
				return err
			}
		}
	}
	for _, res := range m.ClusterScoped() {
		err := w.write(dirPath, fileName(res), res)
		if err != nil {
			return err
		}
	}
	return nil
}

func (w Writer) write(path, fName string, res *resource.Resource) error {
	m, err := res.Map()
	if err != nil {
		return err
	}
	yml, err := yaml.Marshal(m)
	if err != nil {
		return err
	}
	return w.fSys.WriteFile(filepath.Join(path, fName), yml)
}

func fileName(res *resource.Resource) string {
	return strings.ToLower(res.GetGvk().StringWoEmptyField()) +
		"_" + strings.ToLower(res.GetName()) + ".yaml"
}
