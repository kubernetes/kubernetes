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
	"sigs.k8s.io/kustomize/api/k8sdeps/kunstruct"
	"sigs.k8s.io/kustomize/api/k8sdeps/transformer"
	"sigs.k8s.io/kustomize/api/k8sdeps/validator"
	fLdr "sigs.k8s.io/kustomize/api/loader"
	"sigs.k8s.io/kustomize/api/plugins/builtins"
	"sigs.k8s.io/kustomize/api/plugins/config"
	pLdr "sigs.k8s.io/kustomize/api/plugins/loader"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/resource"
	"sigs.k8s.io/kustomize/api/target"
)

const (
	// TODO: get this from command line (arg or flag or env).
	// When true, this means sort the resources before emitting them,
	// per a particular sort order.  When false, don't do the sort,
	// and instead respect the depth-first resource input order as
	// specified by the kustomization files in the input tree.
	doLegacyResourceSort    = true
)

// RunKustomizeBuild runs kustomize build given a filesystem and a path
func RunKustomizeBuild(
	out io.Writer,
	fSys filesys.FileSystem,
	path string) error {
	pf := transformer.NewFactoryImpl()
	rf := resmap.NewFactory(
		resource.NewFactory(
			kunstruct.NewKunstructuredFactoryImpl()),
		pf)
	ldr, err := fLdr.NewLoader(
		fLdr.RestrictionRootOnly, path, fSys)
	defer ldr.Cleanup()
	kt, err := target.NewKustTarget(
		ldr,
		validator.NewKustValidator(),
		rf,
		pf,
		pLdr.NewLoader(config.DefaultPluginConfig(), rf),
	)
	if err != nil {
		return err
	}
	m, err := kt.MakeCustomizedResMap()
	if err != nil {
		return err
	}
	if doLegacyResourceSort {
		builtins.NewLegacyOrderTransformerPlugin().Transform(m)
	}
	res, err := m.AsYaml()
	if err != nil {
		return err
	}
	_, err = out.Write(res)
	return err
}
