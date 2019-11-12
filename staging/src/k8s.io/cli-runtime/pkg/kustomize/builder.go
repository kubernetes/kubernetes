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
	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/krusty"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/pkg/commands/build"
	"sigs.k8s.io/kustomize/pkg/fs"
)

// LegacyRunKustomizeBuild depends on a very old release of kustomize.
// Do not use in new code.  Switch to RunKustomizeBuild below.
func LegacyRunKustomizeBuild(out io.Writer, fSys fs.FileSystem, path string) error {
	f := k8sdeps.NewFactory()
	o := build.NewOptions(path, "")
	return o.RunBuild(out, fSys, f.ResmapF, f.TransformerF)
}

// RunKustomizeBuild runs kustomize build given a filesystem and a path
func RunKustomizeBuild(out io.Writer, fSys filesys.FileSystem, path string) error {
	opts := &krusty.Options{
		// When true, this means sort the resources before emitting them,
		// per a particular sort order.  When false, don't do the sort,
		// and instead respect the depth-first resource input order as
		// specified by the kustomization files in the input tree.
		// TODO: get this from shell (arg, flag or env).
		DoLegacyResourceSort: true,
		// In the kubectl context, avoid security issues.
		LoadRestrictions: types.LoadRestrictionsRootOnly,
		// In the kubectl context, avoid security issues.
		PluginConfig: konfig.DisabledPluginConfig(),
	}
	m, err := krusty.MakeKustomizer(fSys, opts).Run(path)
	if err != nil {
		return err
	}
	res, err := m.AsYaml()
	if err != nil {
		return err
	}
	_, err = out.Write(res)
	return err
}
