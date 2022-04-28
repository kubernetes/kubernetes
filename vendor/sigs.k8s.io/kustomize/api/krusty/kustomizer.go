// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package krusty

import (
	"fmt"
	"path/filepath"

	"sigs.k8s.io/kustomize/api/internal/builtins"
	pLdr "sigs.k8s.io/kustomize/api/internal/plugins/loader"
	"sigs.k8s.io/kustomize/api/internal/target"
	"sigs.k8s.io/kustomize/api/internal/utils"
	"sigs.k8s.io/kustomize/api/konfig"
	fLdr "sigs.k8s.io/kustomize/api/loader"
	"sigs.k8s.io/kustomize/api/provenance"
	"sigs.k8s.io/kustomize/api/provider"
	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/filesys"
	"sigs.k8s.io/kustomize/kyaml/openapi"
)

// Kustomizer performs kustomizations.
//
// It's meant to behave similarly to the kustomize CLI, and can be
// used instead of performing an exec to a kustomize CLI subprocess.
// To use, load a filesystem with kustomization files (any
// number of overlays and bases), then make a Kustomizer
// injected with the given filesystem, then call Run.
type Kustomizer struct {
	options     *Options
	depProvider *provider.DepProvider
}

// MakeKustomizer returns an instance of Kustomizer.
func MakeKustomizer(o *Options) *Kustomizer {
	return &Kustomizer{
		options:     o,
		depProvider: provider.NewDepProvider(),
	}
}

// Run performs a kustomization.
//
// It reads given path from the given file system, interprets it as
// a kustomization.yaml file, perform the kustomization it represents,
// and return the resulting resources.
//
// Any files referenced by the kustomization must be present on the
// filesystem.  One may call Run any number of times, on any number
// of internal paths (e.g. the filesystem may contain multiple overlays,
// and Run can be called on each of them).
func (b *Kustomizer) Run(
	fSys filesys.FileSystem, path string) (resmap.ResMap, error) {
	resmapFactory := resmap.NewFactory(b.depProvider.GetResourceFactory())
	lr := fLdr.RestrictionNone
	if b.options.LoadRestrictions == types.LoadRestrictionsRootOnly {
		lr = fLdr.RestrictionRootOnly
	}
	ldr, err := fLdr.NewLoader(lr, path, fSys)
	if err != nil {
		return nil, err
	}
	defer ldr.Cleanup()
	kt := target.NewKustTarget(
		ldr,
		b.depProvider.GetFieldValidator(),
		resmapFactory,
		// The plugin configs are always located on disk, regardless of the fSys passed in
		pLdr.NewLoader(b.options.PluginConfig, resmapFactory, filesys.MakeFsOnDisk()),
	)
	err = kt.Load()
	if err != nil {
		return nil, err
	}
	var bytes []byte
	if openApiPath, exists := kt.Kustomization().OpenAPI["path"]; exists {
		bytes, err = ldr.Load(filepath.Join(ldr.Root(), openApiPath))
		if err != nil {
			return nil, err
		}
	}
	err = openapi.SetSchema(kt.Kustomization().OpenAPI, bytes, true)
	if err != nil {
		return nil, err
	}
	var m resmap.ResMap
	m, err = kt.MakeCustomizedResMap()
	if err != nil {
		return nil, err
	}
	if b.options.DoLegacyResourceSort {
		err = builtins.NewLegacyOrderTransformerPlugin().Transform(m)
		if err != nil {
			return nil, err
		}
	}
	if b.options.AddManagedbyLabel || utils.StringSliceContains(kt.Kustomization().BuildMetadata, types.ManagedByLabelOption) {
		t := builtins.LabelTransformerPlugin{
			Labels: map[string]string{
				konfig.ManagedbyLabelKey: fmt.Sprintf("kustomize-%s", provenance.GetProvenance().Semver()),
			},
			FieldSpecs: []types.FieldSpec{{
				Path:               "metadata/labels",
				CreateIfNotPresent: true,
			}},
		}
		err = t.Transform(m)
		if err != nil {
			return nil, err
		}
	}
	m.RemoveBuildAnnotations()
	if !utils.StringSliceContains(kt.Kustomization().BuildMetadata, types.OriginAnnotations) {
		m.RemoveOriginAnnotations()
	}
	if !utils.StringSliceContains(kt.Kustomization().BuildMetadata, types.TransformerAnnotations) {
		m.RemoveTransformerAnnotations()
	}
	return m, nil
}
