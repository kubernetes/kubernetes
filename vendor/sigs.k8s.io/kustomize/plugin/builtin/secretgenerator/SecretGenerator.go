// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

//go:generate go run sigs.k8s.io/kustomize/cmd/pluginator
package main

import (
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/yaml"
)

type plugin struct {
	ldr ifc.Loader
	rf  *resmap.Factory
	types.GeneratorOptions
	types.SecretArgs
}

//noinspection GoUnusedGlobalVariable
var KustomizePlugin plugin

func (p *plugin) Config(
	ldr ifc.Loader, rf *resmap.Factory, config []byte) (err error) {
	p.GeneratorOptions = types.GeneratorOptions{}
	p.SecretArgs = types.SecretArgs{}
	err = yaml.Unmarshal(config, p)
	p.ldr = ldr
	p.rf = rf
	return
}

func (p *plugin) Generate() (resmap.ResMap, error) {
	return p.rf.FromSecretArgs(p.ldr, &p.GeneratorOptions, p.SecretArgs)
}
