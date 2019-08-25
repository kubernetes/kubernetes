// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/yaml"
)

// A secret generator example that gets data
// from a database (simulated by a hardcoded map).
type plugin struct {
	rf        *resmap.Factory
	ldr       ifc.Loader
	Name      string `json:"name,omitempty" yaml:"name,omitempty"`
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
	// List of keys to use in database lookups
	Keys []string `json:"keys,omitempty" yaml:"keys,omitempty"`
}

//noinspection GoUnusedGlobalVariable
//nolint: golint
var KustomizePlugin plugin

var database = map[string]string{
	"TREE":      "oak",
	"ROCKET":    "SaturnV",
	"FRUIT":     "apple",
	"VEGETABLE": "carrot",
	"SIMPSON":   "homer",
}

func (p *plugin) Config(
	ldr ifc.Loader, rf *resmap.Factory, c []byte) error {
	p.rf = rf
	p.ldr = ldr
	return yaml.Unmarshal(c, p)
}

// The plan here is to convert the plugin's input
// into the format used by the builtin secret generator plugin.
func (p *plugin) Generate() (resmap.ResMap, error) {
	args := types.SecretArgs{}
	args.Name = p.Name
	args.Namespace = p.Namespace
	for _, k := range p.Keys {
		if v, ok := database[k]; ok {
			args.LiteralSources = append(
				args.LiteralSources, k+"="+v)
		}
	}
	return p.rf.FromSecretArgs(p.ldr, nil, args)
}
