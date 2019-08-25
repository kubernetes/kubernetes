// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
	"sigs.k8s.io/kustomize/plugin/builtin"
	"sigs.k8s.io/yaml"
)

// Add a date prefix to the name.
// A plugin that adapts another plugin.
type plugin struct {
	t transformers.Transformer
}

//noinspection GoUnusedGlobalVariable
//nolint: golint
var KustomizePlugin plugin

func (p *plugin) makePrefixSuffixPluginConfig() ([]byte, error) {
	var s struct {
		Prefix     string
		Suffix     string
		FieldSpecs []config.FieldSpec
	}
	s.Prefix = getDate() + "-"
	s.FieldSpecs = []config.FieldSpec{
		{Path: "metadata/name"},
	}
	return yaml.Marshal(s)
}

func (p *plugin) Config(
	ldr ifc.Loader, rf *resmap.Factory, _ []byte) error {
	// Ignore the incoming c, compute new config.
	c, err := p.makePrefixSuffixPluginConfig()
	if err != nil {
		return errors.Wrapf(
			err, "dateprefixer makeconfig")
	}
	prefixer := builtin.NewPrefixSuffixTransformerPlugin()
	err = prefixer.Config(ldr, rf, c)
	if err != nil {
		return errors.Wrapf(
			err, "prefixsuffix configure")
	}
	p.t = prefixer
	return nil
}

// Returns a constant, rather than
//   time.Now().Format("2006-01-02")
// to make tests happy.
// This is just an example.
func getDate() string {
	return "2018-05-11"
}

func (p *plugin) Transform(m resmap.ResMap) error {
	return p.t.Transform(m)
}
