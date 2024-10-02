// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package patchjson6902

import (
	"strings"

	jsonpatch "gopkg.in/evanphx/json-patch.v4"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
	k8syaml "sigs.k8s.io/yaml"
)

type Filter struct {
	Patch string

	decodedPatch jsonpatch.Patch
}

var _ kio.Filter = Filter{}

func (pf Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	decodedPatch, err := pf.decodePatch()
	if err != nil {
		return nil, err
	}
	pf.decodedPatch = decodedPatch
	return kio.FilterAll(yaml.FilterFunc(pf.run)).Filter(nodes)
}

func (pf Filter) decodePatch() (jsonpatch.Patch, error) {
	patch := pf.Patch
	// If the patch doesn't look like a JSON6902 patch, we
	// try to parse it to json.
	if !strings.HasPrefix(pf.Patch, "[") {
		p, err := k8syaml.YAMLToJSON([]byte(patch))
		if err != nil {
			return nil, err
		}
		patch = string(p)
	}
	decodedPatch, err := jsonpatch.DecodePatch([]byte(patch))
	if err != nil {
		return nil, err
	}
	return decodedPatch, nil
}

func (pf Filter) run(node *yaml.RNode) (*yaml.RNode, error) {
	// We don't actually use the kyaml library for manipulating the
	// yaml here. We just marshal it to json and rely on the
	// jsonpatch library to take care of applying the patch.
	// This means ordering might not be preserved with this filter.
	b, err := node.MarshalJSON()
	if err != nil {
		return nil, err
	}
	res, err := pf.decodedPatch.Apply(b)
	if err != nil {
		return nil, err
	}
	err = node.UnmarshalJSON(res)
	return node, err
}
