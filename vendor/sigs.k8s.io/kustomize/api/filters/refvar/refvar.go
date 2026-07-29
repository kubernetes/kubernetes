// Copyright 2022 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package refvar

import (
	"fmt"
	"strconv"

	"sigs.k8s.io/kustomize/api/filters/fieldspec"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Filter updates $(VAR) style variables with values.
// The fieldSpecs are the places to look for occurrences of $(VAR).
type Filter struct {
	MappingFunc MappingFunc     `json:"mappingFunc,omitempty" yaml:"mappingFunc,omitempty"`
	FieldSpec   types.FieldSpec `json:"fieldSpec,omitempty" yaml:"fieldSpec,omitempty"`
}

func (f Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	return kio.FilterAll(yaml.FilterFunc(f.run)).Filter(nodes)
}

func (f Filter) run(node *yaml.RNode) (*yaml.RNode, error) {
	err := node.PipeE(fieldspec.Filter{
		FieldSpec: f.FieldSpec,
		SetValue:  f.set,
	})
	return node, err
}

func (f Filter) set(node *yaml.RNode) error {
	if yaml.IsMissingOrNull(node) {
		return nil
	}
	switch node.YNode().Kind {
	case yaml.ScalarNode:
		return f.setScalar(node)
	case yaml.MappingNode:
		return f.setMap(node)
	case yaml.SequenceNode:
		return f.setSeq(node)
	default:
		return fmt.Errorf("invalid type encountered %v", node.YNode().Kind)
	}
}

func updateNodeValue(node *yaml.Node, newValue interface{}) {
	switch newValue := newValue.(type) {
	case int:
		node.Value = strconv.FormatInt(int64(newValue), 10)
		node.Tag = yaml.NodeTagInt
	case int32:
		node.Value = strconv.FormatInt(int64(newValue), 10)
		node.Tag = yaml.NodeTagInt
	case int64:
		node.Value = strconv.FormatInt(newValue, 10)
		node.Tag = yaml.NodeTagInt
	case bool:
		node.SetString(strconv.FormatBool(newValue))
		node.Tag = yaml.NodeTagBool
	case float32:
		node.SetString(strconv.FormatFloat(float64(newValue), 'f', -1, 32))
		node.Tag = yaml.NodeTagFloat
	case float64:
		node.SetString(strconv.FormatFloat(newValue, 'f', -1, 64))
		node.Tag = yaml.NodeTagFloat
	default:
		node.SetString(newValue.(string))
		node.Tag = yaml.NodeTagString
	}
	node.Style = 0
}

func (f Filter) setScalar(node *yaml.RNode) error {
	if !yaml.IsYNodeString(node.YNode()) {
		return nil
	}
	v := DoReplacements(node.YNode().Value, f.MappingFunc)
	updateNodeValue(node.YNode(), v)
	return nil
}

func (f Filter) setMap(node *yaml.RNode) error {
	contents := node.YNode().Content
	for i := 0; i < len(contents); i += 2 {
		if !yaml.IsYNodeString(contents[i]) {
			return fmt.Errorf(
				"invalid map key: value='%s', tag='%s'",
				contents[i].Value, contents[i].Tag)
		}
		if !yaml.IsYNodeString(contents[i+1]) {
			continue
		}
		newValue := DoReplacements(contents[i+1].Value, f.MappingFunc)
		updateNodeValue(contents[i+1], newValue)
	}
	return nil
}

func (f Filter) setSeq(node *yaml.RNode) error {
	for _, item := range node.YNode().Content {
		if !yaml.IsYNodeString(item) {
			return fmt.Errorf("invalid value type expect a string")
		}
		newValue := DoReplacements(item.Value, f.MappingFunc)
		updateNodeValue(item, newValue)
	}
	return nil
}
