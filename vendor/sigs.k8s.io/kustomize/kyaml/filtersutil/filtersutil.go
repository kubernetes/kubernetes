// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filtersutil

import (
	"encoding/json"

	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// ApplyToJSON applies the filter to the json objects.
//
// ApplyToJSON marshals the objects into a slice of yaml.RNodes, runs
// the filter on the slice, and then unmarshals the values back.
//
// The filter must not create or delete objects because the objects
// are updated in place.
func ApplyToJSON(filter kio.Filter, objs ...marshalerUnmarshaler) error {
	var nodes []*yaml.RNode

	// convert the json objects to rnodes
	for i := range objs {
		node, err := GetRNode(objs[i])
		if err != nil {
			return err
		}
		nodes = append(nodes, node)
	}

	// apply the filter
	nodes, err := filter.Filter(nodes)
	if err != nil {
		return err
	}
	if len(nodes) != len(objs) {
		return errors.Errorf("filter cannot create or delete objects")
	}

	// convert the rnodes to json objects
	for i := range nodes {
		err = setRNode(objs[i], nodes[i])
		if err != nil {
			return err
		}
	}

	return nil
}

type marshalerUnmarshaler interface {
	json.Unmarshaler
	json.Marshaler
}

// GetRNode converts k into an RNode
func GetRNode(k json.Marshaler) (*yaml.RNode, error) {
	j, err := k.MarshalJSON()
	if err != nil {
		return nil, err
	}
	return yaml.Parse(string(j))
}

// setRNode marshals node into k
func setRNode(k json.Unmarshaler, node *yaml.RNode) error {
	s, err := node.String()
	if err != nil {
		return err
	}
	m := map[string]interface{}{}
	if err := yaml.Unmarshal([]byte(s), &m); err != nil {
		return err
	}

	b, err := json.Marshal(m)
	if err != nil {
		return err
	}
	return k.UnmarshalJSON(b)
}
