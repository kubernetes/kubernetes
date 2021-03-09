// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package hasher

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sort"

	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// SortArrayAndComputeHash sorts a string array and
// returns a hash for it
func SortArrayAndComputeHash(s []string) (string, error) {
	sort.Strings(s)
	data, err := json.Marshal(s)
	if err != nil {
		return "", err
	}
	return Encode(Hash(string(data)))
}

// Copied from https://github.com/kubernetes/kubernetes
// /blob/master/pkg/kubectl/util/hash/hash.go
func Encode(hex string) (string, error) {
	if len(hex) < 10 {
		return "", fmt.Errorf(
			"input length must be at least 10")
	}
	enc := []rune(hex[:10])
	for i := range enc {
		switch enc[i] {
		case '0':
			enc[i] = 'g'
		case '1':
			enc[i] = 'h'
		case '3':
			enc[i] = 'k'
		case 'a':
			enc[i] = 'm'
		case 'e':
			enc[i] = 't'
		}
	}
	return string(enc), nil
}

// Hash returns the hex form of the sha256 of the argument.
func Hash(data string) string {
	return fmt.Sprintf("%x", sha256.Sum256([]byte(data)))
}

// HashRNode returns the hash value of input RNode
func HashRNode(node *yaml.RNode) (string, error) {
	// get node kind
	kindNode, err := node.Pipe(yaml.FieldMatcher{Name: "kind"})
	if err != nil {
		return "", err
	}
	kind := kindNode.YNode().Value

	// calculate hash for different kinds
	encoded := ""
	switch kind {
	case "ConfigMap":
		encoded, err = encodeConfigMap(node)
	case "Secret":
		encoded, err = encodeSecret(node)
	default:
		var encodedBytes []byte
		encodedBytes, err = json.Marshal(node.YNode())
		encoded = string(encodedBytes)
	}
	if err != nil {
		return "", err
	}
	return Encode(Hash(encoded))
}

func getNodeValues(node *yaml.RNode, paths []string) (map[string]interface{}, error) {
	values := make(map[string]interface{})
	for _, p := range paths {
		vn, err := node.Pipe(yaml.Lookup(p))
		if err != nil {
			return map[string]interface{}{}, err
		}
		if vn == nil {
			values[p] = ""
			continue
		}
		if vn.YNode().Kind != yaml.ScalarNode {
			vs, err := vn.MarshalJSON()
			if err != nil {
				return map[string]interface{}{}, err
			}
			// data, binaryData and stringData are all maps
			var v map[string]interface{}
			json.Unmarshal(vs, &v)
			values[p] = v
		} else {
			values[p] = vn.YNode().Value
		}
	}
	return values, nil
}

// encodeConfigMap encodes a ConfigMap.
// Data, Kind, and Name are taken into account.
// BinaryData is included if it's not empty to avoid useless key in output.
func encodeConfigMap(node *yaml.RNode) (string, error) {
	// get fields
	paths := []string{"metadata/name", "data", "binaryData"}
	values, err := getNodeValues(node, paths)
	if err != nil {
		return "", err
	}
	m := map[string]interface{}{"kind": "ConfigMap", "name": values["metadata/name"],
		"data": values["data"]}
	if _, ok := values["binaryData"].(map[string]interface{}); ok {
		m["binaryData"] = values["binaryData"]
	}

	// json.Marshal sorts the keys in a stable order in the encoding
	data, err := json.Marshal(m)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// encodeSecret encodes a Secret.
// Data, Kind, Name, and Type are taken into account.
// StringData is included if it's not empty to avoid useless key in output.
func encodeSecret(node *yaml.RNode) (string, error) {
	// get fields
	paths := []string{"type", "metadata/name", "data", "stringData"}
	values, err := getNodeValues(node, paths)
	if err != nil {
		return "", err
	}
	m := map[string]interface{}{"kind": "Secret", "type": values["type"],
		"name": values["metadata/name"], "data": values["data"]}
	if _, ok := values["stringData"].(map[string]interface{}); ok {
		m["stringData"] = values["stringData"]
	}

	// json.Marshal sorts the keys in a stable order in the encoding
	data, err := json.Marshal(m)
	if err != nil {
		return "", err
	}
	return string(data), nil
}
