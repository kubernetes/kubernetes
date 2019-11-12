// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kunstruct

import (
	"encoding/json"
	"fmt"

	"sigs.k8s.io/kustomize/api/hasher"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/pseudo/k8s/api/core/v1"
	"sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/apis/meta/v1/unstructured"
)

// kustHash computes a hash of an unstructured object.
type kustHash struct{}

// NewKustHash returns a kustHash object
func NewKustHash() *kustHash {
	return &kustHash{}
}

// Hash returns a hash of either a ConfigMap or a Secret
func (h *kustHash) Hash(m ifc.Kunstructured) (string, error) {
	u := unstructured.Unstructured{
		Object: m.Map(),
	}
	kind := u.GetKind()
	switch kind {
	case "ConfigMap":
		cm, err := unstructuredToConfigmap(u)
		if err != nil {
			return "", err
		}
		return configMapHash(cm)
	case "Secret":
		sec, err := unstructuredToSecret(u)

		if err != nil {
			return "", err
		}
		return secretHash(sec)
	default:
		return "", fmt.Errorf(
			"type %s is not supported for hashing in %v",
			kind, m.Map())
	}
}

// configMapHash returns a hash of the ConfigMap.
// The Data, Kind, and Name are taken into account.
func configMapHash(cm *v1.ConfigMap) (string, error) {
	encoded, err := encodeConfigMap(cm)
	if err != nil {
		return "", err
	}
	h, err := hasher.Encode(hasher.Hash(encoded))
	if err != nil {
		return "", err
	}
	return h, nil
}

// SecretHash returns a hash of the Secret.
// The Data, Kind, Name, and Type are taken into account.
func secretHash(sec *v1.Secret) (string, error) {
	encoded, err := encodeSecret(sec)
	if err != nil {
		return "", err
	}
	h, err := hasher.Encode(hasher.Hash(encoded))
	if err != nil {
		return "", err
	}
	return h, nil
}

// encodeConfigMap encodes a ConfigMap.
// Data, Kind, and Name are taken into account.
func encodeConfigMap(cm *v1.ConfigMap) (string, error) {
	// json.Marshal sorts the keys in a stable order in the encoding
	m := map[string]interface{}{"kind": "ConfigMap", "name": cm.Name, "data": cm.Data}
	if len(cm.BinaryData) > 0 {
		m["binaryData"] = cm.BinaryData
	}
	data, err := json.Marshal(m)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// encodeSecret encodes a Secret.
// Data, Kind, Name, and Type are taken into account.
func encodeSecret(sec *v1.Secret) (string, error) {
	// json.Marshal sorts the keys in a stable order in the encoding
	data, err := json.Marshal(map[string]interface{}{"kind": "Secret", "type": sec.Type, "name": sec.Name, "data": sec.Data})
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func unstructuredToConfigmap(u unstructured.Unstructured) (*v1.ConfigMap, error) {
	marshaled, err := json.Marshal(u.Object)
	if err != nil {
		return nil, err
	}
	var out v1.ConfigMap
	err = json.Unmarshal(marshaled, &out)
	return &out, err
}

func unstructuredToSecret(u unstructured.Unstructured) (*v1.Secret, error) {
	marshaled, err := json.Marshal(u.Object)
	if err != nil {
		return nil, err
	}
	var out v1.Secret
	err = json.Unmarshal(marshaled, &out)
	return &out, err
}
