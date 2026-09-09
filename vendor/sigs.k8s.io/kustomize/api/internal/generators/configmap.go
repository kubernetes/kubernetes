// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package generators

import (
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// MakeConfigMap makes a configmap.
//
// ConfigMap: https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.19/#configmap-v1-core
//
// ConfigMaps and Secrets are similar.
//
// Both objects have a `data` field, which contains a map from keys to
// values that must be UTF-8 valid strings.  Such data might be simple text,
// or whoever made the data may have done so by performing a base64 encoding
// on binary data. Regardless, k8s has no means to know this, so it treats
// the data field as a string.
//
// The ConfigMap has an additional field `binaryData`, also a map, but its
// values are _intended_ to be interpreted as a base64 encoding of []byte,
// by whatever makes use of the ConfigMap.
//
// In a ConfigMap, any key used in `data` cannot also be used in `binaryData`
// and vice-versa.  A key must be unique across both maps.
func MakeConfigMap(
	ldr ifc.KvLoader, args *types.ConfigMapArgs) (rn *yaml.RNode, err error) {
	rn, err = makeBaseNode("ConfigMap", args.Name, args.Namespace)
	if err != nil {
		return nil, err
	}
	m, err := makeValidatedDataMap(ldr, args.Name, args.KvPairSources)
	if err != nil {
		return nil, err
	}
	if err = rn.LoadMapIntoConfigMapData(m); err != nil {
		return nil, err
	}
	err = copyLabelsAndAnnotations(rn, args.Options)
	if err != nil {
		return nil, err
	}
	err = setImmutable(rn, args.Options)
	if err != nil {
		return nil, err
	}
	return rn, nil
}
