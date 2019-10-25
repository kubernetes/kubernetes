// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package configmapandsecret generates configmaps and secrets per generator rules.
package configmapandsecret

import (
	"fmt"
	"unicode/utf8"

	"github.com/pkg/errors"
	"k8s.io/api/core/v1"
	"sigs.k8s.io/kustomize/api/types"
)

func makeFreshConfigMap(
	args *types.ConfigMapArgs) *v1.ConfigMap {
	cm := &v1.ConfigMap{}
	cm.APIVersion = "v1"
	cm.Kind = "ConfigMap"
	cm.Name = args.Name
	cm.Namespace = args.Namespace
	cm.Data = map[string]string{}
	return cm
}

// MakeConfigMap returns a new ConfigMap, or nil and an error.
func (f *Factory) MakeConfigMap(
	args *types.ConfigMapArgs) (*v1.ConfigMap, error) {
	all, err := f.kvLdr.Load(args.KvPairSources)
	if err != nil {
		return nil, errors.Wrap(err, "loading KV pairs")
	}
	cm := makeFreshConfigMap(args)
	for _, p := range all {
		err = f.addKvToConfigMap(cm, p)
		if err != nil {
			return nil, errors.Wrap(err, "trouble mapping")
		}
	}
	if f.options != nil {
		cm.SetLabels(f.options.Labels)
		cm.SetAnnotations(f.options.Annotations)
	}
	return cm, nil
}

// addKvToConfigMap adds the given key and data to the given config map.
// Error if key invalid, or already exists.
func (f *Factory) addKvToConfigMap(configMap *v1.ConfigMap, p types.Pair) error {
	if err := f.kvLdr.Validator().ErrIfInvalidKey(p.Key); err != nil {
		return err
	}
	// If the configmap data contains byte sequences that are all in the UTF-8
	// range, we will write it to .Data
	if utf8.Valid([]byte(p.Value)) {
		if _, entryExists := configMap.Data[p.Key]; entryExists {
			return fmt.Errorf(keyExistsErrorMsg, p.Key, configMap.Data)
		}
		configMap.Data[p.Key] = p.Value
		return nil
	}
	// otherwise, it's BinaryData
	if configMap.BinaryData == nil {
		configMap.BinaryData = map[string][]byte{}
	}
	if _, entryExists := configMap.BinaryData[p.Key]; entryExists {
		return fmt.Errorf(keyExistsErrorMsg, p.Key, configMap.BinaryData)
	}
	configMap.BinaryData[p.Key] = []byte(p.Value)
	return nil
}
