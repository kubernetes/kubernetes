/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package configmapandsecret generates configmaps and secrets per generator rules.
package configmapandsecret

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"github.com/pkg/errors"
	"k8s.io/api/core/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/cli-runtime/pkg/kustomize/k8sdeps/kv"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/types"
)

// ConfigMapFactory makes ConfigMaps.
type ConfigMapFactory struct {
	ldr ifc.Loader
}

// NewConfigMapFactory returns a new ConfigMapFactory.
func NewConfigMapFactory(l ifc.Loader) *ConfigMapFactory {
	return &ConfigMapFactory{ldr: l}
}

func (f *ConfigMapFactory) makeFreshConfigMap(
	args *types.ConfigMapArgs) *corev1.ConfigMap {
	cm := &corev1.ConfigMap{}
	cm.APIVersion = "v1"
	cm.Kind = "ConfigMap"
	cm.Name = args.Name
	cm.Namespace = args.Namespace
	cm.Data = map[string]string{}
	return cm
}

// MakeConfigMap returns a new ConfigMap, or nil and an error.
func (f *ConfigMapFactory) MakeConfigMap(
	args *types.ConfigMapArgs, options *types.GeneratorOptions) (*corev1.ConfigMap, error) {
	var all []kv.Pair
	var err error
	cm := f.makeFreshConfigMap(args)

	pairs, err := keyValuesFromEnvFile(f.ldr, args.EnvSource)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf(
			"env source file: %s",
			args.EnvSource))
	}
	all = append(all, pairs...)

	pairs, err = keyValuesFromLiteralSources(args.LiteralSources)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf(
			"literal sources %v", args.LiteralSources))
	}
	all = append(all, pairs...)

	pairs, err = keyValuesFromFileSources(f.ldr, args.FileSources)
	if err != nil {
		return nil, errors.Wrap(err, fmt.Sprintf(
			"file sources: %v", args.FileSources))
	}
	all = append(all, pairs...)

	for _, p := range all {
		err = addKvToConfigMap(cm, p.Key, p.Value)
		if err != nil {
			return nil, err
		}
	}
	if options != nil {
		cm.SetLabels(options.Labels)
		cm.SetAnnotations(options.Annotations)
	}
	return cm, nil
}

// addKvToConfigMap adds the given key and data to the given config map.
// Error if key invalid, or already exists.
func addKvToConfigMap(configMap *v1.ConfigMap, keyName, data string) error {
	// Note, the rules for ConfigMap keys are the exact same as the ones for SecretKeys.
	if errs := validation.IsConfigMapKey(keyName); len(errs) != 0 {
		return fmt.Errorf("%q is not a valid key name for a ConfigMap: %s", keyName, strings.Join(errs, ";"))
	}

	keyExistsErrorMsg := "cannot add key %s, another key by that name already exists: %v"

	// If the configmap data contains byte sequences that are all in the UTF-8
	// range, we will write it to .Data
	if utf8.Valid([]byte(data)) {
		if _, entryExists := configMap.Data[keyName]; entryExists {
			return fmt.Errorf(keyExistsErrorMsg, keyName, configMap.Data)
		}
		configMap.Data[keyName] = data
		return nil
	}

	// otherwise, it's BinaryData
	if configMap.BinaryData == nil {
		configMap.BinaryData = map[string][]byte{}
	}
	if _, entryExists := configMap.BinaryData[keyName]; entryExists {
		return fmt.Errorf(keyExistsErrorMsg, keyName, configMap.BinaryData)
	}
	configMap.BinaryData[keyName] = []byte(data)
	return nil
}
