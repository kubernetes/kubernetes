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
	"path"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/api/core/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/types"
)

// ConfigMapFactory makes ConfigMaps.
type ConfigMapFactory struct {
	fSys fs.FileSystem
	ldr  ifc.Loader
}

// NewConfigMapFactory returns a new ConfigMapFactory.
func NewConfigMapFactory(
	fSys fs.FileSystem, l ifc.Loader) *ConfigMapFactory {
	return &ConfigMapFactory{fSys: fSys, ldr: l}
}

func (f *ConfigMapFactory) makeFreshConfigMap(
	args *types.ConfigMapArgs) *corev1.ConfigMap {
	cm := &corev1.ConfigMap{}
	cm.APIVersion = "v1"
	cm.Kind = "ConfigMap"
	cm.Name = args.Name
	cm.Data = map[string]string{}
	return cm
}

// MakeConfigMap returns a new ConfigMap, or nil and an error.
func (f *ConfigMapFactory) MakeConfigMap(
	args *types.ConfigMapArgs, options *types.GeneratorOptions) (*corev1.ConfigMap, error) {
	var all []kvPair
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

	for _, kv := range all {
		err = addKvToConfigMap(cm, kv.key, kv.value)
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

func keyValuesFromLiteralSources(sources []string) ([]kvPair, error) {
	var kvs []kvPair
	for _, s := range sources {
		k, v, err := parseLiteralSource(s)
		if err != nil {
			return nil, err
		}
		kvs = append(kvs, kvPair{key: k, value: v})
	}
	return kvs, nil
}

func keyValuesFromFileSources(ldr ifc.Loader, sources []string) ([]kvPair, error) {
	var kvs []kvPair
	for _, s := range sources {
		k, fPath, err := parseFileSource(s)
		if err != nil {
			return nil, err
		}
		content, err := ldr.Load(fPath)
		if err != nil {
			return nil, err
		}
		kvs = append(kvs, kvPair{key: k, value: string(content)})
	}
	return kvs, nil
}

func keyValuesFromEnvFile(l ifc.Loader, path string) ([]kvPair, error) {
	if path == "" {
		return nil, nil
	}
	content, err := l.Load(path)
	if err != nil {
		return nil, err
	}
	return keyValuesFromLines(content)
}

// addKvToConfigMap adds the given key and data to the given config map.
// Error if key invalid, or already exists.
func addKvToConfigMap(configMap *v1.ConfigMap, keyName, data string) error {
	// Note, the rules for ConfigMap keys are the exact same as the ones for SecretKeys.
	if errs := validation.IsConfigMapKey(keyName); len(errs) != 0 {
		return fmt.Errorf("%q is not a valid key name for a ConfigMap: %s", keyName, strings.Join(errs, ";"))
	}
	if _, entryExists := configMap.Data[keyName]; entryExists {
		return fmt.Errorf("cannot add key %s, another key by that name already exists: %v", keyName, configMap.Data)
	}
	configMap.Data[keyName] = data
	return nil
}

// parseFileSource parses the source given.
//
//  Acceptable formats include:
//   1.  source-path: the basename will become the key name
//   2.  source-name=source-path: the source-name will become the key name and
//       source-path is the path to the key file.
//
// Key names cannot include '='.
func parseFileSource(source string) (keyName, filePath string, err error) {
	numSeparators := strings.Count(source, "=")
	switch {
	case numSeparators == 0:
		return path.Base(source), source, nil
	case numSeparators == 1 && strings.HasPrefix(source, "="):
		return "", "", fmt.Errorf("key name for file path %v missing", strings.TrimPrefix(source, "="))
	case numSeparators == 1 && strings.HasSuffix(source, "="):
		return "", "", fmt.Errorf("file path for key name %v missing", strings.TrimSuffix(source, "="))
	case numSeparators > 1:
		return "", "", errors.New("key names or file paths cannot contain '='")
	default:
		components := strings.Split(source, "=")
		return components[0], components[1], nil
	}
}

// parseLiteralSource parses the source key=val pair into its component pieces.
// This functionality is distinguished from strings.SplitN(source, "=", 2) since
// it returns an error in the case of empty keys, values, or a missing equals sign.
func parseLiteralSource(source string) (keyName, value string, err error) {
	// leading equal is invalid
	if strings.Index(source, "=") == 0 {
		return "", "", fmt.Errorf("invalid literal source %v, expected key=value", source)
	}
	// split after the first equal (so values can have the = character)
	items := strings.SplitN(source, "=", 2)
	if len(items) != 2 {
		return "", "", fmt.Errorf("invalid literal source %v, expected key=value", source)
	}
	return items[0], strings.Trim(items[1], "\"'"), nil
}
