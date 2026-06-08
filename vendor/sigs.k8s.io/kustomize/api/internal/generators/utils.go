// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package generators

import (
	"fmt"
	"path"
	"strings"

	"github.com/go-errors/errors"
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

func makeBaseNode(kind, name, namespace string) (*yaml.RNode, error) {
	rn, err := yaml.Parse(fmt.Sprintf(`
apiVersion: v1
kind: %s
`, kind))
	if err != nil {
		return nil, err
	}
	if name == "" {
		return nil, errors.Errorf("a configmap must have a name")
	}
	if _, err := rn.Pipe(yaml.SetK8sName(name)); err != nil {
		return nil, err
	}
	if namespace != "" {
		if _, err := rn.Pipe(yaml.SetK8sNamespace(namespace)); err != nil {
			return nil, err
		}
	}
	return rn, nil
}

func makeValidatedDataMap(
	ldr ifc.KvLoader, name string, sources types.KvPairSources) (map[string]string, error) {
	pairs, err := ldr.Load(sources)
	if err != nil {
		return nil, errors.WrapPrefix(err, "loading KV pairs", 0)
	}
	knownKeys := make(map[string]string)
	for _, p := range pairs {
		// legal key: alphanumeric characters, '-', '_' or '.'
		if err := ldr.Validator().ErrIfInvalidKey(p.Key); err != nil {
			return nil, err
		}
		if _, ok := knownKeys[p.Key]; ok {
			return nil, errors.Errorf(
				"configmap %s illegally repeats the key `%s`", name, p.Key)
		}
		knownKeys[p.Key] = p.Value
	}
	return knownKeys, nil
}

// copyLabelsAndAnnotations copies labels and annotations from
// GeneratorOptions into the given object.
func copyLabelsAndAnnotations(
	rn *yaml.RNode, opts *types.GeneratorOptions) error {
	if opts == nil {
		return nil
	}
	for _, k := range yaml.SortedMapKeys(opts.Labels) {
		v := opts.Labels[k]
		if _, err := rn.Pipe(yaml.SetLabel(k, v)); err != nil {
			return err
		}
	}
	for _, k := range yaml.SortedMapKeys(opts.Annotations) {
		v := opts.Annotations[k]
		if _, err := rn.Pipe(yaml.SetAnnotation(k, v)); err != nil {
			return err
		}
	}
	return nil
}

func setImmutable(
	rn *yaml.RNode, opts *types.GeneratorOptions) error {
	if opts == nil {
		return nil
	}
	if opts.Immutable {
		n := &yaml.Node{
			Kind:  yaml.ScalarNode,
			Value: "true",
			Tag:   yaml.NodeTagBool,
		}
		if _, err := rn.Pipe(yaml.FieldSetter{Name: "immutable", Value: yaml.NewRNode(n)}); err != nil {
			return err
		}
	}

	return nil
}

// ParseFileSource parses the source given.
//
// Acceptable formats include:
//  1. source-path: the basename will become the key name
//  2. source-name=source-path: the source-name will become the key name and
//     source-path is the path to the key file.
//
// Key names cannot include '='.
func ParseFileSource(source string) (keyName, filePath string, err error) {
	numSeparators := strings.Count(source, "=")
	switch {
	case numSeparators == 0:
		return path.Base(source), source, nil
	case numSeparators == 1 && strings.HasPrefix(source, "="):
		return "", "", errors.Errorf("missing key name for file path %q in source %q", strings.TrimPrefix(source, "="), source)
	case numSeparators == 1 && strings.HasSuffix(source, "="):
		return "", "", errors.Errorf("missing file path for key name %q in source %q", strings.TrimSuffix(source, "="), source)
	case numSeparators > 1:
		return "", "", errors.Errorf("source %q key name or file path contains '='", source)
	default:
		components := strings.Split(source, "=")
		return components[0], components[1], nil
	}
}
