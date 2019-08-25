// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package target

import (
	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/pkg/image"
	"sigs.k8s.io/kustomize/pkg/plugins"
	"sigs.k8s.io/kustomize/pkg/transformers"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/kustomize/plugin/builtin"
	"sigs.k8s.io/yaml"
)

// Functions dedicated to configuring the builtin
// transformer and generator plugins using config data
// read from a kustomization file.
//
// Non-builtin plugins will get their configuration
// from their own dedicated structs and yaml files.
//
// There are some loops in the functions below because
// the kustomization file would, say, allow one to
// request multiple secrets be made, or run multiple
// image tag transforms, so we need to run the plugins
// N times (plugins are easier to write, configure and
// test if they do just one thing).
//
// TODO: Push code down into the plugins, as the first pass
//     at this writes plugins as thin layers over calls
//     into existing packages.  The builtin plugins should
//     be viewed as examples, and the packages they access
//     directory should be public, while everything else
//     should go into internal.

type generatorConfigurator func() ([]transformers.Generator, error)
type transformerConfigurator func(
	tConfig *config.TransformerConfig) ([]transformers.Transformer, error)

func (kt *KustTarget) configureBuiltinGenerators() (
	[]transformers.Generator, error) {
	configurators := []generatorConfigurator{
		kt.configureBuiltinConfigMapGenerator,
		kt.configureBuiltinSecretGenerator,
	}
	var result []transformers.Generator
	for _, f := range configurators {
		r, err := f()
		if err != nil {
			return nil, err
		}
		result = append(result, r...)
	}
	return result, nil
}

func (kt *KustTarget) configureBuiltinTransformers(
	tConfig *config.TransformerConfig) (
	[]transformers.Transformer, error) {
	// TODO: Convert remaining legacy transformers to plugins
	//   with tests:
	//   - patch SMP
	configurators := []transformerConfigurator{
		kt.configureBuiltinNamespaceTransformer,
		kt.configureBuiltinNameTransformer,
		kt.configureBuiltinImageTagTransformer,
		kt.configureBuiltinLabelTransformer,
		kt.configureBuiltinAnnotationsTransformer,
		kt.configureBuiltinPatchJson6902Transformer,
		kt.configureBuiltinReplicaCountTransformer,
	}
	var result []transformers.Transformer
	for _, f := range configurators {
		r, err := f(tConfig)
		if err != nil {
			return nil, err
		}
		result = append(result, r...)
	}

	return result, nil
}

func (kt *KustTarget) configureBuiltinSecretGenerator() (
	result []transformers.Generator, err error) {
	var c struct {
		types.GeneratorOptions
		types.SecretArgs
	}
	if kt.kustomization.GeneratorOptions != nil {
		c.GeneratorOptions = *kt.kustomization.GeneratorOptions
	}
	for _, args := range kt.kustomization.SecretGenerator {
		c.SecretArgs = args
		p := builtin.NewSecretGeneratorPlugin()
		err = kt.configureBuiltinPlugin(p, c, "secret")
		if err != nil {
			return nil, err
		}
		result = append(result, p)
	}
	return
}

func (kt *KustTarget) configureBuiltinConfigMapGenerator() (
	result []transformers.Generator, err error) {
	var c struct {
		types.GeneratorOptions
		types.ConfigMapArgs
	}
	if kt.kustomization.GeneratorOptions != nil {
		c.GeneratorOptions = *kt.kustomization.GeneratorOptions
	}
	for _, args := range kt.kustomization.ConfigMapGenerator {
		c.ConfigMapArgs = args
		p := builtin.NewConfigMapGeneratorPlugin()
		err = kt.configureBuiltinPlugin(p, c, "configmap")
		if err != nil {
			return nil, err
		}
		result = append(result, p)
	}
	return
}

func (kt *KustTarget) configureBuiltinNamespaceTransformer(
	tConfig *config.TransformerConfig) (
	result []transformers.Transformer, err error) {
	var c struct {
		Namespace  string
		FieldSpecs []config.FieldSpec
	}
	c.Namespace = kt.kustomization.Namespace
	c.FieldSpecs = tConfig.NameSpace
	p := builtin.NewNamespaceTransformerPlugin()
	err = kt.configureBuiltinPlugin(p, c, "namespace")
	if err != nil {
		return nil, err
	}
	result = append(result, p)
	return
}

func (kt *KustTarget) configureBuiltinPatchJson6902Transformer(
	tConfig *config.TransformerConfig) (
	result []transformers.Transformer, err error) {
	var c struct {
		Patches []types.PatchJson6902
	}
	c.Patches = kt.kustomization.PatchesJson6902
	p := builtin.NewPatchJson6902TransformerPlugin()
	err = kt.configureBuiltinPlugin(p, c, "patchJson6902")
	if err != nil {
		return nil, err
	}
	result = append(result, p)
	return
}

func (kt *KustTarget) configureBuiltinLabelTransformer(
	tConfig *config.TransformerConfig) (
	result []transformers.Transformer, err error) {
	var c struct {
		Labels     map[string]string
		FieldSpecs []config.FieldSpec
	}
	c.Labels = kt.kustomization.CommonLabels
	c.FieldSpecs = tConfig.CommonLabels
	p := builtin.NewLabelTransformerPlugin()
	err = kt.configureBuiltinPlugin(p, c, "label")
	if err != nil {
		return nil, err
	}
	result = append(result, p)
	return
}

func (kt *KustTarget) configureBuiltinAnnotationsTransformer(
	tConfig *config.TransformerConfig) (
	result []transformers.Transformer, err error) {
	var c struct {
		Annotations map[string]string
		FieldSpecs  []config.FieldSpec
	}
	c.Annotations = kt.kustomization.CommonAnnotations
	c.FieldSpecs = tConfig.CommonAnnotations
	p := builtin.NewAnnotationsTransformerPlugin()
	err = kt.configureBuiltinPlugin(p, c, "annotations")
	if err != nil {
		return nil, err
	}
	result = append(result, p)
	return
}

func (kt *KustTarget) configureBuiltinNameTransformer(
	tConfig *config.TransformerConfig) (
	result []transformers.Transformer, err error) {
	var c struct {
		Prefix     string
		Suffix     string
		FieldSpecs []config.FieldSpec
	}
	c.Prefix = kt.kustomization.NamePrefix
	c.Suffix = kt.kustomization.NameSuffix
	c.FieldSpecs = tConfig.NamePrefix
	p := builtin.NewPrefixSuffixTransformerPlugin()
	err = kt.configureBuiltinPlugin(p, c, "prefixsuffix")
	if err != nil {
		return nil, err
	}
	result = append(result, p)
	return
}

func (kt *KustTarget) configureBuiltinImageTagTransformer(
	tConfig *config.TransformerConfig) (
	result []transformers.Transformer, err error) {
	var c struct {
		ImageTag   image.Image
		FieldSpecs []config.FieldSpec
	}
	for _, args := range kt.kustomization.Images {
		c.ImageTag = args
		c.FieldSpecs = tConfig.Images
		p := builtin.NewImageTagTransformerPlugin()
		err = kt.configureBuiltinPlugin(p, c, "imageTag")
		if err != nil {
			return nil, err
		}
		result = append(result, p)
	}
	return
}

func (kt *KustTarget) configureBuiltinReplicaCountTransformer(
	tConfig *config.TransformerConfig) (
	result []transformers.Transformer, err error) {
	var c struct {
		Replica    types.Replica
		FieldSpecs []config.FieldSpec
	}
	for _, args := range kt.kustomization.Replicas {
		c.Replica = args
		c.FieldSpecs = tConfig.Replicas
		p := builtin.NewReplicaCountTransformerPlugin()
		err = kt.configureBuiltinPlugin(p, c, "replica")
		if err != nil {
			return nil, err
		}
		result = append(result, p)
	}
	return
}

func (kt *KustTarget) configureBuiltinPlugin(
	p plugins.Configurable, c interface{}, id string) (err error) {
	var y []byte
	if c != nil {
		y, err = yaml.Marshal(c)
		if err != nil {
			return errors.Wrapf(
				err, "builtin %s marshal", id)
		}
	}
	err = p.Config(kt.ldr, kt.rFactory, y)
	if err != nil {
		return errors.Wrapf(err, "builtin %s config: %v", id, y)
	}
	return nil
}
