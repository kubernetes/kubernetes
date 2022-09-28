// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinhelpers

import (
	"sigs.k8s.io/kustomize/api/internal/builtins"
	"sigs.k8s.io/kustomize/api/resmap"
)

//go:generate stringer -type=BuiltinPluginType
type BuiltinPluginType int

const (
	Unknown BuiltinPluginType = iota
	AnnotationsTransformer
	ConfigMapGenerator
	IAMPolicyGenerator
	HashTransformer
	ImageTagTransformer
	LabelTransformer
	LegacyOrderTransformer
	NamespaceTransformer
	PatchJson6902Transformer
	PatchStrategicMergeTransformer
	PatchTransformer
	PrefixSuffixTransformer
	PrefixTransformer
	SuffixTransformer
	ReplicaCountTransformer
	SecretGenerator
	ValueAddTransformer
	HelmChartInflationGenerator
	ReplacementTransformer
)

var stringToBuiltinPluginTypeMap map[string]BuiltinPluginType

func init() { //nolint:gochecknoinits
	stringToBuiltinPluginTypeMap = makeStringToBuiltinPluginTypeMap()
}

func makeStringToBuiltinPluginTypeMap() (result map[string]BuiltinPluginType) {
	result = make(map[string]BuiltinPluginType, 23)
	for k := range GeneratorFactories {
		result[k.String()] = k
	}
	for k := range TransformerFactories {
		result[k.String()] = k
	}
	return
}

func GetBuiltinPluginType(n string) BuiltinPluginType {
	result, ok := stringToBuiltinPluginTypeMap[n]
	if ok {
		return result
	}
	return Unknown
}

var GeneratorFactories = map[BuiltinPluginType]func() resmap.GeneratorPlugin{
	ConfigMapGenerator:          builtins.NewConfigMapGeneratorPlugin,
	IAMPolicyGenerator:          builtins.NewIAMPolicyGeneratorPlugin,
	SecretGenerator:             builtins.NewSecretGeneratorPlugin,
	HelmChartInflationGenerator: builtins.NewHelmChartInflationGeneratorPlugin,
}

type MultiTransformer struct {
	transformers []resmap.TransformerPlugin
}

func (t *MultiTransformer) Transform(m resmap.ResMap) error {
	for _, transformer := range t.transformers {
		if err := transformer.Transform(m); err != nil {
			return err
		}
	}
	return nil
}

func (t *MultiTransformer) Config(h *resmap.PluginHelpers, b []byte) error {
	for _, transformer := range t.transformers {
		if err := transformer.Config(h, b); err != nil {
			return err
		}
	}
	return nil
}

func NewMultiTransformer() resmap.TransformerPlugin {
	return &MultiTransformer{[]resmap.TransformerPlugin{
		builtins.NewPrefixTransformerPlugin(),
		builtins.NewSuffixTransformerPlugin(),
	}}
}

var TransformerFactories = map[BuiltinPluginType]func() resmap.TransformerPlugin{
	AnnotationsTransformer:         builtins.NewAnnotationsTransformerPlugin,
	HashTransformer:                builtins.NewHashTransformerPlugin,
	ImageTagTransformer:            builtins.NewImageTagTransformerPlugin,
	LabelTransformer:               builtins.NewLabelTransformerPlugin,
	LegacyOrderTransformer:         builtins.NewLegacyOrderTransformerPlugin,
	NamespaceTransformer:           builtins.NewNamespaceTransformerPlugin,
	PatchJson6902Transformer:       builtins.NewPatchJson6902TransformerPlugin,
	PatchStrategicMergeTransformer: builtins.NewPatchStrategicMergeTransformerPlugin,
	PatchTransformer:               builtins.NewPatchTransformerPlugin,
	PrefixSuffixTransformer:        NewMultiTransformer,
	PrefixTransformer:              builtins.NewPrefixTransformerPlugin,
	SuffixTransformer:              builtins.NewSuffixTransformerPlugin,
	ReplacementTransformer:         builtins.NewReplacementTransformerPlugin,
	ReplicaCountTransformer:        builtins.NewReplicaCountTransformerPlugin,
	ValueAddTransformer:            builtins.NewValueAddTransformerPlugin,
}
