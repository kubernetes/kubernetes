// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// GeneratorOptions modify behavior of all ConfigMap and Secret generators.
type GeneratorOptions struct {
	// Labels to add to all generated resources.
	Labels map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`

	// Annotations to add to all generated resources.
	Annotations map[string]string `json:"annotations,omitempty" yaml:"annotations,omitempty"`

	// DisableNameSuffixHash if true disables the default behavior of adding a
	// suffix to the names of generated resources that is a hash of the
	// resource contents.
	DisableNameSuffixHash bool `json:"disableNameSuffixHash,omitempty" yaml:"disableNameSuffixHash,omitempty"`
}

// MergeGlobalOptionsIntoLocal merges two instances of GeneratorOptions.
// Values in the first 'local' argument cannot be overridden by the second
// 'global' argument, except in the case of booleans.
//
// With booleans, there's no way to distinguish an 'intentional'
// false from 'default' false.  So the rule is, if the global value
// of the value of a boolean is true, i.e. disable, it trumps the
// local value.  If the global value is false, then the local value is
// respected.  Bottom line: a local false cannot override a global true.
//
// boolean fields are always a bad idea; should always use enums instead.
func MergeGlobalOptionsIntoLocal(
	localOpts *GeneratorOptions,
	globalOpts *GeneratorOptions) *GeneratorOptions {
	if globalOpts == nil {
		return localOpts
	}
	if localOpts == nil {
		localOpts = &GeneratorOptions{}
	}
	overrideMap(&localOpts.Labels, globalOpts.Labels)
	overrideMap(&localOpts.Annotations, globalOpts.Annotations)
	if globalOpts.DisableNameSuffixHash {
		localOpts.DisableNameSuffixHash = true
	}
	return localOpts
}

func overrideMap(localMap *map[string]string, globalMap map[string]string) {
	if *localMap == nil {
		if globalMap != nil {
			*localMap = CopyMap(globalMap)
		}
		return
	}
	for k, v := range globalMap {
		_, ok := (*localMap)[k]
		if !ok {
			(*localMap)[k] = v
		}
	}
}

// CopyMap copies a map.
func CopyMap(in map[string]string) map[string]string {
	out := make(map[string]string)
	for k, v := range in {
		out[k] = v
	}
	return out
}
