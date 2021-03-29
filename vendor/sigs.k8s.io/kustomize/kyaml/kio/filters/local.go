// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filters

import (
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

const LocalConfigAnnotation = "config.kubernetes.io/local-config"

// IsLocalConfig filters Resources using the config.kubernetes.io/local-config annotation
type IsLocalConfig struct {
	// IncludeLocalConfig will include local-config if set to true
	IncludeLocalConfig bool `yaml:"includeLocalConfig,omitempty"`

	// ExcludeNonLocalConfig will exclude non local-config if set to true
	ExcludeNonLocalConfig bool `yaml:"excludeNonLocalConfig,omitempty"`
}

// Filter implements kio.Filter
func (c *IsLocalConfig) Filter(inputs []*yaml.RNode) ([]*yaml.RNode, error) {
	var out []*yaml.RNode
	for i := range inputs {
		meta, err := inputs[i].GetMeta()
		if err != nil {
			return nil, err
		}
		_, local := meta.Annotations[LocalConfigAnnotation]

		if local && c.IncludeLocalConfig {
			out = append(out, inputs[i])
		} else if !local && !c.ExcludeNonLocalConfig {
			out = append(out, inputs[i])
		}
	}
	return out, nil
}
