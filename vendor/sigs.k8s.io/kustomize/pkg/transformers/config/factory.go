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

package config

import (
	"log"

	"github.com/ghodss/yaml"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/transformers/config/defaultconfig"
)

// Factory makes instances of TransformerConfig.
type Factory struct {
	ldr ifc.Loader
}

func NewFactory(l ifc.Loader) *Factory {
	return &Factory{ldr: l}
}

func (tf *Factory) loader() ifc.Loader {
	if tf.ldr.(ifc.Loader) == nil {
		log.Fatal("no loader")
	}
	return tf.ldr
}

// FromFiles returns a TranformerConfig object from a list of files
func (tf *Factory) FromFiles(
	paths []string) (*TransformerConfig, error) {
	result := &TransformerConfig{}
	for _, path := range paths {
		data, err := tf.loader().Load(path)
		if err != nil {
			return nil, err
		}
		t, err := makeTransformerConfigFromBytes(data)
		if err != nil {
			return nil, err
		}
		result = result.Merge(t)
	}
	return result, nil
}

// makeTransformerConfigFromBytes returns a TransformerConfig object from bytes
func makeTransformerConfigFromBytes(data []byte) (*TransformerConfig, error) {
	var t TransformerConfig
	err := yaml.Unmarshal(data, &t)
	if err != nil {
		return nil, err
	}
	t.sortFields()
	return &t, nil
}

// EmptyConfig returns an empty TransformerConfig object
func (tf *Factory) EmptyConfig() *TransformerConfig {
	return &TransformerConfig{}
}

// DefaultConfig returns a default TransformerConfig.
// This should never fail, hence the Fatal panic.
func (tf *Factory) DefaultConfig() *TransformerConfig {
	c, err := makeTransformerConfigFromBytes(
		defaultconfig.GetDefaultFieldSpecs())
	if err != nil {
		log.Fatalf("Unable to make default transformconfig: %v", err)
	}
	return c
}
