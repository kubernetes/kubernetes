/*
Copyright 2019 The Kubernetes Authors.

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

package registry

import (
	"fmt"
	"sync"

	"k8s.io/apiserver/pkg/storage/value"
)

var DefaultCompressionRegistry = NewCompressionRegistry()

type CompressionRegistry interface {
	// RegisterCompressionTransformer registers a transformer for the provided resource.
	RegisterCompressionTransformer(transformer CompressionTransformer) error
	// GetCompressionTransformer returns the transformer for the provided name
	GetCompressionTransformer(compressionTransformerName string) CompressionTransformer
	// GetAllCompressionTransformers returns all the registered transformers
	GetAllCompressionTransformers() []CompressionTransformer
}

type CompressionTransformer interface {
	Name() string
	value.Transformer
}

var _ CompressionRegistry = &compressionRegistry{}

func NewCompressionRegistry() CompressionRegistry {
	return &compressionRegistry{
		transformers: map[string]CompressionTransformer{},
	}
}

type compressionRegistry struct {
	sync.RWMutex
	transformers map[string]CompressionTransformer
}

func (c *compressionRegistry) RegisterCompressionTransformer(transformer CompressionTransformer) error {
	c.Lock()
	defer c.Unlock()
	if _, ok := c.transformers[transformer.Name()]; ok {
		return fmt.Errorf("compression transformer %q already registered", transformer.Name())
	}
	c.transformers[transformer.Name()] = transformer
	return nil
}

func (c *compressionRegistry) GetCompressionTransformer(compressionTransformerName string) CompressionTransformer {
	c.RLock()
	defer c.RUnlock()
	return c.transformers[compressionTransformerName]
}

func (c *compressionRegistry) GetAllCompressionTransformers() []CompressionTransformer {
	c.RLock()
	defer c.RUnlock()
	transformers := make([]CompressionTransformer, 0, len(c.transformers))
	for _, t := range c.transformers {
		transformers = append(transformers, t)
	}
	return transformers
}
