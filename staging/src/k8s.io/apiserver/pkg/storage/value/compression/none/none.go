/*
Copyright 2017 The Kubernetes Authors.

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

// Package snappy transforms values for storage in snappy compression
package snappy

import (
	"context"

	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/compression/registry"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics"
)

func init() {
	value.RegisterMetrics()
	metrics.RegisterMetrics()
}

func init() {
	runtime.Must(registry.DefaultCompressionRegistry.RegisterCompressionTransformer(newNoneCompressionTransformer()))
}

const transformerName = "none"

// newNoneCompressionTransformer returns a transformer which does nothing
func newNoneCompressionTransformer() registry.CompressionTransformer {
	return &noneCompressionTransformer{}
}

var _ registry.CompressionTransformer = &noneCompressionTransformer{}

type noneCompressionTransformer struct {
}

func (t *noneCompressionTransformer) Name() string {
	return transformerName
}

func (t *noneCompressionTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	return data, false, nil
}

func (t *noneCompressionTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	return data, nil
}
