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

package compression

import (
	"context"

	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/compression/registry"
)

func NewNegotiateCompressionTransformer(storageCompression string, delegatedTransformer value.Transformer) value.Transformer {
	return &negotiateCompressionTransformer{
		storageCompression: storageCompression,
	}
}

var _ value.Transformer = &negotiateCompressionTransformer{}

type negotiateCompressionTransformer struct {
	delegatedTransformer value.Transformer
	storageCompression   string
}

func (n *negotiateCompressionTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) (out []byte, stale bool, err error) {
	transformers := registry.DefaultCompressionRegistry.GetAllCompressionTransformers()
	out = data
	for _, t := range transformers {
		out, _, err = t.TransformFromStorage(ctx, out, dataCtx)
		if err != nil {
			return
		}
	}

	// call delegated transformer after compression transformer
	return n.delegatedTransformer.TransformFromStorage(ctx, out, dataCtx)
}

func (n *negotiateCompressionTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) (out []byte, err error) {
	// call delegated transformer before compression transformer
	transformedData, err := n.delegatedTransformer.TransformToStorage(ctx, data, dataCtx)
	if err != nil {
		return nil, err
	}

	compressionTransformer := registry.DefaultCompressionRegistry.GetCompressionTransformer(n.storageCompression)
	return compressionTransformer.TransformToStorage(ctx, transformedData, dataCtx)
}
