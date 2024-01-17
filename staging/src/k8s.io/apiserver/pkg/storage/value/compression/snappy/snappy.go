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
	"bytes"
	"context"

	"github.com/golang/snappy"
	"k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/compression/registry"
	"k8s.io/apiserver/pkg/storage/value/encrypt/envelope/metrics"
	"k8s.io/klog/v2"
)

func init() {
	value.RegisterMetrics()
	metrics.RegisterMetrics()
}

func init() {
	runtime.Must(registry.DefaultCompressionRegistry.RegisterCompressionTransformer(newSnappyCompressionTransformer()))
}

const (
	transformerName    = "snappy"
	snappyPrefix       = "snappy:"
	snappyPrefixLength = len(snappyPrefix)
)

// newSnappyCompressionTransformer returns a transformer which does compression/decompression with snappy
func newSnappyCompressionTransformer() registry.CompressionTransformer {
	return &snappyCompressionTransformer{}
}

var _ registry.CompressionTransformer = &snappyCompressionTransformer{}

type snappyCompressionTransformer struct {
}

func (t *snappyCompressionTransformer) Name() string {
	return transformerName
}

// TransformFromStorage decompress data with snappy from original data
func (t *snappyCompressionTransformer) TransformFromStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, bool, error) {
	if !bytes.HasPrefix(data, []byte(snappyPrefix)) {
		return data, false, nil
	}

	decoded, err := snappy.Decode(nil, data[snappyPrefixLength:])
	if err != nil {
		klog.Errorf("snappy.Decode err:%v", err)
		return nil, false, err
	}
	return decoded, false, nil
}

// TransformToStorage compress data with snappy to be written
func (t *snappyCompressionTransformer) TransformToStorage(ctx context.Context, data []byte, dataCtx value.Context) ([]byte, error) {
	// compression with snappy
	encoded := snappy.Encode(nil, data)
	return append([]byte(snappyPrefix), encoded...), nil
}
