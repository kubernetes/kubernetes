/*
Copyright 2021 The Kubernetes Authors.

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

// Package gzip transforms values for storage at rest using gzip compression.
package gzip

import (
	"bytes"
	"compress/gzip"
	"io"

	"k8s.io/apiserver/pkg/storage/value"
)

// These constants are copied from the compress/gzip package, so that code that imports
// "gzip" does not also have to import "compress/gzip".
const (
	NoCompression      = gzip.NoCompression
	BestSpeed          = gzip.BestSpeed
	BestCompression    = gzip.BestCompression
	DefaultCompression = gzip.DefaultCompression
	HuffmanOnly        = gzip.HuffmanOnly
)

// gzip implements compression for storage of values
type gzipTransformer struct {
	Level int
}

// NewGzipTransformer compresses values with gzip.
func NewGzipTransformer(level int) value.Transformer {
	return &gzipTransformer{
		Level: level,
	}
}

func (gzipTransformer) TransformFromStorage(b []byte, ctx value.Context) ([]byte, bool, error) {
	r, err := gzip.NewReader(bytes.NewBuffer(b))
	if err != nil {
		return nil, false, err
	}
	out, err := io.ReadAll(r)
	if err != nil {
		return nil, false, err
	}
	return out, false, nil
}

func (t gzipTransformer) TransformToStorage(b []byte, ctx value.Context) ([]byte, error) {
	var buf bytes.Buffer
	w, err := gzip.NewWriterLevel(&buf, t.Level)
	if err != nil {
		return nil, err
	}
	_, err = w.Write(b)
	if err != nil {
		return nil, err
	}
	if err := w.Close(); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}
