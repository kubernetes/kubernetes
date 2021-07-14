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

package gzip

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"testing"

	"k8s.io/apiserver/pkg/storage/value"
)

func TestGzipLevelRotation(t *testing.T) {
	testErr := fmt.Errorf("test error")
	p := value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewGzipTransformer(gzip.BestCompression)},
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewGzipTransformer(gzip.BestSpeed)},
	)
	context := value.DefaultContext("authenticated_data")
	out, err := p.TransformToStorage([]byte("firstvalue"), context)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.HasPrefix(out, []byte("first:")) {
		t.Fatalf("unexpected prefix: %q", out)
	}
	from, stale, err := p.TransformFromStorage(out, context)
	if err != nil {
		t.Fatal(err)
	}
	if stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}

	// reverse the order, use the second key
	p = value.NewPrefixTransformers(testErr,
		value.PrefixTransformer{Prefix: []byte("second:"), Transformer: NewGzipTransformer(gzip.BestSpeed)},
		value.PrefixTransformer{Prefix: []byte("first:"), Transformer: NewGzipTransformer(gzip.BestCompression)},
	)
	from, stale, err = p.TransformFromStorage(out, context)
	if err != nil {
		t.Fatal(err)
	}
	if !stale || !bytes.Equal([]byte("firstvalue"), from) {
		t.Fatalf("unexpected data: %t %q", stale, from)
	}
}
