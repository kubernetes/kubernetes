/*
Copyright 2024 The Kubernetes Authors.

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

package modes

import (
	"bytes"
	"sync"
)

var buffers = BufferProvider{p: new(sync.Pool)}

type buffer struct {
	bytes.Buffer
}

type pool interface {
	Get() interface{}
	Put(interface{})
}

type BufferProvider struct {
	p pool
}

func (b *BufferProvider) Get() *buffer {
	if buf, ok := b.p.Get().(*buffer); ok {
		return buf
	}
	return &buffer{}
}

func (b *BufferProvider) Put(buf *buffer) {
	if buf.Cap() > 3*1024*1024 /* Default MaxRequestBodyBytes */ {
		// Objects in a sync.Pool are assumed to be fungible. This is not a good assumption
		// for pools of *bytes.Buffer because a *bytes.Buffer's underlying array grows as
		// needed to accommodate writes. In Kubernetes, apiservers tend to encode "small"
		// objects very frequently and much larger objects (especially large lists) only
		// occasionally. Under steady load, pooled buffers tend to be borrowed frequently
		// enough to prevent them from being released. Over time, each buffer is used to
		// encode a large object and its capacity increases accordingly. The result is that
		// practically all buffers in the pool retain much more capacity than needed to
		// encode most objects.

		// As a basic mitigation for the worst case, buffers with more capacity than the
		// default max request body size are never returned to the pool.
		// TODO: Optimize for higher buffer utilization.
		return
	}
	buf.Reset()
	b.p.Put(buf)
}
