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

package cache

import (
	"bytes"
	"sync"
	"testing"

	fcache "k8s.io/client-go/tools/cache/testing"
)

func newFakeControllerSource(tb testing.TB) *fcache.FakeControllerSource {
	source := fcache.NewFakeControllerSource()
	tb.Cleanup(source.Shutdown)
	return source
}

// threadSafeBuffer is a thread-safe wrapper around bytes.Buffer.
type threadSafeBuffer struct {
	buffer bytes.Buffer
	mu     sync.Mutex
}

func (b *threadSafeBuffer) Write(p []byte) (n int, err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buffer.Write(p)
}

func (b *threadSafeBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buffer.String()
}
