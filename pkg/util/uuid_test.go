/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package util

import (
	"sync"
	"testing"

	"k8s.io/kubernetes/pkg/types"
)

func TestUUID(t *testing.T) {
	itrs := 1000

	ids := map[types.UID]bool{}
	for i := 0; i < itrs; i++ {
		ids[NewUUID()] = true
	}

	if len(ids) != itrs {
		t.Errorf("unexpected number of UUIDs: %d, expected %d", len(ids), itrs)
	}
}

func BenchmarkNewUUID(b *testing.B) {
	ids := map[types.UID]bool{}
	wg := sync.WaitGroup{}
	wg.Add(b.N)
	for i := 0; i < b.N; i++ {
		go func() {
			ids[NewUUID()] = true
			wg.Done()
		}()
	}
	wg.Wait()

	if len(ids) != b.N {
		b.Errorf("unexpected number of UUIDs: %d, expected %d", len(ids), b.N)
	}
}
