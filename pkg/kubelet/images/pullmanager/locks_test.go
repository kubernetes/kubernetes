/*
Copyright The Kubernetes Authors.

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

package pullmanager

import (
	"fmt"
	"testing"
)

func TestStripedLockSetNormalizesSize(t *testing.T) {
	tests := []struct {
		name string
		size int32
		want int32
	}{
		{
			name: "negative",
			size: -1,
			want: 1,
		},
		{
			name: "zero",
			size: 0,
			want: 1,
		},
		{
			name: "below max",
			size: 10,
			want: 10,
		},
		{
			name: "above max",
			size: 500,
			want: 31,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lockSet := NewStripedLockSet(tt.size)
			if len(lockSet.locks) != int(tt.want) {
				t.Fatalf("len(NewStripedLockSet(%d).locks) = %d, want %d", tt.size, len(lockSet.locks), tt.want)
			}
		})
	}
}

func TestStripedLockSetLargeRequestedSizeDoesNotPanic(t *testing.T) {
	const requestedSize int32 = 500
	lockSet := NewStripedLockSet(requestedSize)
	key := keyWithOriginalStripeOutsideAllocatedRange(t, requestedSize, len(lockSet.locks))

	lockSet.Lock(key)
	lockSet.Unlock(key)
}

func keyWithOriginalStripeOutsideAllocatedRange(t *testing.T, requestedSize int32, allocatedLocks int) string {
	t.Helper()

	for i := range 1000 {
		key := fmt.Sprintf("image-ref-%d", i)
		if keyToID(key, requestedSize) >= uint32(allocatedLocks) {
			return key
		}
	}
	t.Fatalf("failed to find key with stripe outside allocated range")
	return ""
}
