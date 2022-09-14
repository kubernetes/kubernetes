/*
Copyright 2023 The Kubernetes Authors.

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

// Package kmsv2 transforms values for storage at rest using a Envelope v2 provider
package kmsv2

import (
	"testing"
	"time"

	"k8s.io/apiserver/pkg/storage/value"
	testingclock "k8s.io/utils/clock/testing"
)

func TestSimpleCacheSetError(t *testing.T) {
	fakeClock := testingclock.NewFakeClock(time.Now())
	cache := newSimpleCache(fakeClock, time.Second)

	tests := []struct {
		name        string
		key         []byte
		transformer value.Transformer
	}{
		{
			name:        "empty key",
			key:         []byte{},
			transformer: nil,
		},
		{
			name:        "nil transformer",
			key:         []byte("key"),
			transformer: nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r == nil {
					t.Errorf("The code did not panic")
				}
			}()
			cache.set(test.key, test.transformer)
		})
	}
}
