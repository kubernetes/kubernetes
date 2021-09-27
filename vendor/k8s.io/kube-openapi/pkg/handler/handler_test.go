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

package handler

import (
	"errors"
	"testing"
)

func TestCache(t *testing.T) {
	calledCount := 0
	expectedBytes := []byte("ABC")
	cacheObj := cache{
		BuildCache: func() ([]byte, string, error) {
			calledCount++
			return expectedBytes, "", nil
		},
	}
	bytes, _, _ := cacheObj.Get()
	if string(bytes) != string(expectedBytes) {
		t.Fatalf("got value of %q from cache (expected %q)", bytes, expectedBytes)
	}
	cacheObj.Get()
	if calledCount != 1 {
		t.Fatalf("expected BuildCache to be called once (called %d times)", calledCount)
	}
}

func TestCacheError(t *testing.T) {
	cacheObj := cache{
		BuildCache: func() ([]byte, string, error) {
			return nil, "", errors.New("cache error")
		},
	}
	_, _, err := cacheObj.Get()
	if err == nil {
		t.Fatalf("expected non-nil err from cache.Get()")
	}
}
