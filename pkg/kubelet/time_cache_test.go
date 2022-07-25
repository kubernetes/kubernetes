/*
Copyright 2020 The Kubernetes Authors.

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

package kubelet

import (
	"testing"
	"time"

	"github.com/golang/groupcache/lru"
)

func TestTimeCache(t *testing.T) {
	cache := &timeCache{cache: lru.New(2)}
	if a, ok := cache.Get("123"); ok {
		t.Errorf("expected cache miss, got %v, %v", a, ok)
	}

	now := time.Now()
	soon := now.Add(time.Minute)
	cache.Add("now", now)
	cache.Add("soon", soon)

	if a, ok := cache.Get("now"); !ok || !a.Equal(now) {
		t.Errorf("expected cache hit matching %v, got %v, %v", now, a, ok)
	}
	if a, ok := cache.Get("soon"); !ok || !a.Equal(soon) {
		t.Errorf("expected cache hit matching %v, got %v, %v", soon, a, ok)
	}

	then := now.Add(-time.Minute)
	cache.Add("then", then)
	if a, ok := cache.Get("now"); ok {
		t.Errorf("expected cache miss from oldest evicted value, got %v, %v", a, ok)
	}
	if a, ok := cache.Get("soon"); !ok || !a.Equal(soon) {
		t.Errorf("expected cache hit matching %v, got %v, %v", soon, a, ok)
	}
	if a, ok := cache.Get("then"); !ok || !a.Equal(then) {
		t.Errorf("expected cache hit matching %v, got %v, %v", then, a, ok)
	}
}
