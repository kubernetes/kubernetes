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

package credentialprovider

import (
	"testing"
	"time"
)

type testProvider struct {
	Count int
}

// Enabled implements dockerConfigProvider
func (d *testProvider) Enabled() bool {
	return true
}

// Provide implements dockerConfigProvider
func (d *testProvider) Provide(image string) DockerConfig {
	d.Count++
	return DockerConfig{}
}

func TestCachingProvider(t *testing.T) {
	provider := &testProvider{
		Count: 0,
	}

	cache := &CachingDockerConfigProvider{
		Provider: provider,
		Lifetime: 1 * time.Second,
	}

	image := "image"

	if provider.Count != 0 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}
	cache.Provide(image)
	cache.Provide(image)
	cache.Provide(image)
	cache.Provide(image)
	if provider.Count != 1 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}

	time.Sleep(cache.Lifetime)
	cache.Provide(image)
	cache.Provide(image)
	cache.Provide(image)
	cache.Provide(image)
	if provider.Count != 2 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}

	time.Sleep(cache.Lifetime)
	cache.Provide(image)
	cache.Provide(image)
	cache.Provide(image)
	cache.Provide(image)
	if provider.Count != 3 {
		t.Errorf("Unexpected number of Provide calls: %v", provider.Count)
	}
}
