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

package informers

import (
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
)

func TestWithReflectionOptionsSetsFactory(t *testing.T) {
	client := fake.NewClientset()

	cfg := cache.ReflectionOptions{
		Backoff: wait.Backoff{
			Duration: 500 * time.Millisecond,
			Factor:   2.0,
			Jitter:   1.0,
			Steps:    10,
			Cap:      30 * time.Second,
		},
		BackoffResetDuration: 2 * time.Minute,
		MinWatchTimeout:      6 * time.Minute,
		MaxWatchTimeout:      12 * time.Minute,
	}

	factory := NewSharedInformerFactoryWithOptions(client, 0, WithReflectionOptions(cfg))
	f := factory.(*sharedInformerFactory)

	if f.reflectionOptions.Backoff == (wait.Backoff{}) {
		t.Error("expected Backoff to be set on factory")
	}
	if f.reflectionOptions.MinWatchTimeout != 6*time.Minute {
		t.Errorf("expected MinWatchTimeout=6m, got %v", f.reflectionOptions.MinWatchTimeout)
	}
	if f.reflectionOptions.MaxWatchTimeout != 12*time.Minute {
		t.Errorf("expected MaxWatchTimeout=12m, got %v", f.reflectionOptions.MaxWatchTimeout)
	}
	if f.reflectionOptions.BackoffResetDuration != 2*time.Minute {
		t.Errorf("expected BackoffResetDuration=2m, got %v", f.reflectionOptions.BackoffResetDuration)
	}
}

