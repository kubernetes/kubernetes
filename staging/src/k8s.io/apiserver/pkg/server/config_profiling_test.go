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

package server

import (
	"path/filepath"
	goruntime "runtime"
	"testing"

	"k8s.io/client-go/rest"
	basecompatibility "k8s.io/component-base/compatibility"
)

func TestContentionProfilingEnablesMutexProfile(t *testing.T) {
	tests := []struct {
		name                      string
		enableProfiling           bool
		debugSocket               bool
		enableContentionProfiling bool
		wantMutexProfileFraction  int
	}{
		{name: "profiling with contention profiling", enableProfiling: true, enableContentionProfiling: true, wantMutexProfileFraction: 1},
		{name: "debug socket with contention profiling", enableProfiling: false, debugSocket: true, enableContentionProfiling: true, wantMutexProfileFraction: 1},
		{name: "contention profiling disabled", enableProfiling: true, enableContentionProfiling: false, wantMutexProfileFraction: 0},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// The profile rates are process-global, so reset them around each case.
			prev := goruntime.SetMutexProfileFraction(0)
			t.Cleanup(func() {
				goruntime.SetMutexProfileFraction(prev)
				goruntime.SetBlockProfileRate(0)
			})

			config := NewConfig(codecs)
			config.ExternalAddress = "192.168.10.4:443"
			config.EffectiveVersion = basecompatibility.NewEffectiveVersionFromString("", "", "")
			config.LoopbackClientConfig = &rest.Config{}
			config.EnableProfiling = tt.enableProfiling
			config.EnableContentionProfiling = tt.enableContentionProfiling
			if tt.debugSocket {
				// New only records the path; the socket is not opened until the server runs.
				config.DebugSocketPath = filepath.Join(t.TempDir(), "sock")
			}

			if _, err := config.Complete(nil).New("test", NewEmptyDelegate()); err != nil {
				t.Fatal(err)
			}

			// A negative value reads the current fraction without changing it.
			if got := goruntime.SetMutexProfileFraction(-1); got != tt.wantMutexProfileFraction {
				t.Errorf("mutex profile fraction = %d, want %d", got, tt.wantMutexProfileFraction)
			}
		})
	}
}
