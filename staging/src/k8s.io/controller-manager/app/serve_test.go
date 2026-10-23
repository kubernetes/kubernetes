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

package app

import (
	"net/http"
	goruntime "runtime"
	"testing"

	componentbaseconfig "k8s.io/component-base/config"
)

func TestNewBaseHandlerContentionProfiling(t *testing.T) {
	tests := []struct {
		name                      string
		enableProfiling           bool
		enableContentionProfiling bool
		wantMutexProfileFraction  int
	}{
		{name: "profiling and contention profiling enabled", enableProfiling: true, enableContentionProfiling: true, wantMutexProfileFraction: 1},
		{name: "profiling disabled", enableProfiling: false, enableContentionProfiling: true, wantMutexProfileFraction: 0},
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

			cfg := &componentbaseconfig.DebuggingConfiguration{
				EnableProfiling:           tt.enableProfiling,
				EnableContentionProfiling: tt.enableContentionProfiling,
			}
			NewBaseHandler(cfg, http.NotFoundHandler())

			// A negative value reads the current fraction without changing it.
			if got := goruntime.SetMutexProfileFraction(-1); got != tt.wantMutexProfileFraction {
				t.Errorf("mutex profile fraction = %d, want %d", got, tt.wantMutexProfileFraction)
			}
		})
	}
}
