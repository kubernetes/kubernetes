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

package environment

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/version"
)

// BenchmarkLoadBaseEnv is expected to be very fast, because a
// a cached environment is loaded for each MustBaseEnvSet call.
func BenchmarkLoadBaseEnv(b *testing.B) {
	ver := DefaultCompatibilityVersion()
	MustBaseEnvSet(ver)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MustBaseEnvSet(ver)
	}
}

// BenchmarkLoadBaseEnvDifferentVersions is expected to be relatively slow, because a
// a new environment must be created for each MustBaseEnvSet call.
func BenchmarkLoadBaseEnvDifferentVersions(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		MustBaseEnvSet(version.MajorMinor(1, uint(i)))
	}
}
