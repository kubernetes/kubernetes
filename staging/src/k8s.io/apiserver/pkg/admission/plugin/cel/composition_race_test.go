/*
Copyright 2025 The Kubernetes Authors.

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

package cel

import (
	"sync"
	"testing"

	"github.com/google/cel-go/cel"

	"k8s.io/apiserver/pkg/cel/environment"
)

// TestCompositionEnvTemplateRace demonstrates a race condition in the shared
// CompositionEnv template used by ValidatingAdmissionPolicy.
//
// There is a data race due to a shared MapType between compilers.
// When one goroutine compiles a policy (calling AddField to write to MapType.Fields)
// while another goroutine resolves types during CEL evaluation (reading MapType.Fields),
// a concurrent map read/write occurs.
//
// Production scenario:
//   - Policy informer triggers compilePolicy() which calls AddField()
//   - Concurrently, admission requests evaluate policies, and CEL type resolution
//     calls FindStructFieldType() which reads from the same Fields map
//
// There is a policySource lock, but AFAICT it only serializes compilations; it does NOT protect
// reads from MapType.Fields during CEL evaluation.
//
// Run with: go test -race -run TestCompositionEnvTemplateRace
func TestCompositionEnvTemplateRace(t *testing.T) {
	baseEnvSet := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion())

	// Create the shared template (like getCompositionEnvTemplateWithStrictCost singleton)
	template, err := NewCompositionEnv(VariablesTypeName, baseEnvSet)
	if err != nil {
		t.Fatal(err)
	}

	var wg sync.WaitGroup

	// Writer: AddField is called during CompileAndStoreVariables
	// In production: informer callback -> compilePolicy -> CompileAndStoreVariables -> AddField
	wg.Go(func() {
		for i := 0; i < 1000; i++ {
			template.AddField("foo", cel.StringType)
		}
	})

	// Reader: FindField accesses the same map as FindStructFieldType
	// In production: admission request -> CEL eval -> type resolution -> FindStructFieldType
	wg.Go(func() {
		for i := 0; i < 1000; i++ {
			template.MapType.FindField("foo")
		}
	})

	wg.Wait()
}
