/*
Copyright 2024 The Kubernetes Authors.

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

package mutation

import (
	"testing"

	"github.com/google/cel-go/cel"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/cel/environment"
)

// mustCreateEnv creates the default env for testing, with given option.
// it fatally fails the test if the env fails to set up.
func mustCreateEnv(t testing.TB, envOptions ...cel.EnvOption) *cel.Env {
	envSet, err := environment.MustBaseEnvSet(environment.DefaultCompatibilityVersion()).
		Extend(environment.VersionedOptions{
			IntroducedVersion: version.MajorMinor(1, 30),
			EnvOptions:        envOptions,
		})
	if err != nil {
		t.Fatalf("fail to create env set: %v", err)
	}
	env, err := envSet.Env(environment.StoredExpressions)
	if err != nil {
		t.Fatalf("fail to setup env: %v", env)
	}
	return env
}

// mustCreateEnvWithOptional creates the default env for testing, with given option,
// and set up the optional library with default configuration.
// it fatally fails the test if the env fails to set up.
func mustCreateEnvWithOptional(t testing.TB, envOptions ...cel.EnvOption) *cel.Env {
	return mustCreateEnv(t, append(envOptions, cel.OptionalTypes())...)
}
