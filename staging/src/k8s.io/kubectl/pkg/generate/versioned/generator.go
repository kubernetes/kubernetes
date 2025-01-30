/*
Copyright 2018 The Kubernetes Authors.

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

package versioned

import (
	"k8s.io/kubectl/pkg/generate"
)

// GeneratorFn gives a way to easily override the function for unit testing if needed
var GeneratorFn generate.GeneratorFunc = DefaultGenerators

const (
	// TODO(sig-cli): Enforce consistent naming for generators here.
	// See discussion in https://github.com/kubernetes/kubernetes/issues/46237
	// before you add any more.
	RunPodV1GeneratorName  = "run-pod/v1"
	ServiceV2GeneratorName = "service/v2"
)

// DefaultGenerators returns the set of default generators for use in Factory instances
func DefaultGenerators(cmdName string) map[string]generate.Generator {
	var generator map[string]generate.Generator
	switch cmdName {
	case "expose":
		generator = map[string]generate.Generator{
			ServiceV2GeneratorName: ServiceGeneratorV2{},
		}
	case "run":
		generator = map[string]generate.Generator{
			RunPodV1GeneratorName: BasicPod{},
		}
	}

	return generator
}
