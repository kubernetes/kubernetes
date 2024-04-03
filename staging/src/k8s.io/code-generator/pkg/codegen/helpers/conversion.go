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

package helpers

import (
	"k8s.io/code-generator/cmd/conversion-gen/generators"
)

func (g *Generator) generateConversion(args *Args) error {
	return g.generateGen(args, genConf{
		name:             "conversion",
		genFn:            generators.GenerateConversion,
		useExtraPeerDirs: true,
		searchPattern:    "+k8s:conversion-gen=",
	})
}
