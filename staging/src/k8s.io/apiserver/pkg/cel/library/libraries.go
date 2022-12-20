/*
Copyright 2022 The Kubernetes Authors.

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

package library

import (
	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/ext"
	"github.com/google/cel-go/interpreter"
)

// ExtensionLibs declares the set of CEL extension libraries available everywhere CEL is used in Kubernetes.
var ExtensionLibs = append(k8sExtensionLibs, ext.Strings())

var k8sExtensionLibs = []cel.EnvOption{
	URLs(),
	Regex(),
	Lists(),
}

var ExtensionLibRegexOptimizations = []*interpreter.RegexOptimization{FindRegexOptimization, FindAllRegexOptimization}
