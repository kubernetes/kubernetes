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

// Package transformers has implementations of resmap.ResMap transformers.
package transformers

import "sigs.k8s.io/kustomize/pkg/resmap"

// A Transformer modifies an instance of resmap.ResMap.
type Transformer interface {
	// Transform modifies data in the argument, e.g. adding labels to resources that can be labelled.
	Transform(m resmap.ResMap) error
}
