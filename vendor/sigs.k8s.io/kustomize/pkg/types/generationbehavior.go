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

package types

// GenerationBehavior specifies generation behavior of configmaps, secrets and maybe other resources.
type GenerationBehavior int

const (
	// BehaviorUnspecified is an Unspecified behavior; typically treated as a Create.
	BehaviorUnspecified GenerationBehavior = iota
	// BehaviorCreate makes a new resource.
	BehaviorCreate
	// BehaviorReplace replaces a resource.
	BehaviorReplace
	// BehaviorMerge attempts to merge a new resource with an existing resource.
	BehaviorMerge
)

// String converts a GenerationBehavior to a string.
func (b GenerationBehavior) String() string {
	switch b {
	case BehaviorReplace:
		return "replace"
	case BehaviorMerge:
		return "merge"
	case BehaviorCreate:
		return "create"
	default:
		return "unspecified"
	}
}

// NewGenerationBehavior converts a string to a GenerationBehavior.
func NewGenerationBehavior(s string) GenerationBehavior {
	switch s {
	case "replace":
		return BehaviorReplace
	case "merge":
		return BehaviorMerge
	case "create":
		return BehaviorCreate
	default:
		return BehaviorUnspecified
	}
}
