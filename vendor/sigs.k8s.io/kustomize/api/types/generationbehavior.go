// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

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
