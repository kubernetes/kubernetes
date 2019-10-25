// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

//go:generate stringer -type=GarbagePolicy
type GarbagePolicy int

const (
	GarbageIgnore GarbagePolicy = iota + 1
	GarbageCollect
)
