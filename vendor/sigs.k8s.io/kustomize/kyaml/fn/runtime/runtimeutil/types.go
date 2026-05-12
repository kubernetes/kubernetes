// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package runtimeutil

type DeferFailureFunction interface {
	GetExit() error
}
