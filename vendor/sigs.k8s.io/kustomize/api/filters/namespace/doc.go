// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package namespace contains a kio.Filter implementation of the kustomize
// namespace transformer.
//
// Special cases for known Kubernetes resources have been hardcoded in addition
// to those defined by the FsSlice.
package namespace
