// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// +build tools

// This file exists to declare that its containing
// package explicitly depends on the pluginator
// tool (via go:generate directives)
package builtins

// TODO: replace this, with the appropriate version
// once the API is launched, and the new pluginator
// has been compiled against it and released.
//
//  import (
//  	_ "sigs.k8s.io/kustomize/pluginator"
// )
