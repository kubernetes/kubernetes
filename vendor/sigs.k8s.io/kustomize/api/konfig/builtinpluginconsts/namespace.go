// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinpluginconsts

const (
	namespaceFieldSpecs = `
namespace:
- path: metadata/namespace
  create: true
- path: subjects
  kind: RoleBinding
- path: subjects
  kind: ClusterRoleBinding
`
)
