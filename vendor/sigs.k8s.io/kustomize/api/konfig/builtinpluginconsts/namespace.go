// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinpluginconsts

const (
	namespaceFieldSpecs = `
namespace:
- path: metadata/namespace
  create: true
- path: metadata/name
  kind: Namespace
  create: true
- path: subjects
  kind: RoleBinding
- path: subjects
  kind: ClusterRoleBinding
- path: spec/service/namespace
  group: apiregistration.k8s.io
  kind: APIService
  create: true
`
)
