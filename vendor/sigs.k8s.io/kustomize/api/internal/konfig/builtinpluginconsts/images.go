// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinpluginconsts

const (
	imagesFieldSpecs = `
images:
- path: spec/containers[]/image
  create: true
- path: spec/initContainers[]/image
  create: true
- path: spec/volumes[]/image/reference
  create: true
- path: spec/template/spec/containers[]/image
  create: true
- path: spec/template/spec/initContainers[]/image
  create: true
- path: spec/template/spec/volumes[]/image/reference
  create: true
`
)
