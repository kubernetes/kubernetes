// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinpluginconsts

const volumeClaimTemplateLabelFieldSpecs = `
volumeClaimTemplateLabels:
` + volumeClaimTemplatesMetadataLabels

const volumeClaimTemplatesMetadataLabels = `
- path: spec/volumeClaimTemplates[]/metadata/labels
  create: true
  group: apps
  kind: StatefulSet
`
