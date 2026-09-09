// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinpluginconsts

const metadataLabelsFieldSpecs = `
- path: metadata/labels
  create: true

- path: spec/template/metadata/labels
  create: true
  version: v1
  kind: ReplicationController

- path: spec/template/metadata/labels
  create: true
  kind: Deployment

- path: spec/template/metadata/labels
  create: true
  kind: ReplicaSet

- path: spec/template/metadata/labels
  create: true
  kind: DaemonSet

- path: spec/template/metadata/labels
  create: true
  group: apps
  kind: StatefulSet

- path: spec/volumeClaimTemplates[]/metadata/labels
  create: true
  group: apps
  kind: StatefulSet

- path: spec/template/metadata/labels
  create: true
  group: batch
  kind: Job

- path: spec/jobTemplate/metadata/labels
  create: true
  group: batch
  kind: CronJob

- path: spec/jobTemplate/spec/template/metadata/labels
  create: true
  group: batch
  kind: CronJob
`
