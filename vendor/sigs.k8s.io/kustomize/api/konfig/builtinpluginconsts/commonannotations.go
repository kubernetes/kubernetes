// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinpluginconsts

const commonAnnotationFieldSpecs = `
commonAnnotations:
- path: metadata/annotations
  create: true

- path: spec/template/metadata/annotations
  create: true
  version: v1
  kind: ReplicationController

- path: spec/template/metadata/annotations
  create: true
  kind: Deployment

- path: spec/template/metadata/annotations
  create: true
  kind: ReplicaSet

- path: spec/template/metadata/annotations
  create: true
  kind: DaemonSet

- path: spec/template/metadata/annotations
  create: true
  kind: StatefulSet

- path: spec/template/metadata/annotations
  create: true
  group: batch
  kind: Job

- path: spec/jobTemplate/metadata/annotations
  create: true
  group: batch
  kind: CronJob

- path: spec/jobTemplate/spec/template/metadata/annotations
  create: true
  group: batch
  kind: CronJob

`
