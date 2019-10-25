// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package consts

const replicasFieldSpecs = `
replicas:
- path: spec/replicas
  create: true
  kind: Deployment

- path: spec/replicas
  create: true
  kind: ReplicationController

- path: spec/replicas
  create: true
  kind: ReplicaSet

- path: spec/replicas
  create: true
  kind: StatefulSet
`
