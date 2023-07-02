// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinpluginconsts

const commonLabelFieldSpecs = `
commonLabels:
- path: spec/selector
  create: true
  version: v1
  kind: Service

- path: spec/selector
  create: true
  version: v1
  kind: ReplicationController
- path: spec/selector/matchLabels
  create: true
  kind: Deployment

- path: spec/template/spec/affinity/podAffinity/preferredDuringSchedulingIgnoredDuringExecution/podAffinityTerm/labelSelector/matchLabels
  create: false
  group: apps
  kind: Deployment

- path: spec/template/spec/affinity/podAffinity/requiredDuringSchedulingIgnoredDuringExecution/labelSelector/matchLabels
  create: false
  group: apps
  kind: Deployment

- path: spec/template/spec/affinity/podAntiAffinity/preferredDuringSchedulingIgnoredDuringExecution/podAffinityTerm/labelSelector/matchLabels
  create: false
  group: apps
  kind: Deployment

- path: spec/template/spec/affinity/podAntiAffinity/requiredDuringSchedulingIgnoredDuringExecution/labelSelector/matchLabels
  create: false
  group: apps
  kind: Deployment

- path: spec/template/spec/topologySpreadConstraints/labelSelector/matchLabels
  create: false
  group: apps
  kind: Deployment

- path: spec/selector/matchLabels
  create: true
  kind: ReplicaSet

- path: spec/selector/matchLabels
  create: true
  kind: DaemonSet

- path: spec/selector/matchLabels
  create: true
  group: apps
  kind: StatefulSet

- path: spec/template/spec/affinity/podAffinity/preferredDuringSchedulingIgnoredDuringExecution/podAffinityTerm/labelSelector/matchLabels
  create: false
  group: apps
  kind: StatefulSet

- path: spec/template/spec/affinity/podAffinity/requiredDuringSchedulingIgnoredDuringExecution/labelSelector/matchLabels
  create: false
  group: apps
  kind: StatefulSet

- path: spec/template/spec/affinity/podAntiAffinity/preferredDuringSchedulingIgnoredDuringExecution/podAffinityTerm/labelSelector/matchLabels
  create: false
  group: apps
  kind: StatefulSet

- path: spec/template/spec/affinity/podAntiAffinity/requiredDuringSchedulingIgnoredDuringExecution/labelSelector/matchLabels
  create: false
  group: apps
  kind: StatefulSet

- path: spec/template/spec/topologySpreadConstraints/labelSelector/matchLabels
  create: false
  group: apps
  kind: StatefulSet

- path: spec/selector/matchLabels
  create: false
  group: batch
  kind: Job

- path: spec/jobTemplate/spec/selector/matchLabels
  create: false
  group: batch
  kind: CronJob

- path: spec/selector/matchLabels
  create: false
  group: policy
  kind: PodDisruptionBudget

- path: spec/podSelector/matchLabels
  create: false
  group: networking.k8s.io
  kind: NetworkPolicy

- path: spec/ingress/from/podSelector/matchLabels
  create: false
  group: networking.k8s.io
  kind: NetworkPolicy

- path: spec/egress/to/podSelector/matchLabels
  create: false
  group: networking.k8s.io
  kind: NetworkPolicy
` + metadataLabelsFieldSpecs
