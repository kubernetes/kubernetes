/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package defaultconfig

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
