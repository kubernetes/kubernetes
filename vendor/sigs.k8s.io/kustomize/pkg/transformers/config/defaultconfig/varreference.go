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

const (
	varReferenceFieldSpecs = `
varReference:
- path: spec/template/spec/initContainers/command
  kind: StatefulSet

- path: spec/template/spec/containers/command
  kind: StatefulSet

- path: spec/template/spec/initContainers/command
  kind: Deployment

- path: spec/template/spec/containers/command
  kind: Deployment

- path: spec/template/spec/initContainers/command
  kind: DaemonSet

- path: spec/template/spec/containers/command
  kind: DaemonSet

- path: spec/template/spec/containers/command
  kind: Job

- path: spec/jobTemplate/spec/template/spec/containers/command
  kind: CronJob

- path: spec/template/spec/initContainers/args
  kind: StatefulSet
 
- path: spec/template/spec/containers/args
  kind: StatefulSet

- path: spec/template/spec/initContainers/args
  kind: Deployment

- path: spec/template/spec/containers/args
  kind: Deployment

- path: spec/template/spec/initContainers/args
  kind: DaemonSet

- path: spec/template/spec/containers/args
  kind: DaemonSet

- path: spec/template/spec/containers/args
  kind: Job

- path: spec/jobTemplate/spec/template/spec/containers/args
  kind: CronJob

- path: spec/template/spec/initContainers/env/value
  kind: StatefulSet

- path: spec/template/spec/containers/env/value
  kind: StatefulSet

- path: spec/template/spec/initContainers/env/value
  kind: Deployment

- path: spec/template/spec/containers/env/value
  kind: Deployment

- path: spec/template/spec/initContainers/env/value
  kind: DaemonSet

- path: spec/template/spec/containers/env/value
  kind: DaemonSet

- path: spec/template/spec/containers/env/value
  kind: Job

- path: spec/jobTemplate/spec/template/spec/containers/env/value
  kind: CronJob

- path: spec/containers/command
  kind: Pod

- path: spec/containers/args
  kind: Pod

- path: spec/containers/env/value
  kind: Pod

- path: spec/rules/host
  kind: Ingress

- path: spec/tls/hosts
  kind: Ingress
`
)
