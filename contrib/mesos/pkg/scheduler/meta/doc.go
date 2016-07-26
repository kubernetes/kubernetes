/*
Copyright 2015 The Kubernetes Authors.

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

// Package meta defines framework constants used as keys in k8s annotations
// that are attached to k8s pods. The scheduler uses some of these annotations
// for reconciliation upon failover. Other annotations are used as part of
// the host-to-pod port-mapping implementation understood by the k8s-mesos
// scheduler and custom endpoints-controller implementation.
package meta // import "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/meta"
