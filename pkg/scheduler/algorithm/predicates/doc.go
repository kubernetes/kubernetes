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

// Package predicates are a set of strategies applied one by one to filter out
// inappropriate nodes. All of them are mandatory strategies. They
// traverse all the Nodes and filter out the matching Node list
// according to the specific pre-selection strategy. If no Node meets
// the Predicate strategies, the Pod will be suspended until a Node
// can satisfy.
package predicates // import "k8s.io/kubernetes/pkg/scheduler/algorithm/predicates"
