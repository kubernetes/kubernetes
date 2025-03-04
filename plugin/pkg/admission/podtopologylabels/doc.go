/*
Copyright 2024 The Kubernetes Authors.

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

// Package podtopologylabels is a plugin that mutates `pod/binding` requests
// to set `topology.k8s.io` labels (including subdomains) from the Node object
// referenced in the Binding to the Binding, which causes the Pod to also
// have these values set.
// If the binding target is NOT a Node object, no action is taken.
// If the referenced Node object does not exist, no action is taken.
package podtopologylabels // import "k8s.io/kubernetes/plugin/pkg/admission/podtopologylabels"
