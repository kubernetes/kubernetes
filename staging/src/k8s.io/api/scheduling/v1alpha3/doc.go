/*
Copyright The Kubernetes Authors.

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

// +k8s:deepcopy-gen=package
// +k8s:protobuf-gen=package
// +k8s:openapi-gen=true
// +k8s:openapi-model-package=io.k8s.api.scheduling.v1alpha3

// +groupName=scheduling.k8s.io

// +k8s:validation-gen=*
// +k8s:validation-gen-input=k8s.io/api/scheduling/v1alpha3
// +k8s:validation-gen-scheme-registry=nil

// A non-registering copy of the declarative validation is generated in this
// package so out-of-tree consumers can run the same validation the
// kube-apiserver enforces without importing k8s.io/kubernetes/pkg or mutating
// a shared scheme on import. The registered copy lives in
// k8s.io/kubernetes/pkg/apis/scheduling/v1alpha3.
// The `*` selector generates validators for all types, including the
// WorkloadPodGroup* building blocks that are not reachable from a TypeMeta
// root.

package v1alpha3
