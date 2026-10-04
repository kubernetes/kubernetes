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

// +k8s:conversion-gen=k8s.io/kubernetes/pkg/apis/scheduling
// +k8s:conversion-gen-external-types=k8s.io/api/scheduling/v1alpha3
// +groupName=scheduling.k8s.io
// +k8s:defaulter-gen=TypeMeta
// +k8s:defaulter-gen-input=k8s.io/api/scheduling/v1alpha3
// +k8s:validation-gen=TypesWithField=TypeMeta
// +k8s:validation-gen-input=k8s.io/api/scheduling/v1alpha3

// This is the registered copy of the scheduling/v1alpha3 declarative
// validation. It registers with the apiserver scheme and is the package
// cross-package references resolve to. A non-registering copy is generated
// in-package under k8s.io/api/scheduling/v1alpha3 for out-of-tree consumers.

package v1alpha3
