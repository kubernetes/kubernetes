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

// Note: The referenced generic ComponentConfig packages with conversions
// between the types (e.g. the external package) needs to be given as an
// input to conversion-gen for it to find the native conversation funcs to
// call.

// +k8s:deepcopy-gen=package
// +k8s:conversion-gen=k8s.io/kubernetes/cmd/cloud-controller-manager/app/apis/config
// +k8s:conversion-gen=k8s.io/apimachinery/pkg/apis/config/v1alpha1
// +k8s:conversion-gen=k8s.io/apiserver/pkg/apis/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/apis/config/v1alpha1
// +k8s:openapi-gen=true
// +k8s:defaulter-gen=TypeMeta
// +groupName=cloudcontrollermanager.config.k8s.io

package v1alpha1 // import "k8s.io/kubernetes/cmd/cloud-controller-manager/app/apis/config/v1alpha1"
