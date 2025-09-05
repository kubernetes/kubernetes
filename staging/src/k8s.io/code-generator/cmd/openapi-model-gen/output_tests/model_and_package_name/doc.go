/*
Copyright 2025 The Kubernetes Authors.

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

// +k8s:openapi-model-gen=true
// +modelPackageName=should_not_be_used

package model_and_package_name

type TypeMeta int

// +modelName=io.k8s.api.core.v1.T1
type T1 struct {
	TypeMeta
}

// +modelName=io.k8s.api.core.v1.T2
type T2 struct {
	TypeMeta
}
