/*
Copyright 2017 The Kubernetes Authors.

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
// +groupName=example2.k8s.io
//
// package example2 contains an example API whose internal version is defined in
// another group ("example"). This happens if a type is moved to a different
// group. It's not recommended to move types across groups, though Kubernetes
// have a few cases due to historical reasons. This package is for tests.
package example2
