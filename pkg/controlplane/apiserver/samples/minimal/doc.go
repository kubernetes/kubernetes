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

// sample-minimal-controlplane is a kube-like generic control plane
// - without CRDs
// - with a limited set of built-in resources
// - with aggregation (TODO: remove)
// - without the container domain specific APIs.
//
// TODO(sttts): remove/disable aggregation
package main
