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

// Package samples contains two kube-like generic control plane apiserver, one
// with CRDs (generic) and one without (minimum).
//
// They are here mainly to preserve the feasibility to construct these kind of
// control planes. Eventually, we might promote them to be example for 3rd parties
// to follow.
package samples
