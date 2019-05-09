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

// Package openapi is a collection of libraries for fetching the openapi spec
// from a Kubernetes server and then indexing the type definitions.
// The openapi spec contains the object model definitions and extensions metadata
// such as the patchStrategy and patchMergeKey for creating patches.
package openapi // k8s.io/kubernetes/pkg/kubectl/cmd/util/openapi
