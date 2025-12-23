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

// Package constants provides canonical definitions for Kubernetes constants.
//
// This module contains:
//   - RFC-defined limits (DNS naming constraints from RFC 1123, RFC 1035)
//   - Kubernetes-specific limits (label value length, field manager length)
//   - Well-known label keys (topology, node labels)
//   - Well-known annotation keys
//   - Well-known taint keys
//
// This module has ZERO dependencies, making it suitable for any package that
// needs Kubernetes constants without pulling in the full apimachinery or api modules.
//
// Other k8s.io modules re-export these constants for backwards compatibility,
// but new code should import directly from this package for minimal dependencies.
package constants
