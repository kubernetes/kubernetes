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

// Package v1alpha1 contains the alpha implementation of the DRA health gRPC
// interface. It is superseded by k8s.io/kubelet/pkg/apis/dra-health/v1,
// which uses the same proto. This package is kept for compatibility during
// the transition and gets removed in the 1.40 era.
//
// This intentionally does not use a "Deprecated:" marker: the kubelet and
// the kubeletplugin helper have to keep using this package while it is
// supported, and the marker would flag those imports in linters.
package v1alpha1
