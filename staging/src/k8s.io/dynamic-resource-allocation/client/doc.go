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

// Package clients provides a wrapper around client-go such that consumers of
// this package can use the latest resource.k8s.io API. Under the hood those
// types get converted to and from the most recent API version supported by the
// apiserver.
//
// Patching and server-side-apply are not supported and return the
// [ErrNotImplemented] error. It would be necessary to convert the patch or
// apply configuration, which is close to impossible (patch) and more code
// (apply configuration).
package client
