/*
Copyright 2021 The Kubernetes Authors.

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

// Package apiserver provides the machinery for building Kubernetes-style API servers.
//
// This library is the foundation for the Kubernetes API server (`kube-apiserver`),
// and is also the primary framework for developers building custom API servers to extend
// the Kubernetes API.
//
// An extension API server is a user-provided, standalone web server that registers itself
// with the main kube-apiserver to handle specific API groups. This allows developers to
// extend Kubernetes with their own APIs that behave like core Kubernetes APIs, complete
// with typed clients, authentication, authorization, and discovery.
//
// # Key Packages
//
// The `apiserver` library is composed of several key packages:
//
//   - `pkg/server`: This is the core of the library, providing the `GenericAPIServer`
//     and the main machinery for building the server.
//   - `pkg/admission`: This package contains the admission control framework. Developers
//     can use this to build custom admission plugins that can validate or mutate
//     requests to enforce custom policies. This is a common way to extend Kubernetes
//     behavior without adding a full API server.
//   - `pkg/authentication`: This package provides the framework for authenticating
//     requests.
//   - `pkg/authorization`: This package provides the framework for authorizing
//     requests.
//   - `pkg/endpoints`: This package contains the machinery for building the REST
//     endpoints for the API server.
//   - `pkg/registry`: This package provides the storage interface for the API server.
//
// # Instantiating a GenericAPIServer
//
// The `GenericAPIServer` struct is the heart of any extension server. It is responsible
// for assembling and running the HTTP serving stack. See the runnable example for a
// demonstration of how to instantiate a `GenericAPIServer`.
//
// # Building an Extension API Server (API Aggregation)
//
// The mechanism that enables extension API servers is API aggregation. The
// primary apiserver (typically the kube-apiserver) acts as a proxy, forwarding
// requests for a specific API group (e.g., /apis/myextension.io/v1) to a
// registered extension server. The apiserver is configured using
// APIService objects.
//
// For most use cases, custom resources (CustomResourceDefinitions) are the
// preferred way to extend the Kubernetes API.
//
// # Building an Admission Plugin
//
// The `pkg/admission` package provides a way to add admission policies directly
// into an apiserver. Admission plugins can be used to validate or mutate objects
// during write operations. The kube-apiserver uses admission plugins to provide
// a variety of core system capabilities.
//
// For most extension use cases dynamic admission control using policies
// (ValidatingAdmissionPolicies or MutatingAdmissionPolicies) or
// webhooks (ValidatingWebhookConfiguration and MutatingWebhookConfiguration) are the
// preferred way to extend admission control.
package apiserver
