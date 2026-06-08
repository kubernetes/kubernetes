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

// Package clientgo is the official Go client for the Kubernetes API. It provides
// a standard set of clients and tools for building applications, controllers,
// and operators that communicate with a Kubernetes cluster.
//
// # Key Packages
//
//   - kubernetes: Contains the typed Clientset for interacting with built-in,
//     versioned API objects (e.g., Pods, Deployments). This is the most common
//     starting point.
//
//   - dynamic: Provides a dynamic client that can perform operations on any
//     Kubernetes object, including Custom Resources (CRs). It is essential for
//     building controllers that work with CRDs.
//
//   - discovery: Used to discover the API groups, versions, and resources
//     supported by a Kubernetes cluster.
//
//   - tools/cache: The foundation of the controller pattern. This package provides
//     efficient caching and synchronization mechanisms (Informers and Listers)
//     for building controllers.
//
//   - tools/clientcmd: Provides methods for loading client configuration from
//     kubeconfig files. This is essential for out-of-cluster applications and CLI tools.
//
//   - rest: Provides a lower-level RESTClient that manages the details of
//     communicating with the Kubernetes API server. It is useful for advanced
//     use cases that require fine-grained control over requests, such as working
//     with non-standard REST verbs.
//
// # Connecting to the API
//
// There are two primary ways to configure a client to connect to the API server:
//
//  1. In-Cluster Configuration: For applications running inside a Kubernetes pod,
//     the `rest.InClusterConfig()` function provides a straightforward way to
//     configure the client. It automatically uses the pod's service account for
//     authentication and is the recommended approach for controllers and operators.
//
//  2. Out-of-Cluster Configuration: For local development or command-line tools,
//     the `clientcmd` package is used to load configuration from a
//     kubeconfig file.
//
// The `rest.Config` object allows for fine-grained control over client-side
// performance and reliability. Key settings include:
//
//   - QPS: The maximum number of queries per second to the API server.
//   - Burst: The maximum number of queries that can be issued in a single burst.
//   - Timeout: The timeout for individual requests.
//
// # Interacting with API Objects
//
// Once configured, a client can be used to interact with objects in the cluster.
//
//   - The Typed Clientset (`kubernetes` package) provides a strongly typed
//     interface for working with built-in Kubernetes objects.
//
//   - The Dynamic Client (`dynamic` package) can work with any object, including
//     Custom Resources, using `unstructured.Unstructured` types.
//
//   - For Custom Resources (CRDs), the `k8s.io/code-generator` repository
//     contains the tools to generate typed clients, informers, and listers. The
//     `sample-controller` is the canonical example of this pattern.
//
//   - Server-Side Apply is a patching strategy that allows multiple actors to
//     share management of an object by tracking field ownership. This prevents
//     actors from inadvertently overwriting each other's changes and provides
//     a mechanism for resolving conflicts. The `applyconfigurations` package
//     provides the necessary tools for this declarative approach.
//
// # Handling API Errors
//
// Robust error handling is essential when interacting with the API. The
// `k8s.io/apimachinery/pkg/api/errors` package provides functions to inspect
// errors and check for common conditions, such as whether a resource was not
// found or already exists. This allows controllers to implement robust,
// idempotent reconciliation logic.
//
// # Building Controllers
//
// The controller pattern is central to Kubernetes. A controller observes the
// state of the cluster and works to bring it to the desired state.
//
//   - The `tools/cache` package provides the building blocks for this pattern.
//     Informers watch the API server and maintain a local cache, Listers provide
//     read-only access to the cache, and Workqueues decouple event detection
//     from processing.
//
//   - In a high-availability deployment where multiple instances of a controller
//     are running, leader election (`tools/leaderelection`) is used to ensure
//     that only one instance is active at a time.
//
//   - Client-side feature gates allow for enabling or disabling experimental
//     features in `client-go`. They can be configured via the `rest.Config` object.
package clientgo
