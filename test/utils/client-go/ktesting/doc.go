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

// Package ktesting in client-go adds supports for testing with client-go
// to ktesting:
// - Storing and retrieving client instances and REST configuration.
// - TODO: Namespace creation.
//
// Those can of course also be passed as separate parameters.
// When using TContext, a single parameter is enough.
//
// TODO: move to k8s.io/client-go/ktesting
package ktesting
