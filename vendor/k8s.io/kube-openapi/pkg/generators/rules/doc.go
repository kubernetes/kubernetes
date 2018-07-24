/*
Copyright 2018 The Kubernetes Authors.

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

// Package rules contains API rules that are enforced in OpenAPI spec generation
// as part of the machinery. Files under this package implement APIRule interface
// which evaluates Go type and produces list of API rule violations.
//
// Implementations of APIRule should be added to API linter under openAPIGen code-
// generator to get integrated in the generation process.
package rules
