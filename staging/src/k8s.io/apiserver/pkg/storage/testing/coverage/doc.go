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

// Package coverage classifies calls made through storage.Interface into a
// small set of API states (ResourceVersion mode, ResourceVersionMatch,
// pagination, selectors, preconditions, terminal error outcome) and reports
// which of those states existing test suites actually exercise.
//
// Wrap decorates a storage.Interface so that Get, GetList, Watch and Delete
// calls are classified and recorded, while every other method passes through
// untouched. It has no dependency beyond storage/apimachinery/stdlib, so it
// can be dropped around any storage.Interface implementation's test suite
// without pulling in new test-only dependencies.
package coverage
