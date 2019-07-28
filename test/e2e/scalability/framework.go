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

// DEPRECATED.
// We already migrated all periodic and presubmit tests to ClusterLoader2.
// We are still keeping this directory for optional functionality, but once
// this is supported in ClusterLoader2 tests, this test will be removed
// (hopefully in 1.16 release).
// Please don't add new functionality to this directory and instead see:
// https://github.com/kubernetes/perf-tests/tree/master/clusterloader2

package scalability

import "github.com/onsi/ginkgo"

// SIGDescribe is the entry point for the sig-scalability e2e framework
func SIGDescribe(text string, body func()) bool {
	return ginkgo.Describe("[sig-scalability] "+text, body)
}
