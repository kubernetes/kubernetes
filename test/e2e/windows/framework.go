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

package windows

import (
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo/v2"
)

// sigDescribe annotates the test with the SIG label.
// Use this together with skipUnlessWindows to define
// tests that only run if the node OS is Windows:
//
//	sigDescribe("foo", skipUnlessWindows(func() { ... }))
var sigDescribe = framework.SIGDescribe("windows")

// skipUnlessWindows wraps some other Ginkgo callback such that
// a BeforeEach runs before tests defined by that callback which
// skips those tests unless the node OS is Windows.
func skipUnlessWindows(cb func()) func() {
	return func() {
		ginkgo.BeforeEach(func() {
			// all tests in this package are Windows specific
			e2eskipper.SkipUnlessNodeOSDistroIs("windows")
		})

		cb()
	}
}
