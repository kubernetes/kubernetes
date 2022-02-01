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
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
)

// SIGDescribe annotates the test with the SIG label.
func SIGDescribe(text string, body func()) bool {
	return ginkgo.Describe("[sig-windows] "+text, func() {
		ginkgo.BeforeEach(func() {
			// all tests in this package are Windows specific
			e2eskipper.SkipUnlessNodeOSDistroIs("windows")
		})

		// enable HostProcessContainers by default (it is Beta and on by default in beta in 1.23)
		// this allows us to use host process containers in some of the tests but allows for
		// someone that disables it to still run a subset of the tests
		if framework.TestContext.FeatureGates == nil {
			framework.TestContext.FeatureGates = map[string]bool{}
			framework.TestContext.FeatureGates[string(features.WindowsHostProcessContainers)] = true
		}
		_, exists := framework.TestContext.FeatureGates[string(features.WindowsHostProcessContainers)]
		if !exists {
			framework.TestContext.FeatureGates[string(features.WindowsHostProcessContainers)] = true
		}

		body()
	})
}
