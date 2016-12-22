/*
Copyright 2016 The Kubernetes Authors.

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

package e2e

import (
	"time"

	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
)

var _ = framework.KubeDescribe("Empty [Feature:Empty]", func() {
	f := framework.NewDefaultFramework("empty")

	BeforeEach(func() {
		c := f.ClientSet
		ns := f.Namespace.Name

		// TODO: respect --allow-notready-nodes flag in those functions.
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
		framework.WaitForAllNodesHealthy(c, time.Minute)

		err := framework.CheckTestingNSDeletedExcept(c, ns)
		framework.ExpectNoError(err)
	})

	It("does nothing", func() {})
})
