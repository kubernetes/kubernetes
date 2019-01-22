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

package network

import (
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/network/scale"

	. "github.com/onsi/ginkgo"
)

var _ = SIGDescribe("Loadbalancing: L7 Scalability", func() {
	defer GinkgoRecover()
	var (
		ns string
	)
	f := framework.NewDefaultFramework("ingress-scale")

	BeforeEach(func() {
		ns = f.Namespace.Name
	})

	Describe("GCE [Slow] [Serial] [Feature:IngressScale]", func() {
		var (
			scaleFramework *scale.IngressScaleFramework
		)

		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce", "gke")

			scaleFramework = scale.NewIngressScaleFramework(f.ClientSet, ns, framework.TestContext.CloudConfig)
			if err := scaleFramework.PrepareScaleTest(); err != nil {
				framework.Failf("Unexpected error while preraring ingress scale test: %v", err)
			}
		})

		AfterEach(func() {
			if errs := scaleFramework.CleanupScaleTest(); len(errs) != 0 {
				framework.Failf("Unexpected error while cleaning up ingress scale test: %v", errs)
			}
		})

		It("Creating and updating ingresses should happen promptly with small/medium/large amount of ingresses", func() {
			if errs := scaleFramework.RunScaleTest(); len(errs) != 0 {
				framework.Failf("Unexpected error while running ingress scale test: %v", errs)
			}

		})
	})
})
