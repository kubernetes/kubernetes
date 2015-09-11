/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/labels"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/kubernetes/pkg/fields"
)

var _ = Describe("Mesos", func() {
	framework := NewFramework("pods")

	BeforeEach(func() {
		SkipUnlessProviderIs("mesos/docker")
	})

	It("applies slave attributes as labels", func() {
		nodeClient := framework.Client.Nodes()

		rackA := labels.SelectorFromSet(map[string]string{"k8s.mesosphere.io/attribute-rack": "1"})
		nodes, err := nodeClient.List(rackA, fields.Everything())
		if err != nil {
			Failf("Failed to query for node: %v", err)
		}
		Expect(len(nodes.Items)).To(Equal(1))

		var addr string
		for _, a := range nodes.Items[0].Status.Addresses {
			if a.Type == api.NodeInternalIP {
				addr = a.Address
			}
		}
		Expect(len(addr)).NotTo(Equal(""))
	})
})
