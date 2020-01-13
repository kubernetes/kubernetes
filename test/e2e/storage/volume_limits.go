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

package storage

import (
	"github.com/onsi/ginkgo"
	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("Volume limits", func() {
	var (
		c clientset.Interface
	)
	f := framework.NewDefaultFramework("volume-limits-on-node")
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("aws", "gce", "gke")
		c = f.ClientSet
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
	})

	ginkgo.It("should verify that all nodes have volume limits", func() {
		nodeList, err := e2enode.GetReadySchedulableNodes(f.ClientSet)
		framework.ExpectNoError(err)
		for _, node := range nodeList.Items {
			volumeLimits := getVolumeLimit(&node)
			if len(volumeLimits) == 0 {
				framework.Failf("Expected volume limits to be set")
			}
		}
	})
})

func getVolumeLimit(node *v1.Node) map[v1.ResourceName]int64 {
	volumeLimits := map[v1.ResourceName]int64{}
	nodeAllocatables := node.Status.Allocatable
	for k, v := range nodeAllocatables {
		if v1helper.IsAttachableVolumeResourceName(k) {
			volumeLimits[k] = v.Value()
		}
	}
	return volumeLimits
}
