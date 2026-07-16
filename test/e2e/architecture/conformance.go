/*
Copyright 2021 The Kubernetes Authors.

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

package architecture

import (
	"context"
	"time"

	"github.com/onsi/ginkgo/v2"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Conformance Tests", func() {
	f := framework.NewDefaultFramework("conformance-tests")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
		Release: v1.23
		Testname: Conformance tests minimum number of nodes.
		Description: Conformance tests requires at least two untainted nodes where pods can be scheduled.
	*/
	framework.ConformanceIt("should have at least two untainted nodes", func(ctx context.Context) {
		ginkgo.By("Getting node addresses")
		framework.ExpectNoError(e2enode.WaitForAllNodesSchedulable(ctx, f.ClientSet, 10*time.Minute))
		nodeList, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if len(nodeList.Items) < 2 {
			framework.Failf("Conformance requires at least two nodes")
		}
	})
})
