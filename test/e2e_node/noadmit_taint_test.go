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

package e2e_node

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// NOTE(harry): this test will taint a node, which means adding other specs into this context
// may be influenced when testing with -node=N.
var _ = framework.KubeDescribe("[Serial] NoAdmitTaint", func() {
	f := framework.NewDefaultFramework("admit-pod")
	Context("when create a static pod", func() {
		var ns, staticPodName, mirrorPodName, nodeName, taintName, taintValue string
		var taint api.Taint
		BeforeEach(func() {
			nodeName = framework.TestContext.NodeName
			ns = f.Namespace.Name
			staticPodName = "static-pod-" + string(uuid.NewUUID())
			// we need to check the mirror pod name (suffixed by nodeName)
			mirrorPodName = staticPodName + "-" + nodeName
		})
		It("should be rejected when node is tainted with NoAdmit effect ", func() {
			By("set NoAdmit taint for the node")
			taintName = fmt.Sprintf("kubernetes.io/e2e-taint-key-%s", string(uuid.NewUUID()))
			taintValue = "testing-taint-value"
			taintEffect := api.TaintEffectNoScheduleNoAdmit
			taint = api.Taint{
				Key:    taintName,
				Value:  taintValue,
				Effect: taintEffect,
			}
			framework.AddOrUpdateTaintOnNode(f.Client, nodeName, taint)
			framework.ExpectNodeHasTaint(f.Client, nodeName, taint)

			By("create the static pod")
			err := createStaticPod(framework.TestContext.ManifestPath, staticPodName, ns, "nginx", api.RestartPolicyAlways)
			Expect(err).ShouldNot(HaveOccurred())

			By("Waiting for static pod rejected event")
			eventFound := false

			_, controller := cache.NewInformer(
				&cache.ListWatch{
					ListFunc: func(options api.ListOptions) (runtime.Object, error) {
						return f.Client.Events(f.Namespace.Name).List(options)
					},
					WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
						return f.Client.Events(f.Namespace.Name).Watch(options)
					},
				},
				&api.Event{},
				0,
				cache.ResourceEventHandlerFuncs{
					AddFunc: func(obj interface{}) {
						if e, ok := obj.(*api.Event); ok {
							if e.InvolvedObject.Kind == "Pod" && e.Reason == "PodToleratesNodeTaints" && strings.Contains(e.Message,
								"Taint Toleration unmatched with SomeUntoleratedTaintIsNoAdmit is: true") {
								By("PodToleratesNodeTaints event found")
								eventFound = true
							}
						}
					},
				},
			)

			stopCh := make(chan struct{})
			defer func() {
				close(stopCh)
			}()
			go controller.Run(stopCh)

			// Check if the PodToleratesNodeTaints event is found
			for start := time.Now(); time.Since(start) < 4*time.Minute; time.Sleep(2 * time.Second) {
				if eventFound {
					break
				}
			}
			Expect(eventFound).Should(Equal(true))
		})

		AfterEach(func() {
			By("delete the static pod")
			err := deleteStaticPod(framework.TestContext.ManifestPath, staticPodName, ns)
			Expect(err).ShouldNot(HaveOccurred())

			By("clear taint")
			framework.RemoveTaintOffNode(f.Client, nodeName, taint)
		})
	})
})
