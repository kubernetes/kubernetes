/*
Copyright 2019 The Kubernetes Authors.

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
	"os"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeapi "k8s.io/kubernetes/pkg/apis/core"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = framework.KubeDescribe("SystemNodeCriticalPod [Slow] [Serial] [Disruptive] [NodeFeature:SystemNodeCriticalPod]", func() {
	f := framework.NewDefaultFramework("system-node-critical-pod-test")
	// this test only manipulates pods in kube-system
	f.SkipNamespaceCreation = true

	Context("when create a system-node-critical pod", func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			diskConsumed := resource.MustParse("200Mi")
			summary := eventuallyGetSummary()
			availableBytes := *(summary.Node.Fs.AvailableBytes)
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalNodeFsAvailable): fmt.Sprintf("%d", availableBytes-uint64(diskConsumed.Value()))}
			initialConfig.EvictionMinimumReclaim = map[string]string{}
		})

		// Place the remainder of the test within a context so that the kubelet config is set before and after the test.
		Context("", func() {
			var staticPodName, mirrorPodName, podPath string
			ns := kubeapi.NamespaceSystem

			BeforeEach(func() {
				By("create a static system-node-critical pod")
				staticPodName = "static-disk-hog-" + string(uuid.NewUUID())
				mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName
				podPath = framework.TestContext.KubeletConfig.StaticPodPath
				// define a static pod consuming disk gradually
				// the upper limit is 1024 (iterations) * 10485760 bytes (10MB) = 10GB
				err := createStaticSystemNodeCriticalPod(
					podPath, staticPodName, ns, busyboxImage, v1.RestartPolicyNever, 1024,
					"dd if=/dev/urandom of=file${i} bs=10485760 count=1 2>/dev/null; sleep .1;",
				)
				Expect(err).ShouldNot(HaveOccurred())

				By("wait for the mirror pod to be running")
				Eventually(func() error {
					return checkMirrorPodRunning(f.ClientSet, mirrorPodName, ns)
				}, time.Minute, time.Second*2).Should(BeNil())
			})

			It("should not be evicted upon DiskPressure", func() {
				By("wait for the node to have DiskPressure condition")
				Eventually(func() error {
					if hasNodeCondition(f, v1.NodeDiskPressure) {
						return nil
					}
					msg := fmt.Sprintf("NodeCondition: %s not encountered yet", v1.NodeDiskPressure)
					framework.Logf(msg)
					return fmt.Errorf(msg)
				}, time.Minute*2, time.Second*4).Should(BeNil())

				By("check if it's running all the time")
				Consistently(func() error {
					err := checkMirrorPodRunning(f.ClientSet, mirrorPodName, ns)
					if err == nil {
						framework.Logf("mirror pod %q is running", mirrorPodName)
					} else {
						framework.Logf(err.Error())
					}
					return err
				}, time.Minute*8, time.Second*4).ShouldNot(HaveOccurred())
			})
			AfterEach(func() {
				By("delete the static pod")
				err := deleteStaticPod(podPath, staticPodName, ns)
				Expect(err).ShouldNot(HaveOccurred())

				By("wait for the mirror pod to disappear")
				Eventually(func() error {
					return checkMirrorPodDisappear(f.ClientSet, mirrorPodName, ns)
				}, time.Minute, time.Second*2).Should(BeNil())
			})
		})
	})
})

func createStaticSystemNodeCriticalPod(dir, name, namespace, image string, restart v1.RestartPolicy,
	iterations int, command string) error {
	template := `
apiVersion: v1
kind: Pod
metadata:
  name: %s
  namespace: %s
spec:
  priorityClassName: system-node-critical
  containers:
  - name: %s
    image: %s
    restartPolicy: %s
    command: ["sh", "-c", "i=0; while [ $i -lt %d ]; do %s i=$(($i+1)); done; while true; do sleep 5; done"]
`
	file := staticPodPath(dir, name, namespace)
	podYaml := fmt.Sprintf(template, name, namespace, name, image, string(restart), iterations, command)

	f, err := os.OpenFile(file, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.WriteString(podYaml)
	return err
}
