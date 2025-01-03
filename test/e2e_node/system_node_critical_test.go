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

package e2enode

import (
	"context"
	"errors"
	"fmt"
	"os"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeapi "k8s.io/kubernetes/pkg/apis/core"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/nodefeature"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("SystemNodeCriticalPod", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), nodefeature.SystemNodeCriticalPod, feature.SystemNodeCriticalPod, nodefeature.Eviction, feature.Eviction, func() {
	f := framework.NewDefaultFramework("system-node-critical-pod-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	// this test only manipulates pods in kube-system
	f.SkipNamespaceCreation = true

	ginkgo.AfterEach(func(ctx context.Context) {
		if framework.TestContext.PrepullImages {
			// The test may cause the prepulled images to be evicted,
			// prepull those images again to ensure this test not affect following tests.
			err := PrePullAllImages(ctx)
			gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
		}
	})
	ginkgo.Context("when create a system-node-critical pod", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			diskConsumed := resource.MustParse("200Mi")
			summary := eventuallyGetSummary(ctx)
			availableBytes := *(summary.Node.Fs.AvailableBytes)
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalNodeFsAvailable): fmt.Sprintf("%d", availableBytes-uint64(diskConsumed.Value()))}
			initialConfig.EvictionMinimumReclaim = map[string]string{}
		})

		// Place the remainder of the test within a context so that the kubelet config is set before and after the test.
		ginkgo.Context("", func() {
			var staticPodName, mirrorPodName, podPath string
			ns := kubeapi.NamespaceSystem

			ginkgo.BeforeEach(func(ctx context.Context) {
				ginkgo.By("create a static system-node-critical pod")
				staticPodName = "static-disk-hog-" + string(uuid.NewUUID())
				mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName
				podPath = kubeletCfg.StaticPodPath
				// define a static pod consuming disk gradually
				// the upper limit is 1024 (iterations) * 10485760 bytes (10MB) = 10GB
				err := createStaticSystemNodeCriticalPod(
					podPath, staticPodName, ns, busyboxImage, v1.RestartPolicyNever, 1024,
					"dd if=/dev/urandom of=file${i} bs=10485760 count=1 2>/dev/null; sleep .1;",
				)
				gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

				ginkgo.By("wait for the mirror pod to be running")
				gomega.Eventually(ctx, func(ctx context.Context) error {
					return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
				}, time.Minute, time.Second*2).Should(gomega.Succeed())
			})

			ginkgo.It("should not be evicted upon DiskPressure", func(ctx context.Context) {
				ginkgo.By("wait for the node to have DiskPressure condition")
				gomega.Eventually(ctx, func(ctx context.Context) error {
					if hasNodeCondition(ctx, f, v1.NodeDiskPressure) {
						return nil
					}
					msg := fmt.Sprintf("NodeCondition: %s not encountered yet", v1.NodeDiskPressure)
					return errors.New(msg)
				}, time.Minute*2, time.Second*4).Should(gomega.Succeed())

				ginkgo.By("check if it's running all the time")
				gomega.Consistently(ctx, func(ctx context.Context) error {
					err := checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
					if err == nil {
						framework.Logf("mirror pod %q is running", mirrorPodName)
					} else {
						framework.Logf("%s", err.Error())
					}
					return err
				}, time.Minute*8, time.Second*4).ShouldNot(gomega.HaveOccurred())
			})
			ginkgo.AfterEach(func(ctx context.Context) {
				defer func() {
					if framework.TestContext.PrepullImages {
						// The test may cause the prepulled images to be evicted,
						// prepull those images again to ensure this test not affect following tests.
						err := PrePullAllImages(ctx)
						gomega.Expect(err).ShouldNot(gomega.HaveOccurred())
					}
				}()
				ginkgo.By("delete the static pod")
				err := deleteStaticPod(podPath, staticPodName, ns)
				gomega.Expect(err).ShouldNot(gomega.HaveOccurred())

				ginkgo.By("wait for the mirror pod to disappear")
				gomega.Eventually(ctx, func(ctx context.Context) error {
					return checkMirrorPodDisappear(ctx, f.ClientSet, mirrorPodName, ns)
				}, time.Minute, time.Second*2).Should(gomega.Succeed())

				ginkgo.By("making sure that node no longer has DiskPressure")
				gomega.Eventually(ctx, func(ctx context.Context) error {
					if hasNodeCondition(ctx, f, v1.NodeDiskPressure) {
						return fmt.Errorf("Conditions haven't returned to normal, node still has DiskPressure")
					}
					return nil
				}, pressureDisappearTimeout, evictionPollInterval).Should(gomega.Succeed())
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
    command: ["sh", "-c", "i=0; while [ $i -lt %d ]; do %s i=$(($i+1)); done; while true; do sleep 5; done"]
  restartPolicy: %s
`
	file := staticPodPath(dir, name, namespace)
	podYaml := fmt.Sprintf(template, name, namespace, name, image, iterations, command, string(restart))

	f, err := os.OpenFile(file, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.WriteString(podYaml)
	return err
}
