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

package e2enode

import (
	"context"
	"errors"
	"os"
	"time"

	"fmt"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

var _ = SIGDescribe("MirrorPod", func() {
	f := framework.NewDefaultFramework("mirror-pod")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline
	ginkgo.Context("when create a mirror pod check readiness ", func() {
		var ns, podPath, staticPodName, mirrorPodName string
		ginkgo.BeforeEach(func(ctx context.Context) {
			ns = f.Namespace.Name
			staticPodName = "static-pod-" + string(uuid.NewUUID())
			mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName

			podPath = framework.TestContext.KubeletConfig.StaticPodPath

			ginkgo.By("create the static pod")
			err := createStaticPodWithReadiness(podPath, staticPodName, ns,
				imageutils.GetE2EImage(imageutils.Perl), v1.RestartPolicyAlways)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be running")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())

			ginkgo.By("wait for pod to be ready")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("expected the mirror pod %q to be present: %v", mirrorPodName, err)
				}
				for _, c := range pod.Status.Conditions {
					if c.Type == v1.PodReady && c.Status == v1.ConditionTrue {
						return nil
					}
				}
				return errors.New("pod needs to be ready")
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
		})
		/*
			Testname: Mirror Pod, readiness
			Description: checks the readiness of a pod on termination
		*/
		ginkgo.It("should be unready with readiness probe failure on termination [NodeConformance]", func(ctx context.Context) {
			// remove the static pod, pod should go unready
			err := deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be not ready")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
				if err != nil {
					return fmt.Errorf("expected the mirror pod %q to be present: %v", mirrorPodName, err)
				}
				for _, c := range pod.Status.Conditions {
					if c.Type == v1.PodReady && c.Status == v1.ConditionFalse {
						return nil
					}
				}
				return errors.New("pod needs to be unready")
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
		})
	})
})

func createStaticPodWithReadiness(dir, name, namespace, image string, restart v1.RestartPolicy) error {
	template := `
apiVersion: v1
kind: Pod
metadata:
  name: %s
  namespace: %s
spec:
  containers:
  - name: test
    image: %s
    restartPolicy: %s
    command:
      - bash
      - -c
      - "_term() { rm -f /tmp/ready ; sleep 10; exit 0; } ; trap _term SIGTERM; touch /tmp/ready ; while true; do echo \"hello\"; sleep 10; done"
    readinessProbe:
      exec:
        command:
        - cat
        - /tmp/ready
      failureThreshold: 1
      initialDelaySeconds: 5
      periodSeconds: 2
`
	file := staticPodPath(dir, name, namespace)
	podYaml := fmt.Sprintf(template, name, namespace, image, string(restart))

	f, err := os.OpenFile(file, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.WriteString(podYaml)
	return err
}
