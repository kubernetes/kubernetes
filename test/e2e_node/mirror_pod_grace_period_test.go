/*
Copyright 2020 The Kubernetes Authors.

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
	"fmt"
	"os"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = SIGDescribe("MirrorPodWithGracePeriod", func() {
	f := framework.NewDefaultFramework("mirror-pod-with-grace-period")
	ginkgo.Context("when create a mirror pod ", func() {
		var ns, podPath, staticPodName, mirrorPodName string
		ginkgo.BeforeEach(func() {
			ns = f.Namespace.Name
			staticPodName = "graceful-pod-" + string(uuid.NewUUID())
			mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName

			podPath = framework.TestContext.KubeletConfig.StaticPodPath

			ginkgo.By("create the static pod")
			err := createStaticPodWithGracePeriod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be running")
			gomega.Eventually(func() error {
				return checkMirrorPodRunning(f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
		})

		ginkgo.It("mirror pod termination should satisfy grace period when static pod is deleted [NodeConformance]", func() {
			ginkgo.By("get mirror pod uid")
			_, err := f.ClientSet.CoreV1().Pods(ns).Get(context.TODO(), mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)

			start := time.Now()

			ginkgo.By("delete the static pod")
			file := staticPodPath(podPath, staticPodName, ns)
			framework.Logf("deleting static pod manifest %q", file)
			err = os.Remove(file)
			framework.ExpectNoError(err)

			for {
				if time.Now().Sub(start).Seconds() > 19 {
					break
				}
				pod, err := f.ClientSet.CoreV1().Pods(ns).Get(context.TODO(), mirrorPodName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				if pod.Status.Phase != v1.PodRunning {
					framework.Failf("expected the mirror pod %q to be running, got %q", mirrorPodName, pod.Status.Phase)
				}
				// have some pause in between the API server queries to avoid throttling
				time.Sleep(time.Duration(200) * time.Millisecond)
			}
		})

		ginkgo.AfterEach(func() {
			ginkgo.By("wait for the mirror pod to disappear")
			gomega.Eventually(func() error {
				return checkMirrorPodDisappear(f.ClientSet, mirrorPodName, ns)
			}, time.Second*19, time.Second).Should(gomega.BeNil())
		})
	})
})

func createStaticPodWithGracePeriod(dir, name, namespace string) error {
	template := `
apiVersion: v1
kind: Pod
metadata:
  name: %s
  namespace: %s
spec:
  terminationGracePeriodSeconds: 20
  containers:
  - name: m-test
    image: busybox:1.31.1
    command:
      - /bin/sh
    args:
      - '-c'
      - |
        _term() {
        echo "Caught SIGTERM signal!"
        sleep 100
        }
        trap _term SIGTERM
        sleep 1000
`
	file := staticPodPath(dir, name, namespace)
	podYaml := fmt.Sprintf(template, name, namespace)

	f, err := os.OpenFile(file, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer f.Close()

	_, err = f.WriteString(podYaml)
	framework.Logf("has written %v", file)
	return err
}
