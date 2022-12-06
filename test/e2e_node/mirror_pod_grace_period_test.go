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

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("MirrorPodWithGracePeriod", func() {
	f := framework.NewDefaultFramework("mirror-pod-with-grace-period")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline
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
			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(context.TODO(), mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			uid := pod.UID

			ginkgo.By("delete the static pod")
			file := staticPodPath(podPath, staticPodName, ns)
			framework.Logf("deleting static pod manifest %q", file)
			err = os.Remove(file)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be running for grace period")
			gomega.Consistently(func() error {
				return checkMirrorPodRunningWithUID(f.ClientSet, mirrorPodName, ns, uid)
			}, 19*time.Second, 200*time.Millisecond).Should(gomega.BeNil())
		})

		ginkgo.It("mirror pod termination should satisfy grace period when static pod is updated [NodeConformance]", func() {
			ginkgo.By("get mirror pod uid")
			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(context.TODO(), mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			uid := pod.UID

			ginkgo.By("update the static pod container image")
			image := imageutils.GetPauseImageName()
			err = createStaticPod(podPath, staticPodName, ns, image, v1.RestartPolicyAlways)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be running for grace period")
			gomega.Consistently(func() error {
				return checkMirrorPodRunningWithUID(f.ClientSet, mirrorPodName, ns, uid)
			}, 19*time.Second, 200*time.Millisecond).Should(gomega.BeNil())

			ginkgo.By("wait for the mirror pod to be updated")
			gomega.Eventually(func() error {
				return checkMirrorPodRecreatedAndRunning(f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())

			ginkgo.By("check the mirror pod container image is updated")
			pod, err = f.ClientSet.CoreV1().Pods(ns).Get(context.TODO(), mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			framework.ExpectEqual(len(pod.Spec.Containers), 1)
			framework.ExpectEqual(pod.Spec.Containers[0].Image, image)
		})

		ginkgo.It("should update a static pod when the static pod is updated multiple times during the graceful termination period [NodeConformance]", func() {
			ginkgo.By("get mirror pod uid")
			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(context.TODO(), mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			uid := pod.UID

			ginkgo.By("update the pod manifest multiple times during the graceful termination period")
			for i := 0; i < 300; i++ {
				err = createStaticPod(podPath, staticPodName, ns,
					fmt.Sprintf("image-%d", i), v1.RestartPolicyAlways)
				framework.ExpectNoError(err)
				time.Sleep(100 * time.Millisecond)
			}
			image := imageutils.GetPauseImageName()
			err = createStaticPod(podPath, staticPodName, ns, image, v1.RestartPolicyAlways)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be updated")
			gomega.Eventually(func() error {
				return checkMirrorPodRecreatedAndRunning(f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())

			ginkgo.By("check the mirror pod container image is updated")
			pod, err = f.ClientSet.CoreV1().Pods(ns).Get(context.TODO(), mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			framework.ExpectEqual(len(pod.Spec.Containers), 1)
			framework.ExpectEqual(pod.Spec.Containers[0].Image, image)
		})

		ginkgo.AfterEach(func() {
			ginkgo.By("delete the static pod")
			err := deleteStaticPod(podPath, staticPodName, ns)
			if !os.IsNotExist(err) {
				framework.ExpectNoError(err)
			}

			ginkgo.By("wait for the mirror pod to disappear")
			gomega.Eventually(func() error {
				return checkMirrorPodDisappear(f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
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

func checkMirrorPodRunningWithUID(cl clientset.Interface, name, namespace string, oUID types.UID) error {
	pod, err := cl.CoreV1().Pods(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("expected the mirror pod %q to appear: %v", name, err)
	}
	if pod.UID != oUID {
		return fmt.Errorf("expected the uid of mirror pod %q to be same, got %q", name, pod.UID)
	}
	if pod.Status.Phase != v1.PodRunning {
		return fmt.Errorf("expected the mirror pod %q to be running, got %q", name, pod.Status.Phase)
	}
	return nil
}
