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
	goerrors "errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var _ = framework.KubeDescribe("MirrorPod", func() {
	f := framework.NewDefaultFramework("mirror-pod")
	Context("when create a mirror pod ", func() {
		var ns, manifestPath, staticPodName, mirrorPodName string
		BeforeEach(func() {
			ns = f.Namespace.Name
			staticPodName = "static-pod-" + string(uuid.NewUUID())
			mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName

			manifestPath = framework.TestContext.KubeletConfig.PodManifestPath

			By("create the static pod")
			err := createStaticPod(manifestPath, staticPodName, ns,
				imageutils.GetE2EImage(imageutils.NginxSlim), v1.RestartPolicyAlways)
			Expect(err).ShouldNot(HaveOccurred())

			By("wait for the mirror pod to be running")
			Eventually(func() error {
				return checkMirrorPodRunning(f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(BeNil())
		})
		It("should be updated when static pod updated [Conformance]", func() {
			By("get mirror pod uid")
			pod, err := f.ClientSet.Core().Pods(ns).Get(mirrorPodName, metav1.GetOptions{})
			Expect(err).ShouldNot(HaveOccurred())
			uid := pod.UID

			By("update the static pod container image")
			image := framework.GetPauseImageNameForHostArch()
			err = createStaticPod(manifestPath, staticPodName, ns, image, v1.RestartPolicyAlways)
			Expect(err).ShouldNot(HaveOccurred())

			By("wait for the mirror pod to be updated")
			Eventually(func() error {
				return checkMirrorPodRecreatedAndRunnig(f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(BeNil())

			By("check the mirror pod container image is updated")
			pod, err = f.ClientSet.Core().Pods(ns).Get(mirrorPodName, metav1.GetOptions{})
			Expect(err).ShouldNot(HaveOccurred())
			Expect(len(pod.Spec.Containers)).Should(Equal(1))
			Expect(pod.Spec.Containers[0].Image).Should(Equal(image))
		})
		It("should be recreated when mirror pod gracefully deleted [Conformance]", func() {
			By("get mirror pod uid")
			pod, err := f.ClientSet.Core().Pods(ns).Get(mirrorPodName, metav1.GetOptions{})
			Expect(err).ShouldNot(HaveOccurred())
			uid := pod.UID

			By("delete the mirror pod with grace period 30s")
			err = f.ClientSet.Core().Pods(ns).Delete(mirrorPodName, metav1.NewDeleteOptions(30))
			Expect(err).ShouldNot(HaveOccurred())

			By("wait for the mirror pod to be recreated")
			Eventually(func() error {
				return checkMirrorPodRecreatedAndRunnig(f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(BeNil())
		})
		It("should be recreated when mirror pod forcibly deleted [Conformance]", func() {
			By("get mirror pod uid")
			pod, err := f.ClientSet.Core().Pods(ns).Get(mirrorPodName, metav1.GetOptions{})
			Expect(err).ShouldNot(HaveOccurred())
			uid := pod.UID

			By("delete the mirror pod with grace period 0s")
			err = f.ClientSet.Core().Pods(ns).Delete(mirrorPodName, metav1.NewDeleteOptions(0))
			Expect(err).ShouldNot(HaveOccurred())

			By("wait for the mirror pod to be recreated")
			Eventually(func() error {
				return checkMirrorPodRecreatedAndRunnig(f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(BeNil())
		})
		AfterEach(func() {
			By("delete the static pod")
			err := deleteStaticPod(manifestPath, staticPodName, ns)
			Expect(err).ShouldNot(HaveOccurred())

			By("wait for the mirror pod to disappear")
			Eventually(func() error {
				return checkMirrorPodDisappear(f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(BeNil())
		})
	})
})

func staticPodPath(dir, name, namespace string) string {
	return filepath.Join(dir, namespace+"-"+name+".yaml")
}

func createStaticPod(dir, name, namespace, image string, restart v1.RestartPolicy) error {
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

func deleteStaticPod(dir, name, namespace string) error {
	file := staticPodPath(dir, name, namespace)
	return os.Remove(file)
}

func checkMirrorPodDisappear(cl clientset.Interface, name, namespace string) error {
	_, err := cl.Core().Pods(namespace).Get(name, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		return nil
	}
	return goerrors.New("pod not disappear")
}

func checkMirrorPodRunning(cl clientset.Interface, name, namespace string) error {
	pod, err := cl.Core().Pods(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("expected the mirror pod %q to appear: %v", name, err)
	}
	if pod.Status.Phase != v1.PodRunning {
		return fmt.Errorf("expected the mirror pod %q to be running, got %q", name, pod.Status.Phase)
	}
	return nil
}

func checkMirrorPodRecreatedAndRunnig(cl clientset.Interface, name, namespace string, oUID types.UID) error {
	pod, err := cl.Core().Pods(namespace).Get(name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("expected the mirror pod %q to appear: %v", name, err)
	}
	if pod.UID == oUID {
		return fmt.Errorf("expected the uid of mirror pod %q to be changed, got %q", name, pod.UID)
	}
	if pod.Status.Phase != v1.PodRunning {
		return fmt.Errorf("expected the mirror pod %q to be running, got %q", name, pod.Status.Phase)
	}
	return nil
}
