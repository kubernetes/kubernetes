/*
Copyright 2023 The Kubernetes Authors.

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
	clientset "k8s.io/client-go/kubernetes"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"os"
	"path/filepath"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
)

var _ = SIGDescribe("Terminate Pods", func() {
	f := framework.NewDefaultFramework("terminate-pods")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.It("should not hang when terminating pods mounting non-existent volumes", func(ctx context.Context) {
		pod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name: "pod1",
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:  "container1",
						Image: busyboxImage,
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "vol1",
								MountPath: "/mnt/vol1",
							},
						},
					},
				},
				Volumes: []v1.Volume{
					{
						Name: "vol1",
						VolumeSource: v1.VolumeSource{
							Secret: &v1.SecretVolumeSource{
								SecretName: "non-existent-" + string(uuid.NewUUID()),
							},
						},
					},
				},
			},
		}
		client := e2epod.NewPodClient(f)
		pod = client.Create(context.TODO(), pod)
		gomega.Expect(pod.Spec.NodeName).ToNot(gomega.BeEmpty())

		gomega.Eventually(ctx, func() bool {
			pod, _ = client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
			for _, c := range pod.Status.Conditions {
				if c.Type == v1.ContainersReady && c.Status == v1.ConditionFalse {
					return true
				}
			}
			return false
		}, 20*time.Second, 1*time.Second).Should(gomega.BeTrue())

		err := client.Delete(context.Background(), pod.Name, metav1.DeleteOptions{})

		// Wait for the pod to disappear from the API server up to 10 seconds, this shouldn't hang for minutes due to
		// non-existent secret being mounted.
		gomega.Eventually(ctx, func() bool {
			_, err := client.Get(context.TODO(), pod.Name, metav1.GetOptions{})
			return apierrors.IsNotFound(err)
		}, 10*time.Second, time.Second).Should(gomega.BeTrue())

		framework.ExpectNoError(err)
	})

	ginkgo.Context("when kubelet restarts and there is a terminating pod", func() {
		var ns, podPath, podName string
		ginkgo.It("should finish terminating", func(ctx context.Context) {
			ns = f.Namespace.Name
			podName = "terminating-pod-" + string(uuid.NewUUID())

			ginkgo.By("create the pod with a deletion timestamp")
			err := createTerminatingPod(podPath, podName, ns, imageutils.GetE2EImage(imageutils.Nginx))
			framework.ExpectNoError(err)

			ginkgo.By("wait for the pod to be terminating")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return verifyPodTerminating(ctx, f.ClientSet, podName, ns)
			}, 2*time.Minute, time.Second*1).Should(gomega.BeNil())

			ginkgo.By("wait for the pod to completely terminate")
			gomega.Eventually(ctx, func() bool {
				_, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, podName, metav1.GetOptions{})
				return apierrors.IsNotFound(err)
			}, 2*time.Minute, time.Second).Should(gomega.BeTrue())
		})
	})
})

func makePodPath(dir, name, namespace string) string {
	return filepath.Join(dir, namespace+"-"+name+".yaml")
}

func createTerminatingPod(dir, name, namespace, image string) error {
	template := `
apiVersion: v1
kind: Pod
metadata:
  name: %s
  namespace: %s
  deletionTimestamp: "2024-05-16T01:29:53Z"
spec:
  containers:
  - name: test
    image: %s
    lifecycle:
      preStop:
        exec:
          command:
          - "sleep"
          - "10"
    volumeMounts:
    - mountPath: /kube-certs
      name: kube-ssl
      readOnly: true
  volumes:
  - name: kube-ssl
    secret:
      defaultMode: 420
      secretName: kube-public-cert
`
	file := makePodPath(dir, name, namespace)
	podYaml := fmt.Sprintf(template, name, namespace, image)

	f, err := os.OpenFile(file, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer func(f *os.File) {
		_ = f.Close()
	}(f)

	_, err = f.WriteString(podYaml)
	return err
}

func verifyPodTerminating(ctx context.Context, cl clientset.Interface, name, namespace string) error {
	pod, err := cl.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("expected the pod %q to appear: %w", name, err)
	}
	if pod.Status.Phase != v1.PodRunning {
		return fmt.Errorf("expected the pod %q to be running, got %q", name, pod.Status.Phase)
	}
	for i := range pod.Status.ContainerStatuses {
		if pod.Status.ContainerStatuses[i].State.Running == nil {
			return fmt.Errorf("expected the pod %q with container %q to be running (got containers=%v)", name, pod.Status.ContainerStatuses[i].Name, pod.Status.ContainerStatuses[i].State)
		}
	}
	return nil
}
