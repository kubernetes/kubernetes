/*
Copyright The Kubernetes Authors.

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
	"encoding/json"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Dynamic Node-Local Ephemeral Volumes", framework.WithFeatureGate(features.DynamicNodeLocalEphemeralVolumes), func() {
	f := framework.NewDefaultFramework("dynamic-ephemeral-volumes")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	updatePodDynamic := func(ctx context.Context, pod *v1.Pod) (*v1.Pod, error) {
		body, err := json.Marshal(pod)
		if err != nil {
			return nil, err
		}
		result := &v1.Pod{}
		err = f.ClientSet.CoreV1().RESTClient().Put().
			Namespace(f.Namespace.Name).
			Resource("pods").
			Name(pod.Name).
			SubResource("dynamic").
			Body(body).
			Do(ctx).
			Into(result)
		return result, err
	}

	ginkgo.It("should dynamically add and remove emptyDir and configMap volumes with RestartContainer policy", func(ctx context.Context) {
		cm := &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "dyn-cm-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Data: map[string]string{
				"key1": "hello-dynamic-volume",
			},
		}
		_, err := f.ClientSet.CoreV1().ConfigMaps(f.Namespace.Name).Create(ctx, cm, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		restartPolicy := &v1.ContainerDynamicVolumePolicy{
			RestartPolicy: v1.DynamicVolumeRestartContainer,
		}
		pod := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "dyn-vol-pod-" + string(uuid.NewUUID()),
				Namespace: f.Namespace.Name,
			},
			Spec: v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:                "c1",
						Image:               busyboxImage,
						Command:             []string{"sh", "-c", "sleep 3600"},
						DynamicVolumePolicy: restartPolicy,
					},
				},
			},
		})

		ginkgo.By("Adding an emptyDir and configMap volume via /dynamic subresource")
		currentPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		updated := currentPod.DeepCopy()
		updated.Spec.Volumes = append(updated.Spec.Volumes,
			v1.Volume{
				Name: "dyn-empty",
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{},
				},
			},
			v1.Volume{
				Name: "dyn-cm",
				VolumeSource: v1.VolumeSource{
					ConfigMap: &v1.ConfigMapVolumeSource{
						LocalObjectReference: v1.LocalObjectReference{Name: cm.Name},
					},
				},
			},
		)
		updated.Spec.Containers[0].VolumeMounts = append(updated.Spec.Containers[0].VolumeMounts,
			v1.VolumeMount{Name: "dyn-empty", MountPath: "/mnt/dyn-empty"},
			v1.VolumeMount{Name: "dyn-cm", MountPath: "/mnt/dyn-cm"},
		)
		_, err = updatePodDynamic(ctx, updated)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for container c1 to restart in-place and mount the dynamic volumes")
		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).NotTo(gomega.HaveOccurred())
			g.Expect(p.Status.ContainerStatuses).To(gomega.HaveLen(1))
			g.Expect(p.Status.ContainerStatuses[0].RestartCount).To(gomega.BeNumerically(">=", 1))
			g.Expect(p.Status.VolumeStatuses).To(gomega.ContainElements(
				gomega.HaveField("Name", "dyn-empty"),
				gomega.HaveField("Name", "dyn-cm"),
			))
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())

		out := e2epod.ExecCommandInContainer(f, pod.Name, "c1", "cat", "/mnt/dyn-cm/key1")
		gomega.Expect(out).To(gomega.Equal("hello-dynamic-volume"))

		ginkgo.By("Removing dyn-cm volume via /dynamic subresource")
		currentPod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		updatedAfterRemove := currentPod.DeepCopy()
		var remainingVols []v1.Volume
		for _, v := range updatedAfterRemove.Spec.Volumes {
			if v.Name != "dyn-cm" {
				remainingVols = append(remainingVols, v)
			}
		}
		updatedAfterRemove.Spec.Volumes = remainingVols
		var remainingMounts []v1.VolumeMount
		for _, vm := range updatedAfterRemove.Spec.Containers[0].VolumeMounts {
			if vm.Name != "dyn-cm" {
				remainingMounts = append(remainingMounts, vm)
			}
		}
		updatedAfterRemove.Spec.Containers[0].VolumeMounts = remainingMounts
		_, err = updatePodDynamic(ctx, updatedAfterRemove)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for dyn-cm to be unmounted and removed from VolumeStatuses")
		gomega.Eventually(ctx, func(g gomega.Gomega) {
			p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			g.Expect(err).NotTo(gomega.HaveOccurred())
			for _, vs := range p.Status.VolumeStatuses {
				g.Expect(vs.Name).NotTo(gomega.Equal("dyn-cm"))
			}
		}, 60*time.Second, 2*time.Second).Should(gomega.Succeed())
	})
})
