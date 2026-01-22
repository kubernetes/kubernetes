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
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/features"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"sigs.k8s.io/yaml"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/printers"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
)

var _ = SIGDescribe("MirrorPod", func() {
	f := framework.NewDefaultFramework("mirror-pod")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Context("when create a mirror pod", func() {
		var ns, podPath, staticPodName, mirrorPodName string
		ginkgo.BeforeEach(func(ctx context.Context) {
			ns = f.Namespace.Name
			staticPodName = "static-pod-" + string(uuid.NewUUID())
			mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName

			podPath = kubeletCfg.StaticPodPath

			ginkgo.By("create the static pod")
			err := createStaticPod(podPath, staticPodName, ns,
				imageutils.GetE2EImage(imageutils.Nginx), v1.RestartPolicyAlways)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be running")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
		})
		/*
			Release: v1.9
			Testname: Mirror Pod, update
			Description: Updating a static Pod MUST recreate an updated mirror Pod. Create a static pod, verify that a mirror pod is created. Update the static pod by changing the container image, the mirror pod MUST be re-created and updated with the new image.
		*/
		f.It("should be updated when static pod updated", f.WithNodeConformance(), func(ctx context.Context) {
			ginkgo.By("get mirror pod uid")
			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			uid := pod.UID

			ginkgo.By("update the static pod container image")
			image := imageutils.GetPauseImageName()
			err = createStaticPod(podPath, staticPodName, ns, image, v1.RestartPolicyAlways)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be updated")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRecreatedAndRunning(ctx, f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())

			ginkgo.By("check the mirror pod container image is updated")
			pod, err = f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod.Spec.Containers).To(gomega.HaveLen(1))
			gomega.Expect(pod.Spec.Containers[0].Image).To(gomega.Equal(image))
		})
		/*
			Release: v1.9
			Testname: Mirror Pod, delete
			Description:  When a mirror-Pod is deleted then the mirror pod MUST be re-created. Create a static pod, verify that a mirror pod is created. Delete the mirror pod, the mirror pod MUST be re-created and running.
		*/
		f.It("should be recreated when mirror pod gracefully deleted", f.WithNodeConformance(), func(ctx context.Context) {
			ginkgo.By("get mirror pod uid")
			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			uid := pod.UID

			ginkgo.By("delete the mirror pod with grace period 30s")
			err = f.ClientSet.CoreV1().Pods(ns).Delete(ctx, mirrorPodName, *metav1.NewDeleteOptions(30))
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be recreated")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRecreatedAndRunning(ctx, f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
		})
		/*
			Release: v1.9
			Testname: Mirror Pod, force delete
			Description: When a mirror-Pod is deleted, forcibly, then the mirror pod MUST be re-created. Create a static pod, verify that a mirror pod is created. Delete the mirror pod with delete wait time set to zero forcing immediate deletion, the mirror pod MUST be re-created and running.
		*/
		f.It("should be recreated when mirror pod forcibly deleted", f.WithNodeConformance(), func(ctx context.Context) {
			ginkgo.By("get mirror pod uid")
			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			uid := pod.UID

			ginkgo.By("delete the mirror pod with grace period 0s")
			err = f.ClientSet.CoreV1().Pods(ns).Delete(ctx, mirrorPodName, *metav1.NewDeleteOptions(0))
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be recreated")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRecreatedAndRunning(ctx, f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
		})
		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("delete the static pod")
			err := deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to disappear")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
		})
	})
	ginkgo.Context("when create a mirror pod without changes", func() {
		var ns, podPath, staticPodName, mirrorPodName string
		ginkgo.BeforeEach(func() {
		})
		/*
			Release: v1.23
			Testname: Mirror Pod, recreate
			Description: When a static pod's manifest is removed and readded, the mirror pod MUST successfully recreate. Create the static pod, verify it is running, remove its manifest and then add it back, and verify the static pod runs again.
		*/
		f.It("should successfully recreate when file is removed and recreated", f.WithNodeConformance(), func(ctx context.Context) {
			ns = f.Namespace.Name
			staticPodName = "static-pod-" + string(uuid.NewUUID())
			mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName

			podPath = kubeletCfg.StaticPodPath
			ginkgo.By("create the static pod")
			err := createStaticPod(podPath, staticPodName, ns,
				imageutils.GetE2EImage(imageutils.Nginx), v1.RestartPolicyAlways)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be running")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())

			ginkgo.By("delete the pod manifest from disk")
			err = deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By("recreate the file")
			err = createStaticPod(podPath, staticPodName, ns,
				imageutils.GetE2EImage(imageutils.Nginx), v1.RestartPolicyAlways)
			framework.ExpectNoError(err)

			ginkgo.By("mirror pod should restart with count 1")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRunningWithRestartCount(ctx, 2*time.Second, 2*time.Minute, f.ClientSet, mirrorPodName, ns, 1)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())

			ginkgo.By("mirror pod should stay running")
			gomega.Consistently(ctx, func(ctx context.Context) error {
				return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
			}, time.Second*30, time.Second*4).Should(gomega.BeNil())

			ginkgo.By("delete the static pod")
			err = deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to disappear")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())
		})
	})
	ginkgo.Context("when recreating a static pod", func() {
		var ns, podPath, staticPodName, mirrorPodName string
		f.It("it should launch successfully even if it temporarily failed termination due to volume failing to unmount", f.WithNodeConformance(), f.WithSerial(), func(ctx context.Context) {
			node := getNodeName(ctx, f)
			ns = f.Namespace.Name
			c := f.ClientSet
			nfsTestConfig, nfsServerPod, nfsServerHost := e2evolume.NewNFSServerWithNodeName(ctx, c, ns, []string{"-G", "777", "/exports"}, node)
			ginkgo.DeferCleanup(func(ctx context.Context) {
				framework.Logf("Cleaning up NFS server pod")
				e2evolume.TestServerCleanup(ctx, f, nfsTestConfig)
			})

			podPath = kubeletCfg.StaticPodPath
			staticPodName = "static-pod-nfs-test-pod" + string(uuid.NewUUID())
			mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName

			ginkgo.By(fmt.Sprintf("Creating nfs test pod: %s", staticPodName))

			err := createStaticPodUsingNfs(nfsServerHost, node, "sleep 999999", podPath, staticPodName, ns)
			framework.ExpectNoError(err)
			ginkgo.By(fmt.Sprintf("Wating for nfs test pod: %s to start running...", staticPodName))
			gomega.Eventually(func() error {
				return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())

			mirrorPod, err := c.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)

			hash, ok := mirrorPod.Annotations[kubetypes.ConfigHashAnnotationKey]
			if !ok || hash == "" {
				framework.Failf("Failed to get hash for mirrorPod")
			}

			ginkgo.By("Stopping the NFS server")
			e2evolume.StopNFSServer(ctx, f, nfsServerPod)

			ginkgo.By(fmt.Sprintf("Deleting the static nfs test pod: %s", staticPodName))
			err = deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			// Wait 5 mins for syncTerminatedPod to fail. We expect that the pod volume should not be cleaned up because the NFS server is down.
			gomega.Consistently(func() bool {
				return podVolumeDirectoryExists(types.UID(hash))
			}, 5*time.Minute, 10*time.Second).Should(gomega.BeTrueBecause("pod volume should exist while nfs server is stopped"))

			ginkgo.By("Start the NFS server")
			e2evolume.RestartNFSServer(ctx, f, nfsServerPod)

			ginkgo.By("Waiting for the pod volume to deleted after the NFS server is started")
			gomega.Eventually(func() bool {
				return podVolumeDirectoryExists(types.UID(hash))
			}, 5*time.Minute, 10*time.Second).Should(gomega.BeFalseBecause("pod volume should be deleted after nfs server is started"))

			// Create the static pod again with the same config and expect it to start running
			err = createStaticPodUsingNfs(nfsServerHost, node, "sleep 999999", podPath, staticPodName, ns)
			framework.ExpectNoError(err)
			ginkgo.By(fmt.Sprintf("Wating for nfs test pod: %s to start running (after being recreated)", staticPodName))
			gomega.Eventually(func() error {
				return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
			}, 5*time.Minute, 5*time.Second).Should(gomega.BeNil())
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("delete the static pod")
			err := deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to disappear")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.BeNil())

		})

	})
})

var _ = SIGDescribe("MirrorPod (Pod Generation)", func() {
	f := framework.NewDefaultFramework("mirror-pod")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("mirror pod updates", func() {
		var ns, podPath, staticPodName, mirrorPodName string
		ginkgo.BeforeEach(func(ctx context.Context) {
			ns = f.Namespace.Name
			staticPodName = "static-pod-" + string(uuid.NewUUID())
			mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName

			podPath = kubeletCfg.StaticPodPath

			ginkgo.By("create the static pod")
			err := createStaticPod(podPath, staticPodName, ns,
				imageutils.GetE2EImage(imageutils.Nginx), v1.RestartPolicyAlways)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be running")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.Succeed())
		})

		f.It("mirror pod: update activeDeadlineSeconds", f.WithNodeConformance(), func(ctx context.Context) {
			ginkgo.By("get mirror pod uid")
			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			uid := pod.UID

			ginkgo.By("updating ActiveDeadlineSeconds")
			framework.ExpectNoError(createStaticPodWithActiveDeadlineSeconds(podPath, staticPodName, ns, imageutils.GetPauseImageName(), v1.RestartPolicyAlways, 3000))

			ginkgo.By("wait for the mirror pod to be updated")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRecreated(ctx, f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(gomega.Succeed())

			ginkgo.By("check mirror pod generation remains at 1")
			pod, err = f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod.Generation).To(gomega.BeEquivalentTo(int64(1)))

			ginkgo.By("check mirror pod observedGeneration is always empty")
			gomega.Expect(pod.Status.ObservedGeneration).To(gomega.BeEquivalentTo(int64(0)))
		})

		f.It("mirror pod: update container image", f.WithNodeConformance(), func(ctx context.Context) {
			ginkgo.By("get mirror pod uid")
			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			uid := pod.UID

			ginkgo.By("updating container image")
			framework.ExpectNoError(createStaticPod(podPath, staticPodName, ns, imageutils.GetPauseImageName(), v1.RestartPolicyAlways))

			ginkgo.By("wait for the mirror pod to be updated")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRecreated(ctx, f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(gomega.Succeed())

			ginkgo.By("check mirror pod generation remains at 1")
			pod, err = f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod.Generation).To(gomega.BeEquivalentTo(int64(1)))

			ginkgo.By("check mirror pod observedGeneration is always empty")
			gomega.Expect(pod.Status.ObservedGeneration).To(gomega.BeEquivalentTo(int64(0)))
		})

		f.It("mirror pod: update initContainer image", f.WithNodeConformance(), func(ctx context.Context) {
			ginkgo.By("get mirror pod uid")
			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			uid := pod.UID

			ginkgo.By("updating initContainer image")
			framework.ExpectNoError(createStaticPodWithInitContainer(podPath, staticPodName, ns, imageutils.GetPauseImageName(), imageutils.GetPauseImageName(), v1.RestartPolicyAlways))

			ginkgo.By("wait for the mirror pod to be updated")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRecreated(ctx, f.ClientSet, mirrorPodName, ns, uid)
			}, 2*time.Minute, time.Second*4).Should(gomega.Succeed())

			ginkgo.By("check mirror pod generation remains at 1")
			pod, err = f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)
			gomega.Expect(pod.Generation).To(gomega.BeEquivalentTo(int64(1)))

			ginkgo.By("check mirror pod observedGeneration is always empty")
			gomega.Expect(pod.Status.ObservedGeneration).To(gomega.BeEquivalentTo(int64(0)))
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("delete the static pod")
			err := deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to disappear")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.Succeed())
		})
	})
})

var _ = SIGDescribe("MirrorPod with EnvFiles", framework.WithNodeConformance(), framework.WithFeatureGate(features.EnvFiles), func() {
	f := framework.NewDefaultFramework("mirror-pod-envfiles")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var ns, podPath, staticPodName, mirrorPodName string

	ginkgo.Context("when creating a static pod with EnvFiles", func() {
		ginkgo.BeforeEach(func() {
			ns = f.Namespace.Name
			staticPodName = "static-pod-envfiles-" + string(uuid.NewUUID())
			mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName
			podPath = kubeletCfg.StaticPodPath
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("delete the static pod")
			err := deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to disappear")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.Succeed())
		})

		ginkgo.It("should be able to consume variables from a file", func(ctx context.Context) {
			podSpec := v1.PodSpec{
				InitContainers: []v1.Container{
					{
						Name:    "setup-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", `echo CONFIG_1=\'value1\' > /data/config.env && echo CONFIG_2=\'value2\' >> /data/config.env`},
						VolumeMounts: []v1.VolumeMount{
							{
								Name:      "config",
								MountPath: "/data",
							},
						},
					},
				},
				Containers: []v1.Container{
					{
						Name:    "use-envfile",
						Image:   imageutils.GetE2EImage(imageutils.BusyBox),
						Command: []string{"sh", "-c", "env | grep -E '(CONFIG_1|CONFIG_2)' | sort"},
						Env: []v1.EnvVar{
							{
								Name: "CONFIG_1",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "CONFIG_1",
									},
								},
							},
							{
								Name: "CONFIG_2",
								ValueFrom: &v1.EnvVarSource{
									FileKeyRef: &v1.FileKeySelector{
										VolumeName: "config",
										Path:       "config.env",
										Key:        "CONFIG_2",
									},
								},
							},
						},
					},
				},
				RestartPolicy: v1.RestartPolicyNever,
				Volumes: []v1.Volume{
					{
						Name: "config",
						VolumeSource: v1.VolumeSource{
							EmptyDir: &v1.EmptyDirVolumeSource{},
						},
					},
				},
			}

			ginkgo.By("create the static pod with envfiles")
			err := createStaticPodWithSpec(podPath, staticPodName, ns, podSpec)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to succeed")
			err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, mirrorPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By("checking the logs of the mirror pod")
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, ns, mirrorPodName, "use-envfile")
			framework.ExpectNoError(err)

			gomega.Expect(logs).To(gomega.ContainSubstring("CONFIG_1=value1"))
			gomega.Expect(logs).To(gomega.ContainSubstring("CONFIG_2=value2"))
		})
	})
})

func podVolumeDirectoryExists(uid types.UID) bool {
	podVolumePath := fmt.Sprintf("/var/lib/kubelet/pods/%s/volumes/", uid)
	var podVolumeDirectoryExists bool

	if _, err := os.Stat(podVolumePath); !os.IsNotExist(err) {
		podVolumeDirectoryExists = true
	}

	return podVolumeDirectoryExists
}

func createStaticPodUsingNfs(nfsIP string, nodeName string, cmd string, dir string, name string, ns string) error {
	ginkgo.By("create pod using nfs volume")

	isPrivileged := true
	cmdLine := []string{"-c", cmd}
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: v1.PodSpec{
			NodeName: nodeName,
			Containers: []v1.Container{
				{
					Name:    "pod-nfs-vol",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"/bin/sh"},
					Args:    cmdLine,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "nfs-vol",
							MountPath: "/mnt",
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &isPrivileged,
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever, //don't restart pod
			Volumes: []v1.Volume{
				{
					Name: "nfs-vol",
					VolumeSource: v1.VolumeSource{
						NFS: &v1.NFSVolumeSource{
							Server:   nfsIP,
							Path:     "/",
							ReadOnly: false,
						},
					},
				},
			},
		},
	}

	file := staticPodPath(dir, name, ns)
	f, err := os.OpenFile(file, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer f.Close()

	y := printers.YAMLPrinter{}
	y.PrintObj(pod, f)

	return nil
}

func staticPodPath(dir, name, namespace string) string {
	return filepath.Join(dir, namespace+"-"+name+".yaml")
}

func createStaticPod(dir, name, namespace, image string, restart v1.RestartPolicy) error {
	podSpec := v1.PodSpec{
		Containers: []v1.Container{{
			Name:  "test",
			Image: image,
		}},
		RestartPolicy: restart,
	}

	return createStaticPodWithSpec(dir, name, namespace, podSpec)
}

func createStaticPodWithSpec(dir, name, namespace string, podSpec v1.PodSpec) error {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: podSpec,
	}

	podBytes, err := json.Marshal(pod)
	if err != nil {
		return err
	}

	podYaml, err := yaml.JSONToYAML(podBytes)
	if err != nil {
		return err
	}

	file := staticPodPath(dir, name, namespace)

	f, err := os.OpenFile(file, os.O_RDWR|os.O_TRUNC|os.O_CREATE, 0666)
	if err != nil {
		return err
	}
	defer func() {
		// Don't mask other errors.
		_ = f.Close()
	}()

	_, err = f.WriteString(string(podYaml))

	return err
}

func createStaticPodWithActiveDeadlineSeconds(dir, name, namespace, image string, restart v1.RestartPolicy, activeDeadlineSeconds int64) error {
	podSpec := v1.PodSpec{
		Containers: []v1.Container{{
			Name:  "test",
			Image: image,
		}},
		RestartPolicy:         restart,
		ActiveDeadlineSeconds: &activeDeadlineSeconds,
	}

	return createStaticPodWithSpec(dir, name, namespace, podSpec)
}

func createStaticPodWithInitContainer(dir, name, namespace, image, initImage string, restart v1.RestartPolicy) error {
	podSpec := v1.PodSpec{
		InitContainers: []v1.Container{{
			Name:  "init-test",
			Image: initImage,
		}},
		Containers: []v1.Container{{
			Name:  "test",
			Image: image,
		}},
		RestartPolicy: restart,
	}

	return createStaticPodWithSpec(dir, name, namespace, podSpec)
}

func deleteStaticPod(dir, name, namespace string) error {
	file := staticPodPath(dir, name, namespace)
	return os.Remove(file)
}

func checkMirrorPodDisappear(ctx context.Context, cl clientset.Interface, name, namespace string) error {
	_, err := cl.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err == nil {
		return fmt.Errorf("mirror pod %v/%v still exists", namespace, name)
	}
	return fmt.Errorf("expect mirror pod %v/%v to not exist but got error: %w", namespace, name, err)
}

func checkMirrorPodRunning(ctx context.Context, cl clientset.Interface, name, namespace string) error {
	pod, err := cl.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("expected the mirror pod %q to appear: %w", name, err)
	}
	if pod.Status.Phase != v1.PodRunning {
		return fmt.Errorf("expected the mirror pod %q to be running, got %q", name, pod.Status.Phase)
	}
	for i := range pod.Status.ContainerStatuses {
		if pod.Status.ContainerStatuses[i].State.Running == nil {
			return fmt.Errorf("expected the mirror pod %q with container %q to be running (got containers=%v)", name, pod.Status.ContainerStatuses[i].Name, pod.Status.ContainerStatuses[i].State)
		}
	}
	return validateMirrorPod(ctx, cl, pod)
}

func checkMirrorPodRunningWithRestartCount(ctx context.Context, interval time.Duration, timeout time.Duration, cl clientset.Interface, name, namespace string, count int32) error {
	var pod *v1.Pod
	var err error
	err = wait.PollUntilContextTimeout(ctx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		pod, err = cl.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, fmt.Errorf("expected the mirror pod %q to appear: %w", name, err)
		}
		if pod.Status.Phase != v1.PodRunning {
			return false, fmt.Errorf("expected the mirror pod %q to be running, got %q", name, pod.Status.Phase)
		}
		for i := range pod.Status.ContainerStatuses {
			if pod.Status.ContainerStatuses[i].State.Waiting != nil {
				// retry if pod is in waiting state
				return false, nil
			}
			if pod.Status.ContainerStatuses[i].State.Running == nil {
				return false, fmt.Errorf("expected the mirror pod %q with container %q to be running (got containers=%v)", name, pod.Status.ContainerStatuses[i].Name, pod.Status.ContainerStatuses[i].State)
			}
			if pod.Status.ContainerStatuses[i].RestartCount == count {
				// found the restart count
				return true, nil
			}
		}
		return false, nil
	})
	if err != nil {
		return err
	}
	return validateMirrorPod(ctx, cl, pod)
}

func checkMirrorPodRecreatedAndRunning(ctx context.Context, cl clientset.Interface, name, namespace string, oUID types.UID) error {
	pod, err := cl.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("expected the mirror pod %q to appear: %w", name, err)
	}
	if pod.UID == oUID {
		return fmt.Errorf("expected the uid of mirror pod %q to be changed, got %q", name, pod.UID)
	}
	if pod.Status.Phase != v1.PodRunning {
		return fmt.Errorf("expected the mirror pod %q to be running, got %q", name, pod.Status.Phase)
	}
	return validateMirrorPod(ctx, cl, pod)
}

func validateMirrorPod(ctx context.Context, cl clientset.Interface, mirrorPod *v1.Pod) error {
	hash, ok := mirrorPod.Annotations[kubetypes.ConfigHashAnnotationKey]
	if !ok || hash == "" {
		return fmt.Errorf("expected mirror pod %q to have a hash annotation", mirrorPod.Name)
	}
	mirrorHash, ok := mirrorPod.Annotations[kubetypes.ConfigMirrorAnnotationKey]
	if !ok || mirrorHash == "" {
		return fmt.Errorf("expected mirror pod %q to have a mirror pod annotation", mirrorPod.Name)
	}
	if hash != mirrorHash {
		return fmt.Errorf("expected mirror pod %q to have a matching mirror pod hash: got %q; expected %q", mirrorPod.Name, mirrorHash, hash)
	}
	source, ok := mirrorPod.Annotations[kubetypes.ConfigSourceAnnotationKey]
	if !ok {
		return fmt.Errorf("expected mirror pod %q to have a source annotation", mirrorPod.Name)
	}
	if source == kubetypes.ApiserverSource {
		return fmt.Errorf("expected mirror pod %q source to not be 'api'; got: %q", mirrorPod.Name, source)
	}

	if len(mirrorPod.OwnerReferences) != 1 {
		return fmt.Errorf("expected mirror pod %q to have a single owner reference: got %d", mirrorPod.Name, len(mirrorPod.OwnerReferences))
	}
	node, err := cl.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to fetch test node: %w", err)
	}

	controller := true
	expectedOwnerRef := metav1.OwnerReference{
		APIVersion: "v1",
		Kind:       "Node",
		Name:       framework.TestContext.NodeName,
		UID:        node.UID,
		Controller: &controller,
	}
	ref := mirrorPod.OwnerReferences[0]
	if !apiequality.Semantic.DeepEqual(ref, expectedOwnerRef) {
		return fmt.Errorf("unexpected mirror pod %q owner ref: %v", mirrorPod.Name, cmp.Diff(expectedOwnerRef, ref))
	}

	return nil
}

func checkMirrorPodRecreated(ctx context.Context, cl clientset.Interface, name, namespace string, oUID types.UID) error {
	pod, err := cl.CoreV1().Pods(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("expected the mirror pod %q to appear: %w", name, err)
	}
	if pod.UID == oUID {
		return fmt.Errorf("expected the uid of mirror pod %q to be changed, got %q", name, pod.UID)
	}
	return nil
}

var _ = SIGDescribe("MirrorPod", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("mirror-pod-serial")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Context("when kubelet restarts", func() {
		var ns, podPath, staticPodName, mirrorPodName string
		ginkgo.BeforeEach(func(ctx context.Context) {
			ns = f.Namespace.Name
			staticPodName = "static-pod-" + string(uuid.NewUUID())
			mirrorPodName = staticPodName + "-" + framework.TestContext.NodeName
			podPath = kubeletCfg.StaticPodPath

			ginkgo.By("create the static pod")
			podSpec := v1.PodSpec{
				Containers: []v1.Container{
					{
						Name:    "container",
						Image:   defaultImage,
						Command: []string{"sleep", "3600"},
						StartupProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{"/bin/true"},
								},
							},
							InitialDelaySeconds: 1,
							PeriodSeconds:       1,
						},
						ReadinessProbe: &v1.Probe{
							ProbeHandler: v1.ProbeHandler{
								Exec: &v1.ExecAction{
									Command: []string{"/bin/true"},
								},
							},
							InitialDelaySeconds: 1,
							PeriodSeconds:       1,
						},
					},
				},
			}

			err := createStaticPodWithSpec(podPath, staticPodName, ns, podSpec)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to be running")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodRunning(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.Succeed())
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			ginkgo.By("delete the static pod")
			err := deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			ginkgo.By("wait for the mirror pod to disappear")
			gomega.Eventually(ctx, func(ctx context.Context) error {
				return checkMirrorPodDisappear(ctx, f.ClientSet, mirrorPodName, ns)
			}, 2*time.Minute, time.Second*4).Should(gomega.Succeed())
		})

		f.It("should not change container status", f.WithNodeConformance(), func(ctx context.Context) {
			ginkgo.By("Waiting for the pod to be running and ready")
			err := e2epod.WaitForPodCondition(ctx, f.ClientSet, ns, mirrorPodName, "PodReady", f.Timeouts.PodStart,
				func(p *v1.Pod) (bool, error) {
					if p.Status.Phase != v1.PodRunning {
						return false, nil
					}
					for _, cond := range p.Status.Conditions {
						if cond.Type == v1.PodReady && cond.Status == v1.ConditionTrue {
							return true, nil
						}
					}
					return false, nil
				})
			framework.ExpectNoError(err)

			pod, err := f.ClientSet.CoreV1().Pods(ns).Get(ctx, mirrorPodName, metav1.GetOptions{})
			framework.ExpectNoError(err)

			ginkgo.By("Double check the initial state before starting the concurrent check")
			gomega.Expect(pod.Status.ContainerStatuses).ToNot(gomega.BeEmpty())
			for _, status := range pod.Status.ContainerStatuses {
				gomega.Expect(status.RestartCount).To(gomega.BeZero())
				gomega.Expect(status.Started).ToNot(gomega.BeNil())
				gomega.Expect(*status.Started).To(gomega.BeTrueBecause("The Started field should be set to true when a pod enters the Ready condition."))
				gomega.Expect(status.Ready).To(gomega.BeTrueBecause("The Ready field should be set to true when a pod enters the Ready condition."))
			}

			// The grace period for kubelet startup is 10 seconds, so we wait here for 11 seconds.
			time.Sleep(time.Second * 11)

			stopCh := make(chan struct{})
			errCh := make(chan error, 1)
			go func() {
				defer ginkgo.GinkgoRecover()
				watcher, err := f.ClientSet.CoreV1().Pods(ns).Watch(ctx, metav1.ListOptions{
					FieldSelector: "metadata.name=" + mirrorPodName,
				})
				if err != nil {
					errCh <- fmt.Errorf("failed to watch pod: %w", err)
					return
				}
				defer watcher.Stop()

				for {
					select {
					case event, ok := <-watcher.ResultChan():
						if !ok {
							return
						}
						if event.Type != watch.Modified {
							continue
						}
						p, ok := event.Object.(*v1.Pod)
						if !ok {
							continue
						}

						if p.Status.Phase != v1.PodRunning {
							errCh <- fmt.Errorf("pod phase is %v, expected %v", p.Status.Phase, v1.PodRunning)
							return
						}
						if len(p.Status.ContainerStatuses) < len(pod.Spec.Containers) {
							continue
						}
						for _, containerStatus := range p.Status.ContainerStatuses {
							if containerStatus.RestartCount > 0 {
								errCh <- fmt.Errorf("container %q restarted %d times", containerStatus.Name, containerStatus.RestartCount)
								return
							}
							if containerStatus.Started == nil || !*containerStatus.Started {
								errCh <- fmt.Errorf("container %q started status is not true", containerStatus.Name)
								return
							}
							if !containerStatus.Ready {
								errCh <- fmt.Errorf("container %q ready status is not true", containerStatus.Name)
								return
							}
						}
					case <-stopCh:
						close(errCh)
						return
					}
				}
			}()

			ginkgo.By("restarting the kubelet")
			restartKubelet := mustStopKubelet(ctx, f)
			restartKubelet(ctx)

			ginkgo.By("ensuring kubelet is healthy")
			gomega.Eventually(ctx, func() bool {
				return kubeletHealthCheck(kubeletHealthCheckURL)
			}, f.Timeouts.PodStart, f.Timeouts.Poll).Should(gomega.BeTrueBecause("kubelet should be started"))

			// Let the goroutine run for a few more seconds to catch any delayed changes
			time.Sleep(5 * time.Second)
			close(stopCh)

			for err := range errCh {
				framework.ExpectNoError(err, "pod status check failed during kubelet restart")
			}
		})
	})
})
