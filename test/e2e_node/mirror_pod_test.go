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
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"k8s.io/cli-runtime/pkg/printers"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
)

var _ = SIGDescribe("MirrorPod", func() {
	f := framework.NewDefaultFramework("mirror-pod")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.Context("when create a mirror pod ", func() {
		var ns, podPath, staticPodName, mirrorPodName string
		ginkgo.BeforeEach(func(ctx context.Context) {
			framework.Fail("DNM NEED A FAILURE")

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
	ginkgo.Context("when create a mirror pod without changes ", func() {
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
			stopNfsServer(f, nfsServerPod)

			ginkgo.By("Waiting for NFS server to stop...")
			time.Sleep(30 * time.Second)

			ginkgo.By(fmt.Sprintf("Deleting the static nfs test pod: %s", staticPodName))
			err = deleteStaticPod(podPath, staticPodName, ns)
			framework.ExpectNoError(err)

			// Wait 5 mins for syncTerminatedPod to fail. We expect that the pod volume should not be cleaned up because the NFS server is down.
			gomega.Consistently(func() bool {
				return podVolumeDirectoryExists(types.UID(hash))
			}, 5*time.Minute, 10*time.Second).Should(gomega.BeTrueBecause("pod volume should exist while nfs server is stopped"))

			ginkgo.By("Start the NFS server")
			restartNfsServer(f, nfsServerPod)

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

func podVolumeDirectoryExists(uid types.UID) bool {
	podVolumePath := fmt.Sprintf("/var/lib/kubelet/pods/%s/volumes/", uid)
	var podVolumeDirectoryExists bool

	if _, err := os.Stat(podVolumePath); !os.IsNotExist(err) {
		podVolumeDirectoryExists = true
	}

	return podVolumeDirectoryExists
}

// Restart the passed-in nfs-server by issuing a `/usr/sbin/rpc.nfsd 1` command in the
// pod's (only) container. This command changes the number of nfs server threads from
// (presumably) zero back to 1, and therefore allows nfs to open connections again.
func restartNfsServer(f *framework.Framework, serverPod *v1.Pod) {
	const startcmd = "/usr/sbin/rpc.nfsd 1"
	_, _, err := e2evolume.PodExec(f, serverPod, startcmd)
	framework.ExpectNoError(err)

}

// Stop the passed-in nfs-server by issuing a `/usr/sbin/rpc.nfsd 0` command in the
// pod's (only) container. This command changes the number of nfs server threads to 0,
// thus closing all open nfs connections.
func stopNfsServer(f *framework.Framework, serverPod *v1.Pod) {
	const stopcmd = "/usr/sbin/rpc.nfsd 0"
	_, _, err := e2evolume.PodExec(f, serverPod, stopcmd)
	framework.ExpectNoError(err)
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
