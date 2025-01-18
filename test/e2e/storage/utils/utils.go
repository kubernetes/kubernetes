/*
Copyright 2017 The Kubernetes Authors.

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

package utils

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"math"
	"math/rand"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// KubeletOpt type definition
type KubeletOpt string

const (
	// NodeStateTimeout defines Timeout
	NodeStateTimeout = 1 * time.Minute
	// KStart defines start value
	KStart KubeletOpt = "start"
	// KStop defines stop value
	KStop KubeletOpt = "stop"
	// KRestart defines restart value
	KRestart     KubeletOpt = "restart"
	minValidSize string     = "1Ki"
	maxValidSize string     = "10Ei"
)

// VerifyFSGroupInPod verifies that the passed in filePath contains the expectedFSGroup
func VerifyFSGroupInPod(ctx context.Context, f *framework.Framework, filePath, expectedFSGroup string, pod *v1.Pod) {
	cmd := fmt.Sprintf("ls -l %s", filePath)
	stdout, stderr, err := e2epod.ExecShellInPodWithFullOutput(ctx, f, pod.Name, cmd)
	framework.ExpectNoError(err)
	framework.Logf("pod %s/%s exec for cmd %s, stdout: %s, stderr: %s", pod.Namespace, pod.Name, cmd, stdout, stderr)
	fsGroupResult := strings.Fields(stdout)[3]
	gomega.Expect(expectedFSGroup).To(gomega.Equal(fsGroupResult), "Expected fsGroup of %s, got %s", expectedFSGroup, fsGroupResult)
}

// getKubeletRunning return if the kubelet is running or not
func getKubeletRunning(ctx context.Context, nodeIP string) bool {
	command := "systemctl show kubelet --property ActiveState --value"
	framework.Logf("Attempting `%s`", command)
	sshResult, err := e2essh.SSH(ctx, command, nodeIP, framework.TestContext.Provider)
	framework.ExpectNoError(err, fmt.Sprintf("SSH to Node %q errored.", nodeIP))
	e2essh.LogResult(sshResult)
	gomega.Expect(sshResult.Code).To(gomega.BeZero(), "Failed to get kubelet status")
	gomega.Expect(sshResult.Stdout).NotTo(gomega.BeEmpty(), "Kubelet status should not be Empty")
	return strings.TrimSpace(sshResult.Stdout) == "active"
}

// TestKubeletRestartsAndRestoresMount tests that a volume mounted to a pod remains mounted after a kubelet restarts
func TestKubeletRestartsAndRestoresMount(ctx context.Context, c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, volumePath string) {
	byteLen := 64
	seed := time.Now().UTC().UnixNano()

	ginkgo.By("Writing to the volume.")
	CheckWriteToPath(ctx, f, clientPod, v1.PersistentVolumeFilesystem, false, volumePath, byteLen, seed)

	ginkgo.By("Restarting kubelet")
	KubeletCommand(ctx, KRestart, c, clientPod)

	ginkgo.By("Wait 20s for the volume to become stable")
	time.Sleep(20 * time.Second)

	ginkgo.By("Testing that written file is accessible.")
	CheckReadFromPath(ctx, f, clientPod, v1.PersistentVolumeFilesystem, false, volumePath, byteLen, seed)

	framework.Logf("Volume mount detected on pod %s and written file %s is readable post-restart.", clientPod.Name, volumePath)
}

// TestKubeletRestartsAndRestoresMap tests that a volume mapped to a pod remains mapped after a kubelet restarts
func TestKubeletRestartsAndRestoresMap(ctx context.Context, c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, volumePath string) {
	byteLen := 64
	seed := time.Now().UTC().UnixNano()

	ginkgo.By("Writing to the volume.")
	CheckWriteToPath(ctx, f, clientPod, v1.PersistentVolumeBlock, false, volumePath, byteLen, seed)

	ginkgo.By("Restarting kubelet")
	KubeletCommand(ctx, KRestart, c, clientPod)

	ginkgo.By("Wait 20s for the volume to become stable")
	time.Sleep(20 * time.Second)

	ginkgo.By("Testing that written pv is accessible.")
	CheckReadFromPath(ctx, f, clientPod, v1.PersistentVolumeBlock, false, volumePath, byteLen, seed)

	framework.Logf("Volume map detected on pod %s and written data %s is readable post-restart.", clientPod.Name, volumePath)
}

// TestVolumeUnmountsFromDeletedPodWithForceOption tests that a volume unmounts if the client pod was deleted while the kubelet was down.
// forceDelete is true indicating whether the pod is forcefully deleted.
// checkSubpath is true indicating whether the subpath should be checked.
// If secondPod is set, it is started when kubelet is down to check that the volume is usable while the old pod is being deleted and the new pod is starting.
func TestVolumeUnmountsFromDeletedPodWithForceOption(ctx context.Context, c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, forceDelete bool, checkSubpath bool, secondPod *v1.Pod, volumePath string) {
	nodeIP, err := getHostAddress(ctx, c, clientPod)
	framework.ExpectNoError(err)
	nodeIP = nodeIP + ":22"

	ginkgo.By("Expecting the volume mount to be found.")
	result, err := e2essh.SSH(ctx, fmt.Sprintf("mount | grep %s | grep -v volume-subpaths", clientPod.UID), nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	gomega.Expect(result.Code).To(gomega.Equal(0), fmt.Sprintf("Expected grep exit code of 0, got %d", result.Code))

	if checkSubpath {
		ginkgo.By("Expecting the volume subpath mount to be found.")
		result, err := e2essh.SSH(ctx, fmt.Sprintf("cat /proc/self/mountinfo | grep %s | grep volume-subpaths", clientPod.UID), nodeIP, framework.TestContext.Provider)
		e2essh.LogResult(result)
		framework.ExpectNoError(err, "Encountered SSH error.")
		gomega.Expect(result.Code).To(gomega.Equal(0), fmt.Sprintf("Expected grep exit code of 0, got %d", result.Code))
	}

	ginkgo.By("Writing to the volume.")
	byteLen := 64
	seed := time.Now().UTC().UnixNano()
	CheckWriteToPath(ctx, f, clientPod, v1.PersistentVolumeFilesystem, false, volumePath, byteLen, seed)

	// This command is to make sure kubelet is started after test finishes no matter it fails or not.
	ginkgo.DeferCleanup(KubeletCommand, KStart, c, clientPod)
	ginkgo.By("Stopping the kubelet.")
	KubeletCommand(ctx, KStop, c, clientPod)

	if secondPod != nil {
		ginkgo.By("Starting the second pod")
		_, err = c.CoreV1().Pods(clientPod.Namespace).Create(context.TODO(), secondPod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "when starting the second pod")
	}

	ginkgo.By(fmt.Sprintf("Deleting Pod %q", clientPod.Name))
	if forceDelete {
		err = c.CoreV1().Pods(clientPod.Namespace).Delete(ctx, clientPod.Name, *metav1.NewDeleteOptions(0))
	} else {
		err = c.CoreV1().Pods(clientPod.Namespace).Delete(ctx, clientPod.Name, metav1.DeleteOptions{})
	}
	framework.ExpectNoError(err)

	ginkgo.By("Starting the kubelet and waiting for pod to delete.")
	KubeletCommand(ctx, KStart, c, clientPod)
	err = e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, clientPod.Name, f.Namespace.Name, f.Timeouts.PodDelete)
	if err != nil {
		framework.ExpectNoError(err, "Expected pod to be not found.")
	}

	if forceDelete {
		// With forceDelete, since pods are immediately deleted from API server, there is no way to be sure when volumes are torn down
		// so wait some time to finish
		time.Sleep(30 * time.Second)
	}

	if secondPod != nil {
		ginkgo.By("Waiting for the second pod.")
		err = e2epod.WaitForPodRunningInNamespace(ctx, c, secondPod)
		framework.ExpectNoError(err, "while waiting for the second pod Running")

		ginkgo.By("Getting the second pod uuid.")
		secondPod, err := c.CoreV1().Pods(secondPod.Namespace).Get(context.TODO(), secondPod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "getting the second UID")

		ginkgo.By("Expecting the volume mount to be found in the second pod.")
		result, err := e2essh.SSH(ctx, fmt.Sprintf("mount | grep %s | grep -v volume-subpaths", secondPod.UID), nodeIP, framework.TestContext.Provider)
		e2essh.LogResult(result)
		framework.ExpectNoError(err, "Encountered SSH error when checking the second pod.")
		gomega.Expect(result.Code).To(gomega.Equal(0), fmt.Sprintf("Expected grep exit code of 0, got %d", result.Code))

		ginkgo.By("Testing that written file is accessible in the second pod.")
		CheckReadFromPath(ctx, f, secondPod, v1.PersistentVolumeFilesystem, false, volumePath, byteLen, seed)
		err = c.CoreV1().Pods(secondPod.Namespace).Delete(context.TODO(), secondPod.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err, "when deleting the second pod")
		err = e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, secondPod.Name, f.Namespace.Name, f.Timeouts.PodDelete)
		framework.ExpectNoError(err, "when waiting for the second pod to disappear")
	}

	ginkgo.By("Expecting the volume mount not to be found.")
	result, err = e2essh.SSH(ctx, fmt.Sprintf("mount | grep %s | grep -v volume-subpaths", clientPod.UID), nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	gomega.Expect(result.Stdout).To(gomega.BeEmpty(), "Expected grep stdout to be empty (i.e. no mount found).")
	framework.Logf("Volume unmounted on node %s", clientPod.Spec.NodeName)

	if checkSubpath {
		ginkgo.By("Expecting the volume subpath mount not to be found.")
		result, err = e2essh.SSH(ctx, fmt.Sprintf("cat /proc/self/mountinfo | grep %s | grep volume-subpaths", clientPod.UID), nodeIP, framework.TestContext.Provider)
		e2essh.LogResult(result)
		framework.ExpectNoError(err, "Encountered SSH error.")
		gomega.Expect(result.Stdout).To(gomega.BeEmpty(), "Expected grep stdout to be empty (i.e. no subpath mount found).")
		framework.Logf("Subpath volume unmounted on node %s", clientPod.Spec.NodeName)
	}

}

// TestVolumeUnmountsFromDeletedPod tests that a volume unmounts if the client pod was deleted while the kubelet was down.
func TestVolumeUnmountsFromDeletedPod(ctx context.Context, c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, volumePath string) {
	TestVolumeUnmountsFromDeletedPodWithForceOption(ctx, c, f, clientPod, false, false, nil, volumePath)
}

// TestVolumeUnmountsFromForceDeletedPod tests that a volume unmounts if the client pod was forcefully deleted while the kubelet was down.
func TestVolumeUnmountsFromForceDeletedPod(ctx context.Context, c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, volumePath string) {
	TestVolumeUnmountsFromDeletedPodWithForceOption(ctx, c, f, clientPod, true, false, nil, volumePath)
}

// TestVolumeUnmapsFromDeletedPodWithForceOption tests that a volume unmaps if the client pod was deleted while the kubelet was down.
// forceDelete is true indicating whether the pod is forcefully deleted.
func TestVolumeUnmapsFromDeletedPodWithForceOption(ctx context.Context, c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, forceDelete bool, devicePath string) {
	nodeIP, err := getHostAddress(ctx, c, clientPod)
	framework.ExpectNoError(err, "Failed to get nodeIP.")
	nodeIP = nodeIP + ":22"

	// Creating command to check whether path exists
	podDirectoryCmd := fmt.Sprintf("ls /var/lib/kubelet/pods/%s/volumeDevices/*/ | grep '.'", clientPod.UID)
	if isSudoPresent(ctx, nodeIP, framework.TestContext.Provider) {
		podDirectoryCmd = fmt.Sprintf("sudo sh -c \"%s\"", podDirectoryCmd)
	}
	// Directories in the global directory have unpredictable names, however, device symlinks
	// have the same name as pod.UID. So just find anything with pod.UID name.
	globalBlockDirectoryCmd := fmt.Sprintf("find /var/lib/kubelet/plugins -name %s", clientPod.UID)
	if isSudoPresent(ctx, nodeIP, framework.TestContext.Provider) {
		globalBlockDirectoryCmd = fmt.Sprintf("sudo sh -c \"%s\"", globalBlockDirectoryCmd)
	}

	ginkgo.By("Expecting the symlinks from PodDeviceMapPath to be found.")
	result, err := e2essh.SSH(ctx, podDirectoryCmd, nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	gomega.Expect(result.Code).To(gomega.Equal(0), fmt.Sprintf("Expected grep exit code of 0, got %d", result.Code))

	ginkgo.By("Expecting the symlinks from global map path to be found.")
	result, err = e2essh.SSH(ctx, globalBlockDirectoryCmd, nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	gomega.Expect(result.Code).To(gomega.Equal(0), fmt.Sprintf("Expected find exit code of 0, got %d", result.Code))

	// This command is to make sure kubelet is started after test finishes no matter it fails or not.
	ginkgo.DeferCleanup(KubeletCommand, KStart, c, clientPod)
	ginkgo.By("Stopping the kubelet.")
	KubeletCommand(ctx, KStop, c, clientPod)

	ginkgo.By(fmt.Sprintf("Deleting Pod %q", clientPod.Name))
	if forceDelete {
		err = c.CoreV1().Pods(clientPod.Namespace).Delete(ctx, clientPod.Name, *metav1.NewDeleteOptions(0))
	} else {
		err = c.CoreV1().Pods(clientPod.Namespace).Delete(ctx, clientPod.Name, metav1.DeleteOptions{})
	}
	framework.ExpectNoError(err, "Failed to delete pod.")

	ginkgo.By("Starting the kubelet and waiting for pod to delete.")
	KubeletCommand(ctx, KStart, c, clientPod)
	err = e2epod.WaitForPodNotFoundInNamespace(ctx, f.ClientSet, clientPod.Name, f.Namespace.Name, f.Timeouts.PodDelete)
	framework.ExpectNoError(err, "Expected pod to be not found.")

	if forceDelete {
		// With forceDelete, since pods are immediately deleted from API server, there is no way to be sure when volumes are torn down
		// so wait some time to finish
		time.Sleep(30 * time.Second)
	}

	ginkgo.By("Expecting the symlink from PodDeviceMapPath not to be found.")
	result, err = e2essh.SSH(ctx, podDirectoryCmd, nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	gomega.Expect(result.Stdout).To(gomega.BeEmpty(), "Expected grep stdout to be empty.")

	ginkgo.By("Expecting the symlinks from global map path not to be found.")
	result, err = e2essh.SSH(ctx, globalBlockDirectoryCmd, nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	gomega.Expect(result.Stdout).To(gomega.BeEmpty(), "Expected find stdout to be empty.")

	framework.Logf("Volume unmaped on node %s", clientPod.Spec.NodeName)
}

// TestVolumeUnmapsFromDeletedPod tests that a volume unmaps if the client pod was deleted while the kubelet was down.
func TestVolumeUnmapsFromDeletedPod(ctx context.Context, c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, devicePath string) {
	TestVolumeUnmapsFromDeletedPodWithForceOption(ctx, c, f, clientPod, false, devicePath)
}

// TestVolumeUnmapsFromForceDeletedPod tests that a volume unmaps if the client pod was forcefully deleted while the kubelet was down.
func TestVolumeUnmapsFromForceDeletedPod(ctx context.Context, c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, devicePath string) {
	TestVolumeUnmapsFromDeletedPodWithForceOption(ctx, c, f, clientPod, true, devicePath)
}

// RunInPodWithVolume runs a command in a pod with given claim mounted to /mnt directory.
func RunInPodWithVolume(ctx context.Context, c clientset.Interface, t *framework.TimeoutContext, ns, claimName, command string) {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-volume-tester-",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "volume-tester",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", command},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "my-volume",
							MountPath: "/mnt/test",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: "my-volume",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: claimName,
							ReadOnly:  false,
						},
					},
				},
			},
		},
	}
	pod, err := c.CoreV1().Pods(ns).Create(ctx, pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	ginkgo.DeferCleanup(e2epod.DeletePodOrFail, c, ns, pod.Name)
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceTimeout(ctx, c, pod.Name, pod.Namespace, t.PodStartSlow))
}

// StartExternalProvisioner create external provisioner pod
func StartExternalProvisioner(ctx context.Context, c clientset.Interface, ns string, externalPluginName string) *v1.Pod {
	podClient := c.CoreV1().Pods(ns)

	provisionerPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "external-provisioner-",
		},

		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "nfs-provisioner",
					Image: imageutils.GetE2EImage(imageutils.NFSProvisioner),
					SecurityContext: &v1.SecurityContext{
						Capabilities: &v1.Capabilities{
							Add: []v1.Capability{"DAC_READ_SEARCH"},
						},
					},
					Args: []string{
						"-provisioner=" + externalPluginName,
						"-grace-period=0",
					},
					Ports: []v1.ContainerPort{
						{Name: "nfs", ContainerPort: 2049},
						{Name: "mountd", ContainerPort: 20048},
						{Name: "rpcbind", ContainerPort: 111},
						{Name: "rpcbind-udp", ContainerPort: 111, Protocol: v1.ProtocolUDP},
					},
					Env: []v1.EnvVar{
						{
							Name: "POD_IP",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									FieldPath: "status.podIP",
								},
							},
						},
					},
					ImagePullPolicy: v1.PullIfNotPresent,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "export-volume",
							MountPath: "/export",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "export-volume",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
		},
	}
	provisionerPod, err := podClient.Create(ctx, provisionerPod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Failed to create %s pod: %v", provisionerPod.Name, err)

	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(ctx, c, provisionerPod))

	ginkgo.By("locating the provisioner pod")
	pod, err := podClient.Get(ctx, provisionerPod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Cannot locate the provisioner pod %v: %v", provisionerPod.Name, err)

	return pod
}

func isSudoPresent(ctx context.Context, nodeIP string, provider string) bool {
	framework.Logf("Checking if sudo command is present")
	sshResult, err := e2essh.SSH(ctx, "sudo --version", nodeIP, provider)
	framework.ExpectNoError(err, "SSH to %q errored.", nodeIP)
	if !strings.Contains(sshResult.Stderr, "command not found") {
		return true
	}
	return false
}

// CheckReadWriteToPath check that path can b e read and written
func CheckReadWriteToPath(ctx context.Context, f *framework.Framework, pod *v1.Pod, volMode v1.PersistentVolumeMode, path string) {
	if volMode == v1.PersistentVolumeBlock {
		// random -> file1
		e2epod.VerifyExecInPodSucceed(ctx, f, pod, "dd if=/dev/urandom of=/tmp/file1 bs=64 count=1")
		// file1 -> dev (write to dev)
		e2epod.VerifyExecInPodSucceed(ctx, f, pod, fmt.Sprintf("dd if=/tmp/file1 of=%s bs=64 count=1", path))
		// dev -> file2 (read from dev)
		e2epod.VerifyExecInPodSucceed(ctx, f, pod, fmt.Sprintf("dd if=%s of=/tmp/file2 bs=64 count=1", path))
		// file1 == file2 (check contents)
		e2epod.VerifyExecInPodSucceed(ctx, f, pod, "diff /tmp/file1 /tmp/file2")
		// Clean up temp files
		e2epod.VerifyExecInPodSucceed(ctx, f, pod, "rm -f /tmp/file1 /tmp/file2")

		// Check that writing file to block volume fails
		e2epod.VerifyExecInPodFail(ctx, f, pod, fmt.Sprintf("echo 'Hello world.' > %s/file1.txt", path), 1)
	} else {
		// text -> file1 (write to file)
		e2epod.VerifyExecInPodSucceed(ctx, f, pod, fmt.Sprintf("echo 'Hello world.' > %s/file1.txt", path))
		// grep file1 (read from file and check contents)
		e2epod.VerifyExecInPodSucceed(ctx, f, pod, readFile("Hello word.", path))
		// Check that writing to directory as block volume fails
		e2epod.VerifyExecInPodFail(ctx, f, pod, fmt.Sprintf("dd if=/dev/urandom of=%s bs=64 count=1", path), 1)
	}
}

func readFile(content, path string) string {
	if framework.NodeOSDistroIs("windows") {
		return fmt.Sprintf("Select-String '%s' %s/file1.txt", content, path)
	}
	return fmt.Sprintf("grep 'Hello world.' %s/file1.txt", path)
}

// genBinDataFromSeed generate binData with random seed
func genBinDataFromSeed(len int, seed int64) []byte {
	binData := make([]byte, len)
	rand.Seed(seed)

	_, err := rand.Read(binData)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	}

	return binData
}

// CheckReadFromPath validate that file can be properly read.
//
// Note: directIO does not work with (default) BusyBox Pods. A requirement for
// directIO to function correctly, is to read whole sector(s) for Block-mode
// PVCs (normally a sector is 512 bytes), or memory pages for files (commonly
// 4096 bytes).
func CheckReadFromPath(ctx context.Context, f *framework.Framework, pod *v1.Pod, volMode v1.PersistentVolumeMode, directIO bool, path string, len int, seed int64) {
	var pathForVolMode string
	var iflag string

	if volMode == v1.PersistentVolumeBlock {
		pathForVolMode = path
	} else {
		pathForVolMode = filepath.Join(path, "file1.txt")
	}

	if directIO {
		iflag = "iflag=direct"
	}

	sum := sha256.Sum256(genBinDataFromSeed(len, seed))

	e2epod.VerifyExecInPodSucceed(ctx, f, pod, fmt.Sprintf("dd if=%s %s bs=%d count=1 | sha256sum", pathForVolMode, iflag, len))
	e2epod.VerifyExecInPodSucceed(ctx, f, pod, fmt.Sprintf("dd if=%s %s bs=%d count=1 | sha256sum | grep -Fq %x", pathForVolMode, iflag, len, sum))
}

// CheckWriteToPath that file can be properly written.
//
// Note: nocache does not work with (default) BusyBox Pods. To read without
// caching, enable directIO with CheckReadFromPath and check the hints about
// the len requirements.
func CheckWriteToPath(ctx context.Context, f *framework.Framework, pod *v1.Pod, volMode v1.PersistentVolumeMode, nocache bool, path string, len int, seed int64) {
	var pathForVolMode string
	var oflag string

	if volMode == v1.PersistentVolumeBlock {
		pathForVolMode = path
	} else {
		pathForVolMode = filepath.Join(path, "file1.txt")
	}

	if nocache {
		oflag = "oflag=nocache"
	}

	encoded := base64.StdEncoding.EncodeToString(genBinDataFromSeed(len, seed))

	e2epod.VerifyExecInPodSucceed(ctx, f, pod, fmt.Sprintf("echo %s | base64 -d | sha256sum", encoded))
	e2epod.VerifyExecInPodSucceed(ctx, f, pod, fmt.Sprintf("echo %s | base64 -d | dd of=%s %s bs=%d count=1", encoded, pathForVolMode, oflag, len))
}

// GetSectorSize returns the sector size of the device.
func GetSectorSize(ctx context.Context, f *framework.Framework, pod *v1.Pod, device string) int {
	stdout, _, err := e2epod.ExecShellInPodWithFullOutput(ctx, f, pod.Name, fmt.Sprintf("blockdev --getss %s", device))
	framework.ExpectNoError(err, "Failed to get sector size of %s", device)
	ss, err := strconv.Atoi(stdout)
	framework.ExpectNoError(err, "Sector size returned by blockdev command isn't integer value.")

	return ss
}

// findMountPoints returns all mount points on given node under specified directory.
func findMountPoints(ctx context.Context, hostExec HostExec, node *v1.Node, dir string) []string {
	result, err := hostExec.IssueCommandWithResult(ctx, fmt.Sprintf(`find %s -type d -exec mountpoint {} \; | grep 'is a mountpoint$' || true`, dir), node)
	framework.ExpectNoError(err, "Encountered HostExec error.")
	var mountPoints []string
	if err != nil {
		for _, line := range strings.Split(result, "\n") {
			if line == "" {
				continue
			}
			mountPoints = append(mountPoints, strings.TrimSuffix(line, " is a mountpoint"))
		}
	}
	return mountPoints
}

// FindVolumeGlobalMountPoints returns all volume global mount points on the node of given pod.
func FindVolumeGlobalMountPoints(ctx context.Context, hostExec HostExec, node *v1.Node) sets.String {
	return sets.NewString(findMountPoints(ctx, hostExec, node, "/var/lib/kubelet/plugins")...)
}

// CreateDriverNamespace creates a namespace for CSI driver installation.
// The namespace is still tracked and ensured that gets deleted when test terminates.
func CreateDriverNamespace(ctx context.Context, f *framework.Framework) *v1.Namespace {
	ginkgo.By(fmt.Sprintf("Building a driver namespace object, basename %s", f.Namespace.Name))
	// The driver namespace will be bound to the test namespace in the prefix
	namespace, err := f.CreateNamespace(ctx, f.Namespace.Name, map[string]string{
		"e2e-framework":      f.BaseName,
		"e2e-test-namespace": f.Namespace.Name,
	})
	framework.ExpectNoError(err)

	if framework.TestContext.VerifyServiceAccount {
		ginkgo.By("Waiting for a default service account to be provisioned in namespace")
		err = framework.WaitForDefaultServiceAccountInNamespace(ctx, f.ClientSet, namespace.Name)
		framework.ExpectNoError(err)
	} else {
		framework.Logf("Skipping waiting for service account")
	}
	return namespace
}

// WaitForGVRDeletion waits until a non-namespaced object has been deleted
func WaitForGVRDeletion(ctx context.Context, c dynamic.Interface, gvr schema.GroupVersionResource, objectName string, poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for %s %s to be deleted", timeout, gvr.Resource, objectName)

	if successful := WaitUntil(poll, timeout, func() bool {
		_, err := c.Resource(gvr).Get(ctx, objectName, metav1.GetOptions{})
		if err != nil && apierrors.IsNotFound(err) {
			framework.Logf("%s %v is not found and has been deleted", gvr.Resource, objectName)
			return true
		} else if err != nil {
			framework.Logf("Get %s returned an error: %v", objectName, err.Error())
		} else {
			framework.Logf("%s %v has been found and is not deleted", gvr.Resource, objectName)
		}

		return false
	}); successful {
		return nil
	}

	return fmt.Errorf("%s %s is not deleted within %v", gvr.Resource, objectName, timeout)
}

// EnsureGVRDeletion checks that no object as defined by the group/version/kind and name is ever found during the given time period
func EnsureGVRDeletion(ctx context.Context, c dynamic.Interface, gvr schema.GroupVersionResource, objectName string, poll, timeout time.Duration, namespace string) error {
	var resourceClient dynamic.ResourceInterface
	if namespace != "" {
		resourceClient = c.Resource(gvr).Namespace(namespace)
	} else {
		resourceClient = c.Resource(gvr)
	}

	err := framework.Gomega().Eventually(ctx, func(ctx context.Context) error {
		_, err := resourceClient.Get(ctx, objectName, metav1.GetOptions{})
		return err
	}).WithTimeout(timeout).WithPolling(poll).Should(gomega.MatchError(apierrors.IsNotFound, fmt.Sprintf("failed to delete %s %s", gvr, objectName)))
	return err
}

// EnsureNoGVRDeletion checks that an object as defined by the group/version/kind and name has not been deleted during the given time period
func EnsureNoGVRDeletion(ctx context.Context, c dynamic.Interface, gvr schema.GroupVersionResource, objectName string, poll, timeout time.Duration, namespace string) error {
	var resourceClient dynamic.ResourceInterface
	if namespace != "" {
		resourceClient = c.Resource(gvr).Namespace(namespace)
	} else {
		resourceClient = c.Resource(gvr)
	}
	err := framework.Gomega().Consistently(ctx, func(ctx context.Context) error {
		_, err := resourceClient.Get(ctx, objectName, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get %s %s: %w", gvr.Resource, objectName, err)
		}
		return nil
	}).WithTimeout(timeout).WithPolling(poll).Should(gomega.Succeed())
	return err
}

// WaitForNamespacedGVRDeletion waits until a namespaced object has been deleted
func WaitForNamespacedGVRDeletion(ctx context.Context, c dynamic.Interface, gvr schema.GroupVersionResource, ns, objectName string, poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for %s %s to be deleted", timeout, gvr.Resource, objectName)

	if successful := WaitUntil(poll, timeout, func() bool {
		_, err := c.Resource(gvr).Namespace(ns).Get(ctx, objectName, metav1.GetOptions{})
		if err != nil && apierrors.IsNotFound(err) {
			framework.Logf("%s %s is not found in namespace %s and has been deleted", gvr.Resource, objectName, ns)
			return true
		} else if err != nil {
			framework.Logf("Get %s in namespace %s returned an error: %v", objectName, ns, err.Error())
		} else {
			framework.Logf("%s %s has been found in namespace %s and is not deleted", gvr.Resource, objectName, ns)
		}

		return false
	}); successful {
		return nil
	}

	return fmt.Errorf("%s %s in namespace %s is not deleted within %v", gvr.Resource, objectName, ns, timeout)
}

// WaitUntil runs checkDone until a timeout is reached
func WaitUntil(poll, timeout time.Duration, checkDone func() bool) bool {
	// TODO (pohly): replace with gomega.Eventually
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(poll) {
		if checkDone() {
			framework.Logf("WaitUntil finished successfully after %v", time.Since(start))
			return true
		}
	}

	framework.Logf("WaitUntil failed after reaching the timeout %v", timeout)
	return false
}

// WaitForGVRFinalizer waits until a object from a given GVR contains a finalizer
// If namespace is empty, assume it is a non-namespaced object
func WaitForGVRFinalizer(ctx context.Context, c dynamic.Interface, gvr schema.GroupVersionResource, objectName, objectNamespace, finalizer string, poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for object %s %s of resource %s to contain finalizer %s", timeout, objectNamespace, objectName, gvr.Resource, finalizer)
	var (
		err      error
		resource *unstructured.Unstructured
	)
	if successful := WaitUntil(poll, timeout, func() bool {
		switch objectNamespace {
		case "":
			resource, err = c.Resource(gvr).Get(ctx, objectName, metav1.GetOptions{})
		default:
			resource, err = c.Resource(gvr).Namespace(objectNamespace).Get(ctx, objectName, metav1.GetOptions{})
		}
		if err != nil {
			framework.Logf("Failed to get object %s %s with err: %v. Will retry in %v", objectNamespace, objectName, err, timeout)
			return false
		}
		for _, f := range resource.GetFinalizers() {
			if f == finalizer {
				return true
			}
		}
		return false
	}); successful {
		return nil
	}
	if err == nil {
		err = fmt.Errorf("finalizer %s not added to object %s %s of resource %s", finalizer, objectNamespace, objectName, gvr)
	}
	return err
}

// VerifyFilePathGIDInPod verfies expected GID of the target filepath
func VerifyFilePathGIDInPod(ctx context.Context, f *framework.Framework, filePath, expectedGID string, pod *v1.Pod) {
	cmd := fmt.Sprintf("ls -l %s", filePath)
	stdout, stderr, err := e2epod.ExecShellInPodWithFullOutput(ctx, f, pod.Name, cmd)
	framework.ExpectNoError(err)
	framework.Logf("pod %s/%s exec for cmd %s, stdout: %s, stderr: %s", pod.Namespace, pod.Name, cmd, stdout, stderr)
	ll := strings.Fields(stdout)
	framework.Logf("stdout split: %v, expected gid: %v", ll, expectedGID)
	gomega.Expect(ll[3]).To(gomega.Equal(expectedGID))
}

// ChangeFilePathGIDInPod changes the GID of the target filepath.
func ChangeFilePathGIDInPod(ctx context.Context, f *framework.Framework, filePath, targetGID string, pod *v1.Pod) {
	cmd := fmt.Sprintf("chgrp %s %s", targetGID, filePath)
	_, _, err := e2epod.ExecShellInPodWithFullOutput(ctx, f, pod.Name, cmd)
	framework.ExpectNoError(err)
	VerifyFilePathGIDInPod(ctx, f, filePath, targetGID, pod)
}

// DeleteStorageClass deletes the passed in StorageClass and catches errors other than "Not Found"
func DeleteStorageClass(ctx context.Context, cs clientset.Interface, className string) error {
	err := cs.StorageV1().StorageClasses().Delete(ctx, className, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// CreateVolumeSource creates a volume source object
func CreateVolumeSource(pvcName string, readOnly bool) *v1.VolumeSource {
	return &v1.VolumeSource{
		PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
			ClaimName: pvcName,
			ReadOnly:  readOnly,
		},
	}
}

// TryFunc try to execute the function and return err if there is any
func TryFunc(f func()) error {
	var err error
	if f == nil {
		return nil
	}
	defer func() {
		if recoverError := recover(); recoverError != nil {
			err = fmt.Errorf("%v", recoverError)
		}
	}()
	f()
	return err
}

// GetSizeRangesIntersection takes two instances of storage size ranges and determines the
// intersection of the intervals (if it exists) and return the minimum of the intersection
// to be used as the claim size for the test.
// if value not set, that means there's no minimum or maximum size limitation and we set default size for it.
func GetSizeRangesIntersection(first e2evolume.SizeRange, second e2evolume.SizeRange) (string, error) {
	var firstMin, firstMax, secondMin, secondMax resource.Quantity
	var err error

	//if SizeRange is not set, assign a minimum or maximum size
	if len(first.Min) == 0 {
		first.Min = minValidSize
	}
	if len(first.Max) == 0 {
		first.Max = maxValidSize
	}
	if len(second.Min) == 0 {
		second.Min = minValidSize
	}
	if len(second.Max) == 0 {
		second.Max = maxValidSize
	}

	if firstMin, err = resource.ParseQuantity(first.Min); err != nil {
		return "", err
	}
	if firstMax, err = resource.ParseQuantity(first.Max); err != nil {
		return "", err
	}
	if secondMin, err = resource.ParseQuantity(second.Min); err != nil {
		return "", err
	}
	if secondMax, err = resource.ParseQuantity(second.Max); err != nil {
		return "", err
	}

	interSectionStart := math.Max(float64(firstMin.Value()), float64(secondMin.Value()))
	intersectionEnd := math.Min(float64(firstMax.Value()), float64(secondMax.Value()))

	// the minimum of the intersection shall be returned as the claim size
	var intersectionMin resource.Quantity

	if intersectionEnd-interSectionStart >= 0 { //have intersection
		intersectionMin = *resource.NewQuantity(int64(interSectionStart), "BinarySI") //convert value to BinarySI format. E.g. 5Gi
		// return the minimum of the intersection as the claim size
		return intersectionMin.String(), nil
	}
	return "", fmt.Errorf("intersection of size ranges %+v, %+v is null", first, second)
}
