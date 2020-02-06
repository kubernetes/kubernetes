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
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"math/rand"
	"path/filepath"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	clientexec "k8s.io/client-go/util/exec"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	imageutils "k8s.io/kubernetes/test/utils/image"
	uexec "k8s.io/utils/exec"
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
	KRestart KubeletOpt = "restart"
)

const (
	// ClusterRole name for e2e test Priveledged Pod Security Policy User
	podSecurityPolicyPrivilegedClusterRoleName = "e2e-test-privileged-psp"
)

// PodExec runs f.ExecCommandInContainerWithFullOutput to execute a shell cmd in target pod
func PodExec(f *framework.Framework, pod *v1.Pod, shExec string) (string, error) {
	stdout, _, err := f.ExecCommandInContainerWithFullOutput(pod.Name, pod.Spec.Containers[0].Name, "/bin/sh", "-c", shExec)
	return stdout, err
}

// VerifyExecInPodSucceed verifies shell cmd in target pod succeed
func VerifyExecInPodSucceed(f *framework.Framework, pod *v1.Pod, shExec string) {
	_, err := PodExec(f, pod, shExec)
	if err != nil {
		if exiterr, ok := err.(uexec.CodeExitError); ok {
			exitCode := exiterr.ExitStatus()
			framework.ExpectNoError(err,
				"%q should succeed, but failed with exit code %d and error message %q",
				shExec, exitCode, exiterr)
		} else {
			framework.ExpectNoError(err,
				"%q should succeed, but failed with error message %q",
				shExec, err)
		}
	}
}

// VerifyExecInPodFail verifies shell cmd in target pod fail with certain exit code
func VerifyExecInPodFail(f *framework.Framework, pod *v1.Pod, shExec string, exitCode int) {
	_, err := PodExec(f, pod, shExec)
	if err != nil {
		if exiterr, ok := err.(clientexec.ExitError); ok {
			actualExitCode := exiterr.ExitStatus()
			framework.ExpectEqual(actualExitCode, exitCode,
				"%q should fail with exit code %d, but failed with exit code %d and error message %q",
				shExec, exitCode, actualExitCode, exiterr)
		} else {
			framework.ExpectNoError(err,
				"%q should fail with exit code %d, but failed with error message %q",
				shExec, exitCode, err)
		}
	}
	framework.ExpectError(err, "%q should fail with exit code %d, but exit without error", shExec, exitCode)
}

func isSudoPresent(nodeIP string, provider string) bool {
	framework.Logf("Checking if sudo command is present")
	sshResult, err := e2essh.SSH("sudo --version", nodeIP, provider)
	framework.ExpectNoError(err, "SSH to %q errored.", nodeIP)
	if !strings.Contains(sshResult.Stderr, "command not found") {
		return true
	}
	return false
}

// getHostAddress gets the node for a pod and returns the first
// address. Returns an error if the node the pod is on doesn't have an
// address.
func getHostAddress(client clientset.Interface, p *v1.Pod) (string, error) {
	node, err := client.CoreV1().Nodes().Get(p.Spec.NodeName, metav1.GetOptions{})
	if err != nil {
		return "", err
	}
	// Try externalAddress first
	for _, address := range node.Status.Addresses {
		if address.Type == v1.NodeExternalIP {
			if address.Address != "" {
				return address.Address, nil
			}
		}
	}
	// If no externalAddress found, try internalAddress
	for _, address := range node.Status.Addresses {
		if address.Type == v1.NodeInternalIP {
			if address.Address != "" {
				return address.Address, nil
			}
		}
	}

	// If not found, return error
	return "", fmt.Errorf("No address for pod %v on node %v",
		p.Name, p.Spec.NodeName)
}

// KubeletCommand performs `start`, `restart`, or `stop` on the kubelet running on the node of the target pod and waits
// for the desired statues..
// - First issues the command via `systemctl`
// - If `systemctl` returns stderr "command not found, issues the command via `service`
// - If `service` also returns stderr "command not found", the test is aborted.
// Allowed kubeletOps are `KStart`, `KStop`, and `KRestart`
func KubeletCommand(kOp KubeletOpt, c clientset.Interface, pod *v1.Pod) {
	command := ""
	systemctlPresent := false
	kubeletPid := ""

	nodeIP, err := getHostAddress(c, pod)
	framework.ExpectNoError(err)
	nodeIP = nodeIP + ":22"

	framework.Logf("Checking if systemctl command is present")
	sshResult, err := e2essh.SSH("systemctl --version", nodeIP, framework.TestContext.Provider)
	framework.ExpectNoError(err, fmt.Sprintf("SSH to Node %q errored.", pod.Spec.NodeName))
	if !strings.Contains(sshResult.Stderr, "command not found") {
		command = fmt.Sprintf("systemctl %s kubelet", string(kOp))
		systemctlPresent = true
	} else {
		command = fmt.Sprintf("service kubelet %s", string(kOp))
	}

	sudoPresent := isSudoPresent(nodeIP, framework.TestContext.Provider)
	if sudoPresent {
		command = fmt.Sprintf("sudo %s", command)
	}

	if kOp == KRestart {
		kubeletPid = getKubeletMainPid(nodeIP, sudoPresent, systemctlPresent)
	}

	framework.Logf("Attempting `%s`", command)
	sshResult, err = e2essh.SSH(command, nodeIP, framework.TestContext.Provider)
	framework.ExpectNoError(err, fmt.Sprintf("SSH to Node %q errored.", pod.Spec.NodeName))
	e2essh.LogResult(sshResult)
	gomega.Expect(sshResult.Code).To(gomega.BeZero(), "Failed to [%s] kubelet:\n%#v", string(kOp), sshResult)

	if kOp == KStop {
		if ok := e2enode.WaitForNodeToBeNotReady(c, pod.Spec.NodeName, NodeStateTimeout); !ok {
			framework.Failf("Node %s failed to enter NotReady state", pod.Spec.NodeName)
		}
	}
	if kOp == KRestart {
		// Wait for a minute to check if kubelet Pid is getting changed
		isPidChanged := false
		for start := time.Now(); time.Since(start) < 1*time.Minute; time.Sleep(2 * time.Second) {
			kubeletPidAfterRestart := getKubeletMainPid(nodeIP, sudoPresent, systemctlPresent)
			if kubeletPid != kubeletPidAfterRestart {
				isPidChanged = true
				break
			}
		}
		framework.ExpectEqual(isPidChanged, true, "Kubelet PID remained unchanged after restarting Kubelet")
		framework.Logf("Noticed that kubelet PID is changed. Waiting for 30 Seconds for Kubelet to come back")
		time.Sleep(30 * time.Second)
	}
	if kOp == KStart || kOp == KRestart {
		// For kubelet start and restart operations, Wait until Node becomes Ready
		if ok := e2enode.WaitForNodeToBeReady(c, pod.Spec.NodeName, NodeStateTimeout); !ok {
			framework.Failf("Node %s failed to enter Ready state", pod.Spec.NodeName)
		}
	}
}

// getKubeletMainPid return the Main PID of the Kubelet Process
func getKubeletMainPid(nodeIP string, sudoPresent bool, systemctlPresent bool) string {
	command := ""
	if systemctlPresent {
		command = "systemctl status kubelet | grep 'Main PID'"
	} else {
		command = "service kubelet status | grep 'Main PID'"
	}
	if sudoPresent {
		command = fmt.Sprintf("sudo %s", command)
	}
	framework.Logf("Attempting `%s`", command)
	sshResult, err := e2essh.SSH(command, nodeIP, framework.TestContext.Provider)
	framework.ExpectNoError(err, fmt.Sprintf("SSH to Node %q errored.", nodeIP))
	e2essh.LogResult(sshResult)
	gomega.Expect(sshResult.Code).To(gomega.BeZero(), "Failed to get kubelet PID")
	gomega.Expect(sshResult.Stdout).NotTo(gomega.BeEmpty(), "Kubelet Main PID should not be Empty")
	return sshResult.Stdout
}

// TestKubeletRestartsAndRestoresMount tests that a volume mounted to a pod remains mounted after a kubelet restarts
func TestKubeletRestartsAndRestoresMount(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod) {
	path := "/mnt/volume1"
	byteLen := 64
	seed := time.Now().UTC().UnixNano()

	ginkgo.By("Writing to the volume.")
	CheckWriteToPath(f, clientPod, v1.PersistentVolumeFilesystem, path, byteLen, seed)

	ginkgo.By("Restarting kubelet")
	KubeletCommand(KRestart, c, clientPod)

	ginkgo.By("Testing that written file is accessible.")
	CheckReadFromPath(f, clientPod, v1.PersistentVolumeFilesystem, path, byteLen, seed)

	framework.Logf("Volume mount detected on pod %s and written file %s is readable post-restart.", clientPod.Name, path)
}

// TestKubeletRestartsAndRestoresMap tests that a volume mapped to a pod remains mapped after a kubelet restarts
func TestKubeletRestartsAndRestoresMap(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod) {
	path := "/mnt/volume1"
	byteLen := 64
	seed := time.Now().UTC().UnixNano()

	ginkgo.By("Writing to the volume.")
	CheckWriteToPath(f, clientPod, v1.PersistentVolumeBlock, path, byteLen, seed)

	ginkgo.By("Restarting kubelet")
	KubeletCommand(KRestart, c, clientPod)

	ginkgo.By("Testing that written pv is accessible.")
	CheckReadFromPath(f, clientPod, v1.PersistentVolumeBlock, path, byteLen, seed)

	framework.Logf("Volume map detected on pod %s and written data %s is readable post-restart.", clientPod.Name, path)
}

// TestVolumeUnmountsFromDeletedPodWithForceOption tests that a volume unmounts if the client pod was deleted while the kubelet was down.
// forceDelete is true indicating whether the pod is forcefully deleted.
// checkSubpath is true indicating whether the subpath should be checked.
func TestVolumeUnmountsFromDeletedPodWithForceOption(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, forceDelete bool, checkSubpath bool) {
	nodeIP, err := getHostAddress(c, clientPod)
	framework.ExpectNoError(err)
	nodeIP = nodeIP + ":22"

	ginkgo.By("Expecting the volume mount to be found.")
	result, err := e2essh.SSH(fmt.Sprintf("mount | grep %s | grep -v volume-subpaths", clientPod.UID), nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	framework.ExpectEqual(result.Code, 0, fmt.Sprintf("Expected grep exit code of 0, got %d", result.Code))

	if checkSubpath {
		ginkgo.By("Expecting the volume subpath mount to be found.")
		result, err := e2essh.SSH(fmt.Sprintf("cat /proc/self/mountinfo | grep %s | grep volume-subpaths", clientPod.UID), nodeIP, framework.TestContext.Provider)
		e2essh.LogResult(result)
		framework.ExpectNoError(err, "Encountered SSH error.")
		framework.ExpectEqual(result.Code, 0, fmt.Sprintf("Expected grep exit code of 0, got %d", result.Code))
	}

	// This command is to make sure kubelet is started after test finishes no matter it fails or not.
	defer func() {
		KubeletCommand(KStart, c, clientPod)
	}()
	ginkgo.By("Stopping the kubelet.")
	KubeletCommand(KStop, c, clientPod)

	ginkgo.By(fmt.Sprintf("Deleting Pod %q", clientPod.Name))
	if forceDelete {
		err = c.CoreV1().Pods(clientPod.Namespace).Delete(clientPod.Name, metav1.NewDeleteOptions(0))
	} else {
		err = c.CoreV1().Pods(clientPod.Namespace).Delete(clientPod.Name, &metav1.DeleteOptions{})
	}
	framework.ExpectNoError(err)

	ginkgo.By("Starting the kubelet and waiting for pod to delete.")
	KubeletCommand(KStart, c, clientPod)
	err = f.WaitForPodNotFound(clientPod.Name, framework.PodDeleteTimeout)
	if err != nil {
		framework.ExpectNoError(err, "Expected pod to be not found.")
	}

	if forceDelete {
		// With forceDelete, since pods are immediately deleted from API server, there is no way to be sure when volumes are torn down
		// so wait some time to finish
		time.Sleep(30 * time.Second)
	}

	ginkgo.By("Expecting the volume mount not to be found.")
	result, err = e2essh.SSH(fmt.Sprintf("mount | grep %s | grep -v volume-subpaths", clientPod.UID), nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	gomega.Expect(result.Stdout).To(gomega.BeEmpty(), "Expected grep stdout to be empty (i.e. no mount found).")
	framework.Logf("Volume unmounted on node %s", clientPod.Spec.NodeName)

	if checkSubpath {
		ginkgo.By("Expecting the volume subpath mount not to be found.")
		result, err = e2essh.SSH(fmt.Sprintf("cat /proc/self/mountinfo | grep %s | grep volume-subpaths", clientPod.UID), nodeIP, framework.TestContext.Provider)
		e2essh.LogResult(result)
		framework.ExpectNoError(err, "Encountered SSH error.")
		gomega.Expect(result.Stdout).To(gomega.BeEmpty(), "Expected grep stdout to be empty (i.e. no subpath mount found).")
		framework.Logf("Subpath volume unmounted on node %s", clientPod.Spec.NodeName)
	}
}

// TestVolumeUnmountsFromDeletedPod tests that a volume unmounts if the client pod was deleted while the kubelet was down.
func TestVolumeUnmountsFromDeletedPod(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod) {
	TestVolumeUnmountsFromDeletedPodWithForceOption(c, f, clientPod, false, false)
}

// TestVolumeUnmountsFromForceDeletedPod tests that a volume unmounts if the client pod was forcefully deleted while the kubelet was down.
func TestVolumeUnmountsFromForceDeletedPod(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod) {
	TestVolumeUnmountsFromDeletedPodWithForceOption(c, f, clientPod, true, false)
}

// TestVolumeUnmapsFromDeletedPodWithForceOption tests that a volume unmaps if the client pod was deleted while the kubelet was down.
// forceDelete is true indicating whether the pod is forcefully deleted.
func TestVolumeUnmapsFromDeletedPodWithForceOption(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, forceDelete bool) {
	nodeIP, err := getHostAddress(c, clientPod)
	framework.ExpectNoError(err, "Failed to get nodeIP.")
	nodeIP = nodeIP + ":22"

	// Creating command to check whether path exists
	podDirectoryCmd := fmt.Sprintf("ls /var/lib/kubelet/pods/%s/volumeDevices/*/ | grep '.'", clientPod.UID)
	if isSudoPresent(nodeIP, framework.TestContext.Provider) {
		podDirectoryCmd = fmt.Sprintf("sudo sh -c \"%s\"", podDirectoryCmd)
	}
	// Directories in the global directory have unpredictable names, however, device symlinks
	// have the same name as pod.UID. So just find anything with pod.UID name.
	globalBlockDirectoryCmd := fmt.Sprintf("find /var/lib/kubelet/plugins -name %s", clientPod.UID)
	if isSudoPresent(nodeIP, framework.TestContext.Provider) {
		globalBlockDirectoryCmd = fmt.Sprintf("sudo sh -c \"%s\"", globalBlockDirectoryCmd)
	}

	ginkgo.By("Expecting the symlinks from PodDeviceMapPath to be found.")
	result, err := e2essh.SSH(podDirectoryCmd, nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	framework.ExpectEqual(result.Code, 0, fmt.Sprintf("Expected grep exit code of 0, got %d", result.Code))

	ginkgo.By("Expecting the symlinks from global map path to be found.")
	result, err = e2essh.SSH(globalBlockDirectoryCmd, nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	framework.ExpectEqual(result.Code, 0, fmt.Sprintf("Expected find exit code of 0, got %d", result.Code))

	// This command is to make sure kubelet is started after test finishes no matter it fails or not.
	defer func() {
		KubeletCommand(KStart, c, clientPod)
	}()
	ginkgo.By("Stopping the kubelet.")
	KubeletCommand(KStop, c, clientPod)

	ginkgo.By(fmt.Sprintf("Deleting Pod %q", clientPod.Name))
	if forceDelete {
		err = c.CoreV1().Pods(clientPod.Namespace).Delete(clientPod.Name, metav1.NewDeleteOptions(0))
	} else {
		err = c.CoreV1().Pods(clientPod.Namespace).Delete(clientPod.Name, &metav1.DeleteOptions{})
	}
	framework.ExpectNoError(err, "Failed to delete pod.")

	ginkgo.By("Starting the kubelet and waiting for pod to delete.")
	KubeletCommand(KStart, c, clientPod)
	err = f.WaitForPodNotFound(clientPod.Name, framework.PodDeleteTimeout)
	framework.ExpectNoError(err, "Expected pod to be not found.")

	if forceDelete {
		// With forceDelete, since pods are immediately deleted from API server, there is no way to be sure when volumes are torn down
		// so wait some time to finish
		time.Sleep(30 * time.Second)
	}

	ginkgo.By("Expecting the symlink from PodDeviceMapPath not to be found.")
	result, err = e2essh.SSH(podDirectoryCmd, nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	gomega.Expect(result.Stdout).To(gomega.BeEmpty(), "Expected grep stdout to be empty.")

	ginkgo.By("Expecting the symlinks from global map path not to be found.")
	result, err = e2essh.SSH(globalBlockDirectoryCmd, nodeIP, framework.TestContext.Provider)
	e2essh.LogResult(result)
	framework.ExpectNoError(err, "Encountered SSH error.")
	gomega.Expect(result.Stdout).To(gomega.BeEmpty(), "Expected find stdout to be empty.")

	framework.Logf("Volume unmaped on node %s", clientPod.Spec.NodeName)
}

// TestVolumeUnmapsFromDeletedPod tests that a volume unmaps if the client pod was deleted while the kubelet was down.
func TestVolumeUnmapsFromDeletedPod(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod) {
	TestVolumeUnmapsFromDeletedPodWithForceOption(c, f, clientPod, false)
}

// TestVolumeUnmapsFromForceDeletedPod tests that a volume unmaps if the client pod was forcefully deleted while the kubelet was down.
func TestVolumeUnmapsFromForceDeletedPod(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod) {
	TestVolumeUnmapsFromDeletedPodWithForceOption(c, f, clientPod, true)
}

// RunInPodWithVolume runs a command in a pod with given claim mounted to /mnt directory.
func RunInPodWithVolume(c clientset.Interface, ns, claimName, command string) {
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
	pod, err := c.CoreV1().Pods(ns).Create(pod)
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	defer func() {
		e2epod.DeletePodOrFail(c, ns, pod.Name)
	}()
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceSlow(c, pod.Name, pod.Namespace))
}

// StartExternalProvisioner create external provisioner pod
func StartExternalProvisioner(c clientset.Interface, ns string, externalPluginName string) *v1.Pod {
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
	provisionerPod, err := podClient.Create(provisionerPod)
	framework.ExpectNoError(err, "Failed to create %s pod: %v", provisionerPod.Name, err)

	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(c, provisionerPod))

	ginkgo.By("locating the provisioner pod")
	pod, err := podClient.Get(provisionerPod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "Cannot locate the provisioner pod %v: %v", provisionerPod.Name, err)

	return pod
}

// PrivilegedTestPSPClusterRoleBinding test Pod Security Policy Role bindings
func PrivilegedTestPSPClusterRoleBinding(client clientset.Interface,
	namespace string,
	teardown bool,
	saNames []string) {
	bindingString := "Binding"
	if teardown {
		bindingString = "Unbinding"
	}
	roleBindingClient := client.RbacV1().RoleBindings(namespace)
	for _, saName := range saNames {
		ginkgo.By(fmt.Sprintf("%v priviledged Pod Security Policy to the service account %s", bindingString, saName))
		binding := &rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "psp-" + saName,
				Namespace: namespace,
			},
			Subjects: []rbacv1.Subject{
				{
					Kind:      rbacv1.ServiceAccountKind,
					Name:      saName,
					Namespace: namespace,
				},
			},
			RoleRef: rbacv1.RoleRef{
				Kind:     "ClusterRole",
				Name:     podSecurityPolicyPrivilegedClusterRoleName,
				APIGroup: "rbac.authorization.k8s.io",
			},
		}

		roleBindingClient.Delete(binding.GetName(), &metav1.DeleteOptions{})
		err := wait.Poll(2*time.Second, 2*time.Minute, func() (bool, error) {
			_, err := roleBindingClient.Get(binding.GetName(), metav1.GetOptions{})
			return apierrors.IsNotFound(err), nil
		})
		framework.ExpectNoError(err, "Timed out waiting for RBAC binding %s deletion: %v", binding.GetName(), err)

		if teardown {
			continue
		}

		_, err = roleBindingClient.Create(binding)
		framework.ExpectNoError(err, "Failed to create %s role binding: %v", binding.GetName(), err)

	}
}

// CheckVolumeModeOfPath check mode of volume
func CheckVolumeModeOfPath(f *framework.Framework, pod *v1.Pod, volMode v1.PersistentVolumeMode, path string) {
	if volMode == v1.PersistentVolumeBlock {
		// Check if block exists
		VerifyExecInPodSucceed(f, pod, fmt.Sprintf("test -b %s", path))

		// Double check that it's not directory
		VerifyExecInPodFail(f, pod, fmt.Sprintf("test -d %s", path), 1)
	} else {
		// Check if directory exists
		VerifyExecInPodSucceed(f, pod, fmt.Sprintf("test -d %s", path))

		// Double check that it's not block
		VerifyExecInPodFail(f, pod, fmt.Sprintf("test -b %s", path), 1)
	}
}

// CheckReadWriteToPath check that path can b e read and written
func CheckReadWriteToPath(f *framework.Framework, pod *v1.Pod, volMode v1.PersistentVolumeMode, path string) {
	if volMode == v1.PersistentVolumeBlock {
		// random -> file1
		VerifyExecInPodSucceed(f, pod, "dd if=/dev/urandom of=/tmp/file1 bs=64 count=1")
		// file1 -> dev (write to dev)
		VerifyExecInPodSucceed(f, pod, fmt.Sprintf("dd if=/tmp/file1 of=%s bs=64 count=1", path))
		// dev -> file2 (read from dev)
		VerifyExecInPodSucceed(f, pod, fmt.Sprintf("dd if=%s of=/tmp/file2 bs=64 count=1", path))
		// file1 == file2 (check contents)
		VerifyExecInPodSucceed(f, pod, "diff /tmp/file1 /tmp/file2")
		// Clean up temp files
		VerifyExecInPodSucceed(f, pod, "rm -f /tmp/file1 /tmp/file2")

		// Check that writing file to block volume fails
		VerifyExecInPodFail(f, pod, fmt.Sprintf("echo 'Hello world.' > %s/file1.txt", path), 1)
	} else {
		// text -> file1 (write to file)
		VerifyExecInPodSucceed(f, pod, fmt.Sprintf("echo 'Hello world.' > %s/file1.txt", path))
		// grep file1 (read from file and check contents)
		VerifyExecInPodSucceed(f, pod, fmt.Sprintf("grep 'Hello world.' %s/file1.txt", path))

		// Check that writing to directory as block volume fails
		VerifyExecInPodFail(f, pod, fmt.Sprintf("dd if=/dev/urandom of=%s bs=64 count=1", path), 1)
	}
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
func CheckReadFromPath(f *framework.Framework, pod *v1.Pod, volMode v1.PersistentVolumeMode, path string, len int, seed int64) {
	var pathForVolMode string
	if volMode == v1.PersistentVolumeBlock {
		pathForVolMode = path
	} else {
		pathForVolMode = filepath.Join(path, "file1.txt")
	}

	sum := sha256.Sum256(genBinDataFromSeed(len, seed))

	VerifyExecInPodSucceed(f, pod, fmt.Sprintf("dd if=%s bs=%d count=1 | sha256sum", pathForVolMode, len))
	VerifyExecInPodSucceed(f, pod, fmt.Sprintf("dd if=%s bs=%d count=1 | sha256sum | grep -Fq %x", pathForVolMode, len, sum))
}

// CheckWriteToPath that file can be properly written.
func CheckWriteToPath(f *framework.Framework, pod *v1.Pod, volMode v1.PersistentVolumeMode, path string, len int, seed int64) {
	var pathForVolMode string
	if volMode == v1.PersistentVolumeBlock {
		pathForVolMode = path
	} else {
		pathForVolMode = filepath.Join(path, "file1.txt")
	}

	encoded := base64.StdEncoding.EncodeToString(genBinDataFromSeed(len, seed))

	VerifyExecInPodSucceed(f, pod, fmt.Sprintf("echo %s | base64 -d | sha256sum", encoded))
	VerifyExecInPodSucceed(f, pod, fmt.Sprintf("echo %s | base64 -d | dd of=%s bs=%d count=1", encoded, pathForVolMode, len))
}

// findMountPoints returns all mount points on given node under specified directory.
func findMountPoints(hostExec HostExec, node *v1.Node, dir string) []string {
	result, err := hostExec.IssueCommandWithResult(fmt.Sprintf(`find %s -type d -exec mountpoint {} \; | grep 'is a mountpoint$' || true`, dir), node)
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
func FindVolumeGlobalMountPoints(hostExec HostExec, node *v1.Node) sets.String {
	return sets.NewString(findMountPoints(hostExec, node, "/var/lib/kubelet/plugins")...)
}
