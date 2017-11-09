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

package storage

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
)

type testBody func(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume)
type disruptiveTest struct {
	testItStmt string
	runTest    testBody
}
type kubeletOpt string

const (
	MinNodes                    = 2
	NodeStateTimeout            = 1 * time.Minute
	kStart           kubeletOpt = "start"
	kStop            kubeletOpt = "stop"
	kRestart         kubeletOpt = "restart"
)

var _ = SIGDescribe("PersistentVolumes[Disruptive][Flaky]", func() {

	f := framework.NewDefaultFramework("disruptive-pv")
	var (
		c                         clientset.Interface
		ns                        string
		nfsServerPod              *v1.Pod
		nfsPVconfig               framework.PersistentVolumeConfig
		pvcConfig                 framework.PersistentVolumeClaimConfig
		nfsServerIP, clientNodeIP string
		clientNode                *v1.Node
		volLabel                  labels.Set
		selector                  *metav1.LabelSelector
	)

	BeforeEach(func() {
		// To protect the NFS volume pod from the kubelet restart, we isolate it on its own node.
		framework.SkipUnlessNodeCountIsAtLeast(MinNodes)
		framework.SkipIfProviderIs("local")

		c = f.ClientSet
		ns = f.Namespace.Name
		volLabel = labels.Set{framework.VolumeSelectorKey: ns}
		selector = metav1.SetAsLabelSelector(volLabel)
		// Start the NFS server pod.
		_, nfsServerPod, nfsServerIP = framework.NewNFSServer(c, ns, []string{"-G", "777", "/exports"})
		nfsPVconfig = framework.PersistentVolumeConfig{
			NamePrefix: "nfs-",
			Labels:     volLabel,
			PVSource: v1.PersistentVolumeSource{
				NFS: &v1.NFSVolumeSource{
					Server:   nfsServerIP,
					Path:     "/exports",
					ReadOnly: false,
				},
			},
		}
		pvcConfig = framework.PersistentVolumeClaimConfig{
			Annotations: map[string]string{
				v1.BetaStorageClassAnnotation: "",
			},
			Selector: selector,
		}
		// Get the first ready node IP that is not hosting the NFS pod.
		if clientNodeIP == "" {
			framework.Logf("Designating test node")
			nodes := framework.GetReadySchedulableNodesOrDie(c)
			for _, node := range nodes.Items {
				if node.Name != nfsServerPod.Spec.NodeName {
					clientNode = &node
					clientNodeIP = framework.GetNodeExternalIP(clientNode)
					break
				}
			}
			Expect(clientNodeIP).NotTo(BeEmpty())
		}
	})

	AfterEach(func() {
		framework.DeletePodWithWait(f, c, nfsServerPod)
	})

	Context("when kube-controller-manager restarts", func() {
		var (
			diskName1, diskName2 string
			err                  error
			pvConfig1, pvConfig2 framework.PersistentVolumeConfig
			pv1, pv2             *v1.PersistentVolume
			pvSource1, pvSource2 *v1.PersistentVolumeSource
			pvc1, pvc2           *v1.PersistentVolumeClaim
			clientPod            *v1.Pod
		)

		BeforeEach(func() {
			framework.SkipUnlessProviderIs("gce")
			framework.SkipUnlessSSHKeyPresent()

			By("Initializing first PD with PVPVC binding")
			pvSource1, diskName1 = framework.CreateGCEVolume()
			Expect(err).NotTo(HaveOccurred())
			pvConfig1 = framework.PersistentVolumeConfig{
				NamePrefix: "gce-",
				Labels:     volLabel,
				PVSource:   *pvSource1,
				Prebind:    nil,
			}
			pv1, pvc1, err = framework.CreatePVPVC(c, pvConfig1, pvcConfig, ns, false)
			Expect(err).NotTo(HaveOccurred())
			framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pv1, pvc1))

			By("Initializing second PD with PVPVC binding")
			pvSource2, diskName2 = framework.CreateGCEVolume()
			Expect(err).NotTo(HaveOccurred())
			pvConfig2 = framework.PersistentVolumeConfig{
				NamePrefix: "gce-",
				Labels:     volLabel,
				PVSource:   *pvSource2,
				Prebind:    nil,
			}
			pv2, pvc2, err = framework.CreatePVPVC(c, pvConfig2, pvcConfig, ns, false)
			Expect(err).NotTo(HaveOccurred())
			framework.ExpectNoError(framework.WaitOnPVandPVC(c, ns, pv2, pvc2))

			By("Attaching both PVC's to a single pod")
			clientPod, err = framework.CreatePod(c, ns, []*v1.PersistentVolumeClaim{pvc1, pvc2}, true, "")
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			// Delete client/user pod first
			framework.ExpectNoError(framework.DeletePodWithWait(f, c, clientPod))

			// Delete PV and PVCs
			if errs := framework.PVPVCCleanup(c, ns, pv1, pvc1); len(errs) > 0 {
				framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pv1, pvc1 = nil, nil
			if errs := framework.PVPVCCleanup(c, ns, pv2, pvc2); len(errs) > 0 {
				framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pv2, pvc2 = nil, nil

			// Delete the actual disks
			if diskName1 != "" {
				framework.ExpectNoError(framework.DeletePDWithRetry(diskName1))
			}
			if diskName2 != "" {
				framework.ExpectNoError(framework.DeletePDWithRetry(diskName2))
			}
		})

		It("should delete a bound PVC from a clientPod, restart the kube-control-manager, and ensure the kube-controller-manager does not crash", func() {
			By("Deleting PVC for volume 2")
			err = framework.DeletePersistentVolumeClaim(c, pvc2.Name, ns)
			Expect(err).NotTo(HaveOccurred())
			pvc2 = nil

			By("Restarting the kube-controller-manager")
			err = framework.RestartControllerManager()
			Expect(err).NotTo(HaveOccurred())
			err = framework.WaitForControllerManagerUp()
			Expect(err).NotTo(HaveOccurred())
			framework.Logf("kube-controller-manager restarted")

			By("Observing the kube-controller-manager healthy for at least 2 minutes")
			// Continue checking for 2 minutes to make sure kube-controller-manager is healthy
			err = framework.CheckForControllerManagerHealthy(2 * time.Minute)
			Expect(err).NotTo(HaveOccurred())
		})

	})

	Context("when kubelet restarts", func() {
		var (
			clientPod *v1.Pod
			pv        *v1.PersistentVolume
			pvc       *v1.PersistentVolumeClaim
		)

		BeforeEach(func() {
			framework.Logf("Initializing test spec")
			clientPod, pv, pvc = initTestCase(f, c, nfsPVconfig, pvcConfig, ns, clientNode.Name)
		})

		AfterEach(func() {
			framework.Logf("Tearing down test spec")
			tearDownTestCase(c, f, ns, clientPod, pvc, pv)
			pv, pvc, clientPod = nil, nil, nil
		})

		// Test table housing the It() title string and test spec.  runTest is type testBody, defined at
		// the start of this file.  To add tests, define a function mirroring the testBody signature and assign
		// to runTest.
		disruptiveTestTable := []disruptiveTest{
			{
				testItStmt: "Should test that a file written to the mount before kubelet restart is readable after restart.",
				runTest:    testKubeletRestartsAndRestoresMount,
			},
			{
				testItStmt: "Should test that a volume mounted to a pod that is deleted while the kubelet is down unmounts when the kubelet returns.",
				runTest:    testVolumeUnmountsFromDeletedPod,
			},
		}

		// Test loop executes each disruptiveTest iteratively.
		for _, test := range disruptiveTestTable {
			func(t disruptiveTest) {
				It(t.testItStmt, func() {
					By("Executing Spec")
					t.runTest(c, f, clientPod, pvc, pv)
				})
			}(test)
		}
	})
})

// testKubeletRestartsAndRestoresMount tests that a volume mounted to a pod remains mounted after a kubelet restarts
func testKubeletRestartsAndRestoresMount(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume) {
	By("Writing to the volume.")
	file := "/mnt/_SUCCESS"
	out, err := podExec(clientPod, fmt.Sprintf("touch %s", file))
	framework.Logf(out)
	Expect(err).NotTo(HaveOccurred())

	By("Restarting kubelet")
	kubeletCommand(kRestart, c, clientPod)

	By("Testing that written file is accessible.")
	out, err = podExec(clientPod, fmt.Sprintf("cat %s", file))
	framework.Logf(out)
	Expect(err).NotTo(HaveOccurred())
	framework.Logf("Volume mount detected on pod %s and written file %s is readable post-restart.", clientPod.Name, file)
}

// testVolumeUnmountsFromDeletedPod tests that a volume unmounts if the client pod was deleted while the kubelet was down.
func testVolumeUnmountsFromDeletedPod(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod, pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume) {
	nodeIP, err := framework.GetHostExternalAddress(c, clientPod)
	Expect(err).NotTo(HaveOccurred())
	nodeIP = nodeIP + ":22"

	By("Expecting the volume mount to be found.")
	result, err := framework.SSH(fmt.Sprintf("mount | grep %s", clientPod.UID), nodeIP, framework.TestContext.Provider)
	framework.LogSSHResult(result)
	Expect(err).NotTo(HaveOccurred(), "Encountered SSH error.")
	Expect(result.Code).To(BeZero(), fmt.Sprintf("Expected grep exit code of 0, got %d", result.Code))

	By("Stopping the kubelet.")
	kubeletCommand(kStop, c, clientPod)
	defer func() {
		if err != nil {
			kubeletCommand(kStart, c, clientPod)
		}
	}()
	By(fmt.Sprintf("Deleting Pod %q", clientPod.Name))
	err = c.CoreV1().Pods(clientPod.Namespace).Delete(clientPod.Name, &metav1.DeleteOptions{})
	Expect(err).NotTo(HaveOccurred())
	By("Starting the kubelet and waiting for pod to delete.")
	kubeletCommand(kStart, c, clientPod)
	err = f.WaitForPodTerminated(clientPod.Name, "")
	if !apierrs.IsNotFound(err) && err != nil {
		Expect(err).NotTo(HaveOccurred(), "Expected pod to terminate.")
	}

	By("Expecting the volume mount not to be found.")
	result, err = framework.SSH(fmt.Sprintf("mount | grep %s", clientPod.UID), nodeIP, framework.TestContext.Provider)
	framework.LogSSHResult(result)
	Expect(err).NotTo(HaveOccurred(), "Encountered SSH error.")
	Expect(result.Stdout).To(BeEmpty(), "Expected grep stdout to be empty (i.e. no mount found).")
	framework.Logf("Volume unmounted on node %s", clientPod.Spec.NodeName)
}

// initTestCase initializes spec resources (pv, pvc, and pod) and returns pointers to be consumed
// by the test.
func initTestCase(f *framework.Framework, c clientset.Interface, pvConfig framework.PersistentVolumeConfig, pvcConfig framework.PersistentVolumeClaimConfig, ns, nodeName string) (*v1.Pod, *v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	pv, pvc, err := framework.CreatePVPVC(c, pvConfig, pvcConfig, ns, false)
	defer func() {
		if err != nil {
			framework.DeletePersistentVolumeClaim(c, pvc.Name, ns)
			framework.DeletePersistentVolume(c, pv.Name)
		}
	}()
	Expect(err).NotTo(HaveOccurred())
	pod := framework.MakePod(ns, []*v1.PersistentVolumeClaim{pvc}, true, "")
	pod.Spec.NodeName = nodeName
	framework.Logf("Creating NFS client pod.")
	pod, err = c.CoreV1().Pods(ns).Create(pod)
	framework.Logf("NFS client Pod %q created on Node %q", pod.Name, nodeName)
	Expect(err).NotTo(HaveOccurred())
	defer func() {
		if err != nil {
			framework.DeletePodWithWait(f, c, pod)
		}
	}()
	err = framework.WaitForPodRunningInNamespace(c, pod)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Pod %q timed out waiting for phase: Running", pod.Name))
	// Return created api objects
	pod, err = c.CoreV1().Pods(ns).Get(pod.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	pv, err = c.CoreV1().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	return pod, pv, pvc
}

// tearDownTestCase destroy resources created by initTestCase.
func tearDownTestCase(c clientset.Interface, f *framework.Framework, ns string, client *v1.Pod, pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume) {
	// Ignore deletion errors.  Failing on them will interrupt test cleanup.
	framework.DeletePodWithWait(f, c, client)
	framework.DeletePersistentVolumeClaim(c, pvc.Name, ns)
	framework.DeletePersistentVolume(c, pv.Name)
}

// kubeletCommand performs `start`, `restart`, or `stop` on the kubelet running on the node of the target pod and waits
// for the desired statues..
// - First issues the command via `systemctl`
// - If `systemctl` returns stderr "command not found, issues the command via `service`
// - If `service` also returns stderr "command not found", the test is aborted.
// Allowed kubeletOps are `kStart`, `kStop`, and `kRestart`
func kubeletCommand(kOp kubeletOpt, c clientset.Interface, pod *v1.Pod) {
	command := ""
	sudoPresent := false
	systemctlPresent := false
	kubeletPid := ""

	nodeIP, err := framework.GetHostExternalAddress(c, pod)
	Expect(err).NotTo(HaveOccurred())
	nodeIP = nodeIP + ":22"

	framework.Logf("Checking if sudo command is present")
	sshResult, err := framework.SSH("sudo --version", nodeIP, framework.TestContext.Provider)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("SSH to Node %q errored.", pod.Spec.NodeName))
	if !strings.Contains(sshResult.Stderr, "command not found") {
		sudoPresent = true
	}

	framework.Logf("Checking if systemctl command is present")
	sshResult, err = framework.SSH("systemctl --version", nodeIP, framework.TestContext.Provider)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("SSH to Node %q errored.", pod.Spec.NodeName))
	if !strings.Contains(sshResult.Stderr, "command not found") {
		command = fmt.Sprintf("systemctl %s kubelet", string(kOp))
		systemctlPresent = true
	} else {
		command = fmt.Sprintf("service kubelet %s", string(kOp))
	}
	if sudoPresent {
		command = fmt.Sprintf("sudo %s", command)
	}

	if kOp == kRestart {
		kubeletPid = getKubeletMainPid(nodeIP, sudoPresent, systemctlPresent)
	}

	framework.Logf("Attempting `%s`", command)
	sshResult, err = framework.SSH(command, nodeIP, framework.TestContext.Provider)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("SSH to Node %q errored.", pod.Spec.NodeName))
	framework.LogSSHResult(sshResult)
	Expect(sshResult.Code).To(BeZero(), "Failed to [%s] kubelet:\n%#v", string(kOp), sshResult)

	if kOp == kStop {
		if ok := framework.WaitForNodeToBeNotReady(c, pod.Spec.NodeName, NodeStateTimeout); !ok {
			framework.Failf("Node %s failed to enter NotReady state", pod.Spec.NodeName)
		}
	}
	if kOp == kRestart {
		// Wait for a minute to check if kubelet Pid is getting changed
		isPidChanged := false
		for start := time.Now(); time.Since(start) < 1*time.Minute; time.Sleep(2 * time.Second) {
			kubeletPidAfterRestart := getKubeletMainPid(nodeIP, sudoPresent, systemctlPresent)
			if kubeletPid != kubeletPidAfterRestart {
				isPidChanged = true
				break
			}
		}
		Expect(isPidChanged).To(BeTrue(), "Kubelet PID remained unchanged after restarting Kubelet")
		framework.Logf("Noticed that kubelet PID is changed. Waiting for 30 Seconds for Kubelet to come back")
		time.Sleep(30 * time.Second)
	}
	if kOp == kStart || kOp == kRestart {
		// For kubelet start and restart operations, Wait until Node becomes Ready
		if ok := framework.WaitForNodeToBeReady(c, pod.Spec.NodeName, NodeStateTimeout); !ok {
			framework.Failf("Node %s failed to enter Ready state", pod.Spec.NodeName)
		}
	}
}

// return the Main PID of the Kubelet Process
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
	sshResult, err := framework.SSH(command, nodeIP, framework.TestContext.Provider)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("SSH to Node %q errored.", nodeIP))
	framework.LogSSHResult(sshResult)
	Expect(sshResult.Code).To(BeZero(), "Failed to get kubelet PID")
	Expect(sshResult.Stdout).NotTo(BeEmpty(), "Kubelet Main PID should not be Empty")
	return sshResult.Stdout
}

// podExec wraps RunKubectl to execute a bash cmd in target pod
func podExec(pod *v1.Pod, bashExec string) (string, error) {
	return framework.RunKubectl("exec", fmt.Sprintf("--namespace=%s", pod.Namespace), pod.Name, "--", "/bin/sh", "-c", bashExec)
}
