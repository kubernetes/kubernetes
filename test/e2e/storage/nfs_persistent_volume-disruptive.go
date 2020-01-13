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
	"net"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2essh "k8s.io/kubernetes/test/e2e/framework/ssh"
	"k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

type testBody func(c clientset.Interface, f *framework.Framework, clientPod *v1.Pod)
type disruptiveTest struct {
	testItStmt string
	runTest    testBody
}

// checkForControllerManagerHealthy checks that the controller manager does not crash within "duration"
func checkForControllerManagerHealthy(duration time.Duration) error {
	var PID string
	cmd := "pidof kube-controller-manager"
	for start := time.Now(); time.Since(start) < duration; time.Sleep(5 * time.Second) {
		result, err := e2essh.SSH(cmd, net.JoinHostPort(framework.GetMasterHost(), sshPort), framework.TestContext.Provider)
		if err != nil {
			// We don't necessarily know that it crashed, pipe could just be broken
			e2essh.LogResult(result)
			return fmt.Errorf("master unreachable after %v", time.Since(start))
		} else if result.Code != 0 {
			e2essh.LogResult(result)
			return fmt.Errorf("SSH result code not 0. actually: %v after %v", result.Code, time.Since(start))
		} else if result.Stdout != PID {
			if PID == "" {
				PID = result.Stdout
			} else {
				//its dead
				return fmt.Errorf("controller manager crashed, old PID: %s, new PID: %s", PID, result.Stdout)
			}
		} else {
			framework.Logf("kube-controller-manager still healthy after %v", time.Since(start))
		}
	}
	return nil
}

var _ = utils.SIGDescribe("NFSPersistentVolumes[Disruptive][Flaky]", func() {

	f := framework.NewDefaultFramework("disruptive-pv")
	var (
		c                         clientset.Interface
		ns                        string
		nfsServerPod              *v1.Pod
		nfsPVconfig               e2epv.PersistentVolumeConfig
		pvcConfig                 e2epv.PersistentVolumeClaimConfig
		nfsServerIP, clientNodeIP string
		clientNode                *v1.Node
		volLabel                  labels.Set
		selector                  *metav1.LabelSelector
	)

	ginkgo.BeforeEach(func() {
		// To protect the NFS volume pod from the kubelet restart, we isolate it on its own node.
		e2eskipper.SkipUnlessNodeCountIsAtLeast(minNodes)
		e2eskipper.SkipIfProviderIs("local")

		c = f.ClientSet
		ns = f.Namespace.Name
		volLabel = labels.Set{e2epv.VolumeSelectorKey: ns}
		selector = metav1.SetAsLabelSelector(volLabel)
		// Start the NFS server pod.
		_, nfsServerPod, nfsServerIP = volume.NewNFSServer(c, ns, []string{"-G", "777", "/exports"})
		nfsPVconfig = e2epv.PersistentVolumeConfig{
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
		emptyStorageClass := ""
		pvcConfig = e2epv.PersistentVolumeClaimConfig{
			Selector:         selector,
			StorageClassName: &emptyStorageClass,
		}
		// Get the first ready node IP that is not hosting the NFS pod.
		if clientNodeIP == "" {
			framework.Logf("Designating test node")
			nodes, err := e2enode.GetReadySchedulableNodes(c)
			framework.ExpectNoError(err)
			for _, node := range nodes.Items {
				if node.Name != nfsServerPod.Spec.NodeName {
					clientNode = &node
					clientNodeIP, err = e2enode.GetExternalIP(clientNode)
					framework.ExpectNoError(err)
					break
				}
			}
			gomega.Expect(clientNodeIP).NotTo(gomega.BeEmpty())
		}
	})

	ginkgo.AfterEach(func() {
		e2epod.DeletePodWithWait(c, nfsServerPod)
	})

	ginkgo.Context("when kube-controller-manager restarts", func() {
		var (
			diskName1, diskName2 string
			err                  error
			pvConfig1, pvConfig2 e2epv.PersistentVolumeConfig
			pv1, pv2             *v1.PersistentVolume
			pvSource1, pvSource2 *v1.PersistentVolumeSource
			pvc1, pvc2           *v1.PersistentVolumeClaim
			clientPod            *v1.Pod
		)

		ginkgo.BeforeEach(func() {
			e2eskipper.SkipUnlessProviderIs("gce")
			e2eskipper.SkipUnlessSSHKeyPresent()

			ginkgo.By("Initializing first PD with PVPVC binding")
			pvSource1, diskName1 = createGCEVolume()
			framework.ExpectNoError(err)
			pvConfig1 = e2epv.PersistentVolumeConfig{
				NamePrefix: "gce-",
				Labels:     volLabel,
				PVSource:   *pvSource1,
				Prebind:    nil,
			}
			pv1, pvc1, err = e2epv.CreatePVPVC(c, pvConfig1, pvcConfig, ns, false)
			framework.ExpectNoError(err)
			framework.ExpectNoError(e2epv.WaitOnPVandPVC(c, ns, pv1, pvc1))

			ginkgo.By("Initializing second PD with PVPVC binding")
			pvSource2, diskName2 = createGCEVolume()
			framework.ExpectNoError(err)
			pvConfig2 = e2epv.PersistentVolumeConfig{
				NamePrefix: "gce-",
				Labels:     volLabel,
				PVSource:   *pvSource2,
				Prebind:    nil,
			}
			pv2, pvc2, err = e2epv.CreatePVPVC(c, pvConfig2, pvcConfig, ns, false)
			framework.ExpectNoError(err)
			framework.ExpectNoError(e2epv.WaitOnPVandPVC(c, ns, pv2, pvc2))

			ginkgo.By("Attaching both PVC's to a single pod")
			clientPod, err = e2epod.CreatePod(c, ns, nil, []*v1.PersistentVolumeClaim{pvc1, pvc2}, true, "")
			framework.ExpectNoError(err)
		})

		ginkgo.AfterEach(func() {
			// Delete client/user pod first
			framework.ExpectNoError(e2epod.DeletePodWithWait(c, clientPod))

			// Delete PV and PVCs
			if errs := e2epv.PVPVCCleanup(c, ns, pv1, pvc1); len(errs) > 0 {
				framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pv1, pvc1 = nil, nil
			if errs := e2epv.PVPVCCleanup(c, ns, pv2, pvc2); len(errs) > 0 {
				framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pv2, pvc2 = nil, nil

			// Delete the actual disks
			if diskName1 != "" {
				framework.ExpectNoError(e2epv.DeletePDWithRetry(diskName1))
			}
			if diskName2 != "" {
				framework.ExpectNoError(e2epv.DeletePDWithRetry(diskName2))
			}
		})

		ginkgo.It("should delete a bound PVC from a clientPod, restart the kube-control-manager, and ensure the kube-controller-manager does not crash", func() {
			e2eskipper.SkipUnlessSSHKeyPresent()

			ginkgo.By("Deleting PVC for volume 2")
			err = e2epv.DeletePersistentVolumeClaim(c, pvc2.Name, ns)
			framework.ExpectNoError(err)
			pvc2 = nil

			ginkgo.By("Restarting the kube-controller-manager")
			err = framework.RestartControllerManager()
			framework.ExpectNoError(err)
			err = framework.WaitForControllerManagerUp()
			framework.ExpectNoError(err)
			framework.Logf("kube-controller-manager restarted")

			ginkgo.By("Observing the kube-controller-manager healthy for at least 2 minutes")
			// Continue checking for 2 minutes to make sure kube-controller-manager is healthy
			err = checkForControllerManagerHealthy(2 * time.Minute)
			framework.ExpectNoError(err)
		})

	})

	ginkgo.Context("when kubelet restarts", func() {
		var (
			clientPod *v1.Pod
			pv        *v1.PersistentVolume
			pvc       *v1.PersistentVolumeClaim
		)

		ginkgo.BeforeEach(func() {
			framework.Logf("Initializing test spec")
			clientPod, pv, pvc = initTestCase(f, c, nfsPVconfig, pvcConfig, ns, clientNode.Name)
		})

		ginkgo.AfterEach(func() {
			framework.Logf("Tearing down test spec")
			tearDownTestCase(c, f, ns, clientPod, pvc, pv, true /* force PV delete */)
			pv, pvc, clientPod = nil, nil, nil
		})

		// Test table housing the ginkgo.It() title string and test spec.  runTest is type testBody, defined at
		// the start of this file.  To add tests, define a function mirroring the testBody signature and assign
		// to runTest.
		disruptiveTestTable := []disruptiveTest{
			{
				testItStmt: "Should test that a file written to the mount before kubelet restart is readable after restart.",
				runTest:    utils.TestKubeletRestartsAndRestoresMount,
			},
			{
				testItStmt: "Should test that a volume mounted to a pod that is deleted while the kubelet is down unmounts when the kubelet returns.",
				runTest:    utils.TestVolumeUnmountsFromDeletedPod,
			},
			{
				testItStmt: "Should test that a volume mounted to a pod that is force deleted while the kubelet is down unmounts when the kubelet returns.",
				runTest:    utils.TestVolumeUnmountsFromForceDeletedPod,
			},
		}

		// Test loop executes each disruptiveTest iteratively.
		for _, test := range disruptiveTestTable {
			func(t disruptiveTest) {
				ginkgo.It(t.testItStmt, func() {
					ginkgo.By("Executing Spec")
					t.runTest(c, f, clientPod)
				})
			}(test)
		}
	})
})

// createGCEVolume creates PersistentVolumeSource for GCEVolume.
func createGCEVolume() (*v1.PersistentVolumeSource, string) {
	diskName, err := e2epv.CreatePDWithRetry()
	framework.ExpectNoError(err)
	return &v1.PersistentVolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:   diskName,
			FSType:   "ext3",
			ReadOnly: false,
		},
	}, diskName
}

// initTestCase initializes spec resources (pv, pvc, and pod) and returns pointers to be consumed
// by the test.
func initTestCase(f *framework.Framework, c clientset.Interface, pvConfig e2epv.PersistentVolumeConfig, pvcConfig e2epv.PersistentVolumeClaimConfig, ns, nodeName string) (*v1.Pod, *v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	pv, pvc, err := e2epv.CreatePVPVC(c, pvConfig, pvcConfig, ns, false)
	defer func() {
		if err != nil {
			e2epv.DeletePersistentVolumeClaim(c, pvc.Name, ns)
			e2epv.DeletePersistentVolume(c, pv.Name)
		}
	}()
	framework.ExpectNoError(err)
	pod := e2epod.MakePod(ns, nil, []*v1.PersistentVolumeClaim{pvc}, true, "")
	pod.Spec.NodeName = nodeName
	framework.Logf("Creating NFS client pod.")
	pod, err = c.CoreV1().Pods(ns).Create(pod)
	framework.Logf("NFS client Pod %q created on Node %q", pod.Name, nodeName)
	framework.ExpectNoError(err)
	defer func() {
		if err != nil {
			e2epod.DeletePodWithWait(c, pod)
		}
	}()
	err = e2epod.WaitForPodRunningInNamespace(c, pod)
	framework.ExpectNoError(err, fmt.Sprintf("Pod %q timed out waiting for phase: Running", pod.Name))
	// Return created api objects
	pod, err = c.CoreV1().Pods(ns).Get(pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	pvc, err = c.CoreV1().PersistentVolumeClaims(ns).Get(pvc.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	pv, err = c.CoreV1().PersistentVolumes().Get(pv.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return pod, pv, pvc
}

// tearDownTestCase destroy resources created by initTestCase.
func tearDownTestCase(c clientset.Interface, f *framework.Framework, ns string, client *v1.Pod, pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume, forceDeletePV bool) {
	// Ignore deletion errors.  Failing on them will interrupt test cleanup.
	e2epod.DeletePodWithWait(c, client)
	e2epv.DeletePersistentVolumeClaim(c, pvc.Name, ns)
	if forceDeletePV && pv != nil {
		e2epv.DeletePersistentVolume(c, pv.Name)
		return
	}
	err := framework.WaitForPersistentVolumeDeleted(c, pv.Name, 5*time.Second, 5*time.Minute)
	framework.ExpectNoError(err, "Persistent Volume %v not deleted by dynamic provisioner", pv.Name)
}
