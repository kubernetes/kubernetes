/*
Copyright 2018 The Kubernetes Authors.

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

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	noProvisioner = "kubernetes.io/no-provisioner"
	pvNamePrefix  = "pv"
)

func generateConfigsForStaticProvisionPVTest(scName string, volBindMode storagev1.VolumeBindingMode,
	volMode v1.PersistentVolumeMode, pvSource v1.PersistentVolumeSource) (*storagev1.StorageClass,
	framework.PersistentVolumeConfig, framework.PersistentVolumeClaimConfig) {
	// StorageClass
	scConfig := &storagev1.StorageClass{
		ObjectMeta: metav1.ObjectMeta{
			Name: scName,
		},
		Provisioner:       noProvisioner,
		VolumeBindingMode: &volBindMode,
	}
	// PV
	pvConfig := framework.PersistentVolumeConfig{
		PVSource:         pvSource,
		NamePrefix:       pvNamePrefix,
		StorageClassName: scName,
		VolumeMode:       &volMode,
	}
	// PVC
	pvcConfig := framework.PersistentVolumeClaimConfig{
		AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		StorageClassName: &scName,
		VolumeMode:       &volMode,
	}

	return scConfig, pvConfig, pvcConfig
}

func createPVTestResource(cs clientset.Interface, ns string,
	scConfig *storagev1.StorageClass, pvConfig framework.PersistentVolumeConfig,
	pvcConfig framework.PersistentVolumeClaimConfig) (*storagev1.StorageClass, *v1.Pod, *v1.PersistentVolume, *v1.PersistentVolumeClaim) {

	By("Creating sc")
	sc, err := cs.StorageV1().StorageClasses().Create(scConfig)
	Expect(err).NotTo(HaveOccurred())

	By("Creating pv and pvc")
	pv, pvc, err := framework.CreatePVPVC(cs, pvConfig, pvcConfig, ns, false)
	framework.ExpectNoError(err)
	framework.ExpectNoError(framework.WaitOnPVandPVC(cs, ns, pv, pvc))

	By("Creating a pod")
	// TODO(mkimuram): Need to set anti-affinity with storage server pod.
	// Otherwise, storage server pod can also be affected on destructive tests.
	pod, err := framework.CreateSecPod(cs, ns, []*v1.PersistentVolumeClaim{pvc}, false, "", false, false, framework.SELinuxLabel, nil, framework.PodStartTimeout)
	Expect(err).NotTo(HaveOccurred())

	return sc, pod, pv, pvc
}

func createPVTestResourceWithFailure(cs clientset.Interface, ns string,
	scConfig *storagev1.StorageClass, pvConfig framework.PersistentVolumeConfig,
	pvcConfig framework.PersistentVolumeClaimConfig) (*storagev1.StorageClass, *v1.Pod, *v1.PersistentVolume, *v1.PersistentVolumeClaim) {

	By("Creating sc")
	sc, err := cs.StorageV1().StorageClasses().Create(scConfig)
	Expect(err).NotTo(HaveOccurred())

	By("Creating pv and pvc")
	pv, pvc, err := framework.CreatePVPVC(cs, pvConfig, pvcConfig, ns, false)
	framework.ExpectNoError(err)
	framework.ExpectNoError(framework.WaitOnPVandPVC(cs, ns, pv, pvc))

	By("Creating a pod")
	pod, err := framework.CreateSecPod(cs, ns, []*v1.PersistentVolumeClaim{pvc}, false, "", false, false, framework.SELinuxLabel, nil, framework.PodStartTimeout)
	Expect(err).To(HaveOccurred())

	return sc, pod, pv, pvc
}

func deletePVTestResource(f *framework.Framework, cs clientset.Interface, ns string, sc *storagev1.StorageClass,
	pod *v1.Pod, pv *v1.PersistentVolume, pvc *v1.PersistentVolumeClaim) {
	By("Deleting pod")
	framework.ExpectNoError(framework.DeletePodWithWait(f, cs, pod))

	By("Deleting pv and pvc")
	errs := framework.PVPVCCleanup(cs, ns, pv, pvc)
	if len(errs) > 0 {
		framework.Failf("Failed to delete PV and/or PVC: %v", utilerrors.NewAggregate(errs))
	}

	By("Deleting sc")
	framework.ExpectNoError(cs.StorageV1().StorageClasses().Delete(sc.Name, nil))
}

func checkVolumeModeOfPath(pod *v1.Pod, volMode v1.PersistentVolumeMode, path string) {
	if volMode == v1.PersistentVolumeBlock {
		// Check if block exists
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("test -b %s", path))

		// Double check that it's not directory
		utils.VerifyExecInPodFail(pod, fmt.Sprintf("test -d %s", path), 1)
	} else {
		// Check if directory exists
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("test -d %s", path))

		// Double check that it's not block
		utils.VerifyExecInPodFail(pod, fmt.Sprintf("test -b %s", path), 1)
	}
}

func checkReadWriteToPath(pod *v1.Pod, volMode v1.PersistentVolumeMode, path string) {
	if volMode == v1.PersistentVolumeBlock {
		// random -> file1
		utils.VerifyExecInPodSucceed(pod, "dd if=/dev/urandom of=/tmp/file1 bs=64 count=1")
		// file1 -> dev (write to dev)
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("dd if=/tmp/file1 of=%s bs=64 count=1", path))
		// dev -> file2 (read from dev)
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("dd if=%s of=/tmp/file2 bs=64 count=1", path))
		// file1 == file2 (check contents)
		utils.VerifyExecInPodSucceed(pod, "diff /tmp/file1 /tmp/file2")
		// Clean up temp files
		utils.VerifyExecInPodSucceed(pod, "rm -f /tmp/file1 /tmp/file2")

		// Check that writing file to block volume fails
		utils.VerifyExecInPodFail(pod, fmt.Sprintf("echo 'Hello world.' > %s/file1.txt", path), 1)
	} else {
		// text -> file1 (write to file)
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("echo 'Hello world.' > %s/file1.txt", path))
		// grep file1 (read from file and check contents)
		utils.VerifyExecInPodSucceed(pod, fmt.Sprintf("grep 'Hello world.' %s/file1.txt", path))

		// Check that writing to directory as block volume fails
		utils.VerifyExecInPodFail(pod, fmt.Sprintf("dd if=/dev/urandom of=%s bs=64 count=1", path), 1)
	}
}

func skipBlockSupportTestIfUnsupported(volMode v1.PersistentVolumeMode, isBlockSupported bool) {
	if volMode == v1.PersistentVolumeBlock && !isBlockSupported {
		framework.Skipf("Skip assertion for block test for block supported plugin.(Block unsupported)")
	}
}

func skipBlockUnsupportTestUnlessUnspported(volMode v1.PersistentVolumeMode, isBlockSupported bool) {
	if !(volMode == v1.PersistentVolumeBlock && !isBlockSupported) {
		framework.Skipf("Skip assertion for block test for block unsupported plugin.(Block suppported or FileSystem test)")
	}
}

var _ = utils.SIGDescribe("PersistentVolumes-volumeMode", func() {
	f := framework.NewDefaultFramework("pv-volmode")
	const (
		pvTestSCPrefix = "pvtest"
	)

	var (
		cs               clientset.Interface
		ns               string
		scName           string
		isBlockSupported bool
		serverIP         string
		secret           *v1.Secret
		serverPod        *v1.Pod
		pvSource         v1.PersistentVolumeSource
		sc               *storagev1.StorageClass
		pod              *v1.Pod
		pv               *v1.PersistentVolume
		pvc              *v1.PersistentVolumeClaim
		volMode          v1.PersistentVolumeMode
		volBindMode      storagev1.VolumeBindingMode
	)

	BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
		volBindMode = storagev1.VolumeBindingImmediate
	})

	AssertCreateDeletePodAndReadWriteVolume := func() {
		// For block supported plugins
		It("should create sc, pod, pv, and pvc, read/write to the pv, and delete all created resources", func() {
			skipBlockSupportTestIfUnsupported(volMode, isBlockSupported)

			scConfig, pvConfig, pvcConfig := generateConfigsForStaticProvisionPVTest(scName, volBindMode, volMode, pvSource)
			sc, pod, pv, pvc = createPVTestResource(cs, ns, scConfig, pvConfig, pvcConfig)
			defer deletePVTestResource(f, cs, ns, sc, pod, pv, pvc)

			By("Checking if persistent volume exists as expected volume mode")
			checkVolumeModeOfPath(pod, volMode, "/mnt/volume1")

			By("Checking if read/write to persistent volume works properly")
			checkReadWriteToPath(pod, volMode, "/mnt/volume1")
		})

		// For block unsupported plugins
		It("should fail to create pod by failing to mount volume", func() {
			skipBlockUnsupportTestUnlessUnspported(volMode, isBlockSupported)

			scConfig, pvConfig, pvcConfig := generateConfigsForStaticProvisionPVTest(scName, volBindMode, volMode, pvSource)
			sc, pod, pv, pvc = createPVTestResourceWithFailure(cs, ns, scConfig, pvConfig, pvcConfig)
			deletePVTestResource(f, cs, ns, sc, pod, pv, pvc)
		})
	}

	verifyAll := func() {
		AssertCreateDeletePodAndReadWriteVolume()
		// TODO(mkimuram): Add more tests
	}

	Describe("NFS", func() {
		const pvTestNFSSCSuffix = "nfs"

		BeforeEach(func() {
			isBlockSupported = false
			scName = fmt.Sprintf("%v-%v-%v", pvTestSCPrefix, ns, pvTestNFSSCSuffix)
			_, serverPod, serverIP = framework.NewNFSServer(cs, ns, []string{})

			pvSource = v1.PersistentVolumeSource{
				NFS: &v1.NFSVolumeSource{
					Server:   serverIP,
					Path:     "/",
					ReadOnly: false,
				},
			}
		})

		AfterEach(func() {
			framework.Logf("AfterEach: deleting NFS server pod %q...", serverPod.Name)
			err := framework.DeletePodWithWait(f, cs, serverPod)
			Expect(err).NotTo(HaveOccurred(), "AfterEach: NFS server pod failed to delete")
		})

		Context("FileSystem volume Test", func() {
			BeforeEach(func() {
				volMode = v1.PersistentVolumeFilesystem
			})

			verifyAll()
		})

		Context("Block volume Test[Feature:BlockVolume]", func() {
			BeforeEach(func() {
				volMode = v1.PersistentVolumeBlock
			})

			verifyAll()
		})
	})

	Describe("iSCSI [Feature:Volumes]", func() {
		const pvTestISCSISCSuffix = "iscsi"

		BeforeEach(func() {
			isBlockSupported = true
			scName = fmt.Sprintf("%v-%v-%v", pvTestSCPrefix, ns, pvTestISCSISCSuffix)
			_, serverPod, serverIP = framework.NewISCSIServer(cs, ns)

			pvSource = v1.PersistentVolumeSource{
				ISCSI: &v1.ISCSIPersistentVolumeSource{
					TargetPortal: serverIP + ":3260",
					IQN:          "iqn.2003-01.org.linux-iscsi.f21.x8664:sn.4b0aae584f7c",
					Lun:          0,
				},
			}
		})

		AfterEach(func() {
			framework.Logf("AfterEach: deleting iSCSI server pod %q...", serverPod.Name)
			err := framework.DeletePodWithWait(f, cs, serverPod)
			Expect(err).NotTo(HaveOccurred(), "AfterEach: iSCSI server pod failed to delete")
		})

		Context("FileSystem volume Test", func() {
			BeforeEach(func() {
				volMode = v1.PersistentVolumeFilesystem
			})

			verifyAll()
		})

		Context("Block volume Test[Feature:BlockVolume]", func() {
			BeforeEach(func() {
				volMode = v1.PersistentVolumeBlock
			})

			verifyAll()
		})
	})

	Describe("Ceph-RBD [Feature:Volumes]", func() {
		const pvTestRBDSCSuffix = "rbd"

		BeforeEach(func() {
			isBlockSupported = true
			scName = fmt.Sprintf("%v-%v-%v", pvTestSCPrefix, ns, pvTestRBDSCSuffix)
			_, serverPod, secret, serverIP = framework.NewRBDServer(cs, ns)

			framework.Logf("namespace: %v, secret.Name: %v", ns, secret.Name)
			pvSource = v1.PersistentVolumeSource{
				RBD: &v1.RBDPersistentVolumeSource{
					CephMonitors: []string{serverIP},
					RBDPool:      "rbd",
					RBDImage:     "foo",
					RadosUser:    "admin",
					SecretRef: &v1.SecretReference{
						Name:      secret.Name,
						Namespace: ns,
					},
					ReadOnly: false,
				},
			}
		})

		AfterEach(func() {
			framework.Logf("AfterEach: deleting Ceph-RDB server secret %q...", secret.Name)
			secErr := cs.CoreV1().Secrets(ns).Delete(secret.Name, &metav1.DeleteOptions{})
			framework.Logf("AfterEach: deleting Ceph-RDB server pod %q...", serverPod.Name)
			err := framework.DeletePodWithWait(f, cs, serverPod)
			if secErr != nil || err != nil {
				if secErr != nil {
					framework.Logf("AfterEach: Ceph-RDB delete secret failed: %v", secErr)
				}
				if err != nil {
					framework.Logf("AfterEach: Ceph-RDB server pod delete failed: %v", err)
				}
				framework.Failf("AfterEach: cleanup failed")
			}
		})

		Context("FileSystem volume Test", func() {
			BeforeEach(func() {
				volMode = v1.PersistentVolumeFilesystem
			})

			verifyAll()
		})

		Context("Block volume Test[Feature:BlockVolume]", func() {
			BeforeEach(func() {
				volMode = v1.PersistentVolumeBlock
			})

			verifyAll()
		})
	})

	Describe("CephFS [Feature:Volumes]", func() {
		const pvTestCephFSSCSuffix = "cephfs"

		BeforeEach(func() {
			isBlockSupported = false
			scName = fmt.Sprintf("%v-%v-%v", pvTestSCPrefix, ns, pvTestCephFSSCSuffix)
			_, serverPod, secret, serverIP = framework.NewRBDServer(cs, ns)

			pvSource = v1.PersistentVolumeSource{
				CephFS: &v1.CephFSPersistentVolumeSource{
					Monitors: []string{serverIP + ":6789"},
					User:     "kube",
					SecretRef: &v1.SecretReference{
						Name:      secret.Name,
						Namespace: ns,
					},
					ReadOnly: false,
				},
			}
		})

		AfterEach(func() {
			framework.Logf("AfterEach: deleting CephFS server secret %q...", secret.Name)
			secErr := cs.CoreV1().Secrets(ns).Delete(secret.Name, &metav1.DeleteOptions{})
			framework.Logf("AfterEach: deleting CephFS server pod %q...", serverPod.Name)
			err := framework.DeletePodWithWait(f, cs, serverPod)
			if secErr != nil || err != nil {
				if secErr != nil {
					framework.Logf("AfterEach: CephFS delete secret failed: %v", secErr)
				}
				if err != nil {
					framework.Logf("AfterEach: CephFS server pod delete failed: %v", err)
				}
				framework.Failf("AfterEach: cleanup failed")
			}
		})

		Context("FileSystem volume Test", func() {
			BeforeEach(func() {
				volMode = v1.PersistentVolumeFilesystem
			})

			verifyAll()
		})

		Context("Block volume Test[Feature:BlockVolume]", func() {
			BeforeEach(func() {
				volMode = v1.PersistentVolumeBlock
			})

			verifyAll()
		})
	})

})
