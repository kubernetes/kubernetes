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

/*
 * This test checks that the plugin VolumeSources are working when pseudo-streaming
 * various write sizes to mounted files. Note that the plugin is defined inline in
 * the pod spec, not via a persistent volume and claim.
 *
 * These tests work only when privileged containers are allowed, exporting various
 * filesystems (NFS, GlusterFS, ...) usually needs some mounting or other privileged
 * magic in the server pod. Note that the server containers are for testing purposes
 * only and should not be used in production.
 */

package storage

import (
	"fmt"
	"math"
	"path"
	"strconv"
	"strings"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	minFileSize = 1 * framework.MiB

	fileSizeSmall  = 1 * framework.MiB
	fileSizeMedium = 100 * framework.MiB
	fileSizeLarge  = 1 * framework.GiB
)

// MD5 hashes of the test file corresponding to each file size.
// Test files are generated in testVolumeIO()
// If test file generation algorithm changes, these must be recomputed.
var md5hashes = map[int64]string{
	fileSizeSmall:  "5c34c2813223a7ca05a3c2f38c0d1710",
	fileSizeMedium: "f2fa202b1ffeedda5f3a58bd1ae81104",
	fileSizeLarge:  "8d763edc71bd16217664793b5a15e403",
}

// Return the plugin's client pod spec. Use an InitContainer to setup the file i/o test env.
func makePodSpec(config framework.VolumeTestConfig, dir, initCmd string, volsrc v1.VolumeSource, podSecContext *v1.PodSecurityContext) *v1.Pod {
	volName := fmt.Sprintf("%s-%s", config.Prefix, "io-volume")

	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Prefix + "-io-client",
			Labels: map[string]string{
				"role": config.Prefix + "-io-client",
			},
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name:  config.Prefix + "-io-init",
					Image: framework.BusyBoxImage,
					Command: []string{
						"/bin/sh",
						"-c",
						initCmd,
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volName,
							MountPath: dir,
						},
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  config.Prefix + "-io-client",
					Image: framework.BusyBoxImage,
					Command: []string{
						"/bin/sh",
						"-c",
						"sleep 3600", // keep pod alive until explicitly deleted
					},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volName,
							MountPath: dir,
						},
					},
				},
			},
			SecurityContext: podSecContext,
			Volumes: []v1.Volume{
				{
					Name:         volName,
					VolumeSource: volsrc,
				},
			},
			RestartPolicy: v1.RestartPolicyNever, // want pod to fail if init container fails
		},
	}
}

// Write `fsize` bytes to `fpath` in the pod, using dd and the `dd_input` file.
func writeToFile(pod *v1.Pod, fpath, dd_input string, fsize int64) error {
	By(fmt.Sprintf("writing %d bytes to test file %s", fsize, fpath))
	loopCnt := fsize / minFileSize
	writeCmd := fmt.Sprintf("i=0; while [ $i -lt %d ]; do dd if=%s bs=%d >>%s 2>/dev/null; let i+=1; done", loopCnt, dd_input, minFileSize, fpath)
	_, err := podExec(pod, writeCmd)

	return err
}

// Verify that the test file is the expected size and contains the expected content.
func verifyFile(pod *v1.Pod, fpath string, expectSize int64, dd_input string) error {
	By("verifying file size")
	rtnstr, err := podExec(pod, fmt.Sprintf("stat -c %%s %s", fpath))
	if err != nil || rtnstr == "" {
		return fmt.Errorf("unable to get file size via `stat %s`: %v", fpath, err)
	}
	size, err := strconv.Atoi(strings.TrimSuffix(rtnstr, "\n"))
	if err != nil {
		return fmt.Errorf("unable to convert string %q to int: %v", rtnstr, err)
	}
	if int64(size) != expectSize {
		return fmt.Errorf("size of file %s is %d, expected %d", fpath, size, expectSize)
	}

	By("verifying file hash")
	rtnstr, err = podExec(pod, fmt.Sprintf("md5sum %s | cut -d' ' -f1", fpath))
	if err != nil {
		return fmt.Errorf("unable to test file hash via `md5sum %s`: %v", fpath, err)
	}
	actualHash := strings.TrimSuffix(rtnstr, "\n")
	expectedHash, ok := md5hashes[expectSize]
	if !ok {
		return fmt.Errorf("File hash is unknown for file size %d. Was a new file size added to the test suite?",
			expectSize)
	}
	if actualHash != expectedHash {
		return fmt.Errorf("MD5 hash is incorrect for file %s with size %d. Expected: `%s`; Actual: `%s`",
			fpath, expectSize, expectedHash, actualHash)
	}

	return nil
}

// Delete `fpath` to save some disk space on host. Delete errors are logged but ignored.
func deleteFile(pod *v1.Pod, fpath string) {
	By(fmt.Sprintf("deleting test file %s...", fpath))
	_, err := podExec(pod, fmt.Sprintf("rm -f %s", fpath))
	if err != nil {
		// keep going, the test dir will be deleted when the volume is unmounted
		framework.Logf("unable to delete test file %s: %v\nerror ignored, continuing test", fpath, err)
	}
}

// Create the client pod and create files of the sizes passed in by the `fsizes` parameter. Delete the
// client pod and the new files when done.
// Note: the file name is appended to "/opt/<Prefix>/<namespace>", eg. "/opt/nfs/e2e-.../<file>".
// Note: nil can be passed for the podSecContext parm, in which case it is ignored.
// Note: `fsizes` values are enforced to each be at least `minFileSize` and a multiple of `minFileSize`
//   bytes.
func testVolumeIO(f *framework.Framework, cs clientset.Interface, config framework.VolumeTestConfig, volsrc v1.VolumeSource, podSecContext *v1.PodSecurityContext, file string, fsizes []int64) (err error) {
	dir := path.Join("/opt", config.Prefix, config.Namespace)
	dd_input := path.Join(dir, "dd_if")
	writeBlk := strings.Repeat("abcdefghijklmnopqrstuvwxyz123456", 32) // 1KiB value
	loopCnt := minFileSize / int64(len(writeBlk))
	// initContainer cmd to create and fill dd's input file. The initContainer is used to create
	// the `dd` input file which is currently 1MiB. Rather than store a 1MiB go value, a loop is
	// used to create a 1MiB file in the target directory.
	initCmd := fmt.Sprintf("i=0; while [ $i -lt %d ]; do echo -n %s >>%s; let i+=1; done", loopCnt, writeBlk, dd_input)

	clientPod := makePodSpec(config, dir, initCmd, volsrc, podSecContext)

	By(fmt.Sprintf("starting %s", clientPod.Name))
	podsNamespacer := cs.CoreV1().Pods(config.Namespace)
	clientPod, err = podsNamespacer.Create(clientPod)
	if err != nil {
		return fmt.Errorf("failed to create client pod %q: %v", clientPod.Name, err)
	}
	defer func() {
		// note the test dir will be removed when the kubelet unmounts it
		By(fmt.Sprintf("deleting client pod %q...", clientPod.Name))
		e := framework.DeletePodWithWait(f, cs, clientPod)
		if e != nil {
			framework.Logf("client pod failed to delete: %v", e)
			if err == nil { // delete err is returned if err is not set
				err = e
			}
		}
	}()

	err = framework.WaitForPodRunningInNamespace(cs, clientPod)
	if err != nil {
		return fmt.Errorf("client pod %q not running: %v", clientPod.Name, err)
	}

	// create files of the passed-in file sizes and verify test file size and content
	for _, fsize := range fsizes {
		// file sizes must be a multiple of `minFileSize`
		if math.Mod(float64(fsize), float64(minFileSize)) != 0 {
			fsize = fsize/minFileSize + minFileSize
		}
		fpath := path.Join(dir, fmt.Sprintf("%s-%d", file, fsize))
		if err = writeToFile(clientPod, fpath, dd_input, fsize); err != nil {
			return err
		}
		if err = verifyFile(clientPod, fpath, fsize, dd_input); err != nil {
			return err
		}
		deleteFile(clientPod, fpath)
	}

	return
}

// These tests need privileged containers which are disabled by default.
// TODO: support all of the plugins tested in storage/volumes.go
var _ = SIGDescribe("Volume plugin streaming [Slow]", func() {
	f := framework.NewDefaultFramework("volume-io")
	var (
		config    framework.VolumeTestConfig
		cs        clientset.Interface
		ns        string
		serverIP  string
		serverPod *v1.Pod
		volSource v1.VolumeSource
	)

	BeforeEach(func() {
		cs = f.ClientSet
		ns = f.Namespace.Name
	})

	////////////////////////////////////////////////////////////////////////
	// NFS
	////////////////////////////////////////////////////////////////////////
	Describe("NFS", func() {
		testFile := "nfs_io_test"
		// client pod uses selinux
		podSec := v1.PodSecurityContext{
			SELinuxOptions: &v1.SELinuxOptions{
				Level: "s0:c0,c1",
			},
		}

		BeforeEach(func() {
			config, serverPod, serverIP = framework.NewNFSServer(cs, ns, []string{})
			volSource = v1.VolumeSource{
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

		It("should write files of various sizes, verify size, validate content", func() {
			fileSizes := []int64{fileSizeSmall, fileSizeMedium, fileSizeLarge}
			err := testVolumeIO(f, cs, config, volSource, &podSec, testFile, fileSizes)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	////////////////////////////////////////////////////////////////////////
	// Gluster
	////////////////////////////////////////////////////////////////////////
	Describe("GlusterFS", func() {
		var name string
		testFile := "gluster_io_test"

		BeforeEach(func() {
			framework.SkipUnlessNodeOSDistroIs("gci")
			// create gluster server and endpoints
			config, serverPod, serverIP = framework.NewGlusterfsServer(cs, ns)
			name = config.Prefix + "-server"
			volSource = v1.VolumeSource{
				Glusterfs: &v1.GlusterfsVolumeSource{
					EndpointsName: name,
					// 'test_vol' comes from test/images/volumes-tester/gluster/run_gluster.sh
					Path:     "test_vol",
					ReadOnly: false,
				},
			}
		})

		AfterEach(func() {
			framework.Logf("AfterEach: deleting Gluster endpoints %q...", name)
			epErr := cs.CoreV1().Endpoints(ns).Delete(name, nil)
			framework.Logf("AfterEach: deleting Gluster server pod %q...", serverPod.Name)
			err := framework.DeletePodWithWait(f, cs, serverPod)
			if epErr != nil || err != nil {
				if epErr != nil {
					framework.Logf("AfterEach: Gluster delete endpoints failed: %v", err)
				}
				if err != nil {
					framework.Logf("AfterEach: Gluster server pod delete failed: %v", err)
				}
				framework.Failf("AfterEach: cleanup failed")
			}
		})

		It("should write files of various sizes, verify size, validate content", func() {
			fileSizes := []int64{fileSizeSmall, fileSizeMedium}
			err := testVolumeIO(f, cs, config, volSource, nil /*no secContext*/, testFile, fileSizes)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	////////////////////////////////////////////////////////////////////////
	// iSCSI
	// The iscsiadm utility and iscsi target kernel modules must be installed on all nodes.
	////////////////////////////////////////////////////////////////////////
	Describe("iSCSI [Feature:Volumes]", func() {
		testFile := "iscsi_io_test"

		BeforeEach(func() {
			config, serverPod, serverIP = framework.NewISCSIServer(cs, ns)
			volSource = v1.VolumeSource{
				ISCSI: &v1.ISCSIVolumeSource{
					TargetPortal: serverIP + ":3260",
					// from test/images/volumes-tester/iscsi/initiatorname.iscsi
					IQN:      "iqn.2003-01.org.linux-iscsi.f21.x8664:sn.4b0aae584f7c",
					Lun:      0,
					FSType:   "ext2",
					ReadOnly: false,
				},
			}
		})

		AfterEach(func() {
			framework.Logf("AfterEach: deleting iSCSI server pod %q...", serverPod.Name)
			err := framework.DeletePodWithWait(f, cs, serverPod)
			Expect(err).NotTo(HaveOccurred(), "AfterEach: iSCSI server pod failed to delete")
		})

		It("should write files of various sizes, verify size, validate content", func() {
			fileSizes := []int64{fileSizeSmall, fileSizeMedium}
			fsGroup := int64(1234)
			podSec := v1.PodSecurityContext{
				FSGroup: &fsGroup,
			}
			err := testVolumeIO(f, cs, config, volSource, &podSec, testFile, fileSizes)
			Expect(err).NotTo(HaveOccurred())
		})
	})

	////////////////////////////////////////////////////////////////////////
	// Ceph RBD
	////////////////////////////////////////////////////////////////////////
	Describe("Ceph-RBD [Feature:Volumes]", func() {
		var (
			secret *v1.Secret
			name   string
		)
		testFile := "ceph-rbd_io_test"

		BeforeEach(func() {
			config, serverPod, serverIP = framework.NewRBDServer(cs, ns)
			name = config.Prefix + "-server"

			// create server secret
			secret = &v1.Secret{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Secret",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: name,
				},
				Data: map[string][]byte{
					// from test/images/volumes-tester/rbd/keyring
					"key": []byte("AQDRrKNVbEevChAAEmRC+pW/KBVHxa0w/POILA=="),
				},
				Type: "kubernetes.io/rbd",
			}
			var err error
			secret, err = cs.CoreV1().Secrets(ns).Create(secret)
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("BeforeEach: failed to create secret %q for Ceph-RBD: %v", name, err))

			volSource = v1.VolumeSource{
				RBD: &v1.RBDVolumeSource{
					CephMonitors: []string{serverIP},
					RBDPool:      "rbd",
					RBDImage:     "foo",
					RadosUser:    "admin",
					SecretRef: &v1.LocalObjectReference{
						Name: name,
					},
					FSType:   "ext2",
					ReadOnly: true,
				},
			}
		})

		AfterEach(func() {
			framework.Logf("AfterEach: deleting Ceph-RDB server secret %q...", name)
			secErr := cs.CoreV1().Secrets(ns).Delete(name, &metav1.DeleteOptions{})
			framework.Logf("AfterEach: deleting Ceph-RDB server pod %q...", serverPod.Name)
			err := framework.DeletePodWithWait(f, cs, serverPod)
			if secErr != nil || err != nil {
				if secErr != nil {
					framework.Logf("AfterEach: Ceph-RDB delete secret failed: %v", err)
				}
				if err != nil {
					framework.Logf("AfterEach: Ceph-RDB server pod delete failed: %v", err)
				}
				framework.Failf("AfterEach: cleanup failed")
			}
		})

		It("should write files of various sizes, verify size, validate content", func() {
			fileSizes := []int64{fileSizeSmall, fileSizeMedium}
			fsGroup := int64(1234)
			podSec := v1.PodSecurityContext{
				FSGroup: &fsGroup,
			}
			err := testVolumeIO(f, cs, config, volSource, &podSec, testFile, fileSizes)
			Expect(err).NotTo(HaveOccurred())
		})
	})
})
