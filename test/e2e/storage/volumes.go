/*
Copyright 2015 The Kubernetes Authors.

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
 * This test checks that various VolumeSources are working.
 *
 * There are two ways, how to test the volumes:
 * 1) With containerized server (NFS, Ceph, Gluster, iSCSI, ...)
 * The test creates a server pod, exporting simple 'index.html' file.
 * Then it uses appropriate VolumeSource to import this file into a client pod
 * and checks that the pod can see the file. It does so by importing the file
 * into web server root and loadind the index.html from it.
 *
 * These tests work only when privileged containers are allowed, exporting
 * various filesystems (NFS, GlusterFS, ...) usually needs some mounting or
 * other privileged magic in the server pod.
 *
 * Note that the server containers are for testing purposes only and should not
 * be used in production.
 *
 * 2) With server outside of Kubernetes (Cinder, ...)
 * Appropriate server (e.g. OpenStack Cinder) must exist somewhere outside
 * the tested Kubernetes cluster. The test itself creates a new volume,
 * and checks, that Kubernetes can use it as a volume.
 */

// test/e2e/common/volumes.go duplicates the GlusterFS test from this file.  Any changes made to this
// test should be made there as well.

package storage

import (
	"os/exec"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere"
	"k8s.io/kubernetes/test/e2e/framework"
)

func DeleteCinderVolume(name string) error {
	// Try to delete the volume for several seconds - it takes
	// a while for the plugin to detach it.
	var output []byte
	var err error
	timeout := time.Second * 120

	framework.Logf("Waiting up to %v for removal of cinder volume %s", timeout, name)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(5 * time.Second) {
		output, err = exec.Command("cinder", "delete", name).CombinedOutput()
		if err == nil {
			framework.Logf("Cinder volume %s deleted", name)
			return nil
		} else {
			framework.Logf("Failed to delete volume %s: %v", name, err)
		}
	}
	framework.Logf("Giving up deleting volume %s: %v\n%s", name, err, string(output[:]))
	return err
}

// These tests need privileged containers, which are disabled by default.
var _ = framework.KubeDescribe("Volumes [Volume]", func() {
	f := framework.NewDefaultFramework("volume")

	// If 'false', the test won't clear its volumes upon completion. Useful for debugging,
	// note that namespace deletion is handled by delete-namespace flag
	clean := true
	// filled in BeforeEach
	var cs clientset.Interface
	var namespace *v1.Namespace

	BeforeEach(func() {
		cs = f.ClientSet
		namespace = f.Namespace
	})

	////////////////////////////////////////////////////////////////////////
	// NFS
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("NFS", func() {
		It("should be mountable", func() {
			config := framework.VolumeTestConfig{
				Namespace:   namespace.Name,
				Prefix:      "nfs",
				ServerImage: framework.NfsServerImage,
				ServerPorts: []int{2049},
			}

			defer func() {
				if clean {
					framework.VolumeTestCleanup(f, config)
				}
			}()
			pod := framework.StartVolumeServer(cs, config)
			serverIP := pod.Status.PodIP
			framework.Logf("NFS server IP address: %v", serverIP)

			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						NFS: &v1.NFSVolumeSource{
							Server:   serverIP,
							Path:     "/",
							ReadOnly: true,
						},
					},
					File: "index.html",
					// Must match content of test/images/volumes-tester/nfs/index.html
					ExpectedContent: "Hello from NFS!",
				},
			}
			framework.TestVolumeClient(cs, config, nil, tests)
		})
	})

	////////////////////////////////////////////////////////////////////////
	// Gluster
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("GlusterFS [Feature:Volumes]", func() {
		It("should be mountable", func() {
			//TODO (copejon) GFS is not supported on debian image.
			framework.SkipUnlessNodeOSDistroIs("gci")

			config := framework.VolumeTestConfig{
				Namespace:   namespace.Name,
				Prefix:      "gluster",
				ServerImage: framework.GlusterfsServerImage,
				ServerPorts: []int{24007, 24008, 49152},
			}

			defer func() {
				if clean {
					framework.VolumeTestCleanup(f, config)
				}
			}()
			pod := framework.StartVolumeServer(cs, config)
			serverIP := pod.Status.PodIP
			framework.Logf("Gluster server IP address: %v", serverIP)

			// create Endpoints for the server
			endpoints := v1.Endpoints{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Endpoints",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: config.Prefix + "-server",
				},
				Subsets: []v1.EndpointSubset{
					{
						Addresses: []v1.EndpointAddress{
							{
								IP: serverIP,
							},
						},
						Ports: []v1.EndpointPort{
							{
								Name:     "gluster",
								Port:     24007,
								Protocol: v1.ProtocolTCP,
							},
						},
					},
				},
			}

			endClient := cs.Core().Endpoints(config.Namespace)

			defer func() {
				if clean {
					endClient.Delete(config.Prefix+"-server", nil)
				}
			}()

			if _, err := endClient.Create(&endpoints); err != nil {
				framework.Failf("Failed to create endpoints for Gluster server: %v", err)
			}

			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						Glusterfs: &v1.GlusterfsVolumeSource{
							EndpointsName: config.Prefix + "-server",
							// 'test_vol' comes from test/images/volumes-tester/gluster/run_gluster.sh
							Path:     "test_vol",
							ReadOnly: true,
						},
					},
					File: "index.html",
					// Must match content of test/images/volumes-tester/gluster/index.html
					ExpectedContent: "Hello from GlusterFS!",
				},
			}
			framework.TestVolumeClient(cs, config, nil, tests)
		})
	})

	////////////////////////////////////////////////////////////////////////
	// iSCSI
	////////////////////////////////////////////////////////////////////////

	// The test needs privileged containers, which are disabled by default.
	// Also, make sure that iscsiadm utility and iscsi target kernel modules
	// are installed on all nodes!
	// Run the test with "go run hack/e2e.go ... --ginkgo.focus=iSCSI"

	framework.KubeDescribe("iSCSI [Feature:Volumes]", func() {
		It("should be mountable", func() {
			config := framework.VolumeTestConfig{
				Namespace:   namespace.Name,
				Prefix:      "iscsi",
				ServerImage: framework.IscsiServerImage,
				ServerPorts: []int{3260},
				ServerVolumes: map[string]string{
					// iSCSI container needs to insert modules from the host
					"/lib/modules": "/lib/modules",
				},
			}

			defer func() {
				if clean {
					framework.VolumeTestCleanup(f, config)
				}
			}()
			pod := framework.StartVolumeServer(cs, config)
			serverIP := pod.Status.PodIP
			framework.Logf("iSCSI server IP address: %v", serverIP)

			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						ISCSI: &v1.ISCSIVolumeSource{
							TargetPortal: serverIP + ":3260",
							// from test/images/volumes-tester/iscsi/initiatorname.iscsi
							IQN:    "iqn.2003-01.org.linux-iscsi.f21.x8664:sn.4b0aae584f7c",
							Lun:    0,
							FSType: "ext2",
						},
					},
					File: "index.html",
					// Must match content of test/images/volumes-tester/iscsi/block.tar.gz
					ExpectedContent: "Hello from iSCSI",
				},
			}
			fsGroup := int64(1234)
			framework.TestVolumeClient(cs, config, &fsGroup, tests)
		})
	})

	////////////////////////////////////////////////////////////////////////
	// Ceph RBD
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("Ceph RBD [Feature:Volumes]", func() {
		It("should be mountable", func() {
			config := framework.VolumeTestConfig{
				Namespace:   namespace.Name,
				Prefix:      "rbd",
				ServerImage: framework.RbdServerImage,
				ServerPorts: []int{6789},
				ServerVolumes: map[string]string{
					// iSCSI container needs to insert modules from the host
					"/lib/modules": "/lib/modules",
					"/sys":         "/sys",
				},
			}

			defer func() {
				if clean {
					framework.VolumeTestCleanup(f, config)
				}
			}()
			pod := framework.StartVolumeServer(cs, config)
			serverIP := pod.Status.PodIP
			framework.Logf("Ceph server IP address: %v", serverIP)

			// create secrets for the server
			secret := v1.Secret{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Secret",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: config.Prefix + "-secret",
				},
				Data: map[string][]byte{
					// from test/images/volumes-tester/rbd/keyring
					"key": []byte("AQDRrKNVbEevChAAEmRC+pW/KBVHxa0w/POILA=="),
				},
				Type: "kubernetes.io/rbd",
			}

			secClient := cs.Core().Secrets(config.Namespace)

			defer func() {
				if clean {
					secClient.Delete(config.Prefix+"-secret", nil)
				}
			}()

			if _, err := secClient.Create(&secret); err != nil {
				framework.Failf("Failed to create secrets for Ceph RBD: %v", err)
			}

			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						RBD: &v1.RBDVolumeSource{
							CephMonitors: []string{serverIP},
							RBDPool:      "rbd",
							RBDImage:     "foo",
							RadosUser:    "admin",
							SecretRef: &v1.LocalObjectReference{
								Name: config.Prefix + "-secret",
							},
							FSType: "ext2",
						},
					},
					File: "index.html",
					// Must match content of test/images/volumes-tester/rbd/create_block.sh
					ExpectedContent: "Hello from RBD",
				},
			}
			fsGroup := int64(1234)
			framework.TestVolumeClient(cs, config, &fsGroup, tests)
		})
	})
	////////////////////////////////////////////////////////////////////////
	// Ceph
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("CephFS [Feature:Volumes]", func() {
		It("should be mountable", func() {
			config := framework.VolumeTestConfig{
				Namespace:   namespace.Name,
				Prefix:      "cephfs",
				ServerImage: framework.CephServerImage,
				ServerPorts: []int{6789},
			}

			defer func() {
				if clean {
					framework.VolumeTestCleanup(f, config)
				}
			}()
			pod := framework.StartVolumeServer(cs, config)
			serverIP := pod.Status.PodIP
			framework.Logf("Ceph server IP address: %v", serverIP)
			By("sleeping a bit to give ceph server time to initialize")
			time.Sleep(20 * time.Second)

			// create ceph secret
			secret := &v1.Secret{
				TypeMeta: metav1.TypeMeta{
					Kind:       "Secret",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: config.Prefix + "-secret",
				},
				// Must use the ceph keyring at contrib/for-tests/volumes-ceph/ceph/init.sh
				// and encode in base64
				Data: map[string][]byte{
					"key": []byte("AQAMgXhVwBCeDhAA9nlPaFyfUSatGD4drFWDvQ=="),
				},
				Type: "kubernetes.io/cephfs",
			}

			defer func() {
				if clean {
					if err := cs.Core().Secrets(namespace.Name).Delete(secret.Name, nil); err != nil {
						framework.Failf("unable to delete secret %v: %v", secret.Name, err)
					}
				}
			}()

			var err error
			if secret, err = cs.Core().Secrets(namespace.Name).Create(secret); err != nil {
				framework.Failf("unable to create test secret %s: %v", secret.Name, err)
			}

			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						CephFS: &v1.CephFSVolumeSource{
							Monitors:  []string{serverIP + ":6789"},
							User:      "kube",
							SecretRef: &v1.LocalObjectReference{Name: config.Prefix + "-secret"},
							ReadOnly:  true,
						},
					},
					File: "index.html",
					// Must match content of test/images/volumes-tester/ceph/index.html
					ExpectedContent: "Hello Ceph!",
				},
			}
			framework.TestVolumeClient(cs, config, nil, tests)
		})
	})

	////////////////////////////////////////////////////////////////////////
	// OpenStack Cinder
	////////////////////////////////////////////////////////////////////////

	// This test assumes that OpenStack client tools are installed
	// (/usr/bin/nova, /usr/bin/cinder and /usr/bin/keystone)
	// and that the usual OpenStack authentication env. variables are set
	// (OS_USERNAME, OS_PASSWORD, OS_TENANT_NAME at least).

	framework.KubeDescribe("Cinder [Feature:Volumes]", func() {
		It("should be mountable", func() {
			framework.SkipUnlessProviderIs("openstack")
			config := framework.VolumeTestConfig{
				Namespace: namespace.Name,
				Prefix:    "cinder",
			}

			// We assume that namespace.Name is a random string
			volumeName := namespace.Name
			By("creating a test Cinder volume")
			output, err := exec.Command("cinder", "create", "--display-name="+volumeName, "1").CombinedOutput()
			outputString := string(output[:])
			framework.Logf("cinder output:\n%s", outputString)
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				// Ignore any cleanup errors, there is not much we can do about
				// them. They were already logged.
				DeleteCinderVolume(volumeName)
			}()

			// Parse 'id'' from stdout. Expected format:
			// |     attachments     |                  []                  |
			// |  availability_zone  |                 nova                 |
			// ...
			// |          id         | 1d6ff08f-5d1c-41a4-ad72-4ef872cae685 |
			volumeID := ""
			for _, line := range strings.Split(outputString, "\n") {
				fields := strings.Fields(line)
				if len(fields) != 5 {
					continue
				}
				if fields[1] != "id" {
					continue
				}
				volumeID = fields[3]
				break
			}
			framework.Logf("Volume ID: %s", volumeID)
			Expect(volumeID).NotTo(Equal(""))

			defer func() {
				if clean {
					framework.Logf("Running volumeTestCleanup")
					framework.VolumeTestCleanup(f, config)
				}
			}()

			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						Cinder: &v1.CinderVolumeSource{
							VolumeID: volumeID,
							FSType:   "ext3",
							ReadOnly: false,
						},
					},
					File: "index.html",
					// Randomize index.html to make sure we don't see the
					// content from previous test runs.
					ExpectedContent: "Hello from Cinder from namespace " + volumeName,
				},
			}

			framework.InjectHtml(cs, config, tests[0].Volume, tests[0].ExpectedContent)

			fsGroup := int64(1234)
			framework.TestVolumeClient(cs, config, &fsGroup, tests)
		})
	})

	////////////////////////////////////////////////////////////////////////
	// GCE PD
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("PD", func() {
		// Flaky issue: #43977
		It("should be mountable [Flaky]", func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			config := framework.VolumeTestConfig{
				Namespace: namespace.Name,
				Prefix:    "pd",
			}

			By("creating a test gce pd volume")
			volumeName, err := framework.CreatePDWithRetry()
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				// - Get NodeName from the pod spec to which the volume is mounted.
				// - Force detach and delete.
				pod, err := f.PodClient().Get(config.Prefix+"-client", metav1.GetOptions{})
				Expect(err).NotTo(HaveOccurred(), "Failed getting pod %q.", config.Prefix+"-client")
				detachAndDeletePDs(volumeName, []types.NodeName{types.NodeName(pod.Spec.NodeName)})
			}()

			defer func() {
				if clean {
					framework.Logf("Running volumeTestCleanup")
					framework.VolumeTestCleanup(f, config)
				}
			}()

			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName:   volumeName,
							FSType:   "ext3",
							ReadOnly: false,
						},
					},
					File: "index.html",
					// Randomize index.html to make sure we don't see the
					// content from previous test runs.
					ExpectedContent: "Hello from GCE from namespace " + volumeName,
				},
			}

			framework.InjectHtml(cs, config, tests[0].Volume, tests[0].ExpectedContent)

			fsGroup := int64(1234)
			framework.TestVolumeClient(cs, config, &fsGroup, tests)
		})
	})

	////////////////////////////////////////////////////////////////////////
	// ConfigMap
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("ConfigMap", func() {
		It("should be mountable", func() {
			config := framework.VolumeTestConfig{
				Namespace: namespace.Name,
				Prefix:    "configmap",
			}

			defer func() {
				if clean {
					framework.VolumeTestCleanup(f, config)
				}
			}()
			configMap := &v1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ConfigMap",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: config.Prefix + "-map",
				},
				Data: map[string]string{
					"first":  "this is the first file",
					"second": "this is the second file",
					"third":  "this is the third file",
				},
			}
			if _, err := cs.Core().ConfigMaps(namespace.Name).Create(configMap); err != nil {
				framework.Failf("unable to create test configmap: %v", err)
			}
			defer func() {
				_ = cs.Core().ConfigMaps(namespace.Name).Delete(configMap.Name, nil)
			}()

			// Test one ConfigMap mounted several times to test #28502
			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{
								Name: config.Prefix + "-map",
							},
							Items: []v1.KeyToPath{
								{
									Key:  "first",
									Path: "firstfile",
								},
							},
						},
					},
					File:            "firstfile",
					ExpectedContent: "this is the first file",
				},
				{
					Volume: v1.VolumeSource{
						ConfigMap: &v1.ConfigMapVolumeSource{
							LocalObjectReference: v1.LocalObjectReference{
								Name: config.Prefix + "-map",
							},
							Items: []v1.KeyToPath{
								{
									Key:  "second",
									Path: "secondfile",
								},
							},
						},
					},
					File:            "secondfile",
					ExpectedContent: "this is the second file",
				},
			}
			framework.TestVolumeClient(cs, config, nil, tests)
		})
	})

	////////////////////////////////////////////////////////////////////////
	// vSphere
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("vsphere [Feature:Volumes]", func() {
		It("should be mountable", func() {
			framework.SkipUnlessProviderIs("vsphere")
			var (
				volumePath string
			)
			config := framework.VolumeTestConfig{
				Namespace: namespace.Name,
				Prefix:    "vsphere",
			}
			By("creating a test vsphere volume")
			vsp, err := vsphere.GetVSphere()
			Expect(err).NotTo(HaveOccurred())

			volumePath, err = createVSphereVolume(vsp, nil)
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				vsp.DeleteVolume(volumePath)
			}()

			defer func() {
				if clean {
					framework.Logf("Running volumeTestCleanup")
					framework.VolumeTestCleanup(f, config)
				}
			}()

			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						VsphereVolume: &v1.VsphereVirtualDiskVolumeSource{
							VolumePath: volumePath,
							FSType:     "ext4",
						},
					},
					File: "index.html",
					// Randomize index.html to make sure we don't see the
					// content from previous test runs.
					ExpectedContent: "Hello from vSphere from namespace " + namespace.Name,
				},
			}

			framework.InjectHtml(cs, config, tests[0].Volume, tests[0].ExpectedContent)

			fsGroup := int64(1234)
			framework.TestVolumeClient(cs, config, &fsGroup, tests)
		})
	})
	////////////////////////////////////////////////////////////////////////
	// Azure Disk
	////////////////////////////////////////////////////////////////////////
	framework.KubeDescribe("Azure Disk [Feature:Volumes]", func() {
		It("should be mountable [Slow]", func() {
			framework.SkipUnlessProviderIs("azure")
			config := framework.VolumeTestConfig{
				Namespace: namespace.Name,
				Prefix:    "azure",
			}

			By("creating a test azure disk volume")
			volumeName, err := framework.CreatePDWithRetry()
			Expect(err).NotTo(HaveOccurred())
			defer func() {
				framework.DeletePDWithRetry(volumeName)
			}()

			defer func() {
				if clean {
					framework.Logf("Running volumeTestCleanup")
					framework.VolumeTestCleanup(f, config)
				}
			}()
			fsType := "ext4"
			readOnly := false
			diskName := volumeName[(strings.LastIndex(volumeName, "/") + 1):]
			tests := []framework.VolumeTest{
				{
					Volume: v1.VolumeSource{
						AzureDisk: &v1.AzureDiskVolumeSource{
							DiskName:    diskName,
							DataDiskURI: volumeName,
							FSType:      &fsType,
							ReadOnly:    &readOnly,
						},
					},
					File: "index.html",
					// Randomize index.html to make sure we don't see the
					// content from previous test runs.
					ExpectedContent: "Hello from Azure from namespace " + volumeName,
				},
			}

			framework.InjectHtml(cs, config, tests[0].Volume, tests[0].ExpectedContent)

			fsGroup := int64(1234)
			framework.TestVolumeClient(cs, config, &fsGroup, tests)
		})
	})
})
