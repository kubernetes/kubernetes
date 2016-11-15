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

package e2e

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/golang/glog"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

// Configuration of one tests. The test consist of:
// - server pod - runs serverImage, exports ports[]
// - client pod - does not need any special configuration
type VolumeTestConfig struct {
	namespace string
	// Prefix of all pods. Typically the test name.
	prefix string
	// Name of container image for the server pod.
	serverImage string
	// Ports to export from the server pod. TCP only.
	serverPorts []int
	// Arguments to pass to the container image.
	serverArgs []string
	// Volumes needed to be mounted to the server container from the host
	// map <host (source) path> -> <container (dst.) path>
	volumes map[string]string
}

// Starts a container specified by config.serverImage and exports all
// config.serverPorts from it. The returned pod should be used to get the server
// IP address and create appropriate VolumeSource.
func startVolumeServer(client clientset.Interface, config VolumeTestConfig) *api.Pod {
	podClient := client.Core().Pods(config.namespace)

	portCount := len(config.serverPorts)
	serverPodPorts := make([]api.ContainerPort, portCount)

	for i := 0; i < portCount; i++ {
		portName := fmt.Sprintf("%s-%d", config.prefix, i)

		serverPodPorts[i] = api.ContainerPort{
			Name:          portName,
			ContainerPort: int32(config.serverPorts[i]),
			Protocol:      api.ProtocolTCP,
		}
	}

	volumeCount := len(config.volumes)
	volumes := make([]api.Volume, volumeCount)
	mounts := make([]api.VolumeMount, volumeCount)

	i := 0
	for src, dst := range config.volumes {
		mountName := fmt.Sprintf("path%d", i)
		volumes[i].Name = mountName
		volumes[i].VolumeSource.HostPath = &api.HostPathVolumeSource{
			Path: src,
		}

		mounts[i].Name = mountName
		mounts[i].ReadOnly = false
		mounts[i].MountPath = dst

		i++
	}

	By(fmt.Sprint("creating ", config.prefix, " server pod"))
	privileged := new(bool)
	*privileged = true
	serverPod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: api.ObjectMeta{
			Name: config.prefix + "-server",
			Labels: map[string]string{
				"role": config.prefix + "-server",
			},
		},

		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  config.prefix + "-server",
					Image: config.serverImage,
					SecurityContext: &api.SecurityContext{
						Privileged: privileged,
					},
					Args:         config.serverArgs,
					Ports:        serverPodPorts,
					VolumeMounts: mounts,
				},
			},
			Volumes: volumes,
		},
	}
	serverPod, err := podClient.Create(serverPod)
	framework.ExpectNoError(err, "Failed to create %s pod: %v", serverPod.Name, err)

	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(client, serverPod))

	By("locating the server pod")
	pod, err := podClient.Get(serverPod.Name)
	framework.ExpectNoError(err, "Cannot locate the server pod %v: %v", serverPod.Name, err)

	By("sleeping a bit to give the server time to start")
	time.Sleep(20 * time.Second)
	return pod
}

// Clean both server and client pods.
func volumeTestCleanup(f *framework.Framework, config VolumeTestConfig) {
	By(fmt.Sprint("cleaning the environment after ", config.prefix))

	defer GinkgoRecover()

	client := f.ClientSet
	podClient := client.Core().Pods(config.namespace)

	err := podClient.Delete(config.prefix+"-client", nil)
	if err != nil {
		// Log the error before failing test: if the test has already failed,
		// framework.ExpectNoError() won't print anything to logs!
		glog.Warningf("Failed to delete client pod: %v", err)
		framework.ExpectNoError(err, "Failed to delete client pod: %v", err)
	}

	if config.serverImage != "" {
		if err := f.WaitForPodTerminated(config.prefix+"-client", ""); !apierrs.IsNotFound(err) {
			framework.ExpectNoError(err, "Failed to wait client pod terminated: %v", err)
		}
		// See issue #24100.
		// Prevent umount errors by making sure making sure the client pod exits cleanly *before* the volume server pod exits.
		By("sleeping a bit so client can stop and unmount")
		time.Sleep(20 * time.Second)

		err = podClient.Delete(config.prefix+"-server", nil)
		if err != nil {
			glog.Warningf("Failed to delete server pod: %v", err)
			framework.ExpectNoError(err, "Failed to delete server pod: %v", err)
		}
	}
}

// Start a client pod using given VolumeSource (exported by startVolumeServer())
// and check that the pod sees the data from the server pod.
func testVolumeClient(client clientset.Interface, config VolumeTestConfig, volume api.VolumeSource, fsGroup *int64, expectedContent string) {
	By(fmt.Sprint("starting ", config.prefix, " client"))
	clientPod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: api.ObjectMeta{
			Name: config.prefix + "-client",
			Labels: map[string]string{
				"role": config.prefix + "-client",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:       config.prefix + "-client",
					Image:      "gcr.io/google_containers/busybox:1.24",
					WorkingDir: "/opt",
					// An imperative and easily debuggable container which reads vol contents for
					// us to scan in the tests or by eye.
					// We expect that /opt is empty in the minimal containers which we use in this test.
					Command: []string{
						"/bin/sh",
						"-c",
						"while true ; do cat /opt/index.html ; sleep 2 ; ls -altrh /opt/  ; sleep 2 ; done ",
					},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      config.prefix + "-volume",
							MountPath: "/opt/",
						},
					},
				},
			},
			SecurityContext: &api.PodSecurityContext{
				SELinuxOptions: &api.SELinuxOptions{
					Level: "s0:c0,c1",
				},
			},
			Volumes: []api.Volume{
				{
					Name:         config.prefix + "-volume",
					VolumeSource: volume,
				},
			},
		},
	}
	podsNamespacer := client.Core().Pods(config.namespace)

	if fsGroup != nil {
		clientPod.Spec.SecurityContext.FSGroup = fsGroup
	}
	clientPod, err := podsNamespacer.Create(clientPod)
	if err != nil {
		framework.Failf("Failed to create %s pod: %v", clientPod.Name, err)
	}
	framework.ExpectNoError(framework.WaitForPodRunningInNamespace(client, clientPod))

	By("Checking that text file contents are perfect.")
	_, err = framework.LookForStringInPodExec(config.namespace, clientPod.Name, []string{"cat", "/opt/index.html"}, expectedContent, time.Minute)
	Expect(err).NotTo(HaveOccurred(), "failed: finding the contents of the mounted file.")

	if fsGroup != nil {

		By("Checking fsGroup is correct.")
		_, err = framework.LookForStringInPodExec(config.namespace, clientPod.Name, []string{"ls", "-ld", "/opt"}, strconv.Itoa(int(*fsGroup)), time.Minute)
		Expect(err).NotTo(HaveOccurred(), "failed: getting the right priviliges in the file %v", int(*fsGroup))
	}
}

// Insert index.html with given content into given volume. It does so by
// starting and auxiliary pod which writes the file there.
// The volume must be writable.
func injectHtml(client clientset.Interface, config VolumeTestConfig, volume api.VolumeSource, content string) {
	By(fmt.Sprint("starting ", config.prefix, " injector"))
	podClient := client.Core().Pods(config.namespace)

	injectPod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: api.ObjectMeta{
			Name: config.prefix + "-injector",
			Labels: map[string]string{
				"role": config.prefix + "-injector",
			},
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    config.prefix + "-injector",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "echo '" + content + "' > /mnt/index.html && chmod o+rX /mnt /mnt/index.html"},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      config.prefix + "-volume",
							MountPath: "/mnt",
						},
					},
				},
			},
			SecurityContext: &api.PodSecurityContext{
				SELinuxOptions: &api.SELinuxOptions{
					Level: "s0:c0,c1",
				},
			},
			RestartPolicy: api.RestartPolicyNever,
			Volumes: []api.Volume{
				{
					Name:         config.prefix + "-volume",
					VolumeSource: volume,
				},
			},
		},
	}

	defer func() {
		podClient.Delete(config.prefix+"-injector", nil)
	}()

	injectPod, err := podClient.Create(injectPod)
	framework.ExpectNoError(err, "Failed to create injector pod: %v", err)
	err = framework.WaitForPodSuccessInNamespace(client, injectPod.Name, injectPod.Namespace)
	Expect(err).NotTo(HaveOccurred())
}

func deleteCinderVolume(name string) error {
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

// These tests need privileged containers, which are disabled by default.  Run
// the test with "go run hack/e2e.go ... --ginkgo.focus=[Feature:Volumes]"
var _ = framework.KubeDescribe("Volumes [Feature:Volumes]", func() {
	f := framework.NewDefaultFramework("volume")

	// If 'false', the test won't clear its volumes upon completion. Useful for debugging,
	// note that namespace deletion is handled by delete-namespace flag
	clean := true
	// filled in BeforeEach
	var cs clientset.Interface
	var namespace *api.Namespace

	BeforeEach(func() {
		cs = f.ClientSet
		namespace = f.Namespace
	})

	////////////////////////////////////////////////////////////////////////
	// NFS
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("NFS", func() {
		It("should be mountable", func() {
			config := VolumeTestConfig{
				namespace:   namespace.Name,
				prefix:      "nfs",
				serverImage: "gcr.io/google_containers/volume-nfs:0.8",
				serverPorts: []int{2049},
			}

			defer func() {
				if clean {
					volumeTestCleanup(f, config)
				}
			}()
			pod := startVolumeServer(cs, config)
			serverIP := pod.Status.PodIP
			framework.Logf("NFS server IP address: %v", serverIP)

			volume := api.VolumeSource{
				NFS: &api.NFSVolumeSource{
					Server:   serverIP,
					Path:     "/",
					ReadOnly: true,
				},
			}
			// Must match content of test/images/volumes-tester/nfs/index.html
			testVolumeClient(cs, config, volume, nil, "Hello from NFS!")
		})
	})

	////////////////////////////////////////////////////////////////////////
	// Gluster
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("GlusterFS", func() {
		It("should be mountable", func() {
			config := VolumeTestConfig{
				namespace:   namespace.Name,
				prefix:      "gluster",
				serverImage: "gcr.io/google_containers/volume-gluster:0.2",
				serverPorts: []int{24007, 24008, 49152},
			}

			defer func() {
				if clean {
					volumeTestCleanup(f, config)
				}
			}()
			pod := startVolumeServer(cs, config)
			serverIP := pod.Status.PodIP
			framework.Logf("Gluster server IP address: %v", serverIP)

			// create Endpoints for the server
			endpoints := api.Endpoints{
				TypeMeta: unversioned.TypeMeta{
					Kind:       "Endpoints",
					APIVersion: "v1",
				},
				ObjectMeta: api.ObjectMeta{
					Name: config.prefix + "-server",
				},
				Subsets: []api.EndpointSubset{
					{
						Addresses: []api.EndpointAddress{
							{
								IP: serverIP,
							},
						},
						Ports: []api.EndpointPort{
							{
								Name:     "gluster",
								Port:     24007,
								Protocol: api.ProtocolTCP,
							},
						},
					},
				},
			}

			endClient := cs.Core().Endpoints(config.namespace)

			defer func() {
				if clean {
					endClient.Delete(config.prefix+"-server", nil)
				}
			}()

			if _, err := endClient.Create(&endpoints); err != nil {
				framework.Failf("Failed to create endpoints for Gluster server: %v", err)
			}

			volume := api.VolumeSource{
				Glusterfs: &api.GlusterfsVolumeSource{
					EndpointsName: config.prefix + "-server",
					// 'test_vol' comes from test/images/volumes-tester/gluster/run_gluster.sh
					Path:     "test_vol",
					ReadOnly: true,
				},
			}
			// Must match content of test/images/volumes-tester/gluster/index.html
			testVolumeClient(cs, config, volume, nil, "Hello from GlusterFS!")
		})
	})

	////////////////////////////////////////////////////////////////////////
	// iSCSI
	////////////////////////////////////////////////////////////////////////

	// The test needs privileged containers, which are disabled by default.
	// Also, make sure that iscsiadm utility and iscsi target kernel modules
	// are installed on all nodes!
	// Run the test with "go run hack/e2e.go ... --ginkgo.focus=iSCSI"

	framework.KubeDescribe("iSCSI", func() {
		It("should be mountable", func() {
			config := VolumeTestConfig{
				namespace:   namespace.Name,
				prefix:      "iscsi",
				serverImage: "gcr.io/google_containers/volume-iscsi:0.1",
				serverPorts: []int{3260},
				volumes: map[string]string{
					// iSCSI container needs to insert modules from the host
					"/lib/modules": "/lib/modules",
				},
			}

			defer func() {
				if clean {
					volumeTestCleanup(f, config)
				}
			}()
			pod := startVolumeServer(cs, config)
			serverIP := pod.Status.PodIP
			framework.Logf("iSCSI server IP address: %v", serverIP)

			volume := api.VolumeSource{
				ISCSI: &api.ISCSIVolumeSource{
					TargetPortal: serverIP + ":3260",
					// from test/images/volumes-tester/iscsi/initiatorname.iscsi
					IQN:    "iqn.2003-01.org.linux-iscsi.f21.x8664:sn.4b0aae584f7c",
					Lun:    0,
					FSType: "ext2",
				},
			}

			fsGroup := int64(1234)
			// Must match content of test/images/volumes-tester/iscsi/block.tar.gz
			testVolumeClient(cs, config, volume, &fsGroup, "Hello from iSCSI")
		})
	})

	////////////////////////////////////////////////////////////////////////
	// Ceph RBD
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("Ceph RBD", func() {
		It("should be mountable", func() {
			config := VolumeTestConfig{
				namespace:   namespace.Name,
				prefix:      "rbd",
				serverImage: "gcr.io/google_containers/volume-rbd:0.1",
				serverPorts: []int{6789},
				volumes: map[string]string{
					// iSCSI container needs to insert modules from the host
					"/lib/modules": "/lib/modules",
					"/sys":         "/sys",
				},
			}

			defer func() {
				if clean {
					volumeTestCleanup(f, config)
				}
			}()
			pod := startVolumeServer(cs, config)
			serverIP := pod.Status.PodIP
			framework.Logf("Ceph server IP address: %v", serverIP)

			// create secrets for the server
			secret := api.Secret{
				TypeMeta: unversioned.TypeMeta{
					Kind:       "Secret",
					APIVersion: "v1",
				},
				ObjectMeta: api.ObjectMeta{
					Name: config.prefix + "-secret",
				},
				Data: map[string][]byte{
					// from test/images/volumes-tester/rbd/keyring
					"key": []byte("AQDRrKNVbEevChAAEmRC+pW/KBVHxa0w/POILA=="),
				},
				Type: "kubernetes.io/rbd",
			}

			secClient := cs.Core().Secrets(config.namespace)

			defer func() {
				if clean {
					secClient.Delete(config.prefix+"-secret", nil)
				}
			}()

			if _, err := secClient.Create(&secret); err != nil {
				framework.Failf("Failed to create secrets for Ceph RBD: %v", err)
			}

			volume := api.VolumeSource{
				RBD: &api.RBDVolumeSource{
					CephMonitors: []string{serverIP},
					RBDPool:      "rbd",
					RBDImage:     "foo",
					RadosUser:    "admin",
					SecretRef: &api.LocalObjectReference{
						Name: config.prefix + "-secret",
					},
					FSType: "ext2",
				},
			}
			fsGroup := int64(1234)

			// Must match content of test/images/volumes-tester/gluster/index.html
			testVolumeClient(cs, config, volume, &fsGroup, "Hello from RBD")

		})
	})
	////////////////////////////////////////////////////////////////////////
	// Ceph
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("CephFS", func() {
		It("should be mountable", func() {
			config := VolumeTestConfig{
				namespace:   namespace.Name,
				prefix:      "cephfs",
				serverImage: "gcr.io/google_containers/volume-ceph:0.1",
				serverPorts: []int{6789},
			}

			defer func() {
				if clean {
					volumeTestCleanup(f, config)
				}
			}()
			pod := startVolumeServer(cs, config)
			serverIP := pod.Status.PodIP
			framework.Logf("Ceph server IP address: %v", serverIP)
			By("sleeping a bit to give ceph server time to initialize")
			time.Sleep(20 * time.Second)

			// create ceph secret
			secret := &api.Secret{
				TypeMeta: unversioned.TypeMeta{
					Kind:       "Secret",
					APIVersion: "v1",
				},
				ObjectMeta: api.ObjectMeta{
					Name: config.prefix + "-secret",
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

			volume := api.VolumeSource{
				CephFS: &api.CephFSVolumeSource{
					Monitors:  []string{serverIP + ":6789"},
					User:      "kube",
					SecretRef: &api.LocalObjectReference{Name: config.prefix + "-secret"},
					ReadOnly:  true,
				},
			}
			// Must match content of contrib/for-tests/volumes-ceph/ceph/index.html
			testVolumeClient(cs, config, volume, nil, "Hello Ceph!")
		})
	})

	////////////////////////////////////////////////////////////////////////
	// OpenStack Cinder
	////////////////////////////////////////////////////////////////////////

	// This test assumes that OpenStack client tools are installed
	// (/usr/bin/nova, /usr/bin/cinder and /usr/bin/keystone)
	// and that the usual OpenStack authentication env. variables are set
	// (OS_USERNAME, OS_PASSWORD, OS_TENANT_NAME at least).

	framework.KubeDescribe("Cinder", func() {
		It("should be mountable", func() {
			framework.SkipUnlessProviderIs("openstack")
			config := VolumeTestConfig{
				namespace: namespace.Name,
				prefix:    "cinder",
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
				deleteCinderVolume(volumeName)
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
					volumeTestCleanup(f, config)
				}
			}()
			volume := api.VolumeSource{
				Cinder: &api.CinderVolumeSource{
					VolumeID: volumeID,
					FSType:   "ext3",
					ReadOnly: false,
				},
			}

			// Insert index.html into the test volume with some random content
			// to make sure we don't see the content from previous test runs.
			content := "Hello from Cinder from namespace " + volumeName
			injectHtml(cs, config, volume, content)

			fsGroup := int64(1234)
			testVolumeClient(cs, config, volume, &fsGroup, content)
		})
	})

	////////////////////////////////////////////////////////////////////////
	// GCE PD
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("PD", func() {
		It("should be mountable", func() {
			framework.SkipUnlessProviderIs("gce", "gke")
			config := VolumeTestConfig{
				namespace: namespace.Name,
				prefix:    "pd",
			}

			By("creating a test gce pd volume")
			volumeName, err := createPDWithRetry()
			Expect(err).NotTo(HaveOccurred())

			defer func() {
				deletePDWithRetry(volumeName)
			}()

			defer func() {
				if clean {
					framework.Logf("Running volumeTestCleanup")
					volumeTestCleanup(f, config)
				}
			}()
			volume := api.VolumeSource{
				GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
					PDName:   volumeName,
					FSType:   "ext3",
					ReadOnly: false,
				},
			}

			// Insert index.html into the test volume with some random content
			// to make sure we don't see the content from previous test runs.
			content := "Hello from GCE PD from namespace " + volumeName
			injectHtml(cs, config, volume, content)

			fsGroup := int64(1234)
			testVolumeClient(cs, config, volume, &fsGroup, content)
		})
	})
})
