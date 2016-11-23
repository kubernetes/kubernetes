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

package common

import (
	"fmt"
	"os/exec"
	"regexp"
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
func startVolumeServer(f *framework.Framework, config VolumeTestConfig) *api.Pod {
	podClient := f.PodClient()

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
	serverPod = podClient.CreateSync(serverPod)

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

	podClient := f.PodClient()

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
func testVolumeClient(f *framework.Framework, config VolumeTestConfig, volume api.VolumeSource, fsGroup *int64, expectedContent string) {
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
	podClient := f.PodClient()

	if fsGroup != nil {
		clientPod.Spec.SecurityContext.FSGroup = fsGroup
	}
	clientPod = podClient.CreateSync(clientPod)

	By("Checking that text file contents are perfect.")
	result := f.ExecCommandInPod(clientPod.Name, "cat", "/opt/index.html")
	var err error
	if !strings.Contains(result, expectedContent) {
		err = fmt.Errorf("Failed to find \"%s\", last result: \"%s\"", expectedContent, result)
	}
	Expect(err).NotTo(HaveOccurred(), "failed: finding the contents of the mounted file.")

	if fsGroup != nil {

		By("Checking fsGroup is correct.")
		_, err := framework.LookForStringInPodExec(config.namespace, clientPod.Name, []string{"ls", "-ld", "/opt"}, strconv.Itoa(int(*fsGroup)), time.Minute)
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
var _ = framework.KubeDescribe("GCP Volumes", func() {
	f := framework.NewDefaultFramework("gcp-volume")

	// If 'false', the test won't clear its volumes upon completion. Useful for debugging,
	// note that namespace deletion is handled by delete-namespace flag
	clean := true
	// filled in BeforeEach
	var namespace *api.Namespace

	BeforeEach(func() {
		if !isTestEnabled(f.ClientSet) {
			framework.Skipf("NFS tests are not supported for this distro")
		}
		namespace = f.Namespace
	})

	////////////////////////////////////////////////////////////////////////
	// NFS
	////////////////////////////////////////////////////////////////////////

	framework.KubeDescribe("NFSv4", func() {
		It("should be mountable for NFSv4", func() {
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
			pod := startVolumeServer(f, config)
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
			testVolumeClient(f, config, volume, nil, "Hello from NFS!")
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
			pod := startVolumeServer(f, config)
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

			endClient := f.ClientSet.Core().Endpoints(config.namespace)

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
			testVolumeClient(f, config, volume, nil, "Hello from GlusterFS!")
		})
	})
})

func isTestEnabled(c clientset.Interface) bool {
	// Enable the test on node e2e if the node image is GCI.
	nodeName := framework.TestContext.NodeName
	if nodeName != "" {
		if strings.Contains(nodeName, "-gci-dev-") {
			gciVersionRe := regexp.MustCompile("-gci-dev-([0-9]+)-")
			matches := gciVersionRe.FindStringSubmatch(framework.TestContext.NodeName)
			if len(matches) == 2 {
				version, err := strconv.Atoi(matches[1])
				if err != nil {
					glog.Errorf("Error parsing GCI version from NodeName %q: %v", nodeName, err)
					return false
				}
				return version >= 54
			}
		}
		return false
	}

	// For cluster e2e test, because nodeName is empty, retrieve the node objects from api server
	// and check their images. Only run NFSv4 and GlusterFS if nodes are using GCI image for now.
	nodes := framework.GetReadySchedulableNodesOrDie(c)
	for _, node := range nodes.Items {
		if !strings.Contains(node.Status.NodeInfo.OSImage, "Google Container-VM") {
			return false
		}
	}
	return true
}
