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

package volume

import (
	"fmt"
	"path/filepath"
	"strconv"
	"time"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	// Kb is byte size of kilobyte
	Kb int64 = 1000
	// Mb is byte size of megabyte
	Mb int64 = 1000 * Kb
	// Gb is byte size of gigabyte
	Gb int64 = 1000 * Mb
	// Tb is byte size of terabyte
	Tb int64 = 1000 * Gb
	// KiB is byte size of kibibyte
	KiB int64 = 1024
	// MiB is byte size of mebibyte
	MiB int64 = 1024 * KiB
	// GiB is byte size of gibibyte
	GiB int64 = 1024 * MiB
	// TiB is byte size of tebibyte
	TiB int64 = 1024 * GiB

	// VolumeServerPodStartupTimeout is a waiting period for volume server (Ceph, ...) to initialize itself.
	VolumeServerPodStartupTimeout = 3 * time.Minute

	// PodCleanupTimeout is a waiting period for pod to be cleaned up and unmount its volumes so we
	// don't tear down containers with NFS/Ceph/Gluster server too early.
	PodCleanupTimeout = 20 * time.Second

	// Template for iSCSI IQN.
	iSCSIIQNTemplate = "iqn.2003-01.io.k8s:e2e.%s"
)

// TestConfig is a struct for configuration of one tests. The test consist of:
// - server pod - runs serverImage, exports ports[]
// - client pod - does not need any special configuration
type TestConfig struct {
	Namespace string
	// Prefix of all pods. Typically the test name.
	Prefix string
	// Name of container image for the server pod.
	ServerImage string
	// Ports to export from the server pod. TCP only.
	ServerPorts []int
	// Commands to run in the container image.
	ServerCmds []string
	// Arguments to pass to the container image.
	ServerArgs []string
	// Volumes needed to be mounted to the server container from the host
	// map <host (source) path> -> <container (dst.) path>
	// if <host (source) path> is empty, mount a tmpfs emptydir
	ServerVolumes map[string]string
	// Message to wait for before starting clients
	ServerReadyMessage string
	// Use HostNetwork for the server
	ServerHostNetwork bool
	// Wait for the pod to terminate successfully
	// False indicates that the pod is long running
	WaitForCompletion bool
	// ServerNodeName is the spec.nodeName to run server pod on.  Default is any node.
	ServerNodeName string
	// ClientNodeName is the spec.nodeName to run client pod on.  Default is any node.
	ClientNodeName string
	// NodeSelector to use in pod spec (server, client and injector pods).
	NodeSelector map[string]string
}

// Test contains a volume to mount into a client pod and its
// expected content.
type Test struct {
	Volume v1.VolumeSource
	Mode   v1.PersistentVolumeMode
	// Name of file to read/write in FileSystem mode
	File            string
	ExpectedContent string
}

// NewNFSServer is a NFS-specific wrapper for CreateStorageServer.
func NewNFSServer(cs clientset.Interface, namespace string, args []string) (config TestConfig, pod *v1.Pod, ip string) {
	config = TestConfig{
		Namespace:          namespace,
		Prefix:             "nfs",
		ServerImage:        imageutils.GetE2EImage(imageutils.VolumeNFSServer),
		ServerPorts:        []int{2049},
		ServerVolumes:      map[string]string{"": "/exports"},
		ServerReadyMessage: "NFS started",
	}
	if len(args) > 0 {
		config.ServerArgs = args
	}
	pod, ip = CreateStorageServer(cs, config)
	return config, pod, ip
}

// NewGlusterfsServer is a GlusterFS-specific wrapper for CreateStorageServer. Also creates the gluster endpoints object.
func NewGlusterfsServer(cs clientset.Interface, namespace string) (config TestConfig, pod *v1.Pod, ip string) {
	config = TestConfig{
		Namespace:   namespace,
		Prefix:      "gluster",
		ServerImage: imageutils.GetE2EImage(imageutils.VolumeGlusterServer),
		ServerPorts: []int{24007, 24008, 49152},
	}
	pod, ip = CreateStorageServer(cs, config)

	ginkgo.By("creating Gluster endpoints")
	endpoints := &v1.Endpoints{
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
						IP: ip,
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
	_, err := cs.CoreV1().Endpoints(namespace).Create(endpoints)
	framework.ExpectNoError(err, "failed to create endpoints for Gluster server")

	return config, pod, ip
}

// NewISCSIServer is an iSCSI-specific wrapper for CreateStorageServer.
func NewISCSIServer(cs clientset.Interface, namespace string) (config TestConfig, pod *v1.Pod, ip, iqn string) {
	// Generate cluster-wide unique IQN
	iqn = fmt.Sprintf(iSCSIIQNTemplate, namespace)
	config = TestConfig{
		Namespace:   namespace,
		Prefix:      "iscsi",
		ServerImage: imageutils.GetE2EImage(imageutils.VolumeISCSIServer),
		ServerArgs:  []string{iqn},
		ServerVolumes: map[string]string{
			// iSCSI container needs to insert modules from the host
			"/lib/modules": "/lib/modules",
			// iSCSI container needs to configure kernel
			"/sys/kernel": "/sys/kernel",
			// iSCSI source "block devices" must be available on the host
			"/srv/iscsi": "/srv/iscsi",
		},
		ServerReadyMessage: "iscsi target started",
		ServerHostNetwork:  true,
	}
	pod, ip = CreateStorageServer(cs, config)
	// Make sure the client runs on the same node as server so we don't need to open any firewalls.
	config.ClientNodeName = pod.Spec.NodeName
	return config, pod, ip, iqn
}

// NewRBDServer is a CephRBD-specific wrapper for CreateStorageServer.
func NewRBDServer(cs clientset.Interface, namespace string) (config TestConfig, pod *v1.Pod, secret *v1.Secret, ip string) {
	config = TestConfig{
		Namespace:   namespace,
		Prefix:      "rbd",
		ServerImage: imageutils.GetE2EImage(imageutils.VolumeRBDServer),
		ServerPorts: []int{6789},
		ServerVolumes: map[string]string{
			"/lib/modules": "/lib/modules",
		},
		ServerReadyMessage: "Ceph is ready",
	}
	pod, ip = CreateStorageServer(cs, config)
	// create secrets for the server
	secret = &v1.Secret{
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

	secret, err := cs.CoreV1().Secrets(config.Namespace).Create(secret)
	if err != nil {
		framework.Failf("Failed to create secrets for Ceph RBD: %v", err)
	}

	return config, pod, secret, ip
}

// CreateStorageServer is a wrapper for StartVolumeServer(). A storage server config is passed in, and a pod pointer
// and ip address string are returned.
// Note: Expect() is called so no error is returned.
func CreateStorageServer(cs clientset.Interface, config TestConfig) (pod *v1.Pod, ip string) {
	pod = StartVolumeServer(cs, config)
	gomega.Expect(pod).NotTo(gomega.BeNil(), "storage server pod should not be nil")
	ip = pod.Status.PodIP
	gomega.Expect(len(ip)).NotTo(gomega.BeZero(), fmt.Sprintf("pod %s's IP should not be empty", pod.Name))
	framework.Logf("%s server pod IP address: %s", config.Prefix, ip)
	return pod, ip
}

// StartVolumeServer starts a container specified by config.serverImage and exports all
// config.serverPorts from it. The returned pod should be used to get the server
// IP address and create appropriate VolumeSource.
func StartVolumeServer(client clientset.Interface, config TestConfig) *v1.Pod {
	podClient := client.CoreV1().Pods(config.Namespace)

	portCount := len(config.ServerPorts)
	serverPodPorts := make([]v1.ContainerPort, portCount)

	for i := 0; i < portCount; i++ {
		portName := fmt.Sprintf("%s-%d", config.Prefix, i)

		serverPodPorts[i] = v1.ContainerPort{
			Name:          portName,
			ContainerPort: int32(config.ServerPorts[i]),
			Protocol:      v1.ProtocolTCP,
		}
	}

	volumeCount := len(config.ServerVolumes)
	volumes := make([]v1.Volume, volumeCount)
	mounts := make([]v1.VolumeMount, volumeCount)

	i := 0
	for src, dst := range config.ServerVolumes {
		mountName := fmt.Sprintf("path%d", i)
		volumes[i].Name = mountName
		if src == "" {
			volumes[i].VolumeSource.EmptyDir = &v1.EmptyDirVolumeSource{}
		} else {
			volumes[i].VolumeSource.HostPath = &v1.HostPathVolumeSource{
				Path: src,
			}
		}

		mounts[i].Name = mountName
		mounts[i].ReadOnly = false
		mounts[i].MountPath = dst

		i++
	}

	serverPodName := fmt.Sprintf("%s-server", config.Prefix)
	ginkgo.By(fmt.Sprint("creating ", serverPodName, " pod"))
	privileged := new(bool)
	*privileged = true

	restartPolicy := v1.RestartPolicyAlways
	if config.WaitForCompletion {
		restartPolicy = v1.RestartPolicyNever
	}
	serverPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: serverPodName,
			Labels: map[string]string{
				"role": serverPodName,
			},
		},

		Spec: v1.PodSpec{
			HostNetwork: config.ServerHostNetwork,
			Containers: []v1.Container{
				{
					Name:  serverPodName,
					Image: config.ServerImage,
					SecurityContext: &v1.SecurityContext{
						Privileged: privileged,
					},
					Command:      config.ServerCmds,
					Args:         config.ServerArgs,
					Ports:        serverPodPorts,
					VolumeMounts: mounts,
				},
			},
			Volumes:       volumes,
			RestartPolicy: restartPolicy,
			NodeName:      config.ServerNodeName,
			NodeSelector:  config.NodeSelector,
		},
	}

	var pod *v1.Pod
	serverPod, err := podClient.Create(serverPod)
	// ok if the server pod already exists. TODO: make this controllable by callers
	if err != nil {
		if apierrs.IsAlreadyExists(err) {
			framework.Logf("Ignore \"already-exists\" error, re-get pod...")
			ginkgo.By(fmt.Sprintf("re-getting the %q server pod", serverPodName))
			serverPod, err = podClient.Get(serverPodName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Cannot re-get the server pod %q: %v", serverPodName, err)
			pod = serverPod
		} else {
			framework.ExpectNoError(err, "Failed to create %q pod: %v", serverPodName, err)
		}
	}
	if config.WaitForCompletion {
		framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(client, serverPod.Name, serverPod.Namespace))
		framework.ExpectNoError(podClient.Delete(serverPod.Name, nil))
	} else {
		framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(client, serverPod))
		if pod == nil {
			ginkgo.By(fmt.Sprintf("locating the %q server pod", serverPodName))
			pod, err = podClient.Get(serverPodName, metav1.GetOptions{})
			framework.ExpectNoError(err, "Cannot locate the server pod %q: %v", serverPodName, err)
		}
	}
	if config.ServerReadyMessage != "" {
		_, err := framework.LookForStringInLog(pod.Namespace, pod.Name, serverPodName, config.ServerReadyMessage, VolumeServerPodStartupTimeout)
		framework.ExpectNoError(err, "Failed to find %q in pod logs: %s", config.ServerReadyMessage, err)
	}
	return pod
}

// CleanUpVolumeServer is a wrapper of cleanup function for volume server without secret created by specific CreateStorageServer function.
func CleanUpVolumeServer(f *framework.Framework, serverPod *v1.Pod) {
	CleanUpVolumeServerWithSecret(f, serverPod, nil)
}

// CleanUpVolumeServerWithSecret is a wrapper of cleanup function for volume server with secret created by specific CreateStorageServer function.
func CleanUpVolumeServerWithSecret(f *framework.Framework, serverPod *v1.Pod, secret *v1.Secret) {
	cs := f.ClientSet
	ns := f.Namespace

	if secret != nil {
		framework.Logf("Deleting server secret %q...", secret.Name)
		err := cs.CoreV1().Secrets(ns.Name).Delete(secret.Name, &metav1.DeleteOptions{})
		if err != nil {
			framework.Logf("Delete secret failed: %v", err)
		}
	}

	e2elog.Logf("Deleting server pod %q...", serverPod.Name)
	err := e2epod.DeletePodWithWait(cs, serverPod)
	if err != nil {
		framework.Logf("Server pod delete failed: %v", err)
	}
}

// TestCleanup cleans both server and client pods.
func TestCleanup(f *framework.Framework, config TestConfig) {
	ginkgo.By(fmt.Sprint("cleaning the environment after ", config.Prefix))

	defer ginkgo.GinkgoRecover()

	cs := f.ClientSet

	err := e2epod.DeletePodWithWaitByName(cs, config.Prefix+"-client", config.Namespace)
	gomega.Expect(err).To(gomega.BeNil(), "Failed to delete pod %v in namespace %v", config.Prefix+"-client", config.Namespace)

	if config.ServerImage != "" {
		err := e2epod.DeletePodWithWaitByName(cs, config.Prefix+"-server", config.Namespace)
		gomega.Expect(err).To(gomega.BeNil(), "Failed to delete pod %v in namespace %v", config.Prefix+"-server", config.Namespace)
	}
}

func runVolumeTesterPod(client clientset.Interface, config TestConfig, podSuffix string, fsGroup *int64, tests []Test) (*v1.Pod, error) {
	ginkgo.By(fmt.Sprint("starting ", config.Prefix, "-", podSuffix))
	var gracePeriod int64 = 1
	var command string

	if !framework.NodeOSDistroIs("windows") {
		command = "while true ; do sleep 2; done "
	} else {
		command = "while(1) {sleep 2}"
	}
	seLinuxOptions := &v1.SELinuxOptions{Level: "s0:c0,c1"}
	clientPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: config.Prefix + "-" + podSuffix,
			Labels: map[string]string{
				"role": config.Prefix + "-" + podSuffix,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:       config.Prefix + "-" + podSuffix,
					Image:      GetTestImage(framework.BusyBoxImage),
					WorkingDir: "/opt",
					// An imperative and easily debuggable container which reads/writes vol contents for
					// us to scan in the tests or by eye.
					// We expect that /opt is empty in the minimal containers which we use in this test.
					Command:      GenerateScriptCmd(command),
					VolumeMounts: []v1.VolumeMount{},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
			SecurityContext:               GeneratePodSecurityContext(fsGroup, seLinuxOptions),
			Volumes:                       []v1.Volume{},
			NodeName:                      config.ClientNodeName,
			NodeSelector:                  config.NodeSelector,
		},
	}

	for i, test := range tests {
		volumeName := fmt.Sprintf("%s-%s-%d", config.Prefix, "volume", i)
		if test.Mode == v1.PersistentVolumeBlock {
			clientPod.Spec.Containers[0].VolumeDevices = append(clientPod.Spec.Containers[0].VolumeDevices, v1.VolumeDevice{
				Name:       volumeName,
				DevicePath: fmt.Sprintf("/opt/%d", i),
			})
		} else {
			clientPod.Spec.Containers[0].VolumeMounts = append(clientPod.Spec.Containers[0].VolumeMounts, v1.VolumeMount{
				Name:      volumeName,
				MountPath: fmt.Sprintf("/opt/%d", i),
			})
		}
		clientPod.Spec.Volumes = append(clientPod.Spec.Volumes, v1.Volume{
			Name:         volumeName,
			VolumeSource: test.Volume,
		})
	}
	podsNamespacer := client.CoreV1().Pods(config.Namespace)
	clientPod, err := podsNamespacer.Create(clientPod)
	if err != nil {
		return nil, err
	}
	err = e2epod.WaitForPodRunningInNamespace(client, clientPod)
	if err != nil {
		e2epod.WaitForPodToDisappear(client, clientPod.Namespace, clientPod.Name, labels.Everything(), framework.Poll, framework.PodDeleteTimeout)
		return nil, err
	}
	return clientPod, nil
}

func testVolumeContent(client clientset.Interface, pod *v1.Pod, fsGroup *int64, fsType string, tests []Test) {
	ginkgo.By("Checking that text file contents are perfect.")
	for i, test := range tests {
		if test.Mode == v1.PersistentVolumeBlock {
			// Block: check content
			deviceName := fmt.Sprintf("/opt/%d", i)
			commands := GenerateReadBlockCmd(deviceName, len(test.ExpectedContent))
			_, err := framework.LookForStringInPodExec(pod.Namespace, pod.Name, commands, test.ExpectedContent, time.Minute)
			framework.ExpectNoError(err, "failed: finding the contents of the block device %s.", deviceName)

			// Check that it's a real block device
			utils.CheckVolumeModeOfPath(pod, test.Mode, deviceName)
		} else {
			// Filesystem: check content
			fileName := fmt.Sprintf("/opt/%d/%s", i, test.File)
			commands := GenerateReadFileCmd(fileName)
			_, err := framework.LookForStringInPodExec(pod.Namespace, pod.Name, commands, test.ExpectedContent, time.Minute)
			framework.ExpectNoError(err, "failed: finding the contents of the mounted file %s.", fileName)

			// Check that a directory has been mounted
			dirName := filepath.Dir(fileName)
			utils.CheckVolumeModeOfPath(pod, test.Mode, dirName)

			if !framework.NodeOSDistroIs("windows") {
				// Filesystem: check fsgroup
				if fsGroup != nil {
					ginkgo.By("Checking fsGroup is correct.")
					_, err = framework.LookForStringInPodExec(pod.Namespace, pod.Name, []string{"ls", "-ld", dirName}, strconv.Itoa(int(*fsGroup)), time.Minute)
					framework.ExpectNoError(err, "failed: getting the right privileges in the file %v", int(*fsGroup))
				}

				// Filesystem: check fsType
				if fsType != "" {
					ginkgo.By("Checking fsType is correct.")
					_, err = framework.LookForStringInPodExec(pod.Namespace, pod.Name, []string{"grep", " " + dirName + " ", "/proc/mounts"}, fsType, time.Minute)
					framework.ExpectNoError(err, "failed: getting the right fsType %s", fsType)
				}
			}
		}
	}
}

// TestVolumeClient start a client pod using given VolumeSource (exported by startVolumeServer())
// and check that the pod sees expected data, e.g. from the server pod.
// Multiple Tests can be specified to mount multiple volumes to a single
// pod.
func TestVolumeClient(client clientset.Interface, config TestConfig, fsGroup *int64, fsType string, tests []Test) {
	clientPod, err := runVolumeTesterPod(client, config, "client", fsGroup, tests)
	if err != nil {
		framework.Failf("Failed to create client pod: %v", err)

	}
	framework.ExpectNoError(e2epod.WaitForPodRunningInNamespace(client, clientPod))
	testVolumeContent(client, clientPod, fsGroup, fsType, tests)
}

// InjectContent inserts index.html with given content into given volume. It does so by
// starting and auxiliary pod which writes the file there.
// The volume must be writable.
func InjectContent(client clientset.Interface, config TestConfig, fsGroup *int64, fsType string, tests []Test) {
	injectorPod, err := runVolumeTesterPod(client, config, "injector", fsGroup, tests)
	if err != nil {
		framework.Failf("Failed to create injector pod: %v", err)
		return
	}
	defer func() {
		e2epod.DeletePodOrFail(client, injectorPod.Namespace, injectorPod.Name)
		e2epod.WaitForPodToDisappear(client, injectorPod.Namespace, injectorPod.Name, labels.Everything(), framework.Poll, framework.PodDeleteTimeout)
	}()

	ginkgo.By("Writing text file contents in the container.")
	for i, test := range tests {
		commands := []string{"exec", injectorPod.Name, fmt.Sprintf("--namespace=%v", injectorPod.Namespace), "--"}
		if test.Mode == v1.PersistentVolumeBlock {
			// Block: write content
			deviceName := fmt.Sprintf("/opt/%d", i)
			commands = append(commands, GenerateWriteBlockCmd(test.ExpectedContent, deviceName)...)

		} else {
			// Filesystem: write content
			fileName := fmt.Sprintf("/opt/%d/%s", i, test.File)
			commands = append(commands, GenerateWriteFileCmd(test.ExpectedContent, fileName)...)
		}
		out, err := framework.RunKubectl(commands...)
		framework.ExpectNoError(err, "failed: writing the contents: %s", out)
	}

	// Check that the data have been really written in this pod.
	// This tests non-persistent volume types
	testVolumeContent(client, injectorPod, fsGroup, fsType, tests)
}

// CreateGCEVolume creates PersistentVolumeSource for GCEVolume.
func CreateGCEVolume() (*v1.PersistentVolumeSource, string) {
	diskName, err := framework.CreatePDWithRetry()
	framework.ExpectNoError(err)
	return &v1.PersistentVolumeSource{
		GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
			PDName:   diskName,
			FSType:   "ext3",
			ReadOnly: false,
		},
	}, diskName
}

// GenerateScriptCmd generates the corresponding command lines to execute a command.
// Depending on the Node OS is Windows or linux, the command will use powershell or /bin/sh
func GenerateScriptCmd(command string) []string {
	var commands []string
	if !framework.NodeOSDistroIs("windows") {
		commands = []string{"/bin/sh", "-c", command}
	} else {
		commands = []string{"powershell", "/c", command}
	}
	return commands
}

// GenerateWriteBlockCmd generates the corresponding command lines to write to a block device the given content.
// Depending on the Node OS is Windows or linux, the command will use powershell or /bin/sh
func GenerateWriteBlockCmd(content, fullPath string) []string {
	var commands []string
	if !framework.NodeOSDistroIs("windows") {
		commands = []string{"/bin/sh", "-c", "echo '" + content + "' > " + fullPath}
	} else {
		commands = []string{"powershell", "/c", "echo '" + content + "' > " + fullPath}
	}
	return commands
}

// GenerateWriteFileCmd generates the corresponding command lines to write a file with the given content and file path.
// Depending on the Node OS is Windows or linux, the command will use powershell or /bin/sh
func GenerateWriteFileCmd(content, fullPath string) []string {
	var commands []string
	if !framework.NodeOSDistroIs("windows") {
		commands = []string{"/bin/sh", "-c", "echo '" + content + "' > " + fullPath}
	} else {
		commands = []string{"powershell", "/c", "echo '" + content + "' > " + fullPath}
	}
	return commands
}

// GenerateReadFileCmd generates the corresponding command lines to read from a file with the given file path.
// Depending on the Node OS is Windows or linux, the command will use powershell or /bin/sh
func GenerateReadFileCmd(fullPath string) []string {
	var commands []string
	if !framework.NodeOSDistroIs("windows") {
		commands = []string{"cat", fullPath}
	} else {
		commands = []string{"powershell", "/c", "type " + fullPath}
	}
	return commands
}

// GenerateReadBlockCmd generates the corresponding command lines to read from a block device with the given file path.
// Depending on the Node OS is Windows or linux, the command will use powershell or /bin/sh
func GenerateReadBlockCmd(fullPath string, numberOfCharacters int) []string {
	var commands []string
	if !framework.NodeOSDistroIs("windows") {
		commands = []string{"head", "-c", strconv.Itoa(numberOfCharacters), fullPath}
	} else {
		// TODO: is there a way on windows to get the first X bytes from a device?
		commands = []string{"powershell", "/c", "type " + fullPath}
	}
	return commands
}

// GenerateWriteandExecuteScriptFileCmd generates the corresponding command lines to write a file with the given file path
// and also execute this file.
// Depending on the Node OS is Windows or linux, the command will use powershell or /bin/sh
func GenerateWriteandExecuteScriptFileCmd(content, fileName, filePath string) []string {
	// for windows cluster, modify the Pod spec.
	if framework.NodeOSDistroIs("windows") {
		scriptName := fmt.Sprintf("%s.ps1", fileName)
		fullPath := filepath.Join(filePath, scriptName)

		cmd := "echo \"" + content + "\" > " + fullPath + "; .\\" + fullPath
		framework.Logf("generated pod command %s", cmd)
		return []string{"powershell", "/c", cmd}
	}
	scriptName := fmt.Sprintf("%s.sh", fileName)
	fullPath := filepath.Join(filePath, scriptName)
	cmd := fmt.Sprintf("echo \"%s\" > %s; chmod u+x %s; %s;", content, fullPath, fullPath, fullPath)
	return []string{"/bin/sh", "-ec", cmd}
}

// GenerateSecurityContext generates the corresponding container security context with the given inputs
// If the Node OS is windows, currently we will ignore the inputs and return nil.
// TODO: Will modify it after windows has its own security context
func GenerateSecurityContext(privileged bool) *v1.SecurityContext {
	if framework.NodeOSDistroIs("windows") {
		return nil
	}
	return &v1.SecurityContext{
		Privileged: &privileged,
	}
}

// GeneratePodSecurityContext generates the corresponding pod security context with the given inputs
// If the Node OS is windows, currently we will ignore the inputs and return nil.
// TODO: Will modify it after windows has its own security context
func GeneratePodSecurityContext(fsGroup *int64, seLinuxOptions *v1.SELinuxOptions) *v1.PodSecurityContext {
	if framework.NodeOSDistroIs("windows") {
		return nil
	}
	return &v1.PodSecurityContext{
		SELinuxOptions: seLinuxOptions,
		FSGroup:        fsGroup,
	}
}

// GetTestImage returns the image name with the given input
// If the Node OS is windows, currently we return Agnhost image for Windows node
// due to the issue of #https://github.com/kubernetes-sigs/windows-testing/pull/35.
func GetTestImage(image string) string {
	if framework.NodeOSDistroIs("windows") {
		return imageutils.GetE2EImage(imageutils.Agnhost)
	}
	return image
}
