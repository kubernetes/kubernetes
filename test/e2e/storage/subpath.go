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
	"path/filepath"
	"strings"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var (
	volumePath    = "/test-volume"
	volumeName    = "test-volume"
	fileName      = "test-file"
	retryDuration = 180
	mountImage    = imageutils.GetE2EImage(imageutils.Mounttest)
)

type volInfo struct {
	source *v1.VolumeSource
	node   string
}

type volSource interface {
	createVolume(f *framework.Framework) volInfo
	cleanupVolume(f *framework.Framework)
}

var initVolSources = map[string]func() volSource{
	"hostPath":         initHostpath,
	"hostPathSymlink":  initHostpathSymlink,
	"emptyDir":         initEmptydir,
	"gcePD":            initGCEPD,
	"gcePDPartitioned": initGCEPDPartition,
	"nfs":              initNFS,
	"nfsPVC":           initNFSPVC,
	"gluster":          initGluster,
}

var _ = SIGDescribe("Subpath", func() {
	var (
		subPath           string
		subPathDir        string
		filePathInSubpath string
		filePathInVolume  string
		pod               *v1.Pod
		vol               volSource
	)

	f := framework.NewDefaultFramework("subpath")

	for volType, volInit := range initVolSources {
		curVolType := volType
		curVolInit := volInit

		Context(fmt.Sprintf("[Volume type: %v]", curVolType), func() {
			BeforeEach(func() {
				By(fmt.Sprintf("Initializing %s volume", curVolType))
				vol = curVolInit()
				subPath = f.Namespace.Name
				subPathDir = filepath.Join(volumePath, subPath)
				filePathInSubpath = filepath.Join(volumePath, fileName)
				filePathInVolume = filepath.Join(subPathDir, fileName)
				volInfo := vol.createVolume(f)
				pod = testPodSubpath(subPath, curVolType, volInfo.source)
				pod.Namespace = f.Namespace.Name
				pod.Spec.NodeName = volInfo.node
			})

			AfterEach(func() {
				By("Deleting pod")
				err := framework.DeletePodWithWait(f, f.ClientSet, pod)
				Expect(err).ToNot(HaveOccurred())

				By("Cleaning up volume")
				vol.cleanupVolume(f)
			})

			It("should support non-existent path", func() {
				// Write the file in the subPath from container 0
				setWriteCommand(filePathInSubpath, &pod.Spec.Containers[0])

				// Read it from outside the subPath from container 1
				testReadFile(f, filePathInVolume, pod, 1)
			})

			It("should support existing directory", func() {
				// Create the directory
				setInitCommand(pod, fmt.Sprintf("mkdir -p %s", subPathDir))

				// Write the file in the subPath from container 0
				setWriteCommand(filePathInSubpath, &pod.Spec.Containers[0])

				// Read it from outside the subPath from container 1
				testReadFile(f, filePathInVolume, pod, 1)
			})

			It("should support existing single file", func() {
				// Create the file in the init container
				setInitCommand(pod, fmt.Sprintf("mkdir -p %s; echo \"mount-tester new file\" > %s", subPathDir, filePathInVolume))

				// Read it from inside the subPath from container 0
				testReadFile(f, filePathInSubpath, pod, 0)
			})

			It("should fail if subpath directory is outside the volume [Slow]", func() {
				// Create the subpath outside the volume
				setInitCommand(pod, fmt.Sprintf("ln -s /bin %s", subPathDir))

				// Pod should fail
				testPodFailSupath(f, pod)
			})

			It("should fail if subpath file is outside the volume [Slow]", func() {
				// Create the subpath outside the volume
				setInitCommand(pod, fmt.Sprintf("ln -s /bin/sh %s", subPathDir))

				// Pod should fail
				testPodFailSupath(f, pod)
			})

			It("should fail if non-existent subpath is outside the volume [Slow]", func() {
				// Create the subpath outside the volume
				setInitCommand(pod, fmt.Sprintf("ln -s /bin/notanexistingpath %s", subPathDir))

				// Pod should fail
				testPodFailSupath(f, pod)
			})

			It("should fail if subpath with backstepping is outside the volume [Slow]", func() {
				// Create the subpath outside the volume
				setInitCommand(pod, fmt.Sprintf("ln -s ../ %s", subPathDir))

				// Pod should fail
				testPodFailSupath(f, pod)
			})

			It("should support creating multiple subpath from same volumes [Slow]", func() {
				subpathDir1 := filepath.Join(volumePath, "subpath1")
				subpathDir2 := filepath.Join(volumePath, "subpath2")
				filepath1 := filepath.Join("/test-subpath1", fileName)
				filepath2 := filepath.Join("/test-subpath2", fileName)
				setInitCommand(pod, fmt.Sprintf("mkdir -p %s; mkdir -p %s", subpathDir1, subpathDir2))

				addSubpathVolumeContainer(&pod.Spec.Containers[0], v1.VolumeMount{
					Name:      volumeName,
					MountPath: "/test-subpath1",
					SubPath:   "subpath1",
				})
				addSubpathVolumeContainer(&pod.Spec.Containers[0], v1.VolumeMount{
					Name:      volumeName,
					MountPath: "/test-subpath2",
					SubPath:   "subpath2",
				})

				addMultipleWrites(&pod.Spec.Containers[0], filepath1, filepath2)
				testMultipleReads(f, pod, 0, filepath1, filepath2)
			})

			It("should support restarting containers [Slow]", func() {
				// Create the directory
				setInitCommand(pod, fmt.Sprintf("mkdir -p %v", subPathDir))

				testPodContainerRestart(f, pod, filePathInVolume, filePathInSubpath)
			})
		})
	}

	// TODO: add a test case for the same disk with two partitions
})

func testPodSubpath(subpath, volumeType string, source *v1.VolumeSource) *v1.Pod {
	suffix := strings.ToLower(fmt.Sprintf("%s-%s", volumeType, rand.String(4)))
	privileged := true
	gracePeriod := int64(1)
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("pod-subpath-test-%s", suffix),
		},
		Spec: v1.PodSpec{
			InitContainers: []v1.Container{
				{
					Name:  fmt.Sprintf("init-volume-%s", suffix),
					Image: "busybox",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumePath,
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
			Containers: []v1.Container{
				{
					Name:  fmt.Sprintf("test-container-subpath-%s", suffix),
					Image: mountImage,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumePath,
							SubPath:   subpath,
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
				{
					Name:  fmt.Sprintf("test-container-volume-%s", suffix),
					Image: mountImage,
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: volumePath,
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
			RestartPolicy:                 v1.RestartPolicyNever,
			TerminationGracePeriodSeconds: &gracePeriod,
			Volumes: []v1.Volume{
				{
					Name:         volumeName,
					VolumeSource: *source,
				},
			},
		},
	}
}

func setInitCommand(pod *v1.Pod, command string) {
	pod.Spec.InitContainers[0].Command = []string{"/bin/sh", "-ec", command}
}

func setWriteCommand(file string, container *v1.Container) {
	container.Args = []string{
		fmt.Sprintf("--new_file_0644=%v", file),
		fmt.Sprintf("--file_mode=%v", file),
	}
}

func addSubpathVolumeContainer(container *v1.Container, volumeMount v1.VolumeMount) {
	existingMounts := container.VolumeMounts
	container.VolumeMounts = append(existingMounts, volumeMount)
}

func addMultipleWrites(container *v1.Container, file1 string, file2 string) {
	container.Args = []string{
		fmt.Sprintf("--new_file_0644=%v", file1),
		fmt.Sprintf("--new_file_0666=%v", file2),
	}
}

func testMultipleReads(f *framework.Framework, pod *v1.Pod, containerIndex int, file1 string, file2 string) {
	By(fmt.Sprintf("Creating pod %s", pod.Name))
	f.TestContainerOutput("multi_subpath", pod, containerIndex, []string{
		"content of file \"" + file1 + "\": mount-tester new file",
		"content of file \"" + file2 + "\": mount-tester new file",
	})
}

func setReadCommand(file string, container *v1.Container) {
	container.Args = []string{
		fmt.Sprintf("--file_content_in_loop=%v", file),
		fmt.Sprintf("--retry_time=%d", retryDuration),
	}
}

func testReadFile(f *framework.Framework, file string, pod *v1.Pod, containerIndex int) {
	setReadCommand(file, &pod.Spec.Containers[containerIndex])

	By(fmt.Sprintf("Creating pod %s", pod.Name))
	f.TestContainerOutput("subpath", pod, containerIndex, []string{
		"content of file \"" + file + "\": mount-tester new file",
	})
}

func testPodFailSupath(f *framework.Framework, pod *v1.Pod) {
	By(fmt.Sprintf("Creating pod %s", pod.Name))
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred())
	defer func() {
		framework.DeletePodWithWait(f, f.ClientSet, pod)
	}()
	err = framework.WaitTimeoutForPodRunningInNamespace(f.ClientSet, pod.Name, pod.Namespace, time.Minute)
	Expect(err).To(HaveOccurred())

	By("Checking for subpath error event")
	selector := fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      pod.Name,
		"involvedObject.namespace": f.Namespace.Name,
		"reason":                   "Failed",
	}.AsSelector().String()
	options := metav1.ListOptions{FieldSelector: selector}
	events, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(options)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(events.Items)).NotTo(Equal(0))
	Expect(events.Items[0].Message).To(ContainSubstring("subPath"))
}

func testPodContainerRestart(f *framework.Framework, pod *v1.Pod, fileInVolume, fileInSubpath string) {
	pod.Spec.RestartPolicy = v1.RestartPolicyOnFailure

	// Add liveness probe to subpath container
	pod.Spec.Containers[0].Image = "busybox"
	pod.Spec.Containers[0].Command = []string{"/bin/sh", "-ec", "sleep 100000"}
	pod.Spec.Containers[0].LivenessProbe = &v1.Probe{
		Handler: v1.Handler{
			Exec: &v1.ExecAction{
				Command: []string{"cat", fileInSubpath},
			},
		},
		InitialDelaySeconds: 15,
		FailureThreshold:    1,
		PeriodSeconds:       2,
	}

	// Set volume container to write file
	writeCmd := fmt.Sprintf("echo test > %v", fileInVolume)
	pod.Spec.Containers[1].Image = "busybox"
	pod.Spec.Containers[1].Command = []string{"/bin/sh", "-ec", fmt.Sprintf("%v; sleep 100000", writeCmd)}

	// Start pod
	By(fmt.Sprintf("Creating pod %s", pod.Name))
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred())

	err = framework.WaitTimeoutForPodRunningInNamespace(f.ClientSet, pod.Name, pod.Namespace, time.Minute)
	Expect(err).ToNot(HaveOccurred())

	By("Failing liveness probe")
	out, err := podContainerExec(pod, 1, fmt.Sprintf("rm %v", fileInVolume))
	framework.Logf("Pod exec output: %v", out)
	Expect(err).ToNot(HaveOccurred())

	// Check that container has restarted
	By("Waiting for container to restart")
	restarts := int32(0)
	err = wait.PollImmediate(10*time.Second, 2*time.Minute, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, status := range pod.Status.ContainerStatuses {
			if status.Name == pod.Spec.Containers[0].Name {
				framework.Logf("Container %v, restarts: %v", status.Name, status.RestartCount)
				restarts = status.RestartCount
				if restarts > 0 {
					framework.Logf("Container has restart count: %v", restarts)
					return true, nil
				}
			}
		}
		return false, nil
	})
	Expect(err).ToNot(HaveOccurred())

	// Fix liveness probe
	By("Rewriting the file")
	writeCmd = fmt.Sprintf("echo test-after > %v", fileInVolume)
	out, err = podContainerExec(pod, 1, writeCmd)
	framework.Logf("Pod exec output: %v", out)
	Expect(err).ToNot(HaveOccurred())

	// Wait for container restarts to stabilize
	By("Waiting for container to stop restarting")
	stableCount := int(0)
	stableThreshold := int(time.Minute / framework.Poll)
	err = wait.PollImmediate(framework.Poll, 2*time.Minute, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(pod.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, status := range pod.Status.ContainerStatuses {
			if status.Name == pod.Spec.Containers[0].Name {
				if status.RestartCount == restarts {
					stableCount++
					if stableCount > stableThreshold {
						framework.Logf("Container restart has stabilized")
						return true, nil
					}
				} else {
					restarts = status.RestartCount
					stableCount = 0
					framework.Logf("Container has restart count: %v", restarts)
				}
				break
			}
		}
		return false, nil
	})
	Expect(err).ToNot(HaveOccurred())

	// Verify content of file in subpath
	out, err = podContainerExec(pod, 0, fmt.Sprintf("cat %v", fileInSubpath))
	framework.Logf("Pod exec output: %v", out)
	Expect(err).ToNot(HaveOccurred())
	Expect(strings.TrimSpace(out)).To(BeEquivalentTo("test-after"))
}

func podContainerExec(pod *v1.Pod, containerIndex int, bashExec string) (string, error) {
	return framework.RunKubectl("exec", fmt.Sprintf("--namespace=%s", pod.Namespace), pod.Name, "--container", pod.Spec.Containers[containerIndex].Name, "--", "/bin/sh", "-c", bashExec)
}

type hostpathSource struct {
}

func initHostpath() volSource {
	return &hostpathSource{}
}

func (s *hostpathSource) createVolume(f *framework.Framework) volInfo {
	return volInfo{
		source: &v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: "/tmp",
			},
		},
	}
}

func (s *hostpathSource) cleanupVolume(f *framework.Framework) {
}

type hostpathSymlinkSource struct {
}

func initHostpathSymlink() volSource {
	return &hostpathSymlinkSource{}
}

func (s *hostpathSymlinkSource) createVolume(f *framework.Framework) volInfo {
	nodes := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
	Expect(len(nodes.Items)).NotTo(BeZero(), "No available nodes for scheduling")

	node0 := &nodes.Items[0]
	sourcePath := fmt.Sprintf("/tmp/%v", f.Namespace.Name)
	targetPath := fmt.Sprintf("/tmp/%v-link", f.Namespace.Name)
	cmd := fmt.Sprintf("mkdir %v -m 777 && ln -s %v %v", sourcePath, sourcePath, targetPath)
	privileged := true

	// Launch pod to initialize hostpath directory and symlink
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("hostpath-symlink-prep-%s", f.Namespace.Name),
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    fmt.Sprintf("init-volume-%s", f.Namespace.Name),
					Image:   "busybox",
					Command: []string{"/bin/sh", "-ec", cmd},
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      volumeName,
							MountPath: "/tmp",
						},
					},
					SecurityContext: &v1.SecurityContext{
						Privileged: &privileged,
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name: volumeName,
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/tmp",
						},
					},
				},
			},
			NodeName: node0.Name,
		},
	}
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(pod)
	Expect(err).ToNot(HaveOccurred())

	err = framework.WaitForPodSuccessInNamespace(f.ClientSet, pod.Name, pod.Namespace)
	Expect(err).ToNot(HaveOccurred())

	err = framework.DeletePodWithWait(f, f.ClientSet, pod)
	Expect(err).ToNot(HaveOccurred())

	return volInfo{
		source: &v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{
				Path: targetPath,
			},
		},
		node: node0.Name,
	}
}

func (s *hostpathSymlinkSource) cleanupVolume(f *framework.Framework) {
}

type emptydirSource struct {
}

func initEmptydir() volSource {
	return &emptydirSource{}
}

func (s *emptydirSource) createVolume(f *framework.Framework) volInfo {
	return volInfo{
		source: &v1.VolumeSource{
			EmptyDir: &v1.EmptyDirVolumeSource{},
		},
	}
}

func (s *emptydirSource) cleanupVolume(f *framework.Framework) {
}

type gcepdSource struct {
	diskName string
}

func initGCEPD() volSource {
	framework.SkipUnlessProviderIs("gce", "gke")
	return &gcepdSource{}
}

func (s *gcepdSource) createVolume(f *framework.Framework) volInfo {
	var err error

	framework.Logf("Creating GCE PD volume")
	s.diskName, err = framework.CreatePDWithRetry()
	framework.ExpectNoError(err, "Error creating PD")

	return volInfo{
		source: &v1.VolumeSource{
			GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{PDName: s.diskName},
		},
	}
}

func (s *gcepdSource) cleanupVolume(f *framework.Framework) {
	if s.diskName != "" {
		err := framework.DeletePDWithRetry(s.diskName)
		framework.ExpectNoError(err, "Error deleting PD")
	}
}

type gcepdPartitionSource struct {
	diskName string
}

func initGCEPDPartition() volSource {
	// Need to manually create, attach, partition, detach the GCE PD
	// with disk name "subpath-partitioned-disk" before running this test
	manual := true
	if manual {
		framework.Skipf("Skipping manual GCE PD partition test")
	}
	framework.SkipUnlessProviderIs("gce", "gke")
	return &gcepdPartitionSource{diskName: "subpath-partitioned-disk"}
}

func (s *gcepdPartitionSource) createVolume(f *framework.Framework) volInfo {
	// TODO: automate partitioned of GCE PD once it supports raw block volumes
	// framework.Logf("Creating GCE PD volume")
	// s.diskName, err = framework.CreatePDWithRetry()
	// framework.ExpectNoError(err, "Error creating PD")

	return volInfo{
		source: &v1.VolumeSource{
			GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
				PDName:    s.diskName,
				Partition: 1,
			},
		},
	}
}

func (s *gcepdPartitionSource) cleanupVolume(f *framework.Framework) {
	if s.diskName != "" {
		// err := framework.DeletePDWithRetry(s.diskName)
		// framework.ExpectNoError(err, "Error deleting PD")
	}
}

type nfsSource struct {
	serverPod *v1.Pod
}

func initNFS() volSource {
	return &nfsSource{}
}

func (s *nfsSource) createVolume(f *framework.Framework) volInfo {
	var serverIP string

	framework.Logf("Creating NFS server")
	_, s.serverPod, serverIP = framework.NewNFSServer(f.ClientSet, f.Namespace.Name, []string{"-G", "777", "/exports"})

	return volInfo{
		source: &v1.VolumeSource{
			NFS: &v1.NFSVolumeSource{
				Server: serverIP,
				Path:   "/exports",
			},
		},
	}
}

func (s *nfsSource) cleanupVolume(f *framework.Framework) {
	if s.serverPod != nil {
		framework.DeletePodWithWait(f, f.ClientSet, s.serverPod)
	}
}

type glusterSource struct {
	serverPod *v1.Pod
}

func initGluster() volSource {
	framework.SkipUnlessNodeOSDistroIs("gci", "ubuntu")
	return &glusterSource{}
}

func (s *glusterSource) createVolume(f *framework.Framework) volInfo {
	framework.Logf("Creating GlusterFS server")
	_, s.serverPod, _ = framework.NewGlusterfsServer(f.ClientSet, f.Namespace.Name)

	return volInfo{
		source: &v1.VolumeSource{
			Glusterfs: &v1.GlusterfsVolumeSource{
				EndpointsName: "gluster-server",
				Path:          "test_vol",
			},
		},
	}
}

func (s *glusterSource) cleanupVolume(f *framework.Framework) {
	if s.serverPod != nil {
		framework.DeletePodWithWait(f, f.ClientSet, s.serverPod)
		err := f.ClientSet.CoreV1().Endpoints(f.Namespace.Name).Delete("gluster-server", nil)
		Expect(err).NotTo(HaveOccurred(), "Gluster delete endpoints failed")
	}
}

// TODO: need a better way to wrap PVC.  A generic framework should support both static and dynamic PV.
// For static PV, can reuse createVolume methods for inline volumes
type nfsPVCSource struct {
	serverPod *v1.Pod
	pvc       *v1.PersistentVolumeClaim
	pv        *v1.PersistentVolume
}

func initNFSPVC() volSource {
	return &nfsPVCSource{}
}

func (s *nfsPVCSource) createVolume(f *framework.Framework) volInfo {
	var serverIP string

	framework.Logf("Creating NFS server")
	_, s.serverPod, serverIP = framework.NewNFSServer(f.ClientSet, f.Namespace.Name, []string{"-G", "777", "/exports"})

	pvConfig := framework.PersistentVolumeConfig{
		NamePrefix:       "nfs-",
		StorageClassName: f.Namespace.Name,
		PVSource: v1.PersistentVolumeSource{
			NFS: &v1.NFSVolumeSource{
				Server: serverIP,
				Path:   "/exports",
			},
		},
	}
	pvcConfig := framework.PersistentVolumeClaimConfig{
		StorageClassName: &f.Namespace.Name,
	}

	framework.Logf("Creating PVC and PV")
	pv, pvc, err := framework.CreatePVCPV(f.ClientSet, pvConfig, pvcConfig, f.Namespace.Name, false)
	Expect(err).NotTo(HaveOccurred(), "PVC, PV creation failed")

	err = framework.WaitOnPVandPVC(f.ClientSet, f.Namespace.Name, pv, pvc)
	Expect(err).NotTo(HaveOccurred(), "PVC, PV failed to bind")

	s.pvc = pvc
	s.pv = pv

	return volInfo{
		source: &v1.VolumeSource{
			PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
				ClaimName: pvc.Name,
			},
		},
	}
}

func (s *nfsPVCSource) cleanupVolume(f *framework.Framework) {
	if s.pvc != nil || s.pv != nil {
		if errs := framework.PVPVCCleanup(f.ClientSet, f.Namespace.Name, s.pv, s.pvc); len(errs) != 0 {
			framework.Failf("Failed to delete PVC or PV: %v", utilerrors.NewAggregate(errs))
		}
	}
	if s.serverPod != nil {
		framework.DeletePodWithWait(f, f.ClientSet, s.serverPod)
	}
}
