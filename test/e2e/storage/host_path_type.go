/*
Copyright 2020 The Kubernetes Authors.

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
	"context"
	"fmt"
	"path"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
)

var _ = utils.SIGDescribe("HostPathType Directory [Slow]", func() {
	f := framework.NewDefaultFramework("host-path-type-directory")

	var (
		ns           string
		hostBaseDir  string
		mountBaseDir string
		basePod      *v1.Pod
		targetDir    string

		hostPathUnset             = v1.HostPathUnset
		hostPathDirectoryOrCreate = v1.HostPathDirectoryOrCreate
		hostPathDirectory         = v1.HostPathDirectory
		hostPathFile              = v1.HostPathFile
		hostPathSocket            = v1.HostPathSocket
		hostPathCharDev           = v1.HostPathCharDev
		hostPathBlockDev          = v1.HostPathBlockDev
	)

	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name

		ginkgo.By("Create a pod for further testing")
		hostBaseDir = path.Join("/tmp", ns)
		mountBaseDir = "/mnt/test"
		basePod = f.PodClient().CreateSync(newHostPathTypeTestPod(map[string]string{}, hostBaseDir, mountBaseDir, &hostPathDirectoryOrCreate))
		ginkgo.By(fmt.Sprintf("running on node %s", basePod.Spec.NodeName))
		targetDir = path.Join(hostBaseDir, "adir")
		ginkgo.By("Should automatically create a new directory 'adir' when HostPathType is HostPathDirectoryOrCreate")
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetDir, &hostPathDirectoryOrCreate)
	})

	ginkgo.It("Should fail on mounting non-existent directory 'does-not-exist-dir' when HostPathType is HostPathDirectory", func() {
		dirPath := path.Join(hostBaseDir, "does-not-exist-dir")
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			dirPath, fmt.Sprintf("%s is not a directory", dirPath), &hostPathDirectory)
	})

	ginkgo.It("Should be able to mount directory 'adir' successfully when HostPathType is HostPathDirectory", func() {
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetDir, &hostPathDirectory)
	})

	ginkgo.It("Should be able to mount directory 'adir' successfully when HostPathType is HostPathUnset", func() {
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetDir, &hostPathUnset)
	})

	ginkgo.It("Should fail on mounting directory 'adir' when HostPathType is HostPathFile", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetDir, fmt.Sprintf("%s is not a file", targetDir), &hostPathFile)
	})

	ginkgo.It("Should fail on mounting directory 'adir' when HostPathType is HostPathSocket", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetDir, fmt.Sprintf("%s is not a socket", targetDir), &hostPathSocket)
	})

	ginkgo.It("Should fail on mounting directory 'adir' when HostPathType is HostPathCharDev", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetDir, fmt.Sprintf("%s is not a character device", targetDir), &hostPathCharDev)
	})

	ginkgo.It("Should fail on mounting directory 'adir' when HostPathType is HostPathBlockDev", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetDir, fmt.Sprintf("%s is not a block device", targetDir), &hostPathBlockDev)
	})
})

var _ = utils.SIGDescribe("HostPathType File [Slow]", func() {
	f := framework.NewDefaultFramework("host-path-type-file")

	var (
		ns           string
		hostBaseDir  string
		mountBaseDir string
		basePod      *v1.Pod
		targetFile   string

		hostPathUnset             = v1.HostPathUnset
		hostPathDirectoryOrCreate = v1.HostPathDirectoryOrCreate
		hostPathDirectory         = v1.HostPathDirectory
		hostPathFileOrCreate      = v1.HostPathFileOrCreate
		hostPathFile              = v1.HostPathFile
		hostPathSocket            = v1.HostPathSocket
		hostPathCharDev           = v1.HostPathCharDev
		hostPathBlockDev          = v1.HostPathBlockDev
	)

	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name

		ginkgo.By("Create a pod for further testing")
		hostBaseDir = path.Join("/tmp", ns)
		mountBaseDir = "/mnt/test"
		basePod = f.PodClient().CreateSync(newHostPathTypeTestPod(map[string]string{}, hostBaseDir, mountBaseDir, &hostPathDirectoryOrCreate))
		ginkgo.By(fmt.Sprintf("running on node %s", basePod.Spec.NodeName))
		targetFile = path.Join(hostBaseDir, "afile")
		ginkgo.By("Should automatically create a new file 'afile' when HostPathType is HostPathFileOrCreate")
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetFile, &hostPathFileOrCreate)
	})

	ginkgo.It("Should fail on mounting non-existent file 'does-not-exist-file' when HostPathType is HostPathFile", func() {
		filePath := path.Join(hostBaseDir, "does-not-exist-file")
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			filePath, fmt.Sprintf("%s is not a file", filePath), &hostPathFile)
	})

	ginkgo.It("Should be able to mount file 'afile' successfully when HostPathType is HostPathFile", func() {
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetFile, &hostPathFile)
	})

	ginkgo.It("Should be able to mount file 'afile' successfully when HostPathType is HostPathUnset", func() {
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetFile, &hostPathUnset)
	})

	ginkgo.It("Should fail on mounting file 'afile' when HostPathType is HostPathDirectory", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetFile, fmt.Sprintf("%s is not a directory", targetFile), &hostPathDirectory)
	})

	ginkgo.It("Should fail on mounting file 'afile' when HostPathType is HostPathSocket", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetFile, fmt.Sprintf("%s is not a socket", targetFile), &hostPathSocket)
	})

	ginkgo.It("Should fail on mounting file 'afile' when HostPathType is HostPathCharDev", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetFile, fmt.Sprintf("%s is not a character device", targetFile), &hostPathCharDev)
	})

	ginkgo.It("Should fail on mounting file 'afile' when HostPathType is HostPathBlockDev", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetFile, fmt.Sprintf("%s is not a block device", targetFile), &hostPathBlockDev)
	})
})

var _ = utils.SIGDescribe("HostPathType Socket [Slow]", func() {
	f := framework.NewDefaultFramework("host-path-type-socket")

	var (
		ns           string
		hostBaseDir  string
		mountBaseDir string
		basePod      *v1.Pod
		targetSocket string

		hostPathUnset             = v1.HostPathUnset
		hostPathDirectoryOrCreate = v1.HostPathDirectoryOrCreate
		hostPathDirectory         = v1.HostPathDirectory
		hostPathFile              = v1.HostPathFile
		hostPathSocket            = v1.HostPathSocket
		hostPathCharDev           = v1.HostPathCharDev
		hostPathBlockDev          = v1.HostPathBlockDev
	)

	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name

		ginkgo.By("Create a pod for further testing")
		hostBaseDir = path.Join("/tmp", ns)
		mountBaseDir = "/mnt/test"
		basePod = f.PodClient().CreateSync(newHostPathTypeTestPodWithCommand(map[string]string{}, hostBaseDir, mountBaseDir, &hostPathDirectoryOrCreate, fmt.Sprintf("nc -lU %s", path.Join(mountBaseDir, "asocket"))))
		ginkgo.By(fmt.Sprintf("running on node %s", basePod.Spec.NodeName))
		targetSocket = path.Join(hostBaseDir, "asocket")
	})

	ginkgo.It("Should fail on mounting non-existent socket 'does-not-exist-socket' when HostPathType is HostPathSocket", func() {
		socketPath := path.Join(hostBaseDir, "does-not-exist-socket")
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			socketPath, fmt.Sprintf("%s is not a socket", socketPath), &hostPathSocket)
	})

	ginkgo.It("Should be able to mount socket 'asocket' successfully when HostPathType is HostPathSocket", func() {
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetSocket, &hostPathSocket)
	})

	ginkgo.It("Should be able to mount socket 'asocket' successfully when HostPathType is HostPathUnset", func() {
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetSocket, &hostPathUnset)
	})

	ginkgo.It("Should fail on mounting socket 'asocket' when HostPathType is HostPathDirectory", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetSocket, fmt.Sprintf("%s is not a directory", targetSocket), &hostPathDirectory)
	})

	ginkgo.It("Should fail on mounting socket 'asocket' when HostPathType is HostPathFile", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetSocket, fmt.Sprintf("%s is not a file", targetSocket), &hostPathFile)
	})

	ginkgo.It("Should fail on mounting socket 'asocket' when HostPathType is HostPathCharDev", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetSocket, fmt.Sprintf("%s is not a character device", targetSocket), &hostPathCharDev)
	})

	ginkgo.It("Should fail on mounting socket 'asocket' when HostPathType is HostPathBlockDev", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetSocket, fmt.Sprintf("%s is not a block device", targetSocket), &hostPathBlockDev)
	})
})

var _ = utils.SIGDescribe("HostPathType Character Device [Slow]", func() {
	f := framework.NewDefaultFramework("host-path-type-char-dev")

	var (
		ns            string
		hostBaseDir   string
		mountBaseDir  string
		basePod       *v1.Pod
		targetCharDev string

		hostPathUnset             = v1.HostPathUnset
		hostPathDirectoryOrCreate = v1.HostPathDirectoryOrCreate
		hostPathDirectory         = v1.HostPathDirectory
		hostPathFile              = v1.HostPathFile
		hostPathSocket            = v1.HostPathSocket
		hostPathCharDev           = v1.HostPathCharDev
		hostPathBlockDev          = v1.HostPathBlockDev
	)

	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name

		ginkgo.By("Create a pod for further testing")
		hostBaseDir = path.Join("/tmp", ns)
		mountBaseDir = "/mnt/test"
		basePod = f.PodClient().CreateSync(newHostPathTypeTestPod(map[string]string{}, hostBaseDir, mountBaseDir, &hostPathDirectoryOrCreate))
		ginkgo.By(fmt.Sprintf("running on node %s", basePod.Spec.NodeName))
		targetCharDev = path.Join(hostBaseDir, "achardev")
		ginkgo.By("Create a character device for further testing")
		cmd := fmt.Sprintf("mknod %s c 89 1", path.Join(mountBaseDir, "achardev"))
		stdout, stderr, err := utils.PodExec(f, basePod, cmd)
		framework.ExpectNoError(err, "command: %q, stdout: %s\nstderr: %s", cmd, stdout, stderr)
	})

	ginkgo.It("Should fail on mounting non-existent character device 'does-not-exist-char-dev' when HostPathType is HostPathCharDev", func() {
		charDevPath := path.Join(hostBaseDir, "does-not-exist-char-dev")
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			charDevPath, fmt.Sprintf("%s is not a character device", charDevPath), &hostPathCharDev)
	})

	ginkgo.It("Should be able to mount character device 'achardev' successfully when HostPathType is HostPathCharDev", func() {
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetCharDev, &hostPathCharDev)
	})

	ginkgo.It("Should be able to mount character device 'achardev' successfully when HostPathType is HostPathUnset", func() {
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetCharDev, &hostPathUnset)
	})

	ginkgo.It("Should fail on mounting character device 'achardev' when HostPathType is HostPathDirectory", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetCharDev, fmt.Sprintf("%s is not a directory", targetCharDev), &hostPathDirectory)
	})

	ginkgo.It("Should fail on mounting character device 'achardev' when HostPathType is HostPathFile", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetCharDev, fmt.Sprintf("%s is not a file", targetCharDev), &hostPathFile)
	})

	ginkgo.It("Should fail on mounting character device 'achardev' when HostPathType is HostPathSocket", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetCharDev, fmt.Sprintf("%s is not a socket", targetCharDev), &hostPathSocket)
	})

	ginkgo.It("Should fail on mounting character device 'achardev' when HostPathType is HostPathBlockDev", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetCharDev, fmt.Sprintf("%s is not a block device", targetCharDev), &hostPathBlockDev)
	})
})

var _ = utils.SIGDescribe("HostPathType Block Device [Slow]", func() {
	f := framework.NewDefaultFramework("host-path-type-block-dev")

	var (
		ns             string
		hostBaseDir    string
		mountBaseDir   string
		basePod        *v1.Pod
		targetBlockDev string

		hostPathUnset             = v1.HostPathUnset
		hostPathDirectoryOrCreate = v1.HostPathDirectoryOrCreate
		hostPathDirectory         = v1.HostPathDirectory
		hostPathFile              = v1.HostPathFile
		hostPathSocket            = v1.HostPathSocket
		hostPathCharDev           = v1.HostPathCharDev
		hostPathBlockDev          = v1.HostPathBlockDev
	)

	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name

		ginkgo.By("Create a pod for further testing")
		hostBaseDir = path.Join("/tmp", ns)
		mountBaseDir = "/mnt/test"
		basePod = f.PodClient().CreateSync(newHostPathTypeTestPod(map[string]string{}, hostBaseDir, mountBaseDir, &hostPathDirectoryOrCreate))
		ginkgo.By(fmt.Sprintf("running on node %s", basePod.Spec.NodeName))
		targetBlockDev = path.Join(hostBaseDir, "ablkdev")
		ginkgo.By("Create a block device for further testing")
		cmd := fmt.Sprintf("mknod %s b 89 1", path.Join(mountBaseDir, "ablkdev"))
		stdout, stderr, err := utils.PodExec(f, basePod, cmd)
		framework.ExpectNoError(err, "command %q: stdout: %s\nstderr: %s", cmd, stdout, stderr)
	})

	ginkgo.It("Should fail on mounting non-existent block device 'does-not-exist-blk-dev' when HostPathType is HostPathBlockDev", func() {
		blkDevPath := path.Join(hostBaseDir, "does-not-exist-blk-dev")
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			blkDevPath, fmt.Sprintf("%s is not a block device", blkDevPath), &hostPathBlockDev)
	})

	ginkgo.It("Should be able to mount block device 'ablkdev' successfully when HostPathType is HostPathBlockDev", func() {
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetBlockDev, &hostPathBlockDev)
	})

	ginkgo.It("Should be able to mount block device 'ablkdev' successfully when HostPathType is HostPathUnset", func() {
		verifyPodHostPathType(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName}, targetBlockDev, &hostPathUnset)
	})

	ginkgo.It("Should fail on mounting block device 'ablkdev' when HostPathType is HostPathDirectory", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetBlockDev, fmt.Sprintf("%s is not a directory", targetBlockDev), &hostPathDirectory)
	})

	ginkgo.It("Should fail on mounting block device 'ablkdev' when HostPathType is HostPathFile", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetBlockDev, fmt.Sprintf("%s is not a file", targetBlockDev), &hostPathFile)
	})

	ginkgo.It("Should fail on mounting block device 'ablkdev' when HostPathType is HostPathSocket", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetBlockDev, fmt.Sprintf("%s is not a socket", targetBlockDev), &hostPathSocket)
	})

	ginkgo.It("Should fail on mounting block device 'ablkdev' when HostPathType is HostPathCharDev", func() {
		verifyPodHostPathTypeFailure(f, map[string]string{"kubernetes.io/hostname": basePod.Spec.NodeName},
			targetBlockDev, fmt.Sprintf("%s is not a character device", targetBlockDev), &hostPathCharDev)
	})
})

func newHostPathTypeTestPod(nodeSelector map[string]string, hostDir, mountDir string, hostPathType *v1.HostPathType) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-hostpath-type-",
		},
		Spec: v1.PodSpec{
			NodeSelector:  nodeSelector,
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "host-path-testing",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "host",
							MountPath: mountDir,
							ReadOnly:  false,
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "host",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: hostDir,
							Type: hostPathType,
						},
					},
				},
			},
		},
	}
	return pod
}

func newHostPathTypeTestPodWithCommand(nodeSelector map[string]string, hostDir, mountDir string, hostPathType *v1.HostPathType, command string) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-hostpath-type-",
		},
		Spec: v1.PodSpec{
			NodeSelector:  nodeSelector,
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "host-path-sh-testing",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "host",
							MountPath: mountDir,
							ReadOnly:  false,
						},
					},
					Command: []string{"sh"},
					Args:    []string{"-c", command},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "host",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: hostDir,
							Type: hostPathType,
						},
					},
				},
			},
		},
	}
	return pod
}

func verifyPodHostPathTypeFailure(f *framework.Framework, nodeSelector map[string]string, hostDir, pattern string, hostPathType *v1.HostPathType) {
	pod := newHostPathTypeTestPod(nodeSelector, hostDir, "/mnt/test", hostPathType)
	ginkgo.By(fmt.Sprintf("Creating pod %s", pod.Name))
	pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("Checking for HostPathType error event")
	eventSelector := fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      pod.Name,
		"involvedObject.namespace": f.Namespace.Name,
		"reason":                   events.FailedMountVolume,
	}.AsSelector().String()
	msg := "hostPath type check failed"

	err = e2eevents.WaitTimeoutForEvent(f.ClientSet, f.Namespace.Name, eventSelector, msg, framework.PodStartTimeout)
	// Events are unreliable, don't depend on the event. It's used only to speed up the test.
	if err != nil {
		framework.Logf("Warning: did not get event about FailedMountVolume")
	}

	// Check the pod is still not running
	p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "could not re-read the pod after event (or timeout)")
	framework.ExpectEqual(p.Status.Phase, v1.PodPending, "Pod phase isn't pending")

	f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), pod.Name, *metav1.NewDeleteOptions(0))
}

func verifyPodHostPathType(f *framework.Framework, nodeSelector map[string]string, hostDir string, hostPathType *v1.HostPathType) {
	newPod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(),
		newHostPathTypeTestPod(nodeSelector, hostDir, "/mnt/test", hostPathType), metav1.CreateOptions{})
	framework.ExpectNoError(err)
	framework.ExpectNoError(e2epod.WaitTimeoutForPodRunningInNamespace(f.ClientSet, newPod.Name, newPod.Namespace, framework.PodStartShortTimeout))

	f.ClientSet.CoreV1().Pods(f.Namespace.Name).Delete(context.TODO(), newPod.Name, *metav1.NewDeleteOptions(0))
}
