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

package vsphere

import (
	"context"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	ext4FSType    = "ext4"
	ext3FSType    = "ext3"
	invalidFSType = "ext10"
	execCommand   = "/bin/df -T /mnt/volume1 | /bin/awk 'FNR == 2 {print $2}' > /mnt/volume1/fstype && while true ; do sleep 2 ; done"
)

/*
	Test to verify fstype specified in storage-class is being honored after volume creation.

	Steps
	1. Create StorageClass with fstype set to valid type (default case included).
	2. Create PVC which uses the StorageClass created in step 1.
	3. Wait for PV to be provisioned.
	4. Wait for PVC's status to become Bound.
	5. Create pod using PVC on specific node.
	6. Wait for Disk to be attached to the node.
	7. Execute command in the pod to get fstype.
	8. Delete pod and Wait for Volume Disk to be detached from the Node.
	9. Delete PVC, PV and Storage Class.

	Test to verify if an invalid fstype specified in storage class fails pod creation.

	Steps
	1. Create StorageClass with invalid.
	2. Create PVC which uses the StorageClass created in step 1.
	3. Wait for PV to be provisioned.
	4. Wait for PVC's status to become Bound.
	5. Create pod using PVC.
	6. Verify if the pod creation fails.
	7. Verify if the MountVolume.MountDevice fails because it is unable to find the file system executable file on the node.
*/

var _ = utils.SIGDescribe("Volume FStype", feature.Vsphere, func() {
	f := framework.NewDefaultFramework("volume-fstype")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var (
		client clientset.Interface
	)
	ginkgo.BeforeEach(func(ctx context.Context) {
		e2eskipper.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		gomega.Expect(GetReadySchedulableNodeInfos(ctx, client)).NotTo(gomega.BeEmpty())
	})

	ginkgo.It("verify fstype - ext3 formatted volume", func(ctx context.Context) {
		ginkgo.By("Invoking Test for fstype: ext3")
		invokeTestForFstype(ctx, f, ext3FSType, ext3FSType)
	})

	ginkgo.It("verify fstype - default value should be ext4", func(ctx context.Context) {
		ginkgo.By("Invoking Test for fstype: Default Value - ext4")
		invokeTestForFstype(ctx, f, "", ext4FSType)
	})

	ginkgo.It("verify invalid fstype", func(ctx context.Context) {
		ginkgo.By("Invoking Test for fstype: invalid Value")
		invokeTestForInvalidFstype(ctx, f, client, invalidFSType)
	})
})

func invokeTestForFstype(ctx context.Context, f *framework.Framework, fstype string, expectedContent string) {
	framework.Logf("Invoking Test for fstype: %s", fstype)
	namespace := f.Namespace.Name
	scParameters := make(map[string]string)
	scParameters["fstype"] = fstype

	// Create Persistent Volume
	ginkgo.By("Creating Storage Class With Fstype")
	pvclaim, persistentvolumes := createVolume(ctx, f.ClientSet, f.Timeouts, f.Namespace.Name, scParameters)

	// Create Pod and verify the persistent volume is accessible
	pod := createPodAndVerifyVolumeAccessible(ctx, f, pvclaim, persistentvolumes)
	_, err := e2eoutput.LookForStringInPodExec(namespace, pod.Name, []string{"/bin/cat", "/mnt/volume1/fstype"}, expectedContent, time.Minute)
	framework.ExpectNoError(err)

	// Detach and delete volume
	detachVolume(ctx, f, pod, persistentvolumes[0].Spec.VsphereVolume.VolumePath)
	err = e2epv.DeletePersistentVolumeClaim(ctx, f.ClientSet, pvclaim.Name, namespace)
	framework.ExpectNoError(err)
}

func invokeTestForInvalidFstype(ctx context.Context, f *framework.Framework, client clientset.Interface, fstype string) {
	namespace := f.Namespace.Name
	scParameters := make(map[string]string)
	scParameters["fstype"] = fstype

	// Create Persistent Volume
	ginkgo.By("Creating Storage Class With Invalid Fstype")
	pvclaim, persistentvolumes := createVolume(ctx, client, f.Timeouts, namespace, scParameters)

	ginkgo.By("Creating pod to attach PV to the node")
	var pvclaims []*v1.PersistentVolumeClaim
	pvclaims = append(pvclaims, pvclaim)
	// Create pod to attach Volume to Node
	pod, err := e2epod.CreatePod(ctx, client, namespace, nil, pvclaims, f.NamespacePodSecurityLevel, execCommand)
	framework.ExpectError(err)

	eventList, err := client.CoreV1().Events(namespace).List(ctx, metav1.ListOptions{})
	framework.ExpectNoError(err)

	// Detach and delete volume
	detachVolume(ctx, f, pod, persistentvolumes[0].Spec.VsphereVolume.VolumePath)
	err = e2epv.DeletePersistentVolumeClaim(ctx, client, pvclaim.Name, namespace)
	framework.ExpectNoError(err)

	gomega.Expect(eventList.Items).NotTo(gomega.BeEmpty())
	errorMsg := `MountVolume.MountDevice failed for volume "` + persistentvolumes[0].Name + `" : executable file not found`
	isFound := false
	for _, item := range eventList.Items {
		if strings.Contains(item.Message, errorMsg) {
			isFound = true
		}
	}
	if !isFound {
		framework.Failf("Unable to verify MountVolume.MountDevice failure for volume %s", persistentvolumes[0].Name)
	}
}

func createVolume(ctx context.Context, client clientset.Interface, timeouts *framework.TimeoutContext, namespace string, scParameters map[string]string) (*v1.PersistentVolumeClaim, []*v1.PersistentVolume) {
	storageclass, err := client.StorageV1().StorageClasses().Create(ctx, getVSphereStorageClassSpec("fstype", scParameters, nil, ""), metav1.CreateOptions{})
	framework.ExpectNoError(err)
	ginkgo.DeferCleanup(framework.IgnoreNotFound(client.StorageV1().StorageClasses().Delete), storageclass.Name, metav1.DeleteOptions{})

	ginkgo.By("Creating PVC using the Storage Class")
	pvclaim, err := client.CoreV1().PersistentVolumeClaims(namespace).Create(ctx, getVSphereClaimSpecWithStorageClass(namespace, "2Gi", storageclass), metav1.CreateOptions{})
	framework.ExpectNoError(err)

	var pvclaims []*v1.PersistentVolumeClaim
	pvclaims = append(pvclaims, pvclaim)
	ginkgo.By("Waiting for claim to be in bound phase")
	persistentvolumes, err := e2epv.WaitForPVClaimBoundPhase(ctx, client, pvclaims, timeouts.ClaimProvision)
	framework.ExpectNoError(err)
	return pvclaim, persistentvolumes
}

func createPodAndVerifyVolumeAccessible(ctx context.Context, f *framework.Framework, pvclaim *v1.PersistentVolumeClaim, persistentvolumes []*v1.PersistentVolume) *v1.Pod {
	var pvclaims []*v1.PersistentVolumeClaim
	pvclaims = append(pvclaims, pvclaim)
	ginkgo.By("Creating pod to attach PV to the node")
	// Create pod to attach Volume to Node
	pod, err := e2epod.CreatePod(ctx, f.ClientSet, f.Namespace.Name, nil, pvclaims, f.NamespacePodSecurityLevel, execCommand)
	framework.ExpectNoError(err)

	// Asserts: Right disk is attached to the pod
	ginkgo.By("Verify the volume is accessible and available in the pod")
	verifyVSphereVolumesAccessible(ctx, f.ClientSet, pod, persistentvolumes)
	return pod
}

// detachVolume delete the volume passed in the argument and wait until volume is detached from the node,
func detachVolume(ctx context.Context, f *framework.Framework, pod *v1.Pod, volPath string) {
	pod, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	nodeName := pod.Spec.NodeName
	ginkgo.By("Deleting pod")
	err = e2epod.DeletePodWithWait(ctx, f.ClientSet, pod)
	framework.ExpectNoError(err)

	ginkgo.By("Waiting for volumes to be detached from the node")
	framework.ExpectNoError(waitForVSphereDiskToDetach(ctx, volPath, nodeName))
}
