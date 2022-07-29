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
	"context"
	"fmt"
	"path"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("[Feature:Flexvolumes] Mounted flexvolume volume expand [Slow]", func() {
	var (
		c                 clientset.Interface
		ns                string
		err               error
		pvc               *v1.PersistentVolumeClaim
		resizableSc       *storagev1.StorageClass
		nodeName          string
		isNodeLabeled     bool
		nodeKeyValueLabel map[string]string
		nodeLabelValue    string
		nodeKey           string
		node              *v1.Node
	)

	f := framework.NewDefaultFramework("mounted-flexvolume-expand")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("aws", "gce", "local")
		e2eskipper.SkipUnlessMasterOSDistroIs("debian", "ubuntu", "gci", "custom")
		e2eskipper.SkipUnlessNodeOSDistroIs("debian", "ubuntu", "gci", "custom")
		e2eskipper.SkipUnlessSSHKeyPresent()
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))
		var err error

		node, err = e2enode.GetRandomReadySchedulableNode(f.ClientSet)
		framework.ExpectNoError(err)
		nodeName = node.Name

		nodeKey = "mounted_flexvolume_expand"

		if !isNodeLabeled {
			nodeLabelValue = ns
			nodeKeyValueLabel = make(map[string]string)
			nodeKeyValueLabel[nodeKey] = nodeLabelValue
			framework.AddOrUpdateLabelOnNode(c, nodeName, nodeKey, nodeLabelValue)
			isNodeLabeled = true
		}

		test := testsuites.StorageClassTest{
			Name:                 "flexvolume-resize",
			Timeouts:             f.Timeouts,
			ClaimSize:            "2Gi",
			AllowVolumeExpansion: true,
			Provisioner:          "flex-expand",
		}

		resizableSc, err = c.StorageV1().StorageClasses().Create(context.TODO(), newStorageClass(test, ns, "resizing"), metav1.CreateOptions{})
		if err != nil {
			fmt.Printf("storage class creation error: %v\n", err)
		}
		framework.ExpectNoError(err, "Error creating resizable storage class: %v", err)
		framework.ExpectEqual(*resizableSc.AllowVolumeExpansion, true)

		pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			StorageClassName: &(resizableSc.Name),
			ClaimSize:        "2Gi",
		}, ns)
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(context.TODO(), pvc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating pvc: %v", err)

	})

	framework.AddCleanupAction(func() {
		if len(nodeLabelValue) > 0 {
			framework.RemoveLabelOffNode(c, nodeName, nodeKey)
		}
	})

	ginkgo.AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up resources for mounted volume resize")

		if c != nil {
			if errs := e2epv.PVPVCCleanup(c, ns, nil, pvc); len(errs) > 0 {
				framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pvc, nodeName, isNodeLabeled, nodeLabelValue = nil, "", false, ""
			nodeKeyValueLabel = make(map[string]string)
		}
	})

	ginkgo.It("should be resizable when mounted", func() {
		e2eskipper.SkipUnlessSSHKeyPresent()

		driver := "dummy-attachable"

		ginkgo.By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driver))
		installFlex(c, node, "k8s", driver, path.Join(driverDir, driver))
		ginkgo.By(fmt.Sprintf("installing flexvolume %s on (master) node %s as %s", path.Join(driverDir, driver), node.Name, driver))
		installFlex(c, nil, "k8s", driver, path.Join(driverDir, driver))

		pv := e2epv.MakePersistentVolume(e2epv.PersistentVolumeConfig{
			PVSource: v1.PersistentVolumeSource{
				FlexVolume: &v1.FlexPersistentVolumeSource{
					Driver: "k8s/" + driver,
				}},
			NamePrefix:       "pv-",
			StorageClassName: resizableSc.Name,
			VolumeMode:       pvc.Spec.VolumeMode,
		})

		_, err = e2epv.CreatePV(c, f.Timeouts, pv)
		framework.ExpectNoError(err, "Error creating pv %v", err)

		ginkgo.By("Waiting for PVC to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		var pvs []*v1.PersistentVolume

		pvs, err = e2epv.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)
		framework.ExpectEqual(len(pvs), 1)

		var pod *v1.Pod
		ginkgo.By("Creating pod")
		pod, err = createNginxPod(c, ns, nodeKeyValueLabel, pvcClaims)
		framework.ExpectNoError(err, "Failed to create pod %v", err)
		defer e2epod.DeletePodWithWait(c, pod)

		ginkgo.By("Waiting for pod to go to 'running' state")
		err = e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.ObjectMeta.Name, f.Namespace.Name)
		framework.ExpectNoError(err, "Pod didn't go to 'running' state %v", err)

		ginkgo.By("Expanding current pvc")
		newSize := resource.MustParse("6Gi")
		newPVC, err := testsuites.ExpandPVCSize(pvc, newSize, c)
		framework.ExpectNoError(err, "While updating pvc for more size")
		pvc = newPVC
		gomega.Expect(pvc).NotTo(gomega.BeNil())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) != 0 {
			framework.Failf("error updating pvc size %q", pvc.Name)
		}

		ginkgo.By("Waiting for cloudprovider resize to finish")
		err = testsuites.WaitForControllerVolumeResize(pvc, c, totalResizeWaitPeriod)
		framework.ExpectNoError(err, "While waiting for pvc resize to finish")

		ginkgo.By("Waiting for file system resize to finish")
		pvc, err = testsuites.WaitForFSResize(pvc, c)
		framework.ExpectNoError(err, "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
	})
})

// createNginxPod creates an nginx pod.
func createNginxPod(client clientset.Interface, namespace string, nodeSelector map[string]string, pvclaims []*v1.PersistentVolumeClaim) (*v1.Pod, error) {
	pod := makeNginxPod(namespace, nodeSelector, pvclaims)
	pod, err := client.CoreV1().Pods(namespace).Create(context.TODO(), pod, metav1.CreateOptions{})
	if err != nil {
		return nil, fmt.Errorf("pod Create API error: %v", err)
	}
	// Waiting for pod to be running
	err = e2epod.WaitForPodNameRunningInNamespace(client, pod.Name, namespace)
	if err != nil {
		return pod, fmt.Errorf("pod %q is not Running: %v", pod.Name, err)
	}
	// get fresh pod info
	pod, err = client.CoreV1().Pods(namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	if err != nil {
		return pod, fmt.Errorf("pod Get API error: %v", err)
	}
	return pod, nil
}

// makeNginxPod returns a pod definition based on the namespace using nginx image
func makeNginxPod(ns string, nodeSelector map[string]string, pvclaims []*v1.PersistentVolumeClaim) *v1.Pod {
	podSpec := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "pvc-tester-",
			Namespace:    ns,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "write-pod",
					Image: imageutils.GetE2EImage(imageutils.Nginx),
					Ports: []v1.ContainerPort{
						{
							Name:          "http-server",
							ContainerPort: 80,
						},
					},
				},
			},
		},
	}
	var volumeMounts = make([]v1.VolumeMount, len(pvclaims))
	var volumes = make([]v1.Volume, len(pvclaims))
	for index, pvclaim := range pvclaims {
		volumename := fmt.Sprintf("volume%v", index+1)
		volumeMounts[index] = v1.VolumeMount{Name: volumename, MountPath: "/mnt/" + volumename}
		volumes[index] = v1.Volume{Name: volumename, VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: pvclaim.Name, ReadOnly: false}}}
	}
	podSpec.Spec.Containers[0].VolumeMounts = volumeMounts
	podSpec.Spec.Volumes = volumes
	if nodeSelector != nil {
		podSpec.Spec.NodeSelector = nodeSelector
	}
	return podSpec
}
