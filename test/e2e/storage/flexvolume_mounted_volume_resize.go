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
	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
	"path"
)

var _ = utils.SIGDescribe(feature.Flexvolumes, "Mounted flexvolume expand", framework.WithSlow(), func() {
	var (
		c                 clientset.Interface
		ns                string
		err               error
		pvc               *v1.PersistentVolumeClaim
		resizableSc       *storagev1.StorageClass
		node              *v1.Node
		nodeName          string
		nodeKeyValueLabel map[string]string
		nodeLabelValue    string
		nodeKey           string
	)

	f := framework.NewDefaultFramework("mounted-flexvolume-expand")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	ginkgo.BeforeEach(func(ctx context.Context) {
		e2eskipper.SkipUnlessProviderIs("aws", "gce", "local")
		e2eskipper.SkipUnlessMasterOSDistroIs("debian", "ubuntu", "gci", "custom")
		e2eskipper.SkipUnlessNodeOSDistroIs("debian", "ubuntu", "gci", "custom")
		e2eskipper.SkipUnlessSSHKeyPresent()
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(e2enode.WaitForAllNodesSchedulable(ctx, c, f.Timeouts.NodeSchedulable))

		node, err = e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		nodeName = node.Name

		nodeKey = "mounted_flexvolume_expand_" + ns
		nodeLabelValue = ns
		nodeKeyValueLabel = map[string]string{nodeKey: nodeLabelValue}
		e2enode.AddOrUpdateLabelOnNode(c, nodeName, nodeKey, nodeLabelValue)
		ginkgo.DeferCleanup(e2enode.RemoveLabelOffNode, c, nodeName, nodeKey)

		test := testsuites.StorageClassTest{
			Name:                 "flexvolume-resize",
			Timeouts:             f.Timeouts,
			ClaimSize:            "2Gi",
			AllowVolumeExpansion: true,
			Provisioner:          "flex-expand",
		}

		resizableSc, err = c.StorageV1().StorageClasses().Create(ctx, newStorageClass(test, ns, "resizing"), metav1.CreateOptions{})
		if err != nil {
			fmt.Printf("storage class creation error: %v\n", err)
		}
		framework.ExpectNoError(err, "Error creating resizable storage class")
		if !*resizableSc.AllowVolumeExpansion {
			framework.Failf("Class %s does not allow volume expansion", resizableSc.Name)
		}

		pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			StorageClassName: &(resizableSc.Name),
			ClaimSize:        "2Gi",
		}, ns)
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(ctx, pvc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating pvc")
		ginkgo.DeferCleanup(func(ctx context.Context) {
			framework.Logf("AfterEach: Cleaning up resources for mounted volume resize")
			if errs := e2epv.PVPVCCleanup(ctx, c, ns, nil, pvc); len(errs) > 0 {
				framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
		})
	})

	ginkgo.It("Should verify mounted flex volumes can be resized", func(ctx context.Context) {
		driver := "dummy-attachable"
		ginkgo.By(fmt.Sprintf("installing flexvolume %s on node %s as %s", path.Join(driverDir, driver), node.Name, driver))
		installFlex(ctx, c, node, "k8s", driver, path.Join(driverDir, driver))
		ginkgo.By(fmt.Sprintf("installing flexvolume %s on (master) node %s as %s", path.Join(driverDir, driver), node.Name, driver))
		installFlex(ctx, c, nil, "k8s", driver, path.Join(driverDir, driver))

		pv := e2epv.MakePersistentVolume(e2epv.PersistentVolumeConfig{
			PVSource: v1.PersistentVolumeSource{
				FlexVolume: &v1.FlexPersistentVolumeSource{
					Driver: "k8s/" + driver,
				}},
			NamePrefix:       "pv-",
			StorageClassName: resizableSc.Name,
			VolumeMode:       pvc.Spec.VolumeMode,
		})

		_, err = e2epv.CreatePV(ctx, c, f.Timeouts, pv)
		framework.ExpectNoError(err, "Error creating pv %v", err)

		ginkgo.By("Waiting for PVC to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		var pvs []*v1.PersistentVolume

		pvs, err = e2epv.WaitForPVClaimBoundPhase(ctx, c, pvcClaims, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)
		gomega.Expect(pvs).To(gomega.HaveLen(1))

		ginkgo.By("Creating a deployment with the provisioned volume")
		deployment, err := e2edeployment.CreateDeployment(ctx, c, int32(1), map[string]string{"test": "app"}, nodeKeyValueLabel, ns, pvcClaims, admissionapi.LevelRestricted, "")
		framework.ExpectNoError(err, "Failed creating deployment %v", err)
		ginkgo.DeferCleanup(c.AppsV1().Deployments(ns).Delete, deployment.Name, metav1.DeleteOptions{})

		ginkgo.By("Expanding current pvc")
		newSize := resource.MustParse("6Gi")
		newPVC, err := testsuites.ExpandPVCSize(ctx, pvc, newSize, c)
		framework.ExpectNoError(err, "While updating pvc for more size")
		pvc = newPVC
		gomega.Expect(pvc).NotTo(gomega.BeNil())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) != 0 {
			framework.Failf("error updating pvc size %q", pvc.Name)
		}

		ginkgo.By("Waiting for cloudprovider resize to finish")
		err = testsuites.WaitForControllerVolumeResize(ctx, pvc, c, totalResizeWaitPeriod)
		framework.ExpectNoError(err, "While waiting for pvc resize to finish")

		ginkgo.By("Getting a pod from deployment")
		podList, err := e2edeployment.GetPodsForDeployment(ctx, c, deployment)
		framework.ExpectNoError(err, "While getting pods from deployment")
		gomega.Expect(podList.Items).NotTo(gomega.BeEmpty())
		pod := podList.Items[0]

		ginkgo.By("Deleting the pod from deployment")
		err = e2epod.DeletePodWithWait(ctx, c, &pod)
		framework.ExpectNoError(err, "while deleting pod for resizing")

		ginkgo.By("Waiting for deployment to create new pod")
		pod, err = waitForDeploymentToRecreatePod(ctx, c, deployment)
		framework.ExpectNoError(err, "While waiting for pod to be recreated")

		ginkgo.By("Waiting for file system resize to finish")
		pvc, err = testsuites.WaitForFSResize(ctx, pvc, c)
		framework.ExpectNoError(err, "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		gomega.Expect(pvcConditions).To(gomega.BeEmpty(), "pvc should not have conditions")
	})
})
