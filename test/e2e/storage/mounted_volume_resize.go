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
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/client/conditions"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("Mounted volume expand[Slow]", func() {
	var (
		c                 clientset.Interface
		ns                string
		err               error
		pvc               *v1.PersistentVolumeClaim
		resizableSc       *storage.StorageClass
		nodeName          string
		isNodeLabeled     bool
		nodeKeyValueLabel map[string]string
		nodeLabelValue    string
		nodeKey           string
	)

	f := framework.NewDefaultFramework("mounted-volume-expand")
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("aws", "gce")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))

		nodeList := framework.GetReadySchedulableNodesOrDie(f.ClientSet)
		if len(nodeList.Items) != 0 {
			nodeName = nodeList.Items[0].Name
		} else {
			framework.Failf("Unable to find ready and schedulable Node")
		}

		nodeKey = "mounted_volume_expand"

		if !isNodeLabeled {
			nodeLabelValue = ns
			nodeKeyValueLabel = make(map[string]string)
			nodeKeyValueLabel[nodeKey] = nodeLabelValue
			framework.AddOrUpdateLabelOnNode(c, nodeName, nodeKey, nodeLabelValue)
			isNodeLabeled = true
		}

		test := storageClassTest{
			name:      "default",
			claimSize: "2Gi",
		}
		resizableSc, err = createResizableStorageClass(test, ns, "resizing", c)
		Expect(err).NotTo(HaveOccurred(), "Error creating resizable storage class")
		Expect(*resizableSc.AllowVolumeExpansion).To(BeTrue())

		pvc = newClaim(test, ns, "default")
		pvc.Spec.StorageClassName = &resizableSc.Name
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
		Expect(err).NotTo(HaveOccurred(), "Error creating pvc")
	})

	framework.AddCleanupAction(func() {
		if len(nodeLabelValue) > 0 {
			framework.RemoveLabelOffNode(c, nodeName, nodeKey)
		}
	})

	AfterEach(func() {
		framework.Logf("AfterEach: Cleaning up resources for mounted volume resize")

		if c != nil {
			if errs := framework.PVPVCCleanup(c, ns, nil, pvc); len(errs) > 0 {
				framework.Failf("AfterEach: Failed to delete PVC and/or PV. Errors: %v", utilerrors.NewAggregate(errs))
			}
			pvc, nodeName, isNodeLabeled, nodeLabelValue = nil, "", false, ""
			nodeKeyValueLabel = make(map[string]string)
		}
	})

	It("Should verify mounted devices can be resized", func() {
		By("Waiting for PVC to be in bound phase")
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}
		pvs, err := framework.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		Expect(err).NotTo(HaveOccurred(), "Failed waiting for PVC to be bound %v", err)
		Expect(len(pvs)).To(Equal(1))

		By("Creating a deployment with the provisioned volume")
		deployment, err := framework.CreateDeployment(c, int32(1), map[string]string{"test": "app"}, nodeKeyValueLabel, ns, pvcClaims, "")
		Expect(err).NotTo(HaveOccurred(), "Failed creating deployment %v", err)
		defer c.AppsV1().Deployments(ns).Delete(deployment.Name, &metav1.DeleteOptions{})

		By("Expanding current pvc")
		newSize := resource.MustParse("6Gi")
		pvc, err = expandPVCSize(pvc, newSize, c)
		Expect(err).NotTo(HaveOccurred(), "While updating pvc for more size")
		Expect(pvc).NotTo(BeNil())

		pvcSize := pvc.Spec.Resources.Requests[v1.ResourceStorage]
		if pvcSize.Cmp(newSize) != 0 {
			framework.Failf("error updating pvc size %q", pvc.Name)
		}

		By("Waiting for cloudprovider resize to finish")
		err = waitForControllerVolumeResize(pvc, c)
		Expect(err).NotTo(HaveOccurred(), "While waiting for pvc resize to finish")

		By("Getting a pod from deployment")
		podList, err := framework.GetPodsForDeployment(c, deployment)
		Expect(podList.Items).NotTo(BeEmpty())
		pod := podList.Items[0]

		By("Deleting the pod from deployment")
		err = framework.DeletePodWithWait(f, c, &pod)
		Expect(err).NotTo(HaveOccurred(), "while deleting pod for resizing")

		By("Waiting for deployment to create new pod")
		pod, err = waitForDeploymentToRecreatePod(c, deployment)
		Expect(err).NotTo(HaveOccurred(), "While waiting for pod to be recreated")

		By("Waiting for file system resize to finish")
		pvc, err = waitForFSResize(pvc, c)
		Expect(err).NotTo(HaveOccurred(), "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		Expect(len(pvcConditions)).To(Equal(0), "pvc should not have conditions")
	})
})

func waitForDeploymentToRecreatePod(client clientset.Interface, deployment *apps.Deployment) (v1.Pod, error) {
	var runningPod v1.Pod
	waitErr := wait.PollImmediate(10*time.Second, 5*time.Minute, func() (bool, error) {
		podList, err := framework.GetPodsForDeployment(client, deployment)
		for _, pod := range podList.Items {
			switch pod.Status.Phase {
			case v1.PodRunning:
				runningPod = pod
				return true, nil
			case v1.PodFailed, v1.PodSucceeded:
				return false, conditions.ErrPodCompleted
			}
			return false, nil
		}
		return false, err
	})
	return runningPod, waitErr
}
