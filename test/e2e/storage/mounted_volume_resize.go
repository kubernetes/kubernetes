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
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	admissionapi "k8s.io/pod-security-admission/api"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/client/conditions"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

var _ = utils.SIGDescribe("Mounted volume expand [Feature:StorageProvider]", func() {
	var (
		c                 clientset.Interface
		ns                string
		pvc               *v1.PersistentVolumeClaim
		sc                *storagev1.StorageClass
		cleanStorageClass func()
		nodeName          string
		isNodeLabeled     bool
		nodeKeyValueLabel map[string]string
		nodeLabelValue    string
		nodeKey           string
	)

	f := framework.NewDefaultFramework("mounted-volume-expand")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("aws", "gce")
		c = f.ClientSet
		ns = f.Namespace.Name
		framework.ExpectNoError(framework.WaitForAllNodesSchedulable(c, framework.TestContext.NodeSchedulableTimeout))

		node, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
		framework.ExpectNoError(err)
		nodeName = node.Name

		nodeKey = "mounted_volume_expand"

		if !isNodeLabeled {
			nodeLabelValue = ns
			nodeKeyValueLabel = make(map[string]string)
			nodeKeyValueLabel[nodeKey] = nodeLabelValue
			framework.AddOrUpdateLabelOnNode(c, nodeName, nodeKey, nodeLabelValue)
			isNodeLabeled = true
		}

		test := testsuites.StorageClassTest{
			Name:                 "default",
			Timeouts:             f.Timeouts,
			ClaimSize:            "2Gi",
			AllowVolumeExpansion: true,
			DelayBinding:         true,
			Parameters:           make(map[string]string),
		}

		sc, cleanStorageClass = testsuites.SetupStorageClass(c, newStorageClass(test, ns, "resizing"))
		if !*sc.AllowVolumeExpansion {
			framework.Failf("Class %s does not allow volume expansion", sc.Name)
		}

		pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        test.ClaimSize,
			StorageClassName: &(sc.Name),
			VolumeMode:       &test.VolumeMode,
		}, ns)
		pvc, err = c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(context.TODO(), pvc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Error creating pvc")
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

		cleanStorageClass()
	})

	ginkgo.It("Should verify mounted devices can be resized", func() {
		pvcClaims := []*v1.PersistentVolumeClaim{pvc}

		// The reason we use a node selector is because we do not want pod to move to different node when pod is deleted.
		// Keeping pod on same node reproduces the scenario that volume might already be mounted when resize is attempted.
		// We should consider adding a unit test that exercises this better.
		ginkgo.By("Creating a deployment with selected PVC")
		deployment, err := e2edeployment.CreateDeployment(c, int32(1), map[string]string{"test": "app"}, nodeKeyValueLabel, ns, pvcClaims, "")
		framework.ExpectNoError(err, "Failed creating deployment %v", err)
		defer c.AppsV1().Deployments(ns).Delete(context.TODO(), deployment.Name, metav1.DeleteOptions{})

		// PVC should be bound at this point
		ginkgo.By("Checking for bound PVC")
		pvs, err := e2epv.WaitForPVClaimBoundPhase(c, pvcClaims, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err, "Failed waiting for PVC to be bound %v", err)
		framework.ExpectEqual(len(pvs), 1)

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

		ginkgo.By("Getting a pod from deployment")
		podList, err := e2edeployment.GetPodsForDeployment(c, deployment)
		framework.ExpectNoError(err, "While getting pods from deployment")
		gomega.Expect(podList.Items).NotTo(gomega.BeEmpty())
		pod := podList.Items[0]

		ginkgo.By("Deleting the pod from deployment")
		err = e2epod.DeletePodWithWait(c, &pod)
		framework.ExpectNoError(err, "while deleting pod for resizing")

		ginkgo.By("Waiting for deployment to create new pod")
		pod, err = waitForDeploymentToRecreatePod(c, deployment)
		framework.ExpectNoError(err, "While waiting for pod to be recreated")

		ginkgo.By("Waiting for file system resize to finish")
		pvc, err = testsuites.WaitForFSResize(pvc, c)
		framework.ExpectNoError(err, "while waiting for fs resize to finish")

		pvcConditions := pvc.Status.Conditions
		framework.ExpectEqual(len(pvcConditions), 0, "pvc should not have conditions")
	})
})

func waitForDeploymentToRecreatePod(client clientset.Interface, deployment *appsv1.Deployment) (v1.Pod, error) {
	var runningPod v1.Pod
	waitErr := wait.PollImmediate(10*time.Second, 5*time.Minute, func() (bool, error) {
		podList, err := e2edeployment.GetPodsForDeployment(client, deployment)
		if err != nil {
			return false, fmt.Errorf("failed to get pods for deployment: %v", err)
		}
		for _, pod := range podList.Items {
			switch pod.Status.Phase {
			case v1.PodRunning:
				runningPod = pod
				return true, nil
			case v1.PodFailed, v1.PodSucceeded:
				return false, conditions.ErrPodCompleted
			}
		}
		return false, nil
	})
	if waitErr != nil {
		return runningPod, fmt.Errorf("error waiting for recreated pod: %v", waitErr)
	}
	return runningPod, nil
}
