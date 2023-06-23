/*
Copyright 2023 The Kubernetes Authors.

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
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/drivers"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

/*
The Goal of this test is to do stress testing on NodeOutOfServiceVolumeDetach feature

This test assumes the following:
- The infra is GCP.
- Number of nodes in the Kubernetes cluster is at least 10, otherwise the test will be skipped.
- NodeOutOfServiceVolumeDetach feature is enabled.

This test performs the following:
 1. Deploys a gce-pd csi driver.
 2. Creates a gce-pd csi storage class.
 3. Taints 50% of nodes out of 'N' nodes.
 4. Creates a stateful set with "N/2" number of replicas.
 5. Un-taint the nodes that was done at step 3.
 6. Shutdowns the kubelet of node on which the STS app pods is scheduled.
    This shutdown is a non-graceful shutdown as by default the grace period is 0 on Kubelet.
    Also, this test ensures that at least 50% of nodes are untouched so that the pods
    can successfully get scheduled to these untouched(healthy) nodes.
    To achieve this, step 3 is performed.
 7. Adds `out-of-service` taint on the node which is shut down.
 8. Verifies that pod gets immediately scheduled to other different healthy nodes and gets into running and ready state.
 9. Starts the kubelet back on the other nodes.
 10. Removes the `out-of-service` taint from the node.
*/
const (
	// MinNodeRequired is the number of nodes required in the Kubernetes
	// cluster in order for this test to run.
	MinNodeRequired = 10
)

var _ = utils.SIGDescribe("[Feature:NodeOutOfServiceVolumeDetach] [Scalability] [Disruptive] [LinuxOnly] NonGracefulNodeShutdown", func() {
	var (
		c             clientset.Interface
		ns            string
		nodeListCache *v1.NodeList
	)
	f := framework.NewDefaultFramework("non-graceful-shutdown")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func(ctx context.Context) {
		c = f.ClientSet
		ns = f.Namespace.Name
		e2eskipper.SkipUnlessProviderIs("gce")
		var err error
		nodeListCache, err = e2enode.GetReadySchedulableNodes(ctx, c)
		if err != nil {
			framework.Logf("Failed to list node: %v", err)
		}
		if len(nodeListCache.Items) < MinNodeRequired {
			ginkgo.Skip("At least 2 nodes are required to run the test")
		}
	})

	ginkgo.Describe("[NonGracefulNodeShutdown] pod that uses a persistent volume via gce pd driver", func() {
		ginkgo.It("should get immediately rescheduled to a different node after non graceful node shutdown ", func(ctx context.Context) {
			// Install gce pd csi driver
			ginkgo.By("deploying csi gce-pd driver")
			driver := drivers.InitGcePDCSIDriver()
			config := driver.PrepareTest(ctx, f)
			dDriver, ok := driver.(storageframework.DynamicPVTestDriver)
			if !ok {
				e2eskipper.Skipf("csi driver expected DynamicPVTestDriver but got %v", driver)
			}
			ginkgo.By("Creating a gce-pd storage class")
			sc := dDriver.GetDynamicProvisionStorageClass(ctx, config, "")
			_, err := c.StorageV1().StorageClasses().Create(ctx, sc, metav1.CreateOptions{})
			framework.ExpectNoError(err, "failed to create a storage class")
			scName := &sc.Name

			// Taint 50% of nodes so that STS pods cannot get scheduled to those and hence these 50%
			// nodes does not go through shutdown.
			nodesToBeTainted := nodeListCache.Items[:len(nodeListCache.Items)/2]
			taintNew := v1.Taint{
				Key:    "STSPods",
				Effect: v1.TaintEffectNoSchedule,
			}
			for _, nodeName := range nodesToBeTainted {
				e2enode.AddOrUpdateTaintOnNode(ctx, c, nodeName.Name, taintNew)
			}

			// Create STS with 'N/2' replicas ( 'N' = Number of nodes in the k8s cluster ).
			replicaCount := int32(MinNodeRequired / 2)
			stsName := "sts-app-gcepd"
			podLabels := map[string]string{"app": stsName}
			podsList := createAndVerifySTS(ctx, scName, stsName, ns, &replicaCount, podLabels, c)
			oldNodeNameList := make([]string, 0)
			for _, pod := range podsList {
				oldNodeNameList = append(oldNodeNameList, pod.Spec.NodeName)
			}

			// stop the kubelet on nodes where the sts pods were scheduled.
			for _, pod := range podsList {
				ginkgo.By("Stopping the kubelet non gracefully for pod" + pod.Name)
				utils.KubeletCommand(ctx, utils.KStop, c, &pod)
				ginkgo.DeferCleanup(utils.KubeletCommand, utils.KStart, c, pod)
			}

			// tainting the shutdown nodes with `out-of-service` taint
			taint := v1.Taint{
				Key:    v1.TaintNodeOutOfService,
				Effect: v1.TaintEffectNoExecute,
			}
			for _, oldNodeName := range oldNodeNameList {
				ginkgo.By("Adding out of service taint on node " + oldNodeName)
				// taint this node as out-of-service node
				e2enode.AddOrUpdateTaintOnNode(ctx, c, oldNodeName, taint)
				ginkgo.DeferCleanup(e2enode.RemoveTaintOffNode, c, oldNodeName, taint)
			}

			ginkgo.By(fmt.Sprintf("Checking if the pods of sts %s got rescheduled to a new node", stsName))
			// building field selector such that pods on a particular list of nodes are not selected.
			nodeFieldSelector := make([]fields.Selector, 0)
			for _, v := range oldNodeNameList {
				nodeFieldSelector = append(nodeFieldSelector, fields.OneTermNotEqualSelector("spec.nodeName", v))
			}
			labelSelectorStr := labels.SelectorFromSet(podLabels).String()
			podListOpts := metav1.ListOptions{
				LabelSelector: labelSelectorStr,
				FieldSelector: fields.AndSelectors(nodeFieldSelector...).String(),
			}
			_, err = e2epod.WaitForPods(ctx, c, ns, podListOpts, e2epod.Range{MinMatching: int(replicaCount)}, framework.PodStartTimeout, "be running and ready", e2epod.RunningReady)
			framework.ExpectNoError(err)

			// Bring the nodes back online.
			for _, pod := range podsList {
				utils.KubeletCommand(ctx, utils.KStart, c, &pod)
			}

			// Remove the `out-of-service` taint on the nodes.
			for _, oldNodeName := range oldNodeNameList {
				e2enode.RemoveTaintOffNode(ctx, c, oldNodeName, taint)
			}

			// Verify that pods gets scheduled to the older nodes that was terminated non gracefully and now
			// is back online
			newSTSName := "sts-app-gcepd-new"
			newPodLabels := map[string]string{"app": newSTSName}
			newReplicaCount := int32(MinNodeRequired)
			createAndVerifySTS(ctx, scName, newSTSName, ns, &newReplicaCount, newPodLabels, c)
		})
	})
})

// createAndVerifyStatefulDeployment creates a statefulset
func createAndVerifySTS(ctx context.Context, scName *string, name, ns string, replicas *int32, podLabels map[string]string,
	c clientset.Interface) []v1.Pod {
	ginkgo.By("Creating a sts using the pvc")
	pvcName := name + "-pvc"
	sts := makeSTS(ns, name, pvcName, scName, replicas, podLabels)
	_, err := c.AppsV1().StatefulSets(ns).Create(ctx, sts, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to created the sts")
	ginkgo.By(fmt.Sprintf("Ensuring that the pod of sts %s is running and ready", sts.Name))
	labelSelector := labels.SelectorFromSet(podLabels)
	podList, err := e2epod.WaitForPodsWithLabelRunningReady(ctx, c, ns, labelSelector, 1, framework.PodStartTimeout)
	framework.ExpectNoError(err)
	pod := podList.Items
	return pod
}

// makeSTS returns a statefulset configuration
func makeSTS(ns, name, pvcName string, scName *string, replicas *int32, labels map[string]string) *appsv1.StatefulSet {
	stsApp := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: ns,
		},
		Spec: appsv1.StatefulSetSpec{
			Replicas: replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "sts-pod-nginx",
							Image: imageutils.GetE2EImage(imageutils.Agnhost),
							Command: []string{
								"/bin/sh",
								"-c",
								"while true; do echo $(date) >> /mnt/managed/outfile; sleep 1; done",
							},
							VolumeMounts: []v1.VolumeMount{
								{
									Name:      "managed",
									MountPath: "/mnt/managed",
								},
							},
						},
					},
				},
			},
			VolumeClaimTemplates: []v1.PersistentVolumeClaim{
				{
					ObjectMeta: metav1.ObjectMeta{
						Name: pvcName,
					},
					Spec: v1.PersistentVolumeClaimSpec{
						StorageClassName: scName,
						AccessModes: []v1.PersistentVolumeAccessMode{
							v1.ReadWriteOnce,
						},
						Resources: v1.ResourceRequirements{
							Requests: v1.ResourceList{
								v1.ResourceStorage: resource.MustParse("1Gi"), // Adjust the size as per your requirements
							},
						},
					},
				},
			},
		},
	}
	return stsApp
}
