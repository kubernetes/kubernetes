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

package upgrades

import (
	"context"
	"fmt"
	"time"

	"github.com/google/go-cmp/cmp"
	g "github.com/onsi/ginkgo"
	o "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeploy "k8s.io/kubernetes/test/e2e/framework/deployment"
	"k8s.io/kubernetes/test/e2e/upgrades"

	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	deploymentName = "dp"
	// poll is how often to poll pods, nodes and claims.
	poll            = 2 * time.Second
	pollLongTimeout = 5 * time.Minute
)

// TODO: Test that the deployment stays available during master (and maybe
// node and cluster upgrades).

type DeploymentSnapshot struct {
	Deployment  *appsv1.Deployment
	ReplicaSets map[types.UID]*appsv1.ReplicaSet
	// Pods        map[types.UID]*corev1.Pod
}

func snaphotDeployment(ctx context.Context, c clientset.Interface, namespace, name string) (*DeploymentSnapshot, error) {
	var err error
	var s DeploymentSnapshot

	s.Deployment, err = c.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("can't get deployment: %w", err)
	}

	rsSelector, err := metav1.LabelSelectorAsSelector(s.Deployment.Spec.Selector)
	if err != nil {
		return nil, fmt.Errorf("can't create RS selector: %w", err)
	}
	rsCandidateList, err := c.AppsV1().ReplicaSets(s.Deployment.Namespace).List(ctx, metav1.ListOptions{
		LabelSelector: rsSelector.String(),
	})
	s.ReplicaSets = make(map[types.UID]*appsv1.ReplicaSet, rsCandidateList.Size())
	for i := range rsCandidateList.Items {
		rs := &(rsCandidateList.Items[i])
		if metav1.IsControlledBy(rs, s.Deployment) {
			s.ReplicaSets[rs.UID] = rs
		}
	}

	// s.Pods = make(map[types.UID][types.UID]*corev1.Pod)
	// for _, rs := range s.ReplicaSets {
	// 	podSelector, err := metav1.LabelSelectorAsSelector(rs.Spec.Selector)
	// 	if err != nil {
	// 		return nil, fmt.Errorf("can't create Pod selector: %w", err)
	// 	}
	// 	podCandidateList, err := c.CoreV1().Pods(rs.Namespace).List(ctx, metav1.ListOptions{
	// 		LabelSelector: podSelector.String(),
	// 	})
	// 	for i := range podCandidateList.Items {
	// 		pod := &(podCandidateList.Items[i])
	// 		if metav1.IsControlledBy(pod, rs) {
	// 			s.Pods[pod.UID] = pod
	// 		}
	// 	}
	// }

	return &s, nil
}

// DeploymentUpgradeTest tests that a deployment is using the same replica
// sets before and after a cluster upgrade.
type DeploymentUpgradeTest struct {
	old *DeploymentSnapshot
}

// Name returns the tracking name of the test.
func (DeploymentUpgradeTest) Name() string { return "[sig-apps] deployment-upgrade" }

// Setup creates a deployment and makes sure it has a new and an old replicaset running.
func (t *DeploymentUpgradeTest) Setup(f *framework.Framework) {
	ctx := context.TODO()
	c := f.ClientSet

	g.By("Creating a deployment with 1 replica")
	d := e2edeploy.NewDeployment(
		deploymentName,
		int32(1),
		map[string]string{"test": "upgrade"},
		"nginx",
		imageutils.GetE2EImage(imageutils.Nginx),
		appsv1.RollingUpdateDeploymentStrategyType,
	)
	deployment, err := c.AppsV1().Deployments(f.Namespace.Name).Create(ctx, d, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	g.By("Waiting deployment to complete")
	framework.ExpectNoError(e2edeploy.WaitForDeploymentComplete(c, deployment))

	g.By("Waiting for revision of the deployment to become 1")
	framework.ExpectNoError(waitForDeploymentRevision(c, deployment, "1"))

	// Trigger a new rollout so that we have some history.
	g.By("Triggering a new rollout")
	deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, deployment.Namespace, deploymentName, func(update *appsv1.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = "updated-name"
	})
	framework.ExpectNoError(err)

	g.By("Waiting for revision to become 2")
	framework.ExpectNoError(waitForDeploymentRevision(c, deployment, "2"))

	g.By("Waiting deployment to complete")
	framework.ExpectNoError(e2edeploy.WaitForDeploymentComplete(c, deployment))

	t.old, err = snaphotDeployment(ctx, c, deployment.Namespace, deployment.Name)
	framework.ExpectNoError(err)
	// sanity check
	framework.ExpectEqual(t.old.Deployment.UID, deployment.UID)
}

// Test checks whether the replicasets for a deployment are the same after an upgrade.
func (t *DeploymentUpgradeTest) Test(f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	// Block until upgrade is done
	g.By(fmt.Sprintf("Waiting for upgrade to finish before checking replicasets for deployment %q", deploymentName))
	<-done

	ctx := context.TODO()
	c := f.ClientSet

	old := t.old

	g.By("Making a dummy update on the deployment")
	deployment, err := e2edeploy.UpdateDeploymentWithRetries(c, old.Deployment.Namespace, old.Deployment.Name, func(d *appsv1.Deployment) {
		if d.Annotations == nil {
			d.Annotations = map[string]string{}
		}
		d.Annotations["dummyAnnotationToTestDefaultingOnUpdate"] = "42"
	})
	framework.ExpectNoError(err)

	// If the nodes were restarted during cluster upgrade we need to wait for the deployment to be fully available
	g.By("Waiting deployment to be available")
	framework.ExpectNoError(e2edeploy.WaitForDeploymentComplete(c, old.Deployment))

	g.By("Snaphotting the Deployment and its assets.")
	new, err := snaphotDeployment(ctx, c, old.Deployment.Namespace, old.Deployment.Name)
	framework.ExpectNoError(err)

	g.By("Validating the Deployment.")
	framework.ExpectEqual(new.Deployment.UID, old.Deployment.UID)
	framework.ExpectEqual(new.Deployment.Generation, old.Deployment.Generation)
	framework.ExpectEqual(
		new.Deployment.Annotations[deploymentutil.RevisionAnnotation],
		old.Deployment.Annotations[deploymentutil.RevisionAnnotation],
	)

	g.By("Validating ReplicaSets.")
	framework.ExpectEqual(len(new.ReplicaSets), len(old.Deployment.UID))
	var oldRSUIDs []types.UID
	for uid, _ := range old.ReplicaSets {
		oldRSUIDs = append(oldRSUIDs, uid)
	}
	var newRSUIDs []types.UID
	for uid, _ := range new.ReplicaSets {
		newRSUIDs = append(newRSUIDs, uid)
	}
	framework.ExpectEqual(newRSUIDs, oldRSUIDs, "New ReplicaSet UIDs differ: "+cmp.Diff(newRSUIDs, oldRSUIDs))
	for _, newRS := range new.ReplicaSets {
		oldRS, ok := old.ReplicaSets[newRS.UID]
		o.Expect(ok).To(o.BeTrue())
		framework.ExpectEqual(oldRS.Namespace, newRS.Namespace)
		framework.ExpectEqual(oldRS.Name, newRS.Name)
		framework.ExpectEqual(oldRS.UID, newRS.UID)

		g.By(fmt.Sprintf("Validating ReplicaSet %s/%s (uid=%s)", newRS.Namespace, newRS.Name, newRS.UID))
		framework.ExpectEqual(newRS.Generation, oldRS.Generation)
		framework.ExpectEqual(newRS.Spec.Replicas, oldRS.Spec.Replicas)
		framework.ExpectEqual(newRS.Status.AvailableReplicas, oldRS.Status.AvailableReplicas)
		framework.ExpectEqual(newRS.Status.ObservedGeneration, oldRS.Status.ObservedGeneration)
	}

	// Verify the upgraded deployment is active by scaling up the deployment by 1
	g.By("Scaling up deployment by 1")
	deployment, err = e2edeploy.UpdateDeploymentWithRetries(c, new.Deployment.Namespace, new.Deployment.Name, func(d *appsv1.Deployment) {
		*d.Spec.Replicas = *new.Deployment.Spec.Replicas + 1
	})
	framework.ExpectNoError(err)

	g.By("Waiting for deployment to complete after scaling")
	framework.ExpectNoError(e2edeploy.WaitForDeploymentComplete(c, deployment))
}

// Teardown cleans up any remaining resources.
func (t *DeploymentUpgradeTest) Teardown(f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

// waitForDeploymentRevision waits for becoming the target revision of a delopyment.
func waitForDeploymentRevision(c clientset.Interface, d *appsv1.Deployment, targetRevision string) error {
	err := wait.PollImmediate(poll, pollLongTimeout, func() (bool, error) {
		deployment, err := c.AppsV1().Deployments(d.Namespace).Get(context.TODO(), d.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		revision := deployment.Annotations[deploymentutil.RevisionAnnotation]
		return revision == targetRevision, nil
	})
	if err != nil {
		return fmt.Errorf("error waiting for revision to become %q for deployment %q: %v", targetRevision, d.Name, err)
	}
	return nil
}
