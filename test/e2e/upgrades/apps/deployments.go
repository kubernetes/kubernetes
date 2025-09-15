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

package apps

import (
	"context"
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	deploymentutil "k8s.io/kubernetes/pkg/controller/deployment/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2edeployment "k8s.io/kubernetes/test/e2e/framework/deployment"
	"k8s.io/kubernetes/test/e2e/upgrades"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

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

// DeploymentUpgradeTest tests that a deployment is using the same replica
// sets before and after a cluster upgrade.
type DeploymentUpgradeTest struct {
	oldDeploymentUID types.UID
	oldRSUID         types.UID
	newRSUID         types.UID
}

// Name returns the tracking name of the test.
func (DeploymentUpgradeTest) Name() string { return "[sig-apps] deployment-upgrade" }

// Setup creates a deployment and makes sure it has a new and an old replicaset running.
func (t *DeploymentUpgradeTest) Setup(ctx context.Context, f *framework.Framework) {
	c := f.ClientSet
	nginxImage := imageutils.GetE2EImage(imageutils.Nginx)

	ns := f.Namespace.Name
	deploymentClient := c.AppsV1().Deployments(ns)
	rsClient := c.AppsV1().ReplicaSets(ns)

	ginkgo.By(fmt.Sprintf("Creating a deployment %q with 1 replica in namespace %q", deploymentName, ns))
	d := e2edeployment.NewDeployment(deploymentName, int32(1), map[string]string{"test": "upgrade"}, "nginx", nginxImage, appsv1.RollingUpdateDeploymentStrategyType)
	deployment, err := deploymentClient.Create(ctx, d, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Waiting deployment %q to complete", deploymentName))
	framework.ExpectNoError(e2edeployment.WaitForDeploymentComplete(c, deployment))

	ginkgo.By(fmt.Sprintf("Getting replicaset revision 1 of deployment %q", deploymentName))
	rsSelector, err := metav1.LabelSelectorAsSelector(d.Spec.Selector)
	framework.ExpectNoError(err)
	rsList, err := rsClient.List(ctx, metav1.ListOptions{LabelSelector: rsSelector.String()})
	framework.ExpectNoError(err)
	rss := rsList.Items
	gomega.Expect(rss).To(gomega.HaveLen(1), "expected one replicaset, got %d", len(rss))
	t.oldRSUID = rss[0].UID

	ginkgo.By(fmt.Sprintf("Waiting for revision of the deployment %q to become 1", deploymentName))
	framework.ExpectNoError(waitForDeploymentRevision(ctx, c, deployment, "1"))

	// Trigger a new rollout so that we have some history.
	ginkgo.By(fmt.Sprintf("Triggering a new rollout for deployment %q", deploymentName))
	deployment, err = e2edeployment.UpdateDeploymentWithRetries(c, ns, deploymentName, func(update *appsv1.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = "updated-name"
	})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Waiting deployment %q to complete", deploymentName))
	framework.ExpectNoError(e2edeployment.WaitForDeploymentComplete(c, deployment))

	ginkgo.By(fmt.Sprintf("Getting replicasets revision 1 and 2 of deployment %q", deploymentName))
	rsList, err = rsClient.List(ctx, metav1.ListOptions{LabelSelector: rsSelector.String()})
	framework.ExpectNoError(err)
	rss = rsList.Items
	gomega.Expect(rss).To(gomega.HaveLen(2), "expected 2 replicaset, got %d", len(rss))

	ginkgo.By(fmt.Sprintf("Checking replicaset of deployment %q that is created before rollout survives the rollout", deploymentName))
	switch t.oldRSUID {
	case rss[0].UID:
		t.newRSUID = rss[1].UID
	case rss[1].UID:
		t.newRSUID = rss[0].UID
	default:
		framework.ExpectNoError(fmt.Errorf("old replicaset with UID %q does not survive rollout", t.oldRSUID))
	}

	ginkgo.By(fmt.Sprintf("Waiting for revision of the deployment %q to become 2", deploymentName))
	framework.ExpectNoError(waitForDeploymentRevision(ctx, c, deployment, "2"))

	t.oldDeploymentUID = deployment.UID
}

// Test checks whether the replicasets for a deployment are the same after an upgrade.
func (t *DeploymentUpgradeTest) Test(ctx context.Context, f *framework.Framework, done <-chan struct{}, upgrade upgrades.UpgradeType) {
	// Block until upgrade is done
	ginkgo.By(fmt.Sprintf("Waiting for upgrade to finish before checking replicasets for deployment %q", deploymentName))
	<-done

	c := f.ClientSet
	ns := f.Namespace.Name
	deploymentClient := c.AppsV1().Deployments(ns)
	rsClient := c.AppsV1().ReplicaSets(ns)

	deployment, err := deploymentClient.Get(ctx, deploymentName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Checking UID to verify deployment %q survives upgrade", deploymentName))
	gomega.Expect(deployment.UID).To(gomega.Equal(t.oldDeploymentUID))

	ginkgo.By(fmt.Sprintf("Verifying deployment %q does not create new replicasets", deploymentName))
	rsSelector, err := metav1.LabelSelectorAsSelector(deployment.Spec.Selector)
	framework.ExpectNoError(err)
	rsList, err := rsClient.List(ctx, metav1.ListOptions{LabelSelector: rsSelector.String()})
	framework.ExpectNoError(err)
	rss := rsList.Items
	gomega.Expect(rss).To(gomega.HaveLen(2), "expected 2 replicaset, got %d", len(rss))

	switch t.oldRSUID {
	case rss[0].UID:
		gomega.Expect(rss[1].UID).To(gomega.Equal(t.newRSUID))
	case rss[1].UID:
		gomega.Expect(rss[0].UID).To(gomega.Equal(t.newRSUID))
	default:
		framework.ExpectNoError(fmt.Errorf("new replicasets are created during upgrade of deployment %q", deploymentName))
	}

	ginkgo.By(fmt.Sprintf("Verifying revision of the deployment %q is still 2", deploymentName))
	gomega.Expect(deployment.Annotations).To(gomega.HaveKeyWithValue(deploymentutil.RevisionAnnotation, "2"))

	ginkgo.By(fmt.Sprintf("Waiting for deployment %q to complete adoption", deploymentName))
	framework.ExpectNoError(e2edeployment.WaitForDeploymentComplete(c, deployment))

	// Verify the upgraded deployment is active by scaling up the deployment by 1
	ginkgo.By(fmt.Sprintf("Scaling up replicaset of deployment %q by 1", deploymentName))
	deploymentWithUpdatedReplicas, err := e2edeployment.UpdateDeploymentWithRetries(c, ns, deploymentName, func(deployment *appsv1.Deployment) {
		*deployment.Spec.Replicas = *deployment.Spec.Replicas + 1
	})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Waiting for deployment %q to complete after scaling", deploymentName))
	framework.ExpectNoError(e2edeployment.WaitForDeploymentComplete(c, deploymentWithUpdatedReplicas))
}

// Teardown cleans up any remaining resources.
func (t *DeploymentUpgradeTest) Teardown(ctx context.Context, f *framework.Framework) {
	// rely on the namespace deletion to clean up everything
}

// waitForDeploymentRevision waits for becoming the target revision of a delopyment.
func waitForDeploymentRevision(ctx context.Context, c clientset.Interface, d *appsv1.Deployment, targetRevision string) error {
	err := wait.PollImmediate(poll, pollLongTimeout, func() (bool, error) {
		deployment, err := c.AppsV1().Deployments(d.Namespace).Get(ctx, d.Name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		revision := deployment.Annotations[deploymentutil.RevisionAnnotation]
		return revision == targetRevision, nil
	})
	if err != nil {
		return fmt.Errorf("error waiting for revision to become %q for deployment %q: %w", targetRevision, d.Name, err)
	}
	return nil
}
