/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"
	deploymentutil "k8s.io/kubernetes/pkg/util/deployment"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

const (
	// nginxImage defined in kubectl.go
	nginxImageName = "nginx"
	redisImage     = "gcr.io/google_containers/redis:e2e"
	redisImageName = "redis"
)

var _ = Describe("Deployment", func() {
	f := NewDefaultFramework("deployment")

	It("deployment should create new pods", func() {
		testNewDeployment(f)
	})
	It("RollingUpdateDeployment should delete old pods and create new ones", func() {
		testRollingUpdateDeployment(f)
	})
	It("RollingUpdateDeployment should scale up and down in the right order", func() {
		testRollingUpdateDeploymentEvents(f)
	})
	It("RecreateDeployment should delete old pods and create new ones", func() {
		testRecreateDeployment(f)
	})
	It("deployment should delete old replica sets", func() {
		testDeploymentCleanUpPolicy(f)
	})
	It("deployment should support rollover", func() {
		testRolloverDeployment(f)
	})
	It("paused deployment should be ignored by the controller", func() {
		testPausedDeployment(f)
	})
	It("deployment should support rollback", func() {
		testRollbackDeployment(f)
	})
	It("deployment should support rollback when there's replica set with no revision", func() {
		testRollbackDeploymentRSNoRevision(f)
	})
	It("deployment should label adopted RSs and pods", func() {
		testDeploymentLabelAdopted(f)
	})
})

func newRS(rsName string, replicas int, rsPodLabels map[string]string, imageName string, image string) *extensions.ReplicaSet {
	zero := int64(0)
	return &extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name: rsName,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: replicas,
			Selector: &unversioned.LabelSelector{MatchLabels: rsPodLabels},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: rsPodLabels,
				},
				Spec: api.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []api.Container{
						{
							Name:  imageName,
							Image: image,
						},
					},
				},
			},
		},
	}
}

func newDeployment(deploymentName string, replicas int, podLabels map[string]string, imageName string, image string, strategyType extensions.DeploymentStrategyType, revisionHistoryLimit *int) *extensions.Deployment {
	zero := int64(0)
	return &extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: deploymentName,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: replicas,
			Selector: &unversioned.LabelSelector{MatchLabels: podLabels},
			Strategy: extensions.DeploymentStrategy{
				Type: strategyType,
			},
			RevisionHistoryLimit: revisionHistoryLimit,
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabels,
				},
				Spec: api.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []api.Container{
						{
							Name:  imageName,
							Image: image,
						},
					},
				},
			},
		},
	}
}

func newDeploymentRollback(name string, annotations map[string]string, revision int64) *extensions.DeploymentRollback {
	return &extensions.DeploymentRollback{
		Name:               name,
		UpdatedAnnotations: annotations,
		RollbackTo:         extensions.RollbackConfig{Revision: revision},
	}
}

// checkDeploymentRevision checks if the input deployment's and its new replica set's revision and images are as expected.
func checkDeploymentRevision(c *clientset.Clientset, ns, deploymentName, revision, imageName, image string) (*extensions.Deployment, *extensions.ReplicaSet) {
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// Check revision of the new replica set of this deployment
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRS.Annotations).NotTo(Equal(nil))
	Expect(newRS.Annotations[deploymentutil.RevisionAnnotation]).Should(Equal(revision))
	// Check revision of This deployment
	Expect(deployment.Annotations).NotTo(Equal(nil))
	Expect(deployment.Annotations[deploymentutil.RevisionAnnotation]).Should(Equal(revision))
	if len(imageName) > 0 {
		// Check the image the new replica set creates
		Expect(newRS.Spec.Template.Spec.Containers[0].Name).Should(Equal(imageName))
		Expect(newRS.Spec.Template.Spec.Containers[0].Image).Should(Equal(image))
		// Check the image the deployment creates
		Expect(deployment.Spec.Template.Spec.Containers[0].Name).Should(Equal(imageName))
		Expect(deployment.Spec.Template.Spec.Containers[0].Image).Should(Equal(image))
	}
	return deployment, newRS
}

func stopDeployment(c *clientset.Clientset, oldC client.Interface, ns, deploymentName string) {
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	Logf("deleting deployment %s", deploymentName)
	reaper, err := kubectl.ReaperFor(extensions.Kind("Deployment"), oldC)
	Expect(err).NotTo(HaveOccurred())
	timeout := 1 * time.Minute
	err = reaper.Stop(ns, deployment.Name, timeout, api.NewDeleteOptions(0))
	Expect(err).NotTo(HaveOccurred())

	Logf("ensuring deployment %s was deleted", deploymentName)
	_, err = c.Extensions().Deployments(ns).Get(deployment.Name)
	Expect(err).To(HaveOccurred())
	Expect(errors.IsNotFound(err)).To(BeTrue())
	Logf("ensuring deployment %s rcs were deleted", deploymentName)
	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	Expect(err).NotTo(HaveOccurred())
	options := api.ListOptions{LabelSelector: selector}
	// NOTE: this is cherrypicked to 1.2 only to fix the kubernetes-e2e-gke-1.2-1.4-upgrade-cluster test flakes
	err = waitForReplicaSetsGone(c, ns, options)
	Expect(err).NotTo(HaveOccurred())
	Logf("ensuring deployment %s pods were deleted", deploymentName)
	var pods *api.PodList
	if err := wait.PollImmediate(time.Second, wait.ForeverTestTimeout, func() (bool, error) {
		pods, err = c.Core().Pods(ns).List(api.ListOptions{})
		if err != nil {
			return false, err
		}
		if len(pods.Items) == 0 {
			return true, nil
		}
		return false, nil
	}); err != nil {
		Failf("Err : %s\n. Failed to remove deployment %s pods : %+v", err, deploymentName, pods)
	}
}

// NOTE: this is cherrypicked to 1.2 only to fix the kubernetes-e2e-gke-1.2-1.4-upgrade-cluster test flakes
func waitForReplicaSetsGone(c *clientset.Clientset, ns string, options api.ListOptions) error {
	return wait.Poll(poll, wait.ForeverTestTimeout, func() (bool, error) {
		rss, err := c.Extensions().ReplicaSets(ns).List(options)
		if err != nil {
			return false, err
		}
		if len(rss.Items) != 0 {
			Logf("Waiting for all RSes to be gone, %d RS left", len(rss.Items))
			return false, nil
		}
		return true, nil
	})
}

func testNewDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	c := clientset.FromUnversionedClient(f.Client)

	deploymentName := "test-new-deployment"
	podLabels := map[string]string{"name": nginxImageName}
	replicas := 1
	Logf("Creating simple deployment %s", deploymentName)
	d := newDeployment(deploymentName, replicas, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType, nil)
	d.Annotations = map[string]string{"test": "should-copy-to-replica-set", kubectl.LastAppliedConfigAnnotation: "should-not-copy-to-replica-set"}
	_, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Wait for it to be updated to revision 1
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", nginxImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	// Check new RS annotations
	Expect(newRS.Annotations["test"]).Should(Equal("should-copy-to-replica-set"))
	Expect(newRS.Annotations[kubectl.LastAppliedConfigAnnotation]).Should(Equal(""))
	Expect(deployment.Annotations["test"]).Should(Equal("should-copy-to-replica-set"))
	Expect(deployment.Annotations[kubectl.LastAppliedConfigAnnotation]).Should(Equal("should-not-copy-to-replica-set"))
}

func testRollingUpdateDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod"}
	rsPodLabels := map[string]string{
		"name": "sample-pod",
		"pod":  nginxImageName,
	}

	rsName := "test-rolling-update-controller"
	replicas := 3
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, rsPodLabels, nginxImageName, nginxImage))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "sample-pod", false, 3)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "test-rolling-update-deployment"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment(deploymentName, replicas, deploymentPodLabels, redisImageName, redisImage, extensions.RollingUpdateDeploymentStrategyType, nil))
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Wait for it to be updated to revision 1
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", redisImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// There should be 1 old RS (nginx-controller, which is adopted)
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	_, allOldRSs, err := deploymentutil.GetOldReplicaSets(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(allOldRSs)).Should(Equal(1))
	// The old RS should contain pod-template-hash in its selector, label, and template label
	Expect(len(allOldRSs[0].Labels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
	Expect(len(allOldRSs[0].Spec.Selector.MatchLabels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
	Expect(len(allOldRSs[0].Spec.Template.Labels[extensions.DefaultDeploymentUniqueLabelKey])).Should(BeNumerically(">", 0))
}

func testRollingUpdateDeploymentEvents(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod-2"}
	rsPodLabels := map[string]string{
		"name": "sample-pod-2",
		"pod":  nginxImageName,
	}
	rsName := "test-rolling-scale-controller"
	replicas := 1

	rsRevision := "3546343826724305832"
	annotations := make(map[string]string)
	annotations[deploymentutil.RevisionAnnotation] = rsRevision
	rs := newRS(rsName, replicas, rsPodLabels, nginxImageName, nginxImage)
	rs.Annotations = annotations

	_, err := c.Extensions().ReplicaSets(ns).Create(rs)
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "sample-pod-2", false, 1)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "test-rolling-scale-deployment"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment(deploymentName, replicas, deploymentPodLabels, redisImageName, redisImage, extensions.RollingUpdateDeploymentStrategyType, nil))
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Wait for it to be updated to revision 3546343826724305833
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "3546343826724305833", redisImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())
	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	waitForEvents(unversionedClient, ns, deployment, 2)
	events, err := c.Core().Events(ns).Search(deployment)
	if err != nil {
		Logf("error in listing events: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	// There should be 2 events, one to scale up the new ReplicaSet and then to scale down
	// the old ReplicaSet.
	Expect(len(events.Items)).Should(Equal(2))
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRS).NotTo(Equal(nil))
	Expect(events.Items[0].Message).Should(Equal(fmt.Sprintf("Scaled up replica set %s to 1", newRS.Name)))
	Expect(events.Items[1].Message).Should(Equal(fmt.Sprintf("Scaled down replica set %s to 0", rsName)))
}

func testRecreateDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "sample-pod-3"}
	rsPodLabels := map[string]string{
		"name": "sample-pod-3",
		"pod":  nginxImageName,
	}

	rsName := "test-recreate-controller"
	replicas := 3
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, rsPodLabels, nginxImageName, nginxImage))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "sample-pod-3", false, 3)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "test-recreate-deployment"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment(deploymentName, replicas, deploymentPodLabels, redisImageName, redisImage, extensions.RecreateDeploymentStrategyType, nil))
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Wait for it to be updated to revision 1
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", redisImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, 0, replicas, 0)
	Expect(err).NotTo(HaveOccurred())

	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	waitForEvents(unversionedClient, ns, deployment, 2)
	events, err := c.Core().Events(ns).Search(deployment)
	if err != nil {
		Logf("error in listing events: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	// There should be 2 events, one to scale up the new ReplicaSet and then to scale down the old ReplicaSet.
	Expect(len(events.Items)).Should(Equal(2))
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(newRS).NotTo(Equal(nil))
	Expect(events.Items[0].Message).Should(Equal(fmt.Sprintf("Scaled down replica set %s to 0", rsName)))
	Expect(events.Items[1].Message).Should(Equal(fmt.Sprintf("Scaled up replica set %s to 3", newRS.Name)))
}

// testDeploymentCleanUpPolicy tests that deployment supports cleanup policy
func testDeploymentCleanUpPolicy(f *Framework) {
	ns := f.Namespace.Name
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	deploymentPodLabels := map[string]string{"name": "cleanup-pod"}
	rsPodLabels := map[string]string{
		"name": "cleanup-pod",
		"pod":  nginxImageName,
	}
	rsName := "test-cleanup-controller"
	replicas := 1
	revisionHistoryLimit := util.IntPtr(0)
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, rsPodLabels, nginxImageName, nginxImage))
	Expect(err).NotTo(HaveOccurred())

	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, "cleanup-pod", false, 1)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a deployment to delete nginx pods and instead bring up redis pods.
	deploymentName := "test-cleanup-deployment"
	Logf("Creating deployment %s", deploymentName)

	pods, err := c.Pods(ns).List(api.ListOptions{LabelSelector: labels.Everything()})
	if err != nil {
		Expect(err).NotTo(HaveOccurred(), "Failed to query for pods: %v", err)
	}
	options := api.ListOptions{
		ResourceVersion: pods.ListMeta.ResourceVersion,
	}
	stopCh := make(chan struct{})
	w, err := c.Pods(ns).Watch(options)
	go func() {
		// There should be only one pod being created, which is the pod with the redis image.
		// The old RS shouldn't create new pod when deployment controller adding pod template hash label to its selector.
		numPodCreation := 1
		for {
			select {
			case event, _ := <-w.ResultChan():
				if event.Type != watch.Added {
					continue
				}
				numPodCreation--
				if numPodCreation < 0 {
					Failf("Expect only one pod creation, the second creation event: %#v\n", event)
				}
				pod, ok := event.Object.(*api.Pod)
				if !ok {
					Fail("Expect event Object to be a pod")
				}
				if pod.Spec.Containers[0].Name != redisImageName {
					Failf("Expect the created pod to have container name %s, got pod %#v\n", redisImageName, pod)
				}
			case <-stopCh:
				return
			}
		}
	}()
	_, err = c.Extensions().Deployments(ns).Create(newDeployment(deploymentName, replicas, deploymentPodLabels, redisImageName, redisImage, extensions.RollingUpdateDeploymentStrategyType, revisionHistoryLimit))
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	err = waitForDeploymentOldRSsNum(c, ns, deploymentName, *revisionHistoryLimit)
	Expect(err).NotTo(HaveOccurred())
	close(stopCh)
}

// testRolloverDeployment tests that deployment supports rollover.
// i.e. we can change desired state and kick off rolling update, then change desired state again before it finishes.
func testRolloverDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	podName := "rollover-pod"
	deploymentPodLabels := map[string]string{"name": podName}
	rsPodLabels := map[string]string{
		"name": podName,
		"pod":  nginxImageName,
	}

	rsName := "test-rollover-controller"
	rsReplicas := 4
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, rsReplicas, rsPodLabels, nginxImageName, nginxImage))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, podName, false, rsReplicas)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}
	// Wait for the required pods to be ready for at least minReadySeconds (be available)
	deploymentMinReadySeconds := 5
	err = waitForPodsReady(c, ns, podName, deploymentMinReadySeconds)
	Expect(err).NotTo(HaveOccurred())

	// Create a deployment to delete nginx pods and instead bring up redis-slave pods.
	deploymentName, deploymentImageName := "test-rollover-deployment", "redis-slave"
	deploymentReplicas := 4
	deploymentImage := "gcr.io/google_samples/gb-redisslave:v1"
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	Logf("Creating deployment %s", deploymentName)
	newDeployment := newDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType, nil)
	newDeployment.Spec.MinReadySeconds = deploymentMinReadySeconds
	newDeployment.Spec.Strategy.RollingUpdate = &extensions.RollingUpdateDeployment{
		MaxUnavailable: intstr.FromInt(1),
		MaxSurge:       intstr.FromInt(1),
	}
	_, err = c.Extensions().Deployments(ns).Create(newDeployment)
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Verify that the pods were scaled up and down as expected. We use events to verify that.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// Make sure the deployment starts to scale up and down replica sets
	waitForPartialEvents(unversionedClient, ns, deployment, 2)
	// Check if it's updated to revision 1 correctly
	_, newRS := checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	// Before the deployment finishes, update the deployment to rollover the above 2 ReplicaSets and bring up redis pods.
	// If the deployment already finished here, the test would fail. When this happens, increase its minReadySeconds or replicas to prevent it.
	Expect(newRS.Spec.Replicas).Should(BeNumerically("<", deploymentReplicas))
	updatedDeploymentImageName, updatedDeploymentImage := redisImageName, redisImage
	deployment, err = updateDeploymentWithRetries(c, ns, newDeployment.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pod template update.
	err = waitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 2
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "2", updatedDeploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, deploymentMinReadySeconds)
	Expect(err).NotTo(HaveOccurred())
}

func testPausedDeployment(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	deploymentName := "test-paused-deployment"
	podLabels := map[string]string{"name": nginxImageName}
	d := newDeployment(deploymentName, 1, podLabels, nginxImageName, nginxImage, extensions.RollingUpdateDeploymentStrategyType, nil)
	d.Spec.Paused = true
	Logf("Creating paused deployment %s", deploymentName)
	_, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)
	// Check that deployment is created fine.
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	// Verify that there is no latest state realized for the new deployment.
	rs, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	if rs != nil {
		err = fmt.Errorf("unexpected new rs/%s for deployment/%s", rs.Name, deployment.Name)
		Expect(err).NotTo(HaveOccurred())
	}

	// Update the deployment to run
	deployment, err = updateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Paused = false
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the resume.
	err = waitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	if err != nil {
		Expect(err).NotTo(HaveOccurred())
	}
	opts := api.ListOptions{LabelSelector: selector}
	w, err := c.Extensions().ReplicaSets(ns).Watch(opts)
	Expect(err).NotTo(HaveOccurred())

	select {
	case <-w.ResultChan():
		// this is it
	case <-time.After(time.Minute):
		err = fmt.Errorf("expected a new replica set to be created")
		Expect(err).NotTo(HaveOccurred())
	}

	// Pause the deployment and delete the replica set.
	// The paused deployment shouldn't recreate a new one.
	deployment, err = updateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Paused = true
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pause.
	err = waitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(DeleteReplicaSet(unversionedClient, ns, newRS.Name)).NotTo(HaveOccurred())

	deployment, err = c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())

	if !deployment.Spec.Paused {
		err = fmt.Errorf("deployment %q should be paused", deployment.Name)
		Expect(err).NotTo(HaveOccurred())
	}
	shouldBeNil, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	if shouldBeNil != nil {
		err = fmt.Errorf("deployment %q shouldn't have a replica set but there is %q", deployment.Name, shouldBeNil.Name)
		Expect(err).NotTo(HaveOccurred())
	}
}

// testRollbackDeployment tests that a deployment is created (revision 1) and updated (revision 2), and
// then rollback to revision 1 (should update template to revision 1, and then update revision 1 to 3),
// and then rollback to last revision.
func testRollbackDeployment(f *Framework) {
	ns := f.Namespace.Name
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	podName := "nginx"
	deploymentPodLabels := map[string]string{"name": podName}

	// 1. Create a deployment to create nginx pods.
	deploymentName, deploymentImageName := "test-rollback-deployment", nginxImageName
	deploymentReplicas := 1
	deploymentImage := nginxImage
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	Logf("Creating deployment %s", deploymentName)
	d := newDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType, nil)
	createAnnotation := map[string]string{"action": "create", "author": "minion"}
	d.Annotations = createAnnotation
	_, err := c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Wait for it to be updated to revision 1
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", deploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Current newRS annotation should be "create"
	err = checkNewRSAnnotations(c, ns, deploymentName, createAnnotation)
	Expect(err).NotTo(HaveOccurred())

	// 2. Update the deployment to create redis pods.
	updatedDeploymentImage := redisImage
	updatedDeploymentImageName := redisImageName
	updateAnnotation := map[string]string{"action": "update", "log": "I need to update it"}
	deployment, err := updateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
		update.Annotations = updateAnnotation
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pod template update.
	err = waitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 2
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "2", updatedDeploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Current newRS annotation should be "update"
	err = checkNewRSAnnotations(c, ns, deploymentName, updateAnnotation)
	Expect(err).NotTo(HaveOccurred())

	// 3. Update the deploymentRollback to rollback to revision 1
	revision := int64(1)
	Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback := newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait for the deployment to start rolling back
	err = waitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// TODO: report RollbackDone in deployment status and check it here

	// Wait for it to be updated to revision 3
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "3", deploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Current newRS annotation should be "create", after the rollback
	err = checkNewRSAnnotations(c, ns, deploymentName, createAnnotation)
	Expect(err).NotTo(HaveOccurred())

	// 4. Update the deploymentRollback to rollback to last revision
	revision = 0
	Logf("rolling back deployment %s to last revision", deploymentName)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 4
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "4", updatedDeploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Current newRS annotation should be "update", after the rollback
	err = checkNewRSAnnotations(c, ns, deploymentName, updateAnnotation)
	Expect(err).NotTo(HaveOccurred())
}

// testRollbackDeploymentRSNoRevision tests that deployment supports rollback even when there's old replica set without revision.
// An old replica set without revision is created, and then a deployment is created (v1). The deployment shouldn't add revision
// annotation to the old replica set. Then rollback the deployment to last revision, and it should fail.
// Then update the deployment to v2 and rollback it to v1 should succeed, now the deployment
// becomes v3. Then rollback the deployment to v10 (doesn't exist in history) should fail.
// Finally, rollback the deployment (v3) to v3 should be no-op.
// TODO: When we finished reporting rollback status in deployment status, check the rollback status here in each case.
func testRollbackDeploymentRSNoRevision(f *Framework) {
	ns := f.Namespace.Name
	c := clientset.FromUnversionedClient(f.Client)
	podName := "nginx"
	deploymentPodLabels := map[string]string{"name": podName}
	rsPodLabels := map[string]string{
		"name": podName,
		"pod":  nginxImageName,
	}

	// Create an old RS without revision
	rsName := "test-rollback-no-revision-controller"
	rsReplicas := 0
	rs := newRS(rsName, rsReplicas, rsPodLabels, nginxImageName, nginxImage)
	rs.Annotations = make(map[string]string)
	rs.Annotations["make"] = "difference"
	_, err := c.Extensions().ReplicaSets(ns).Create(rs)
	Expect(err).NotTo(HaveOccurred())

	// 1. Create a deployment to create nginx pods, which have different template than the replica set created above.
	deploymentName, deploymentImageName := "test-rollback-no-revision-deployment", nginxImageName
	deploymentReplicas := 1
	deploymentImage := nginxImage
	deploymentStrategyType := extensions.RollingUpdateDeploymentStrategyType
	Logf("Creating deployment %s", deploymentName)
	d := newDeployment(deploymentName, deploymentReplicas, deploymentPodLabels, deploymentImageName, deploymentImage, deploymentStrategyType, nil)
	_, err = c.Extensions().Deployments(ns).Create(d)
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Wait for it to be updated to revision 1
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", deploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// Check that the replica set we created still doesn't contain revision information
	rs, err = c.Extensions().ReplicaSets(ns).Get(rsName)
	Expect(err).NotTo(HaveOccurred())
	Expect(rs.Annotations[deploymentutil.RevisionAnnotation]).Should(Equal(""))

	// 2. Update the deploymentRollback to rollback to last revision
	//    Since there's only 1 revision in history, it should stay as revision 1
	revision := int64(0)
	Logf("rolling back deployment %s to last revision", deploymentName)
	rollback := newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait for the deployment to start rolling back
	err = waitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// TODO: report RollbackRevisionNotFound in deployment status and check it here

	// The pod template shouldn't change since there's no last revision
	// Check if the deployment is still revision 1 and still has the old pod template
	checkDeploymentRevision(c, ns, deploymentName, "1", deploymentImageName, deploymentImage)

	// 3. Update the deployment to create redis pods.
	updatedDeploymentImage := redisImage
	updatedDeploymentImageName := redisImageName
	deployment, err := updateDeploymentWithRetries(c, ns, d.Name, func(update *extensions.Deployment) {
		update.Spec.Template.Spec.Containers[0].Name = updatedDeploymentImageName
		update.Spec.Template.Spec.Containers[0].Image = updatedDeploymentImage
	})
	Expect(err).NotTo(HaveOccurred())

	// Use observedGeneration to determine if the controller noticed the pod template update.
	err = waitForObservedDeployment(c, ns, deploymentName, deployment.Generation)
	Expect(err).NotTo(HaveOccurred())

	// Wait for it to be updated to revision 2
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "2", updatedDeploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// 4. Update the deploymentRollback to rollback to revision 1
	revision = 1
	Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait for the deployment to start rolling back
	err = waitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// TODO: report RollbackDone in deployment status and check it here

	// The pod template should be updated to the one in revision 1
	// Wait for it to be updated to revision 3
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "3", deploymentImage)
	Expect(err).NotTo(HaveOccurred())

	err = waitForDeploymentStatus(c, ns, deploymentName, deploymentReplicas, deploymentReplicas-1, deploymentReplicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// 5. Update the deploymentRollback to rollback to revision 10
	//    Since there's no revision 10 in history, it should stay as revision 3
	revision = 10
	Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait for the deployment to start rolling back
	err = waitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// TODO: report RollbackRevisionNotFound in deployment status and check it here

	// The pod template shouldn't change since there's no revision 10
	// Check if it's still revision 3 and still has the old pod template
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)

	// 6. Update the deploymentRollback to rollback to revision 3
	//    Since it's already revision 3, it should be no-op
	revision = 3
	Logf("rolling back deployment %s to revision %d", deploymentName, revision)
	rollback = newDeploymentRollback(deploymentName, nil, revision)
	err = c.Extensions().Deployments(ns).Rollback(rollback)
	Expect(err).NotTo(HaveOccurred())

	// Wait for the deployment to start rolling back
	err = waitForDeploymentRollbackCleared(c, ns, deploymentName)
	Expect(err).NotTo(HaveOccurred())
	// TODO: report RollbackTemplateUnchanged in deployment status and check it here

	// The pod template shouldn't change since it's already revision 3
	// Check if it's still revision 3 and still has the old pod template
	checkDeploymentRevision(c, ns, deploymentName, "3", deploymentImageName, deploymentImage)
}

func testDeploymentLabelAdopted(f *Framework) {
	ns := f.Namespace.Name
	// TODO: remove unversionedClient when the refactoring is done. Currently some
	// functions like verifyPod still expects a unversioned#Client.
	unversionedClient := f.Client
	c := clientset.FromUnversionedClient(unversionedClient)
	// Create nginx pods.
	podName := "nginx"
	podLabels := map[string]string{"name": podName}

	rsName := "test-adopted-controller"
	replicas := 3
	image := nginxImage
	_, err := c.Extensions().ReplicaSets(ns).Create(newRS(rsName, replicas, podLabels, podName, image))
	Expect(err).NotTo(HaveOccurred())
	// Verify that the required pods have come up.
	err = verifyPods(unversionedClient, ns, podName, false, 3)
	if err != nil {
		Logf("error in waiting for pods to come up: %s", err)
		Expect(err).NotTo(HaveOccurred())
	}

	// Create a nginx deployment to adopt the old rs.
	deploymentName := "test-adopted-deployment"
	Logf("Creating deployment %s", deploymentName)
	_, err = c.Extensions().Deployments(ns).Create(newDeployment(deploymentName, replicas, podLabels, podName, image, extensions.RollingUpdateDeploymentStrategyType, nil))
	Expect(err).NotTo(HaveOccurred())
	defer stopDeployment(c, f.Client, ns, deploymentName)

	// Wait for it to be updated to revision 1
	err = waitForDeploymentRevisionAndImage(c, ns, deploymentName, "1", image)
	Expect(err).NotTo(HaveOccurred())

	// The RS and pods should be relabeled before the status is updated by syncRollingUpdateDeployment
	err = waitForDeploymentStatus(c, ns, deploymentName, replicas, replicas-1, replicas+1, 0)
	Expect(err).NotTo(HaveOccurred())

	// There should be no old RSs (overlapping RS)
	deployment, err := c.Extensions().Deployments(ns).Get(deploymentName)
	Expect(err).NotTo(HaveOccurred())
	oldRSs, allOldRSs, err := deploymentutil.GetOldReplicaSets(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(oldRSs)).Should(Equal(0))
	Expect(len(allOldRSs)).Should(Equal(0))
	// New RS should contain pod-template-hash in its selector, label, and template label
	newRS, err := deploymentutil.GetNewReplicaSet(deployment, c)
	Expect(err).NotTo(HaveOccurred())
	err = checkRSHashLabel(newRS)
	Expect(err).NotTo(HaveOccurred())
	// All pods targeted by the deployment should contain pod-template-hash in their labels, and there should be only 3 pods
	selector, err := unversioned.LabelSelectorAsSelector(deployment.Spec.Selector)
	Expect(err).NotTo(HaveOccurred())
	options := api.ListOptions{LabelSelector: selector}
	pods, err := c.Core().Pods(ns).List(options)
	Expect(err).NotTo(HaveOccurred())
	err = checkPodHashLabel(pods)
	Expect(err).NotTo(HaveOccurred())
	Expect(len(pods.Items)).Should(Equal(replicas))
}
