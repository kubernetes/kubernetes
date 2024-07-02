/*
Copyright 2015 The Kubernetes Authors.

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
	"encoding/json"
	"errors"
	"fmt"
	"time"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	watch "k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/pkg/controller/replication"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/format"
	"k8s.io/utils/pointer"
)

var _ = SIGDescribe("ReplicationController", func() {
	f := framework.NewDefaultFramework("replication-controller")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var ns string
	var dc dynamic.Interface

	ginkgo.BeforeEach(func() {
		ns = f.Namespace.Name
		dc = f.DynamicClient
	})

	/*
		Release: v1.9
		Testname: Replication Controller, run basic image
		Description: Replication Controller MUST create a Pod with Basic Image and MUST run the service with the provided image. Image MUST be tested by dialing into the service listening through TCP, UDP and HTTP.
	*/
	framework.ConformanceIt("should serve a basic image on each replica with a public image", func(ctx context.Context) {
		TestReplicationControllerServeImageOrFail(ctx, f, "basic", imageutils.GetE2EImage(imageutils.Agnhost))
	})

	ginkgo.It("should serve a basic image on each replica with a private image", func(ctx context.Context) {
		// requires private images
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
		privateimage := imageutils.GetConfig(imageutils.AgnhostPrivate)
		TestReplicationControllerServeImageOrFail(ctx, f, "private", privateimage.GetE2EImage())
	})

	/*
		Release: v1.15
		Testname: Replication Controller, check for issues like exceeding allocated quota
		Description: Attempt to create a Replication Controller with pods exceeding the namespace quota. The creation MUST fail
	*/
	framework.ConformanceIt("should surface a failure condition on a common issue like exceeded quota", func(ctx context.Context) {
		testReplicationControllerConditionCheck(ctx, f)
	})

	/*
		Release: v1.13
		Testname: Replication Controller, adopt matching pods
		Description: An ownerless Pod is created, then a Replication Controller (RC) is created whose label selector will match the Pod. The RC MUST either adopt the Pod or delete and replace it with a new Pod
	*/
	framework.ConformanceIt("should adopt matching pods on creation", func(ctx context.Context) {
		testRCAdoptMatchingOrphans(ctx, f)
	})

	/*
		Release: v1.13
		Testname: Replication Controller, release pods
		Description: A Replication Controller (RC) is created, and its Pods are created. When the labels on one of the Pods change to no longer match the RC's label selector, the RC MUST release the Pod and update the Pod's owner references.
	*/
	framework.ConformanceIt("should release no longer matching pods", func(ctx context.Context) {
		testRCReleaseControlledNotMatching(ctx, f)
	})

	/*
		Release: v1.20
		Testname: Replication Controller, lifecycle
		Description: A Replication Controller (RC) is created, read, patched, and deleted with verification.
	*/
	framework.ConformanceIt("should test the lifecycle of a ReplicationController", func(ctx context.Context) {
		testRcName := "rc-test"
		testRcNamespace := ns
		testRcInitialReplicaCount := int32(1)
		testRcMaxReplicaCount := int32(2)
		rcResource := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "replicationcontrollers"}
		expectedWatchEvents := []watch.Event{
			{Type: watch.Added},
			{Type: watch.Modified},
			{Type: watch.Modified},
			{Type: watch.Modified},
			{Type: watch.Modified},
			{Type: watch.Deleted},
		}

		rcTest := v1.ReplicationController{
			ObjectMeta: metav1.ObjectMeta{
				Name:   testRcName,
				Labels: map[string]string{"test-rc-static": "true"},
			},
			Spec: v1.ReplicationControllerSpec{
				Replicas: &testRcInitialReplicaCount,
				Selector: map[string]string{"test-rc-static": "true"},
				Template: &v1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Name:   testRcName,
						Labels: map[string]string{"test-rc-static": "true"},
					},
					Spec: v1.PodSpec{
						Containers: []v1.Container{{
							Name:  testRcName,
							Image: imageutils.GetE2EImage(imageutils.Nginx),
						}},
					},
				},
			},
		}

		framework.WatchEventSequenceVerifier(ctx, dc, rcResource, testRcNamespace, testRcName, metav1.ListOptions{LabelSelector: "test-rc-static=true"}, expectedWatchEvents, func(retryWatcher *watchtools.RetryWatcher) (actualWatchEvents []watch.Event) {
			ginkgo.By("creating a ReplicationController")
			// Create a ReplicationController
			_, err := f.ClientSet.CoreV1().ReplicationControllers(testRcNamespace).Create(ctx, &rcTest, metav1.CreateOptions{})
			framework.ExpectNoError(err, "Failed to create ReplicationController")

			ginkgo.By("waiting for RC to be added")
			eventFound := false
			ctxUntil, cancel := context.WithTimeout(ctx, 60*time.Second)
			defer cancel()
			_, err = watchUntilWithoutRetry(ctxUntil, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				if watchEvent.Type != watch.Added {
					return false, nil
				}
				actualWatchEvents = append(actualWatchEvents, watchEvent)
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait until condition with watch events should not return an error")
			if !eventFound {
				framework.Failf("failed to find RC %v event", watch.Added)
			}

			ginkgo.By("waiting for available Replicas")
			eventFound = false
			ctxUntil, cancel = context.WithTimeout(ctx, f.Timeouts.PodStart)
			defer cancel()
			_, err = watchUntilWithoutRetry(ctxUntil, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				var rc *v1.ReplicationController
				rcBytes, err := json.Marshal(watchEvent.Object)
				if err != nil {
					return false, err
				}
				err = json.Unmarshal(rcBytes, &rc)
				if err != nil {
					return false, err
				}
				if rc.Status.Replicas != testRcInitialReplicaCount || rc.Status.ReadyReplicas != testRcInitialReplicaCount {
					return false, nil
				}
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait for condition with watch events should not return an error")
			if !eventFound {
				framework.Failf("RC has not reached ReadyReplicas count of %v", testRcInitialReplicaCount)
			}

			rcLabelPatchPayload, err := json.Marshal(v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"test-rc": "patched"},
				},
			})
			framework.ExpectNoError(err, "failed to marshal json of replicationcontroller label patch")
			// Patch the ReplicationController
			ginkgo.By("patching ReplicationController")
			testRcPatched, err := f.ClientSet.CoreV1().ReplicationControllers(testRcNamespace).Patch(ctx, testRcName, types.StrategicMergePatchType, []byte(rcLabelPatchPayload), metav1.PatchOptions{})
			framework.ExpectNoError(err, "Failed to patch ReplicationController")
			gomega.Expect(testRcPatched.ObjectMeta.Labels).To(gomega.HaveKeyWithValue("test-rc", "patched"), "failed to patch RC")
			ginkgo.By("waiting for RC to be modified")
			eventFound = false
			ctxUntil, cancel = context.WithTimeout(ctx, 60*time.Second)
			defer cancel()
			_, err = watchUntilWithoutRetry(ctxUntil, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				if watchEvent.Type != watch.Modified {
					return false, nil
				}
				actualWatchEvents = append(actualWatchEvents, watchEvent)
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait until condition with watch events should not return an error")
			if !eventFound {
				framework.Failf("failed to find RC %v event", watch.Added)
			}

			rcStatusPatchPayload, err := json.Marshal(map[string]interface{}{
				"status": map[string]interface{}{
					"readyReplicas":     0,
					"availableReplicas": 0,
				},
			})
			framework.ExpectNoError(err, "Failed to marshal JSON of ReplicationController label patch")

			// Patch the ReplicationController's status
			ginkgo.By("patching ReplicationController status")
			rcStatus, err := f.ClientSet.CoreV1().ReplicationControllers(testRcNamespace).Patch(ctx, testRcName, types.StrategicMergePatchType, []byte(rcStatusPatchPayload), metav1.PatchOptions{}, "status")
			framework.ExpectNoError(err, "Failed to patch ReplicationControllerStatus")
			gomega.Expect(rcStatus.Status.ReadyReplicas).To(gomega.Equal(int32(0)), "ReplicationControllerStatus's readyReplicas does not equal 0")
			ginkgo.By("waiting for RC to be modified")
			eventFound = false
			ctxUntil, cancel = context.WithTimeout(ctx, 60*time.Second)
			defer cancel()
			_, err = watchUntilWithoutRetry(ctxUntil, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				if watchEvent.Type != watch.Modified {
					return false, nil
				}
				actualWatchEvents = append(actualWatchEvents, watchEvent)
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait until condition with watch events should not return an error")

			if !eventFound {
				framework.Failf("failed to find RC %v event", watch.Added)
			}

			ginkgo.By("waiting for available Replicas")
			_, err = watchUntilWithoutRetry(ctx, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				var rc *v1.ReplicationController
				rcBytes, err := json.Marshal(watchEvent.Object)
				if err != nil {
					return false, err
				}
				err = json.Unmarshal(rcBytes, &rc)
				if err != nil {
					return false, err
				}
				if rc.Status.Replicas != testRcInitialReplicaCount {
					return false, nil
				}
				return true, nil
			})
			framework.ExpectNoError(err, "Failed to find updated ready replica count")
			if !eventFound {
				framework.Fail("Failed to find updated ready replica count")
			}
			ginkgo.By("fetching ReplicationController status")
			rcStatusUnstructured, err := dc.Resource(rcResource).Namespace(testRcNamespace).Get(ctx, testRcName, metav1.GetOptions{}, "status")
			framework.ExpectNoError(err, "Failed to fetch ReplicationControllerStatus")

			rcStatusUjson, err := json.Marshal(rcStatusUnstructured)
			framework.ExpectNoError(err, "Failed to marshal json of replicationcontroller label patch")
			json.Unmarshal(rcStatusUjson, &rcStatus)
			gomega.Expect(rcStatus.Status.Replicas).To(gomega.Equal(testRcInitialReplicaCount), "ReplicationController ReplicaSet cound does not match initial Replica count")

			rcScalePatchPayload, err := json.Marshal(autoscalingv1.Scale{
				Spec: autoscalingv1.ScaleSpec{
					Replicas: testRcMaxReplicaCount,
				},
			})
			framework.ExpectNoError(err, "Failed to marshal json of replicationcontroller label patch")

			// Patch the ReplicationController's scale
			ginkgo.By("patching ReplicationController scale")
			_, err = f.ClientSet.CoreV1().ReplicationControllers(testRcNamespace).Patch(ctx, testRcName, types.StrategicMergePatchType, []byte(rcScalePatchPayload), metav1.PatchOptions{}, "scale")
			framework.ExpectNoError(err, "Failed to patch ReplicationControllerScale")
			ginkgo.By("waiting for RC to be modified")
			eventFound = false
			ctxUntil, cancel = context.WithTimeout(ctx, f.Timeouts.PodStart)
			defer cancel()
			_, err = watchUntilWithoutRetry(ctxUntil, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				if watchEvent.Type != watch.Modified {
					return false, nil
				}
				actualWatchEvents = append(actualWatchEvents, watchEvent)
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait until condition with watch events should not return an error")
			if !eventFound {
				framework.Failf("Failed to find RC %v event", watch.Added)
			}

			ginkgo.By("waiting for ReplicationController's scale to be the max amount")
			eventFound = false
			_, err = watchUntilWithoutRetry(ctx, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				var rc *v1.ReplicationController
				rcBytes, err := json.Marshal(watchEvent.Object)
				if err != nil {
					return false, err
				}
				err = json.Unmarshal(rcBytes, &rc)
				if err != nil {
					return false, err
				}
				if rc.ObjectMeta.Name != testRcName || rc.ObjectMeta.Namespace != testRcNamespace || rc.Status.Replicas != testRcMaxReplicaCount || rc.Status.ReadyReplicas != testRcMaxReplicaCount {
					return false, nil
				}
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait until condition with watch events should not return an error")
			if !eventFound {
				framework.Fail("Failed to find updated ready replica count")
			}

			// Get the ReplicationController
			ginkgo.By("fetching ReplicationController; ensuring that it's patched")
			rc, err := f.ClientSet.CoreV1().ReplicationControllers(testRcNamespace).Get(ctx, testRcName, metav1.GetOptions{})
			framework.ExpectNoError(err, "failed to fetch ReplicationController")
			gomega.Expect(rc.ObjectMeta.Labels).To(gomega.HaveKeyWithValue("test-rc", "patched"), "ReplicationController is missing a label from earlier patch")

			rcStatusUpdatePayload := rc
			rcStatusUpdatePayload.Status.AvailableReplicas = 1
			rcStatusUpdatePayload.Status.ReadyReplicas = 1

			// Replace the ReplicationController's status
			ginkgo.By("updating ReplicationController status")
			_, err = f.ClientSet.CoreV1().ReplicationControllers(testRcNamespace).UpdateStatus(ctx, rcStatusUpdatePayload, metav1.UpdateOptions{})
			framework.ExpectNoError(err, "failed to update ReplicationControllerStatus")

			ginkgo.By("waiting for RC to be modified")
			eventFound = false
			ctxUntil, cancel = context.WithTimeout(ctx, 60*time.Second)
			defer cancel()
			_, err = watchUntilWithoutRetry(ctxUntil, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				if watchEvent.Type != watch.Modified {
					return false, nil
				}
				actualWatchEvents = append(actualWatchEvents, watchEvent)
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait until condition with watch events should not return an error")

			if !eventFound {
				framework.Failf("failed to find RC %v event", watch.Added)
			}

			ginkgo.By("listing all ReplicationControllers")
			rcs, err := f.ClientSet.CoreV1().ReplicationControllers("").List(ctx, metav1.ListOptions{LabelSelector: "test-rc-static=true"})
			framework.ExpectNoError(err, "failed to list ReplicationController")
			gomega.Expect(rcs.Items).ToNot(gomega.BeEmpty(), "Expected to find a ReplicationController but none was found")

			ginkgo.By("checking that ReplicationController has expected values")
			foundRc := false
			for _, rcItem := range rcs.Items {
				if rcItem.ObjectMeta.Name == testRcName &&
					rcItem.ObjectMeta.Namespace == testRcNamespace &&
					rcItem.ObjectMeta.Labels["test-rc-static"] == "true" &&
					rcItem.ObjectMeta.Labels["test-rc"] == "patched" {
					foundRc = true
				}
			}
			if !foundRc {
				framework.Failf("ReplicationController doesn't have expected values.\nValues that are in the ReplicationController list:\n%s", format.Object(rcs.Items, 1))
			}

			// Delete ReplicationController
			ginkgo.By("deleting ReplicationControllers by collection")
			err = f.ClientSet.CoreV1().ReplicationControllers(testRcNamespace).DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "test-rc-static=true"})
			framework.ExpectNoError(err, "Failed to delete ReplicationControllers")

			ginkgo.By("waiting for ReplicationController to have a DELETED watchEvent")
			eventFound = false
			ctxUntil, cancel = context.WithTimeout(ctx, 60*time.Second)
			defer cancel()
			_, err = watchUntilWithoutRetry(ctxUntil, retryWatcher, func(watchEvent watch.Event) (bool, error) {
				if watchEvent.Type != watch.Deleted {
					return false, nil
				}
				actualWatchEvents = append(actualWatchEvents, watchEvent)
				eventFound = true
				return true, nil
			})
			framework.ExpectNoError(err, "Wait until condition with watch events should not return an error")
			if !eventFound {
				framework.Failf("failed to find RC %v event", watch.Added)
			}
			return actualWatchEvents
		}, func() (err error) {
			_ = f.ClientSet.CoreV1().ReplicationControllers(testRcNamespace).DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "test-rc-static=true"})
			return err
		})
	})

	/*
		Release: v1.26
		Testname: Replication Controller, get and update ReplicationController scale
		Description: A ReplicationController is created which MUST succeed. It MUST
		succeed when reading the ReplicationController scale. When updating the
		ReplicationController scale it MUST succeed and the field MUST equal the new value.
	*/
	framework.ConformanceIt("should get and update a ReplicationController scale", func(ctx context.Context) {
		rcClient := f.ClientSet.CoreV1().ReplicationControllers(ns)
		rcName := "e2e-rc-" + utilrand.String(5)
		initialRCReplicaCount := int32(1)
		expectedRCReplicaCount := int32(2)

		ginkgo.By(fmt.Sprintf("Creating ReplicationController %q", rcName))
		rc := newRC(rcName, initialRCReplicaCount, map[string]string{"name": rcName}, WebserverImageName, WebserverImage, nil)
		_, err := rcClient.Create(ctx, rc, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create ReplicationController: %v", err)

		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 1*time.Minute, true, checkReplicationControllerStatusReplicaCount(f, rcName, initialRCReplicaCount))
		framework.ExpectNoError(err, "failed to confirm the quantity of ReplicationController replicas")

		ginkgo.By(fmt.Sprintf("Getting scale subresource for ReplicationController %q", rcName))
		scale, err := rcClient.GetScale(ctx, rcName, metav1.GetOptions{})
		framework.ExpectNoError(err, "Failed to get scale subresource: %v", err)
		gomega.Expect(scale.Status.Replicas).To(gomega.Equal(initialRCReplicaCount), "Failed to get the current replica count")

		ginkgo.By("Updating a scale subresource")
		scale.ResourceVersion = "" // indicate the scale update should be unconditional
		scale.Spec.Replicas = expectedRCReplicaCount
		_, err = rcClient.UpdateScale(ctx, rcName, scale, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "Failed to update scale subresource: %v", err)

		ginkgo.By(fmt.Sprintf("Verifying replicas where modified for replication controller %q", rcName))
		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 1*time.Minute, true, checkReplicationControllerStatusReplicaCount(f, rcName, expectedRCReplicaCount))
		framework.ExpectNoError(err, "failed to confirm the quantity of ReplicationController replicas")
	})
})

func newRC(rsName string, replicas int32, rcPodLabels map[string]string, imageName string, image string, args []string) *v1.ReplicationController {
	zero := int64(0)
	return &v1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{
			Name: rsName,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: pointer.Int32(replicas),
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: rcPodLabels,
				},
				Spec: v1.PodSpec{
					TerminationGracePeriodSeconds: &zero,
					Containers: []v1.Container{
						{
							Name:  imageName,
							Image: image,
							Args:  args,
						},
					},
				},
			},
		},
	}
}

// TestReplicationControllerServeImageOrFail is a basic test to check
// the deployment of an image using a replication controller.
// The image serves its hostname which is checked for each replica.
func TestReplicationControllerServeImageOrFail(ctx context.Context, f *framework.Framework, test string, image string) {
	name := "my-hostname-" + test + "-" + string(uuid.NewUUID())
	replicas := int32(1)

	// Create a replication controller for a service
	// that serves its hostname.
	// The source for the Docker container kubernetes/serve_hostname is
	// in contrib/for-demos/serve_hostname
	ginkgo.By(fmt.Sprintf("Creating replication controller %s", name))
	newRC := newRC(name, replicas, map[string]string{"name": name}, name, image, []string{"serve-hostname"})
	newRC.Spec.Template.Spec.Containers[0].Ports = []v1.ContainerPort{{ContainerPort: 9376}}
	_, err := f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Create(ctx, newRC, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	// Check that pods for the new RC were created.
	// TODO: Maybe switch PodsCreated to just check owner references.
	pods, err := e2epod.PodsCreated(ctx, f.ClientSet, f.Namespace.Name, name, replicas)
	framework.ExpectNoError(err)

	// Wait for the pods to enter the running state and are Ready. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	framework.Logf("Ensuring all pods for ReplicationController %q are running", name)
	running := int32(0)
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil {
			continue
		}
		err = e2epod.WaitTimeoutForPodReadyInNamespace(ctx, f.ClientSet, pod.Name, f.Namespace.Name, framework.PodStartTimeout)
		if err != nil {
			updatePod, getErr := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
			if getErr == nil {
				err = fmt.Errorf("pod %q never run (phase: %s, conditions: %+v): %w", updatePod.Name, updatePod.Status.Phase, updatePod.Status.Conditions, err)
			} else {
				err = fmt.Errorf("pod %q never run: %w", pod.Name, err)
			}
		}
		framework.ExpectNoError(err)
		framework.Logf("Pod %q is running and ready(conditions: %+v)", pod.Name, pod.Status.Conditions)
		running++
	}

	// Sanity check
	gomega.Expect(running).To(gomega.Equal(replicas), "unexpected number of running and ready pods: %+v", pods.Items)

	// Verify that something is listening.
	framework.Logf("Trying to dial the pod")
	framework.ExpectNoError(e2epod.WaitForPodsResponding(ctx, f.ClientSet, f.Namespace.Name, name, true, 2*time.Minute, pods))
}

// 1. Create a quota restricting pods in the current namespace to 2.
// 2. Create a replication controller that wants to run 3 pods.
// 3. Check replication controller conditions for a ReplicaFailure condition.
// 4. Relax quota or scale down the controller and observe the condition is gone.
func testReplicationControllerConditionCheck(ctx context.Context, f *framework.Framework) {
	c := f.ClientSet
	namespace := f.Namespace.Name
	name := "condition-test"

	framework.Logf("Creating quota %q that allows only two pods to run in the current namespace", name)
	quota := newPodQuota(name, "2")
	_, err := c.CoreV1().ResourceQuotas(namespace).Create(ctx, quota, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 1*time.Minute, true, func(ctx context.Context) (bool, error) {
		quota, err = c.CoreV1().ResourceQuotas(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		podQuota := quota.Status.Hard[v1.ResourcePods]
		quantity := resource.MustParse("2")
		return (&podQuota).Cmp(quantity) == 0, nil
	})
	if wait.Interrupted(err) {
		err = fmt.Errorf("resource quota %q never synced", name)
	}
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Creating rc %q that asks for more than the allowed pod quota", name))
	rc := newRC(name, 3, map[string]string{"name": name}, WebserverImageName, WebserverImage, nil)
	rc, err = c.CoreV1().ReplicationControllers(namespace).Create(ctx, rc, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Checking rc %q has the desired failure condition set", name))
	generation := rc.Generation
	conditions := rc.Status.Conditions
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 1*time.Minute, true, func(ctx context.Context) (bool, error) {
		rc, err = c.CoreV1().ReplicationControllers(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		if generation > rc.Status.ObservedGeneration {
			return false, nil
		}
		conditions = rc.Status.Conditions

		cond := replication.GetCondition(rc.Status, v1.ReplicationControllerReplicaFailure)
		return cond != nil, nil
	})
	if wait.Interrupted(err) {
		err = fmt.Errorf("rc manager never added the failure condition for rc %q: %#v", name, conditions)
	}
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Scaling down rc %q to satisfy pod quota", name))
	rc, err = updateReplicationControllerWithRetries(ctx, c, namespace, name, func(update *v1.ReplicationController) {
		x := int32(2)
		update.Spec.Replicas = &x
	})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Checking rc %q has no failure condition set", name))
	generation = rc.Generation
	conditions = rc.Status.Conditions
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 1*time.Minute, true, func(ctx context.Context) (bool, error) {
		rc, err = c.CoreV1().ReplicationControllers(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		if generation > rc.Status.ObservedGeneration {
			return false, nil
		}
		conditions = rc.Status.Conditions

		cond := replication.GetCondition(rc.Status, v1.ReplicationControllerReplicaFailure)
		return cond == nil, nil
	})
	if wait.Interrupted(err) {
		err = fmt.Errorf("rc manager never removed the failure condition for rc %q: %#v", name, conditions)
	}
	framework.ExpectNoError(err)
}

func testRCAdoptMatchingOrphans(ctx context.Context, f *framework.Framework) {
	name := "pod-adoption"
	ginkgo.By(fmt.Sprintf("Given a Pod with a 'name' label %s is created", name))
	p := e2epod.NewPodClient(f).CreateSync(ctx, &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"name": name,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  name,
					Image: WebserverImage,
				},
			},
		},
	})

	ginkgo.By("When a replication controller with a matching selector is created")
	replicas := int32(1)
	rcSt := newRC(name, replicas, map[string]string{"name": name}, name, WebserverImage, nil)
	rcSt.Spec.Selector = map[string]string{"name": name}
	rc, err := f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Create(ctx, rcSt, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("Then the orphan pod is adopted")
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 1*time.Minute, true, func(ctx context.Context) (bool, error) {
		p2, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, p.Name, metav1.GetOptions{})
		// The Pod p should either be adopted or deleted by the RC
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		framework.ExpectNoError(err)
		for _, owner := range p2.OwnerReferences {
			if *owner.Controller && owner.UID == rc.UID {
				// pod adopted
				return true, nil
			}
		}
		// pod still not adopted
		return false, nil
	})
	framework.ExpectNoError(err)
}

func testRCReleaseControlledNotMatching(ctx context.Context, f *framework.Framework) {
	name := "pod-release"
	ginkgo.By("Given a ReplicationController is created")
	replicas := int32(1)
	rcSt := newRC(name, replicas, map[string]string{"name": name}, name, WebserverImage, nil)
	rcSt.Spec.Selector = map[string]string{"name": name}
	rc, err := f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Create(ctx, rcSt, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("When the matched label of one of its pods change")
	pods, err := e2epod.PodsCreated(ctx, f.ClientSet, f.Namespace.Name, rc.Name, replicas)
	framework.ExpectNoError(err)

	p := pods.Items[0]
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 1*time.Minute, true, func(ctx context.Context) (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		pod.Labels = map[string]string{"name": "not-matching-name"}
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Update(ctx, pod, metav1.UpdateOptions{})
		if err != nil && apierrors.IsConflict(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return true, nil
	})
	framework.ExpectNoError(err)

	ginkgo.By("Then the pod is released")
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 1*time.Minute, true, func(ctx context.Context) (bool, error) {
		p2, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		for _, owner := range p2.OwnerReferences {
			if *owner.Controller && owner.UID == rc.UID {
				// pod still belonging to the replication controller
				return false, nil
			}
		}
		// pod already released
		return true, nil
	})
	framework.ExpectNoError(err)
}

type updateRcFunc func(d *v1.ReplicationController)

// updateReplicationControllerWithRetries retries updating the given rc on conflict with the following steps:
// 1. Get latest resource
// 2. applyUpdate
// 3. Update the resource
func updateReplicationControllerWithRetries(ctx context.Context, c clientset.Interface, namespace, name string, applyUpdate updateRcFunc) (*v1.ReplicationController, error) {
	var rc *v1.ReplicationController
	var updateErr error
	pollErr := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 1*time.Minute, true, func(ctx context.Context) (bool, error) {
		var err error
		if rc, err = c.CoreV1().ReplicationControllers(namespace).Get(ctx, name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(rc)
		if rc, err = c.CoreV1().ReplicationControllers(namespace).Update(ctx, rc, metav1.UpdateOptions{}); err == nil {
			framework.Logf("Updating replication controller %q", name)
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if wait.Interrupted(pollErr) {
		pollErr = fmt.Errorf("couldn't apply the provided updated to rc %q: %v", name, updateErr)
	}
	return rc, pollErr
}

// watchUntilWithoutRetry ...
// reads items from the watch until each provided condition succeeds, and then returns the last watch
// encountered. The first condition that returns an error terminates the watch (and the event is also returned).
// If no event has been received, the returned event will be nil.
// Conditions are satisfied sequentially so as to provide a useful primitive for higher level composition.
// Waits until context deadline or until context is canceled.
//
// the same as watchtools.UntilWithoutRetry, just without the closing of the watch - as for the purpose of being paired with WatchEventSequenceVerifier, the watch is needed for continual watch event collection
func watchUntilWithoutRetry(ctx context.Context, watcher watch.Interface, conditions ...watchtools.ConditionFunc) (*watch.Event, error) {
	ch := watcher.ResultChan()
	var lastEvent *watch.Event
	for _, condition := range conditions {
		// check the next condition against the previous event and short circuit waiting for the next watch
		if lastEvent != nil {
			done, err := condition(*lastEvent)
			if err != nil {
				return lastEvent, err
			}
			if done {
				continue
			}
		}
	ConditionSucceeded:
		for {
			select {
			case event, ok := <-ch:
				if !ok {
					return lastEvent, watchtools.ErrWatchClosed
				}
				lastEvent = &event

				done, err := condition(event)
				if err != nil {
					return lastEvent, err
				}
				if done {
					break ConditionSucceeded
				}

			case <-ctx.Done():
				return lastEvent, wait.ErrorInterrupted(errors.New("timed out waiting for the condition"))
			}
		}
	}
	return lastEvent, nil
}

func checkReplicationControllerStatusReplicaCount(f *framework.Framework, rcName string, quantity int32) func(ctx context.Context) (bool, error) {
	return func(ctx context.Context) (bool, error) {

		framework.Logf("Get Replication Controller %q to confirm replicas", rcName)
		rc, err := f.ClientSet.CoreV1().ReplicationControllers(f.Namespace.Name).Get(ctx, rcName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		if rc.Status.Replicas != quantity {
			return false, nil
		}
		framework.Logf("Found %d replicas for %q replication controller", quantity, rc.Name)
		return true, nil
	}
}
