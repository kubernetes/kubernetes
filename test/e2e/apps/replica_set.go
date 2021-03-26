/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/controller/replicaset"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2ereplicaset "k8s.io/kubernetes/test/e2e/framework/replicaset"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

func newRS(rsName string, replicas int32, rsPodLabels map[string]string, imageName string, image string, args []string) *appsv1.ReplicaSet {
	zero := int64(0)
	return &appsv1.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:   rsName,
			Labels: rsPodLabels,
		},
		Spec: appsv1.ReplicaSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: rsPodLabels,
			},
			Replicas: &replicas,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: rsPodLabels,
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

func newPodQuota(name, number string) *v1.ResourceQuota {
	return &v1.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ResourceQuotaSpec{
			Hard: v1.ResourceList{
				v1.ResourcePods: resource.MustParse(number),
			},
		},
	}
}

var _ = SIGDescribe("ReplicaSet", func() {
	f := framework.NewDefaultFramework("replicaset")

	/*
		Release: v1.9
		Testname: Replica Set, run basic image
		Description: Create a ReplicaSet with a Pod and a single Container. Make sure that the Pod is running. Pod SHOULD send a valid response when queried.
	*/
	framework.ConformanceIt("should serve a basic image on each replica with a public image ", func() {
		testReplicaSetServeImageOrFail(f, "basic", framework.ServeHostnameImage)
	})

	ginkgo.It("should serve a basic image on each replica with a private image", func() {
		// requires private images
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
		privateimage := imageutils.GetConfig(imageutils.AgnhostPrivate)
		testReplicaSetServeImageOrFail(f, "private", privateimage.GetE2EImage())
	})

	ginkgo.It("should surface a failure condition on a common issue like exceeded quota", func() {
		testReplicaSetConditionCheck(f)
	})

	/*
		Release: v1.13
		Testname: Replica Set, adopt matching pods and release non matching pods
		Description: A Pod is created, then a Replica Set (RS) whose label selector will match the Pod. The RS MUST either adopt the Pod or delete and replace it with a new Pod. When the labels on one of the Pods owned by the RS change to no longer match the RS's label selector, the RS MUST release the Pod and update the Pod's owner references
	*/
	framework.ConformanceIt("should adopt matching pods on creation and release no longer matching pods", func() {
		testRSAdoptMatchingAndReleaseNotMatching(f)
	})

	/*
		Release: v1.21
		Testname: ReplicaSet, completes the scaling of a ReplicaSet subresource
		Description: Create a ReplicaSet (RS) with a single Pod. The Pod MUST be verified
		that it is running. The RS MUST get and verify the scale subresource count.
		The RS MUST update and verify the scale subresource. The RS MUST patch and verify
		a scale subresource.
	*/
	framework.ConformanceIt("Replicaset should have a working scale subresource", func() {
		testRSScaleSubresources(f)
	})

	/*
		Release: v1.21
		Testname: ReplicaSet, is created, Replaced and Patched
		Description: Create a ReplicaSet (RS) with a single Pod. The Pod MUST be verified
		that it is running. The RS MUST scale to two replicas and verify the scale count
		The RS MUST be patched and verify that patch succeeded.
	*/
	framework.ConformanceIt("Replace and Patch tests", func() {
		testRSLifeCycle(f)
	})
})

// A basic test to check the deployment of an image using a ReplicaSet. The
// image serves its hostname which is checked for each replica.
func testReplicaSetServeImageOrFail(f *framework.Framework, test string, image string) {
	name := "my-hostname-" + test + "-" + string(uuid.NewUUID())
	replicas := int32(1)

	// Create a ReplicaSet for a service that serves its hostname.
	// The source for the Docker containter kubernetes/serve_hostname is
	// in contrib/for-demos/serve_hostname
	framework.Logf("Creating ReplicaSet %s", name)
	newRS := newRS(name, replicas, map[string]string{"name": name}, name, image, []string{"serve-hostname"})
	newRS.Spec.Template.Spec.Containers[0].Ports = []v1.ContainerPort{{ContainerPort: 9376}}
	_, err := f.ClientSet.AppsV1().ReplicaSets(f.Namespace.Name).Create(context.TODO(), newRS, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	// Check that pods for the new RS were created.
	// TODO: Maybe switch PodsCreated to just check owner references.
	pods, err := e2epod.PodsCreated(f.ClientSet, f.Namespace.Name, name, replicas)
	framework.ExpectNoError(err)

	// Wait for the pods to enter the running state. Waiting loops until the pods
	// are running so non-running pods cause a timeout for this test.
	framework.Logf("Ensuring a pod for ReplicaSet %q is running", name)
	running := int32(0)
	for _, pod := range pods.Items {
		if pod.DeletionTimestamp != nil {
			continue
		}
		err = e2epod.WaitForPodNameRunningInNamespace(f.ClientSet, pod.Name, f.Namespace.Name)
		if err != nil {
			updatePod, getErr := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
			if getErr == nil {
				err = fmt.Errorf("pod %q never run (phase: %s, conditions: %+v): %v", updatePod.Name, updatePod.Status.Phase, updatePod.Status.Conditions, err)
			} else {
				err = fmt.Errorf("pod %q never run: %v", pod.Name, err)
			}
		}
		framework.ExpectNoError(err)
		framework.Logf("Pod %q is running (conditions: %+v)", pod.Name, pod.Status.Conditions)
		running++
	}

	// Sanity check
	framework.ExpectEqual(running, replicas, "unexpected number of running pods: %+v", pods.Items)

	// Verify that something is listening.
	framework.Logf("Trying to dial the pod")
	retryTimeout := 2 * time.Minute
	retryInterval := 5 * time.Second
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": name}))
	err = wait.Poll(retryInterval, retryTimeout, e2epod.NewProxyResponseChecker(f.ClientSet, f.Namespace.Name, label, name, true, pods).CheckAllResponses)
	if err != nil {
		framework.Failf("Did not get expected responses within the timeout period of %.2f seconds.", retryTimeout.Seconds())
	}
}

// 1. Create a quota restricting pods in the current namespace to 2.
// 2. Create a replica set that wants to run 3 pods.
// 3. Check replica set conditions for a ReplicaFailure condition.
// 4. Scale down the replica set and observe the condition is gone.
func testReplicaSetConditionCheck(f *framework.Framework) {
	c := f.ClientSet
	namespace := f.Namespace.Name
	name := "condition-test"

	ginkgo.By(fmt.Sprintf("Creating quota %q that allows only two pods to run in the current namespace", name))
	quota := newPodQuota(name, "2")
	_, err := c.CoreV1().ResourceQuotas(namespace).Create(context.TODO(), quota, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		quota, err = c.CoreV1().ResourceQuotas(namespace).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		quantity := resource.MustParse("2")
		podQuota := quota.Status.Hard[v1.ResourcePods]
		return (&podQuota).Cmp(quantity) == 0, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("resource quota %q never synced", name)
	}
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Creating replica set %q that asks for more than the allowed pod quota", name))
	rs := newRS(name, 3, map[string]string{"name": name}, WebserverImageName, WebserverImage, nil)
	rs, err = c.AppsV1().ReplicaSets(namespace).Create(context.TODO(), rs, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Checking replica set %q has the desired failure condition set", name))
	generation := rs.Generation
	conditions := rs.Status.Conditions
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		rs, err = c.AppsV1().ReplicaSets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		if generation > rs.Status.ObservedGeneration {
			return false, nil
		}
		conditions = rs.Status.Conditions

		cond := replicaset.GetCondition(rs.Status, appsv1.ReplicaSetReplicaFailure)
		return cond != nil, nil

	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("rs controller never added the failure condition for replica set %q: %#v", name, conditions)
	}
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Scaling down replica set %q to satisfy pod quota", name))
	rs, err = e2ereplicaset.UpdateReplicaSetWithRetries(c, namespace, name, func(update *appsv1.ReplicaSet) {
		x := int32(2)
		update.Spec.Replicas = &x
	})
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("Checking replica set %q has no failure condition set", name))
	generation = rs.Generation
	conditions = rs.Status.Conditions
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		rs, err = c.AppsV1().ReplicaSets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
		if err != nil {
			return false, err
		}

		if generation > rs.Status.ObservedGeneration {
			return false, nil
		}
		conditions = rs.Status.Conditions

		cond := replicaset.GetCondition(rs.Status, appsv1.ReplicaSetReplicaFailure)
		return cond == nil, nil
	})
	if err == wait.ErrWaitTimeout {
		err = fmt.Errorf("rs controller never removed the failure condition for rs %q: %#v", name, conditions)
	}
	framework.ExpectNoError(err)
}

func testRSAdoptMatchingAndReleaseNotMatching(f *framework.Framework) {
	name := "pod-adoption-release"
	ginkgo.By(fmt.Sprintf("Given a Pod with a 'name' label %s is created", name))
	p := f.PodClient().CreateSync(&v1.Pod{
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

	ginkgo.By("When a replicaset with a matching selector is created")
	replicas := int32(1)
	rsSt := newRS(name, replicas, map[string]string{"name": name}, name, WebserverImage, nil)
	rsSt.Spec.Selector = &metav1.LabelSelector{MatchLabels: map[string]string{"name": name}}
	rs, err := f.ClientSet.AppsV1().ReplicaSets(f.Namespace.Name).Create(context.TODO(), rsSt, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("Then the orphan pod is adopted")
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		p2, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), p.Name, metav1.GetOptions{})
		// The Pod p should either be adopted or deleted by the ReplicaSet
		if apierrors.IsNotFound(err) {
			return true, nil
		}
		framework.ExpectNoError(err)
		for _, owner := range p2.OwnerReferences {
			if *owner.Controller && owner.UID == rs.UID {
				// pod adopted
				return true, nil
			}
		}
		// pod still not adopted
		return false, nil
	})
	framework.ExpectNoError(err)

	ginkgo.By("When the matched label of one of its pods change")
	pods, err := e2epod.PodsCreated(f.ClientSet, f.Namespace.Name, rs.Name, replicas)
	framework.ExpectNoError(err)

	p = &pods.Items[0]
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		pod, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		pod.Labels = map[string]string{"name": "not-matching-name"}
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Update(context.TODO(), pod, metav1.UpdateOptions{})
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
	err = wait.PollImmediate(1*time.Second, 1*time.Minute, func() (bool, error) {
		p2, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), p.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		for _, owner := range p2.OwnerReferences {
			if *owner.Controller && owner.UID == rs.UID {
				// pod still belonging to the replicaset
				return false, nil
			}
		}
		// pod already released
		return true, nil
	})
	framework.ExpectNoError(err)
}

func testRSScaleSubresources(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet

	// Create webserver pods.
	rsPodLabels := map[string]string{
		"name": "sample-pod",
		"pod":  WebserverImageName,
	}

	rsName := "test-rs"
	replicas := int32(1)
	ginkgo.By(fmt.Sprintf("Creating replica set %q that asks for more than the allowed pod quota", rsName))
	rs := newRS(rsName, replicas, rsPodLabels, WebserverImageName, WebserverImage, nil)
	_, err := c.AppsV1().ReplicaSets(ns).Create(context.TODO(), rs, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	// Verify that the required pods have come up.
	err = e2epod.VerifyPodsRunning(c, ns, "sample-pod", false, replicas)
	framework.ExpectNoError(err, "error in waiting for pods to come up: %s", err)

	ginkgo.By("getting scale subresource")
	scale, err := c.AppsV1().ReplicaSets(ns).GetScale(context.TODO(), rsName, metav1.GetOptions{})
	if err != nil {
		framework.Failf("Failed to get scale subresource: %v", err)
	}
	framework.ExpectEqual(scale.Spec.Replicas, int32(1))
	framework.ExpectEqual(scale.Status.Replicas, int32(1))

	ginkgo.By("updating a scale subresource")
	scale.ResourceVersion = "" // indicate the scale update should be unconditional
	scale.Spec.Replicas = 2
	scaleResult, err := c.AppsV1().ReplicaSets(ns).UpdateScale(context.TODO(), rsName, scale, metav1.UpdateOptions{})
	if err != nil {
		framework.Failf("Failed to put scale subresource: %v", err)
	}
	framework.ExpectEqual(scaleResult.Spec.Replicas, int32(2))

	ginkgo.By("verifying the replicaset Spec.Replicas was modified")
	rs, err = c.AppsV1().ReplicaSets(ns).Get(context.TODO(), rsName, metav1.GetOptions{})
	if err != nil {
		framework.Failf("Failed to get statefulset resource: %v", err)
	}
	framework.ExpectEqual(*(rs.Spec.Replicas), int32(2))

	ginkgo.By("Patch a scale subresource")
	scale.ResourceVersion = "" // indicate the scale update should be unconditional
	scale.Spec.Replicas = 4    // should be 2 after "UpdateScale" operation, now Patch to 4
	rsScalePatchPayload, err := json.Marshal(autoscalingv1.Scale{
		Spec: autoscalingv1.ScaleSpec{
			Replicas: scale.Spec.Replicas,
		},
	})
	framework.ExpectNoError(err, "Could not Marshal JSON for patch payload")

	_, err = c.AppsV1().ReplicaSets(ns).Patch(context.TODO(), rsName, types.StrategicMergePatchType, []byte(rsScalePatchPayload), metav1.PatchOptions{}, "scale")
	framework.ExpectNoError(err, "Failed to patch replicaset: %v", err)

	rs, err = c.AppsV1().ReplicaSets(ns).Get(context.TODO(), rsName, metav1.GetOptions{})
	framework.ExpectNoError(err, "Failed to get replicaset resource: %v", err)
	framework.ExpectEqual(*(rs.Spec.Replicas), int32(4), "replicaset should have 4 replicas")

}

// ReplicaSet Replace and Patch tests
func testRSLifeCycle(f *framework.Framework) {
	ns := f.Namespace.Name
	c := f.ClientSet
	zero := int64(0)

	// Create webserver pods.
	rsPodLabels := map[string]string{
		"name": "sample-pod",
		"pod":  WebserverImageName,
	}

	rsName := "test-rs"
	label := "test-rs=patched"
	labelMap := map[string]string{"test-rs": "patched"}
	replicas := int32(1)
	rsPatchReplicas := int32(3)
	rsPatchImage := imageutils.GetE2EImage(imageutils.Pause)

	w := &cache.ListWatch{
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.LabelSelector = label
			return f.ClientSet.AppsV1().ReplicaSets(ns).Watch(context.TODO(), options)
		},
	}
	rsList, err := f.ClientSet.AppsV1().ReplicaSets("").List(context.TODO(), metav1.ListOptions{LabelSelector: label})
	framework.ExpectNoError(err, "failed to list rsList")
	// Create a ReplicaSet
	rs := newRS(rsName, replicas, rsPodLabels, WebserverImageName, WebserverImage, nil)
	_, err = c.AppsV1().ReplicaSets(ns).Create(context.TODO(), rs, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	// Verify that the required pods have come up.
	err = e2epod.VerifyPodsRunning(c, ns, "sample-pod", false, replicas)
	framework.ExpectNoError(err, "Failed to create pods: %s", err)

	// Scale the ReplicaSet
	ginkgo.By(fmt.Sprintf("Scaling up %q replicaset ", rsName))
	_, err = e2ereplicaset.UpdateReplicaSetWithRetries(c, ns, rsName, func(update *appsv1.ReplicaSet) {
		x := int32(2)
		update.Spec.Replicas = &x
	})
	framework.ExpectNoError(err, "ReplicaSet fail to scale to %q replicasets")

	// Patch the PeplicaSet
	ginkgo.By("patching the ReplicaSet")
	rsPatch, err := json.Marshal(map[string]interface{}{
		"metadata": map[string]interface{}{
			"labels": labelMap,
		},
		"spec": map[string]interface{}{
			"replicas": rsPatchReplicas,
			"template": map[string]interface{}{
				"spec": map[string]interface{}{
					"TerminationGracePeriodSeconds": &zero,
					"containers": [1]map[string]interface{}{{
						"name":  rsName,
						"image": rsPatchImage,
					}},
				},
			},
		},
	})
	framework.ExpectNoError(err, "failed to Marshal ReplicaSet JSON patch")
	_, err = f.ClientSet.AppsV1().ReplicaSets(ns).Patch(context.TODO(), rsName, types.StrategicMergePatchType, []byte(rsPatch), metav1.PatchOptions{})
	framework.ExpectNoError(err, "failed to patch ReplicaSet")

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err = watchtools.Until(ctx, rsList.ResourceVersion, w, func(event watch.Event) (bool, error) {
		if rset, ok := event.Object.(*appsv1.ReplicaSet); ok {
			found := rset.ObjectMeta.Name == rsName &&
				rset.ObjectMeta.Labels["test-rs"] == "patched" &&
				rset.Status.ReadyReplicas == rsPatchReplicas &&
				rset.Status.AvailableReplicas == rsPatchReplicas &&
				rset.Spec.Template.Spec.Containers[0].Image == rsPatchImage
			if !found {
				framework.Logf("observed ReplicaSet %v in namespace %v with ReadyReplicas %v, AvailableReplicas %v", rset.ObjectMeta.Name, rset.ObjectMeta.Namespace, rset.Status.ReadyReplicas,
					rset.Status.AvailableReplicas)
			} else {
				framework.Logf("observed Replicaset %v in namespace %v with ReadyReplicas %v found %v", rset.ObjectMeta.Name, rset.ObjectMeta.Namespace, rset.Status.ReadyReplicas, found)
			}
			return found, nil
		}
		return false, nil
	})

	framework.ExpectNoError(err, "failed to see replicas of %v in namespace %v scale to requested amount of %v", rs.Name, ns, rsPatchReplicas)
}
