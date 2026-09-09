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

package scheduling

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/klog/v2"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

const (
	podName = "pfpod"
)

var _ = SIGDescribe("LimitRange", func() {
	f := framework.NewDefaultFramework("limitrange")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	/*
		Release: v1.18
		Testname: LimitRange, resources
		Description: Creating a Limitrange and verifying the creation of Limitrange, updating the Limitrange and validating the Limitrange. Creating Pods with resources and validate the pod resources are applied to the Limitrange
	*/
	framework.ConformanceIt("should create a LimitRange with defaults and ensure pod has those defaults applied.", func(ctx context.Context) {
		ginkgo.By("Creating a LimitRange")
		min := getResourceList("50m", "100Mi", "100Gi")
		max := getResourceList("500m", "500Mi", "500Gi")
		defaultLimit := getResourceList("500m", "500Mi", "500Gi")
		defaultRequest := getResourceList("100m", "200Mi", "200Gi")
		maxLimitRequestRatio := v1.ResourceList{}
		value := strconv.Itoa(time.Now().Nanosecond()) + string(uuid.NewUUID())
		limitRange := newLimitRange("limit-range", value, v1.LimitTypeContainer,
			min, max,
			defaultLimit, defaultRequest,
			maxLimitRequestRatio)

		ginkgo.By("Setting up watch")
		selector := labels.SelectorFromSet(labels.Set(map[string]string{"time": value}))

		options := metav1.ListOptions{LabelSelector: selector.String()}
		limitRanges, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).List(ctx, options)
		framework.ExpectNoError(err, "failed to query for limitRanges")
		gomega.Expect(limitRanges.Items).To(gomega.BeEmpty())

		lw := &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.LabelSelector = selector.String()
				limitRanges, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).List(ctx, options)
				return limitRanges, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = selector.String()
				return f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Watch(ctx, options)
			},
		}
		_, informer, w, _ := watchtools.NewIndexerInformerWatcherWithLogger(klog.FromContext(ctx), lw, &v1.LimitRange{})
		defer w.Stop()

		timeoutCtx, cancel := context.WithTimeout(ctx, wait.ForeverTestTimeout)
		defer cancel()
		if !cache.WaitForCacheSync(timeoutCtx.Done(), informer.HasSynced) {
			framework.Failf("Timeout while waiting for LimitRange informer to sync")
		}

		ginkgo.By("Submitting a LimitRange")
		limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Create(ctx, limitRange, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Verifying LimitRange creation was observed")
		select {
		case event, _ := <-w.ResultChan():
			if event.Type != watch.Added {
				framework.Failf("Failed to observe limitRange creation : %v", event)
			}
		case <-time.After(e2eservice.RespondingTimeout):
			framework.Failf("Timeout while waiting for LimitRange creation")
		}

		ginkgo.By("Fetching the LimitRange to ensure it has proper values")
		limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Get(ctx, limitRange.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		expected := v1.ResourceRequirements{Requests: defaultRequest, Limits: defaultLimit}
		actual := v1.ResourceRequirements{Requests: limitRange.Spec.Limits[0].DefaultRequest, Limits: limitRange.Spec.Limits[0].Default}
		err = equalResourceRequirement(expected, actual)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a Pod with no resource requirements")
		pod := newTestPod("pod-no-resources", v1.ResourceList{}, v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring Pod has resource requirements applied from LimitRange")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		for i := range pod.Spec.Containers {
			err = equalResourceRequirement(expected, pod.Spec.Containers[i].Resources)
			if err != nil {
				// Print the pod to help in debugging.
				framework.Logf("Pod %+v does not have the expected requirements", pod)
				framework.ExpectNoError(err)
			}
		}

		ginkgo.By("Creating a Pod with partial resource requirements")
		pod = newTestPod("pod-partial-resources", getResourceList("", "150Mi", "150Gi"), getResourceList("300m", "", ""))
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring Pod has merged resource requirements applied from LimitRange")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(ctx, pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		// This is an interesting case, so it's worth a comment
		// If you specify a Limit, and no Request, the Limit will default to the Request
		// This means that the LimitRange.DefaultRequest will ONLY take affect if a container.resources.limit is not supplied
		expected = v1.ResourceRequirements{Requests: getResourceList("300m", "150Mi", "150Gi"), Limits: getResourceList("300m", "500Mi", "500Gi")}
		for i := range pod.Spec.Containers {
			err = equalResourceRequirement(expected, pod.Spec.Containers[i].Resources)
			if err != nil {
				// Print the pod to help in debugging.
				framework.Logf("Pod %+v does not have the expected requirements", pod)
				framework.ExpectNoError(err)
			}
		}

		ginkgo.By("Failing to create a Pod with less than min resources")
		pod = newTestPod(podName, getResourceList("10m", "50Mi", "50Gi"), v1.ResourceList{})
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred())

		ginkgo.By("Failing to create a Pod with more than max resources")
		pod = newTestPod(podName, getResourceList("600m", "600Mi", "600Gi"), v1.ResourceList{})
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred())

		ginkgo.By("Updating a LimitRange")
		newMin := getResourceList("9m", "49Mi", "49Gi")
		limitRange.Spec.Limits[0].Min = newMin
		limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Update(ctx, limitRange, metav1.UpdateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Verifying LimitRange updating is effective")
		err = wait.PollUntilContextTimeout(ctx, time.Second*2, time.Second*20, false, func(ctx context.Context) (bool, error) {
			limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Get(ctx, limitRange.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			return reflect.DeepEqual(limitRange.Spec.Limits[0].Min, newMin), nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating a Pod with less than former min resources")
		pod = newTestPod(podName, getResourceList("10m", "50Mi", "50Gi"), v1.ResourceList{})
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Failing to create a Pod with more than max resources")
		pod = newTestPod(podName, getResourceList("600m", "600Mi", "600Gi"), v1.ResourceList{})
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
		gomega.Expect(err).To(gomega.HaveOccurred())

		ginkgo.By("Deleting a LimitRange")
		err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Delete(ctx, limitRange.Name, *metav1.NewDeleteOptions(30))
		framework.ExpectNoError(err)

		ginkgo.By("Verifying the LimitRange was deleted")
		err = wait.PollUntilContextTimeout(ctx, time.Second*5, e2eservice.RespondingTimeout, false, func(ctx context.Context) (bool, error) {
			limitRanges, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).List(ctx, metav1.ListOptions{})

			if err != nil {
				framework.Logf("Unable to retrieve LimitRanges: %v", err)
				return false, nil
			}

			if len(limitRanges.Items) == 0 {
				framework.Logf("limitRange is already deleted")
				return true, nil
			}

			for i := range limitRanges.Items {
				lr := limitRanges.Items[i]
				framework.Logf("LimitRange %v/%v has not yet been deleted", lr.Namespace, lr.Name)
			}

			return false, nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating a Pod with more than former max resources")
		pod = newTestPod(podName+"2", getResourceList("600m", "600Mi", "600Gi"), v1.ResourceList{})
		// When the LimitRanger admission plugin find 0 items from the LimitRange informer cache,
		// it will try to lookup LimitRanges from the local LiveLookupCache which liveTTL is 30s.
		// If a LimitRange was deleted from the apiserver, informer watch the delete event and then
		// handle it lead to the informer cache doesn't have any other items, but the local LiveLookupCache
		// has it and not expired at the same time, the LimitRanger admission plugin will use the
		// deleted LimitRange to validate the request. So the request will be rejected by the plugin
		// till the item is expired.
		//
		// With the following retry, we can make sure the item is expired and the request will be
		// validated as expected.
		err = framework.Gomega().Eventually(ctx, func(ctx context.Context) error {
			_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(ctx, pod, metav1.CreateOptions{})
			return err
		}).WithPolling(5 * time.Second).WithTimeout(30 * time.Second).ShouldNot(gomega.HaveOccurred())
		framework.ExpectNoError(err)
	})

	/*
		Release: v1.26
		Testname: LimitRange, list, patch and delete a LimitRange by collection
		Description: When two limitRanges are created in different namespaces,
		both MUST succeed. Listing limitRanges across all namespaces with a
		labelSelector MUST find both limitRanges. When patching the first limitRange
		it MUST succeed and the fields MUST equal the new values. When deleting
		the limitRange by collection with a labelSelector it MUST delete only one
		limitRange.
	*/
	framework.ConformanceIt("should list, patch and delete a LimitRange by collection", ginkgo.NodeTimeout(wait.ForeverTestTimeout), func(ctx context.Context) {

		ns := f.Namespace.Name
		lrClient := f.ClientSet.CoreV1().LimitRanges(ns)
		lrName := "e2e-limitrange-" + utilrand.String(5)
		e2eLabelSelector := "e2e-test=" + lrName
		patchedLabelSelector := lrName + "=patched"

		min := getResourceList("50m", "100Mi", "100Gi")
		max := getResourceList("500m", "500Mi", "500Gi")
		defaultLimit := getResourceList("500m", "500Mi", "500Gi")
		defaultRequest := getResourceList("100m", "200Mi", "200Gi")
		maxLimitRequestRatio := v1.ResourceList{}

		limitRange := &v1.LimitRange{
			ObjectMeta: metav1.ObjectMeta{
				Name: lrName,
				Labels: map[string]string{
					"e2e-test": lrName,
					lrName:     "created",
				},
			},
			Spec: v1.LimitRangeSpec{
				Limits: []v1.LimitRangeItem{
					{
						Type:                 v1.LimitTypeContainer,
						Min:                  min,
						Max:                  max,
						Default:              defaultLimit,
						DefaultRequest:       defaultRequest,
						MaxLimitRequestRatio: maxLimitRequestRatio,
					},
				},
			},
		}
		// Create a copy to be used in a second namespace
		limitRange2 := &v1.LimitRange{}
		*limitRange2 = *limitRange

		ginkgo.By(fmt.Sprintf("Creating LimitRange %q in namespace %q", lrName, f.Namespace.Name))
		limitRange, err := lrClient.Create(ctx, limitRange, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create limitRange %q", lrName)
		gomega.Expect(limitRange).To(apimachineryutils.HaveValidResourceVersion())

		ginkgo.By("Creating another limitRange in another namespace")
		lrNamespace, err := f.CreateNamespace(ctx, lrName, nil)
		framework.ExpectNoError(err, "failed creating Namespace")
		framework.Logf("Namespace %q created", lrNamespace.ObjectMeta.Name)
		framework.Logf("Creating LimitRange %q in namespace %q", lrName, lrNamespace.Name)
		_, err = f.ClientSet.CoreV1().LimitRanges(lrNamespace.ObjectMeta.Name).Create(ctx, limitRange2, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create limitRange %q in %q namespace", lrName, lrNamespace.ObjectMeta.Name)

		// Listing across all namespaces to verify api endpoint: listCoreV1LimitRangeForAllNamespaces
		ginkgo.By(fmt.Sprintf("Listing all LimitRanges with label %q", e2eLabelSelector))
		limitRangeList, err := f.ClientSet.CoreV1().LimitRanges("").List(ctx, metav1.ListOptions{LabelSelector: e2eLabelSelector})
		framework.ExpectNoError(err, "Failed to list any limitRanges: %v", err)
		gomega.Expect(limitRangeList.Items).To(gomega.HaveLen(2), "Failed to find the correct limitRange count")
		framework.Logf("Found %d limitRanges", len(limitRangeList.Items))

		ginkgo.By(fmt.Sprintf("Patching LimitRange %q in %q namespace", lrName, ns))
		newMin := getResourceList("9m", "49Mi", "49Gi")
		limitRange.Spec.Limits[0].Min = newMin

		limitRangePayload, err := json.Marshal(v1.LimitRange{
			ObjectMeta: metav1.ObjectMeta{
				CreationTimestamp: limitRange.CreationTimestamp,
				Labels: map[string]string{
					lrName: "patched",
				},
			},
			Spec: v1.LimitRangeSpec{
				Limits: limitRange.Spec.Limits,
			},
		})
		framework.ExpectNoError(err, "Failed to marshal limitRange JSON")

		patchedLimitRange, err := lrClient.Patch(ctx, lrName, types.StrategicMergePatchType, []byte(limitRangePayload), metav1.PatchOptions{})
		framework.ExpectNoError(err, "Failed to patch limitRange %q", lrName)
		gomega.Expect(patchedLimitRange.Labels[lrName]).To(gomega.Equal("patched"), "%q label didn't have value 'patched' for this limitRange. Current labels: %v", lrName, patchedLimitRange.Labels)
		checkMinLimitRange := apiequality.Semantic.DeepEqual(patchedLimitRange.Spec.Limits[0].Min, newMin)
		if !checkMinLimitRange {
			framework.Failf("LimitRange does not have the correct min limitRange. Currently is %#v ", patchedLimitRange.Spec.Limits[0].Min)
		}
		framework.Logf("LimitRange %q has been patched", lrName)
		gomega.Expect(resourceversion.CompareResourceVersion(limitRange.ResourceVersion, patchedLimitRange.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		ginkgo.By(fmt.Sprintf("Delete LimitRange %q by Collection with labelSelector: %q", lrName, patchedLabelSelector))
		err = lrClient.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: patchedLabelSelector})
		framework.ExpectNoError(err, "failed to delete the LimitRange by Collection")

		ginkgo.By(fmt.Sprintf("Confirm that the limitRange %q has been deleted", lrName))
		err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, checkLimitRangeListQuantity(f, patchedLabelSelector, 0))
		framework.ExpectNoError(err, "failed to count the required limitRanges")
		framework.Logf("LimitRange %q has been deleted.", lrName)

		ginkgo.By(fmt.Sprintf("Confirm that a single LimitRange still exists with label %q", e2eLabelSelector))
		limitRangeList, err = f.ClientSet.CoreV1().LimitRanges("").List(ctx, metav1.ListOptions{LabelSelector: e2eLabelSelector})
		framework.ExpectNoError(err, "Failed to list any limitRanges: %v", err)
		gomega.Expect(limitRangeList.Items).To(gomega.HaveLen(1), "Failed to find the correct limitRange count")
		framework.Logf("Found %d limitRange", len(limitRangeList.Items))
	})
})

func equalResourceRequirement(expected v1.ResourceRequirements, actual v1.ResourceRequirements) error {
	framework.Logf("Verifying requests: expected %v with actual %v", expected.Requests, actual.Requests)
	err := equalResourceList(expected.Requests, actual.Requests)
	if err != nil {
		return err
	}
	framework.Logf("Verifying limits: expected %v with actual %v", expected.Limits, actual.Limits)
	err = equalResourceList(expected.Limits, actual.Limits)
	return err
}

func equalResourceList(expected v1.ResourceList, actual v1.ResourceList) error {
	for k, v := range expected {
		if actualValue, found := actual[k]; !found || (v.Cmp(actualValue) != 0) {
			return fmt.Errorf("resource %v expected %v actual %v", k, v.String(), actualValue.String())
		}
	}
	for k, v := range actual {
		if expectedValue, found := expected[k]; !found || (v.Cmp(expectedValue) != 0) {
			return fmt.Errorf("resource %v expected %v actual %v", k, expectedValue.String(), v.String())
		}
	}
	return nil
}

func getResourceList(cpu, memory string, ephemeralStorage string) v1.ResourceList {
	res := v1.ResourceList{}
	if cpu != "" {
		res[v1.ResourceCPU] = resource.MustParse(cpu)
	}
	if memory != "" {
		res[v1.ResourceMemory] = resource.MustParse(memory)
	}
	if ephemeralStorage != "" {
		res[v1.ResourceEphemeralStorage] = resource.MustParse(ephemeralStorage)
	}
	return res
}

// newLimitRange returns a limit range with specified data
func newLimitRange(name, value string, limitType v1.LimitType,
	min, max,
	defaultLimit, defaultRequest,
	maxLimitRequestRatio v1.ResourceList) *v1.LimitRange {
	return &v1.LimitRange{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"time": value,
			},
		},
		Spec: v1.LimitRangeSpec{
			Limits: []v1.LimitRangeItem{
				{
					Type:                 limitType,
					Min:                  min,
					Max:                  max,
					Default:              defaultLimit,
					DefaultRequest:       defaultRequest,
					MaxLimitRequestRatio: maxLimitRequestRatio,
				},
			},
		},
	}
}

// newTestPod returns a pod that has the specified requests and limits
func newTestPod(name string, requests v1.ResourceList, limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "pause",
					Image: imageutils.GetPauseImageName(),
					Resources: v1.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
		},
	}
}

func checkLimitRangeListQuantity(f *framework.Framework, label string, quantity int) func(ctx context.Context) (bool, error) {
	return func(ctx context.Context) (bool, error) {
		framework.Logf("Requesting list of LimitRange to confirm quantity")

		list, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).List(ctx, metav1.ListOptions{LabelSelector: label})
		if err != nil {
			return false, err
		}

		if len(list.Items) != quantity {
			return false, nil
		}
		framework.Logf("Found %d LimitRange with label %q", quantity, label)
		return true, nil
	}
}
