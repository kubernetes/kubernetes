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
	"fmt"
	"reflect"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eservice "k8s.io/kubernetes/test/e2e/framework/service"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	podName = "pfpod"
)

var _ = SIGDescribe("LimitRange", func() {
	f := framework.NewDefaultFramework("limitrange")

	/*
		Release : v1.18
		Testname: LimitRange, resources
		Description: Creating a Limitrange and verifying the creation of Limitrange, updating the Limitrange and validating the Limitrange. Creating Pods with resources and validate the pod resources are applied to the Limitrange
	*/
	framework.ConformanceIt("should create a LimitRange with defaults and ensure pod has those defaults applied.", func() {
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
		limitRanges, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).List(context.TODO(), options)
		framework.ExpectNoError(err, "failed to query for limitRanges")
		framework.ExpectEqual(len(limitRanges.Items), 0)
		options = metav1.ListOptions{
			LabelSelector:   selector.String(),
			ResourceVersion: limitRanges.ListMeta.ResourceVersion,
		}

		listCompleted := make(chan bool, 1)
		lw := &cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				options.LabelSelector = selector.String()
				limitRanges, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).List(context.TODO(), options)
				if err == nil {
					select {
					case listCompleted <- true:
						framework.Logf("observed the limitRanges list")
						return limitRanges, err
					default:
						framework.Logf("channel blocked")
					}
				}
				return limitRanges, err
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				options.LabelSelector = selector.String()
				return f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Watch(context.TODO(), options)
			},
		}
		_, _, w, _ := watchtools.NewIndexerInformerWatcher(lw, &v1.LimitRange{})
		defer w.Stop()

		ginkgo.By("Submitting a LimitRange")
		limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Create(context.TODO(), limitRange, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Verifying LimitRange creation was observed")
		select {
		case <-listCompleted:
			select {
			case event, _ := <-w.ResultChan():
				if event.Type != watch.Added {
					framework.Failf("Failed to observe limitRange creation : %v", event)
				}
			case <-time.After(e2eservice.RespondingTimeout):
				framework.Failf("Timeout while waiting for LimitRange creation")
			}
		case <-time.After(e2eservice.RespondingTimeout):
			framework.Failf("Timeout while waiting for LimitRange list complete")
		}

		ginkgo.By("Fetching the LimitRange to ensure it has proper values")
		limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Get(context.TODO(), limitRange.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		expected := v1.ResourceRequirements{Requests: defaultRequest, Limits: defaultLimit}
		actual := v1.ResourceRequirements{Requests: limitRange.Spec.Limits[0].DefaultRequest, Limits: limitRange.Spec.Limits[0].Default}
		err = equalResourceRequirement(expected, actual)
		framework.ExpectNoError(err)

		ginkgo.By("Creating a Pod with no resource requirements")
		pod := newTestPod("pod-no-resources", v1.ResourceList{}, v1.ResourceList{})
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring Pod has resource requirements applied from LimitRange")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
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
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Ensuring Pod has merged resource requirements applied from LimitRange")
		pod, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
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
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectError(err)

		ginkgo.By("Failing to create a Pod with more than max resources")
		pod = newTestPod(podName, getResourceList("600m", "600Mi", "600Gi"), v1.ResourceList{})
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectError(err)

		ginkgo.By("Updating a LimitRange")
		newMin := getResourceList("9m", "49Mi", "49Gi")
		limitRange.Spec.Limits[0].Min = newMin
		limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Update(context.TODO(), limitRange, metav1.UpdateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Verifying LimitRange updating is effective")
		err = wait.Poll(time.Second*2, time.Second*20, func() (bool, error) {
			limitRange, err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Get(context.TODO(), limitRange.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			return reflect.DeepEqual(limitRange.Spec.Limits[0].Min, newMin), nil
		})
		framework.ExpectNoError(err)

		ginkgo.By("Creating a Pod with less than former min resources")
		pod = newTestPod(podName, getResourceList("10m", "50Mi", "50Gi"), v1.ResourceList{})
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Failing to create a Pod with more than max resources")
		pod = newTestPod(podName, getResourceList("600m", "600Mi", "600Gi"), v1.ResourceList{})
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectError(err)

		ginkgo.By("Deleting a LimitRange")
		err = f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).Delete(context.TODO(), limitRange.Name, *metav1.NewDeleteOptions(30))
		framework.ExpectNoError(err)

		ginkgo.By("Verifying the LimitRange was deleted")
		gomega.Expect(wait.Poll(time.Second*5, e2eservice.RespondingTimeout, func() (bool, error) {
			selector := labels.SelectorFromSet(labels.Set(map[string]string{"name": limitRange.Name}))
			options := metav1.ListOptions{LabelSelector: selector.String()}
			limitRanges, err := f.ClientSet.CoreV1().LimitRanges(f.Namespace.Name).List(context.TODO(), options)

			if err != nil {
				framework.Logf("Unable to retrieve LimitRanges: %v", err)
				return false, nil
			}

			if len(limitRanges.Items) == 0 {
				framework.Logf("limitRange is already deleted")
				return true, nil
			}

			if len(limitRanges.Items) > 0 {
				if limitRanges.Items[0].ObjectMeta.DeletionTimestamp == nil {
					framework.Logf("deletion has not yet been observed")
					return false, nil
				}
				return true, nil
			}

			return false, nil

		}))

		framework.ExpectNoError(err, "kubelet never observed the termination notice")

		ginkgo.By("Creating a Pod with more than former max resources")
		pod = newTestPod(podName+"2", getResourceList("600m", "600Mi", "600Gi"), v1.ResourceList{})
		_, err = f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)
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
