/*
Copyright 2025 The Kubernetes Authors.

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

package dra

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/onsi/gomega"
	gtypes "github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

const (
	driverNameSuffix = ".driver"
	classNameSuffix  = ".class"
)

// must can be wrapped around a Create/Update/Patch/Get/Delete call and handles the error checking:
//
//	pod = must(tCtx, tCtx.Client().CoreV1().Pods(namespace).Create, pod, metav1.CreateOptions{})
func must[R, P, O any](tCtx ktesting.TContext, call func(context.Context, P, O) (*R, error), p P, o O) *R {
	tCtx.Helper()
	r, err := call(tCtx, p, o)
	tCtx.ExpectNoError(err)
	return r
}

// createTestNamespace creates a namespace with a name that is derived from the
// current test name:
// - Non-alpha-numeric characters replaced by hyphen.
// - Truncated in the middle to make it short enough for GenerateName.
// - Hyphen plus random suffix added by the apiserver.
func createTestNamespace(tCtx ktesting.TContext, labels map[string]string) string {
	tCtx.Helper()
	name := regexp.MustCompile(`[^[:alnum:]_-]`).ReplaceAllString(tCtx.Name(), "-")
	name = strings.ToLower(name)
	// Make sure the generated name leaves enough room so we
	// can use it as a prefix for the driver name.
	if len(name) > (56 - len(driverNameSuffix)) {
		name = name[:24] + "--" + name[len(name)-24:]
	}
	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{GenerateName: name + "-"}}
	ns.Labels = labels
	ns, err := tCtx.Client().CoreV1().Namespaces().Create(tCtx, ns, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create test namespace")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(tCtx.Client().CoreV1().Namespaces().Delete(tCtx, ns.Name, metav1.DeleteOptions{}))
		// *Not* waiting here. Deleting namespaces is slow.
	})
	return ns.Name
}

// createSlice creates the given ResourceSlice and removes it when the test is done.
func createSlice(tCtx ktesting.TContext, slice *resourceapi.ResourceSlice) *resourceapi.ResourceSlice {
	tCtx.Helper()
	slice, err := tCtx.Client().ResourceV1().ResourceSlices().Create(tCtx, slice, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create ResourceSlice")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Cleaning up ResourceSlice...")
		deleteAndWait(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Delete, tCtx.Client().ResourceV1().ResourceSlices().Get, slice.Name)
	})
	return slice
}

// createTestClass creates a DeviceClass with a driver name derived from the test namespace
func createTestClass(tCtx ktesting.TContext, namespace string) (*resourceapi.DeviceClass, string) {
	tCtx.Helper()
	driverName := namespace + driverNameSuffix
	class := class.DeepCopy()
	class.Name = namespace + classNameSuffix
	class.Spec.Selectors = []resourceapi.DeviceSelector{{
		CEL: &resourceapi.CELDeviceSelector{
			Expression: fmt.Sprintf("device.driver == %q", driverName),
		},
	}}
	_, err := tCtx.Client().ResourceV1().DeviceClasses().Create(tCtx, class, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create class")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Cleaning up DeviceClass...")
		deleteAndWait(tCtx, tCtx.Client().ResourceV1().DeviceClasses().Delete, tCtx.Client().ResourceV1().DeviceClasses().Get, class.Name)
	})

	return class, driverName
}

// createClaim creates a claim and in the namespace.
// The class must already exist and is used for all requests.
func createClaim(tCtx ktesting.TContext, namespace string, suffix string, class *resourceapi.DeviceClass, claim *resourceapi.ResourceClaim) *resourceapi.ResourceClaim {
	tCtx.Helper()
	claim = claim.DeepCopy()
	claim.Namespace = namespace
	claim.Name += suffix
	claimName := claim.Name
	for i := range claim.Spec.Devices.Requests {
		request := &claim.Spec.Devices.Requests[i]
		if request.Exactly != nil && request.Exactly.DeviceClassName != "" {
			request.Exactly.DeviceClassName = class.Name
			continue
		}
		for e := range request.FirstAvailable {
			subRequest := &request.FirstAvailable[e]
			subRequest.DeviceClassName = class.Name
		}
	}
	claim, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create claim "+claimName)
	// TODO: some tests leak claims. Probably they need to be fixed... later.
	// tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
	//		// We want to know when tearing this down gets stuck.
	//	deleteAndWait(tCtx, tCtx.Client().ResourceV1().ResourceClaims(namespace).Delete, tCtx.Client().ResourceV1().ResourceClaims(namespace).Get, claim.Name)
	// })
	return claim
}

// createPod create a pod in the namespace, referencing the given claim.
func createPod(tCtx ktesting.TContext, namespace string, suffix string, pod *v1.Pod, claims ...*resourceapi.ResourceClaim) *v1.Pod {
	tCtx.Helper()
	pod = pod.DeepCopy()
	pod.Name += suffix
	podName := pod.Name
	pod.Namespace = namespace
	var resourceClaims []v1.PodResourceClaim
	for _, claim := range claims {
		resourceClaims = append(resourceClaims, v1.PodResourceClaim{
			Name:              claim.Name,
			ResourceClaimName: &claim.Name,
		})
	}
	pod.Spec.ResourceClaims = resourceClaims
	pod, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, pod, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create pod "+podName)
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.Log("Cleaning up Pod...")
		// We must delete pods before uninstalling our driver.
		// Also, we want to know when stopping it gets stuck.
		deleteAndWait(tCtx, tCtx.Client().CoreV1().Pods(namespace).Delete, tCtx.Client().CoreV1().Pods(namespace).Get, pod.Name)
	})
	return pod
}

func waitForPodScheduled(tCtx ktesting.TContext, namespace, podName string) *v1.Pod {
	tCtx.Helper()

	var pod *v1.Pod
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) {
		p, err := tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, podName, metav1.GetOptions{})
		pod = p
		return p, err
	}).WithTimeout(60*time.Second).Should(
		gomega.HaveField("Status.Conditions", gomega.ContainElement(
			gomega.And(
				gomega.HaveField("Type", gomega.Equal(v1.PodScheduled)),
				gomega.HaveField("Status", gomega.Equal(v1.ConditionTrue)),
			),
		)),
		"Pod %s should have been scheduled.", podName,
	)
	return pod
}

func deleteAndWait[T any](tCtx ktesting.TContext, del func(context.Context, string, metav1.DeleteOptions) error, get func(context.Context, string, metav1.GetOptions) (T, error), name string) {
	tCtx.Helper()

	var t T
	var anyT any = t
	var options metav1.DeleteOptions
	if _, ok := anyT.(*v1.Pod); ok {
		// Special case for pods: we don't have a kubelet which acknowledges
		// shutdown of a scheduled pod, so we have to force-delete.
		options.GracePeriodSeconds = ptr.To(int64(0))
	}

	tCtx.ExpectNoError(del(tCtx, name, options), fmt.Sprintf("delete %T %s", t, name))
	waitForNotFound(tCtx, get, name)
}

func waitForNotFound[T any](tCtx ktesting.TContext, get func(context.Context, string, metav1.GetOptions) (T, error), name string) {
	tCtx.Helper()

	var t T
	tCtx.Eventually(func(tCtx ktesting.TContext) error {
		_, err := get(tCtx, name, metav1.GetOptions{})
		return err
	}).WithTimeout(60*time.Second).Should(gomega.MatchError(apierrors.IsNotFound, "IsNotFound"), "Object %T %s should have been removed.", t, name)
}

func waitForClaim(tCtx ktesting.TContext, namespace, claimName string, timeout time.Duration, match gtypes.GomegaMatcher, description ...any) *resourceapi.ResourceClaim {
	tCtx.Helper()
	var latestClaim *resourceapi.ResourceClaim
	tCtx.Eventually(func(tCtx ktesting.TContext) (*resourceapi.ResourceClaim, error) {
		c, err := tCtx.Client().ResourceV1().ResourceClaims(namespace).Get(tCtx, claimName, metav1.GetOptions{})
		latestClaim = c
		return c, err
	}).WithTimeout(timeout).WithPolling(time.Second).Should(match, description...)
	return latestClaim
}

func waitForClaimAllocatedToDevice(tCtx ktesting.TContext, namespace, claimName string, timeout time.Duration) *resourceapi.ResourceClaim {
	tCtx.Helper()
	return waitForClaim(
		tCtx,
		namespace,
		claimName,
		timeout,
		gomega.HaveField("Status.Allocation", gomega.Not(gomega.BeNil())),
		"Claim should have been allocated.",
	)
}
