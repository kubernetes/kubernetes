/*
Copyright The Kubernetes Authors.

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
	"sync"
	"sync/atomic"
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourcealpha "k8s.io/api/resource/v1alpha3"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	resourcealphainformers "k8s.io/client-go/informers/resource/v1alpha3"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/controller/devicetainteviction"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

// testEvictCluster simulates a cluster with many scheduled pods where each
// pod uses it's own ResourceClaim with one device. Then all those
// devices get tainted with a single DeviceTaintRule (causing eviction of all pods at once)
// or by updating the slices (more gradual).
func testEvictCluster(tCtx ktesting.TContext, useRule bool) {
	tCtx.Parallel()

	var wg sync.WaitGroup
	defer func() {
		tCtx.Cancel("time to shut down")
		wg.Wait()
	}()

	numPods := 1000
	devicesPerSlice := 50
	numSlices := (numPods + devicesPerSlice - 1) / devicesPerSlice
	namespace := createTestNamespace(tCtx, nil)
	nodeName := "node-" + namespace
	driverName := "driver-" + namespace
	poolName := "cluster"

	var slices []*resourceapi.ResourceSlice
	for i := range numSlices {
		slice := &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("slice-%s-%d", namespace, i),
			},
			Spec: resourceapi.ResourceSliceSpec{
				Driver: driverName,
				Pool: resourceapi.ResourcePool{
					Name:               poolName,
					ResourceSliceCount: int64(numSlices),
				},
				AllNodes: ptr.To(true),
			},
		}
		for e := range devicesPerSlice {
			slice.Spec.Devices = append(slice.Spec.Devices, resourceapi.Device{
				Name: fmt.Sprintf("device-%d", i*devicesPerSlice+e),
			})
		}
		slices = append(slices, createSlice(tCtx, slice))
	}

	for i := range numPods {
		suffix := fmt.Sprintf("-%d", i)

		pod := podWithClaimName.DeepCopy()
		pod.Name += suffix
		pod.Namespace = namespace
		*pod.Spec.ResourceClaims[0].ResourceClaimName += suffix
		pod.Spec.NodeName = nodeName // Must be scheduled to be evicted.
		pod = must(tCtx, tCtx.Client().CoreV1().Pods(namespace).Create, pod, metav1.CreateOptions{})

		// Scheduled pods do not get deleted without acknowledgement by the kubelet,
		// so we have to force-delete to clean up as we don't have a kubelet.
		tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
			err := tCtx.Client().CoreV1().Pods(namespace).Delete(tCtx, pod.Name, metav1.DeleteOptions{GracePeriodSeconds: ptr.To(int64(0))})
			if !apierrors.IsNotFound(err) {
				tCtx.ExpectNoError(err)
			}
		})

		// No finalizer, so deleting the namespace will be able to delete the claim.
		claim := claim.DeepCopy()
		claim.Name += suffix
		claim.Namespace = namespace
		claim = must(tCtx, tCtx.Client().ResourceV1().ResourceClaims(namespace).Create, claim, metav1.CreateOptions{})
		claim.Status.Allocation = &resourceapi.AllocationResult{
			Devices: resourceapi.DeviceAllocationResult{
				Results: []resourceapi.DeviceRequestAllocationResult{{
					Request: claim.Spec.Devices.Requests[0].Name,
					Driver:  driverName,
					Pool:    poolName,
					Device:  "device" + suffix,
				}},
			},
		}
		claim.Status.ReservedFor = []resourceapi.ResourceClaimConsumerReference{{
			Resource: "pods",
			Name:     pod.Name,
			UID:      pod.UID,
		}}
		must(tCtx, tCtx.Client().ResourceV1().ResourceClaims(namespace).UpdateStatus, claim, metav1.UpdateOptions{})
	}

	// Create a new factory and sync it so that when the controller starts, it is up-to-date.
	// This works as long as this is the only test running it.
	informerFactory := informers.NewSharedInformerFactory(tCtx.Client(), 0)
	var ruleInformer resourcealphainformers.DeviceTaintRuleInformer
	if utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceTaintRules) {
		ruleInformer = informerFactory.Resource().V1alpha3().DeviceTaintRules()
	}
	controller := devicetainteviction.New(tCtx.Client(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Resource().V1().ResourceClaims(),
		informerFactory.Resource().V1().ResourceSlices(),
		ruleInformer,
		informerFactory.Resource().V1().DeviceClasses(),
		"device-taint-eviction",
	)

	var numExistingPods atomic.Int64
	podsDeleted := make(chan struct{})
	_, _ = informerFactory.Core().V1().Pods().Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc: func(obj any) {
			pod := obj.(*v1.Pod)
			if pod.Namespace == namespace {
				numExistingPods.Add(1)
			}
		},

		UpdateFunc: func(oldObj, newObj any) {
			oldPod, newPod := oldObj.(*v1.Pod), newObj.(*v1.Pod)
			if newPod.Namespace == namespace &&
				oldPod.DeletionTimestamp == nil &&
				newPod.DeletionTimestamp != nil &&
				numExistingPods.Add(-1) == 0 {
				// All pods marked for deletion.
				close(podsDeleted)
			}
		},
	})

	informerFactory.Start(tCtx.Done())
	for t, synced := range informerFactory.WaitForCacheSync(tCtx.Done()) {
		tCtx.Logf("informer %s synced: %v", t, synced)
		if !synced {
			tCtx.Errorf("informer for %s failed to sync", t)
		}
	}
	if tCtx.Failed() {
		tCtx.FailNow()
	}
	wg.Go(func() {
		if err := controller.Run(tCtx, 10 /* workers */); err != nil {
			tCtx.Errorf("Unexpected Run error: %v", err)
		}
	})

	ruleName := "rule-" + namespace
	rule := &resourcealpha.DeviceTaintRule{
		ObjectMeta: metav1.ObjectMeta{
			Name: ruleName,
		},
		Spec: resourcealpha.DeviceTaintRuleSpec{
			DeviceSelector: &resourcealpha.DeviceTaintSelector{
				Driver: &driverName,
			},
			Taint: resourcealpha.DeviceTaint{
				Key:    "testing",
				Effect: resourcealpha.DeviceTaintEffectNoExecute,
			},
		},
	}

	if useRule {
		// Evict through DeviceTaintRule.
		must(tCtx, tCtx.Client().ResourceV1alpha3().DeviceTaintRules().Create, rule, metav1.CreateOptions{})
		tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
			err := tCtx.Client().ResourceV1alpha3().DeviceTaintRules().Delete(tCtx, ruleName, metav1.DeleteOptions{})
			if apierrors.IsNotFound(err) {
				return
			}
			tCtx.ExpectNoError(err)
		})
	} else {
		// Evict by tainting each device.
		for i, slice := range slices {
			slice = slice.DeepCopy()
			slice.Spec.Pool.Generation++
			for i := range slice.Spec.Devices {
				slice.Spec.Devices[i].Taints = []resourceapi.DeviceTaint{{
					Key:    "testing",
					Effect: resourceapi.DeviceTaintEffectNoExecute,
				}}
			}
			slices[i] = must(tCtx, tCtx.Client().ResourceV1().ResourceSlices().Update, slice, metav1.UpdateOptions{})
		}
	}

	getRule := func(tCtx ktesting.TContext) *resourcealpha.DeviceTaintRule {
		rule = must(tCtx, tCtx.Client().ResourceV1alpha3().DeviceTaintRules().Get, ruleName, metav1.GetOptions{})
		return rule
	}

	// Evict and wait for pods to be gone.
	start := time.Now()
	select {
	case <-podsDeleted:
		// Okay.
	case <-tCtx.Done():
		tCtx.Fatalf("Waiting for pod deletion was canceled: %v", context.Cause(tCtx))
	case <-time.After(30 * time.Second):
		tCtx.Fatal("Timed out waiting for pod deletion")
	}
	duration := time.Since(start)
	tCtx.Logf("Evicted %d pods in %s.", numPods, duration)

	if useRule {
		// Check condition.
		ktesting.Eventually(tCtx, getRule).WithPolling(10 * time.Second).Should(gomega.HaveField("Status.Conditions", gomega.ConsistOf(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"Type":    gomega.Equal(resourcealpha.DeviceTaintConditionEvictionInProgress),
			"Status":  gomega.Equal(metav1.ConditionFalse),
			"Message": gomega.Equal(fmt.Sprintf("%d pods evicted since starting the controller.", numPods)),
		}))))
	}
}
