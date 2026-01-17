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
	"fmt"
	"strings"
	"time"

	"github.com/onsi/gomega"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/dynamic-resource-allocation/resourceslice"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestCreateResourceSlices uses the ResourceSlice controller to create slices.
// It runs both as integration and as E2E test, with different number of slices.
func TestCreateResourceSlices(tCtx ktesting.TContext, numSlices int) {
	// Define desired resource slices.
	namespace := tCtx.Namespace()
	driverName := namespace
	devicePrefix := "dev-"
	domainSuffix := ".example.com"
	poolName := "network-attached"
	domain := strings.Repeat("x", resourceapi.DeviceMaxDomainLength-len(domainSuffix)) + domainSuffix
	stringValue := strings.Repeat("v", resourceapi.DeviceAttributeMaxValueLength)
	pool := resourceslice.Pool{
		Slices: make([]resourceslice.Slice, numSlices),
	}
	numDevices := 0
	for i := 0; i < numSlices; i++ {
		devices := make([]resourceapi.Device, resourceapi.ResourceSliceMaxDevices)
		for e := 0; e < resourceapi.ResourceSliceMaxDevices; e++ {
			device := resourceapi.Device{
				Name:       devicePrefix + strings.Repeat("x", validation.DNS1035LabelMaxLength-len(devicePrefix)-6) + fmt.Sprintf("%06d", numDevices),
				Attributes: make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice),
			}
			numDevices++
			for j := 0; j < resourceapi.ResourceSliceMaxAttributesAndCapacitiesPerDevice; j++ {
				name := resourceapi.QualifiedName(domain + "/" + strings.Repeat("x", resourceapi.DeviceMaxIDLength-4) + fmt.Sprintf("%04d", j))
				device.Attributes[name] = resourceapi.DeviceAttribute{
					StringValue: &stringValue,
				}
			}
			devices[e] = device
		}
		pool.Slices[i].Devices = devices
	}
	resources := &resourceslice.DriverResources{
		Pools: map[string]resourceslice.Pool{poolName: pool},
	}
	listSlices := func(tCtx ktesting.TContext) *resourceapi.ResourceSliceList {
		tCtx.Helper()
		// TODO: replicate framework.ListObjects/Get/etc. with ktesting
		slices, err := tCtx.Client().ResourceV1().ResourceSlices().List(tCtx, metav1.ListOptions{
			FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName,
		})
		tCtx.ExpectNoError(err, "list slices")
		return slices
	}

	tCtx.Log("Creating slices")
	mutationCacheTTL := 10 * time.Second
	controller, err := resourceslice.StartController(tCtx, resourceslice.Options{
		DriverName:       driverName,
		KubeClient:       tCtx.Client(),
		Resources:        resources,
		MutationCacheTTL: &mutationCacheTTL,
	})
	tCtx.ExpectNoError(err, "start controller")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		controller.Stop()
		ktesting.Eventually(tCtx, func(tCtx ktesting.TContext) *resourceapi.ResourceSliceList {
			err := tCtx.Client().ResourceV1().ResourceSlices().DeleteCollection(tCtx, metav1.DeleteOptions{}, metav1.ListOptions{
				FieldSelector: resourceapi.ResourceSliceSelectorDriver + "=" + driverName,
			})
			tCtx.ExpectNoError(err, "delete slices")
			return listSlices(tCtx)
		}).Should(gomega.HaveField("Items", gomega.BeEmpty()))
	})

	// Eventually we should have all desired slices.
	ktesting.Eventually(tCtx, listSlices).WithTimeout(3 * time.Minute).Should(gomega.HaveField("Items", gomega.HaveLen(numSlices)))

	// Verify state.
	expectSlices := listSlices(tCtx)
	tCtx.Assert(expectSlices.Items).ShouldNot(gomega.BeEmpty())
	tCtx.Logf("Protobuf size of one slice is %d bytes = %d KB.", expectSlices.Items[0].Size(), expectSlices.Items[0].Size()/1024)
	tCtx.Assert(expectSlices.Items[0].Size()).Should(gomega.BeNumerically(">=", 600*1024), "ResourceSlice size")
	tCtx.Assert(expectSlices.Items[0].Size()).Should(gomega.BeNumerically("<", 1024*1024), "ResourceSlice size")
	expectStats := resourceslice.Stats{NumCreates: int64(numSlices)}
	tCtx.Assert(controller.GetStats()).Should(gomega.Equal(expectStats))

	// No further changes expected now, after checking again.
	getStats := func(tCtx ktesting.TContext) resourceslice.Stats { return controller.GetStats() }
	ktesting.Consistently(tCtx, getStats).WithTimeout(2 * mutationCacheTTL).Should(gomega.Equal(expectStats))

	// Ask the controller to delete all slices except for one empty slice.
	tCtx.Log("Deleting slices")
	resources = resources.DeepCopy()
	resources.Pools[poolName] = resourceslice.Pool{Slices: []resourceslice.Slice{{}}}
	controller.Update(resources)

	// One empty slice should remain, after removing the full ones and adding the empty one.
	emptySlice := gomega.HaveField("Spec.Devices", gomega.BeEmpty())
	ktesting.Eventually(tCtx, listSlices).WithTimeout(2 * time.Minute).Should(gomega.HaveField("Items", gomega.HaveExactElements(emptySlice)))
	expectStats = resourceslice.Stats{NumCreates: int64(numSlices) + 1, NumDeletes: int64(numSlices)}

	// There is a window of time where the ResourceSlice exists and is
	// returned in a list but before that ResourceSlice is accounted for
	// in the controller's stats, consisting mostly of network latency
	// between this test process and the API server. Wait for the stats
	// to converge before asserting there are no further changes.
	ktesting.Eventually(tCtx, getStats).WithTimeout(30 * time.Second).Should(gomega.Equal(expectStats))

	ktesting.Consistently(tCtx, getStats).WithTimeout(2 * mutationCacheTTL).Should(gomega.Equal(expectStats))
}
