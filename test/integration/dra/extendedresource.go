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
	"fmt"
	"time"

	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	gtypes "github.com/onsi/gomega/types"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/utils/client-go/ktesting"
)

func testExtendedResource(tCtx ktesting.TContext, enabled, explicit bool) {
	tCtx.Parallel()

	namespace := createTestNamespace(tCtx, nil)

	// Create a DeviceClass with or without ExtendedResourceName based on whether we're testing explicit or implicit extended resources.
	var resourceName string
	var spec *resourceapi.DeviceClassSpec
	if explicit {
		resourceName = "example.com/" + namespace
		// Set extended resource name in the DeviceClass spec only for explicit resources.
		// It's not required for implicit extended resources.
		spec = &resourceapi.DeviceClassSpec{
			ExtendedResourceName: &resourceName,
		}
	}
	class, driverName := createTestClassWithSpec(tCtx, namespace, spec)
	if explicit {
		if enabled {
			require.NotEmpty(tCtx, class.Spec.ExtendedResourceName, "should store ExtendedResourceName")
		} else {
			require.Empty(tCtx, class.Spec.ExtendedResourceName, "should strip ExtendedResourceName")
		}
	} else {
		// For implicit extended resources, derive the resource name from the class.
		resourceName = resourceapi.ResourceDeviceClassPrefix + class.Name
	}

	slice := st.MakeResourceSlice("worker-0", driverName).Devices(device1)
	createSlice(tCtx, slice.Obj())

	startScheduler(tCtx)

	podWithOneContainer := st.MakePod().Name(podName).Namespace(namespace).Container("test-container").Obj()
	pod := createPodWithExtendedResource(tCtx, namespace, resourceName, "1", podWithOneContainer)

	var schedulingAttempted gtypes.GomegaMatcher
	if enabled {
		// Scheduled using device1 in the slice above.
		schedulingAttempted = gomega.HaveField("Status.Conditions", gomega.ContainElement(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Type":   gomega.Equal(v1.PodScheduled),
				"Status": gomega.Equal(v1.ConditionTrue),
			}),
		))
	} else {
		schedulingAttempted = gomega.HaveField("Status.Conditions", gomega.ContainElement(
			gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
				"Type":    gomega.Equal(v1.PodScheduled),
				"Status":  gomega.Equal(v1.ConditionFalse),
				"Reason":  gomega.Equal("Unschedulable"),
				"Message": gomega.Equal(fmt.Sprintf("0/8 nodes are available: 8 Insufficient %s. no new claims to deallocate, preemption: 0/8 nodes are available: 8 Preemption is not helpful for scheduling.", resourceName)),
			}),
		))
	}
	tCtx.Eventually(func(tCtx ktesting.TContext) (*v1.Pod, error) {
		return tCtx.Client().CoreV1().Pods(namespace).Get(tCtx, pod.Name, metav1.GetOptions{})
	}).WithTimeout(time.Minute).WithPolling(time.Second).Should(schedulingAttempted)
}
