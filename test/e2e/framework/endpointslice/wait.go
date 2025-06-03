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

package endpointslice

import (
	"context"
	"fmt"
	"time"

	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// ControllerUpdateTimeout is the maximum expected wait before a change to a Pod
	// or Service is reflected in its EndpointSlices. (This is the same as
	// e2eservice.ServiceEndpointsTimeout, but duplicated here to avoid import loops.)
	ControllerUpdateTimeout = 2 * time.Minute
)

type EndpointSliceConditionFunc func(ctx context.Context, endpointSlices []discoveryv1.EndpointSlice) (done bool, err error)

// WaitForEndpointSlices is an EndpointSlice-specific wrapper around
// wait.PollUntilContextTimeout that polls conditionFunc with a list of serviceName's
// EndpointSlices.
func WaitForEndpointSlices(ctx context.Context, cs clientset.Interface, namespace, serviceName string, interval, timeout time.Duration, conditionFunc EndpointSliceConditionFunc) error {
	return wait.PollUntilContextTimeout(ctx, interval, timeout, true, func(ctx context.Context) (bool, error) {
		esList, err := cs.DiscoveryV1().EndpointSlices(namespace).List(ctx, metav1.ListOptions{LabelSelector: fmt.Sprintf("%s=%s", discoveryv1.LabelServiceName, serviceName)})
		if err != nil {
			framework.Logf("Unexpected error trying to get EndpointSlices for %s/%s: %v", namespace, serviceName, err)
			return false, nil
		}
		return conditionFunc(ctx, esList.Items)
	})
}

// WaitForEndpointCount waits (up to ControllerUpdateTimeout) for the named service to
// have at least one EndpointSlice, where the total length of the `Endpoints` arrays
// across all EndpointSlices is expectNum. (For simple services, this is equivalent to
// saying that it waits for there to be exactly expectNum pods backing the service, but
// for dual-stack services or some services with named ports, the mapping between
// `Endpoints` and pods may not be 1-to-1. Note also that waiting for 0 endpoints means
// "wait for the Service to exist but not have any endpoints", not "wait for the Service
// to be deleted".)
func WaitForEndpointCount(ctx context.Context, cs clientset.Interface, namespace, serviceName string, expectNum int) error {
	framework.Logf("Waiting for amount of service %s/%s endpoints to be %d", namespace, serviceName, expectNum)
	return WaitForEndpointSlices(ctx, cs, namespace, serviceName, time.Second, ControllerUpdateTimeout, func(ctx context.Context, endpointSlices []discoveryv1.EndpointSlice) (bool, error) {
		if len(endpointSlices) == 0 {
			framework.Logf("Waiting for at least 1 EndpointSlice to exist")
			return false, nil
		}

		// EndpointSlices can contain the same address on multiple Slices
		addresses := sets.Set[string]{}
		for _, epSlice := range endpointSlices {
			for _, ep := range epSlice.Endpoints {
				if len(ep.Addresses) > 0 {
					addresses.Insert(ep.Addresses[0])
				}
			}
		}

		if addresses.Len() != expectNum {
			framework.Logf("Unexpected number of Endpoints on Slices, got %d, expected %d", addresses.Len(), expectNum)
			return false, nil
		}
		return true, nil
	})
}
