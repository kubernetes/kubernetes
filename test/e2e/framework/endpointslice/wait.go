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
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
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
