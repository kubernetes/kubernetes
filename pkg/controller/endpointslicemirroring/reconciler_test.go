/*
Copyright 2019 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/component-base/metrics/testutil"
	mirroringmetrics "k8s.io/kubernetes/pkg/controller/endpointslicemirroring/metrics"
	utilpointer "k8s.io/utils/pointer"
)

var defaultMaxEndpointsPerSlice = int32(100)

// TestReconcile ensures that Endpoints are reconciled into corresponding
// EndpointSlices with appropriate fields.
func TestReconcile(t *testing.T) {
	testCases := []struct {
		testName               string
		subsets                []corev1.EndpointSubset
		existingEndpointSlices []*discovery.EndpointSlice
		expectedNumSlices      int
		expectedClientActions  int
		expectedMetrics        expectedMetrics
	}{{
		testName:               "Endpoints with no subsets",
		subsets:                []corev1.EndpointSubset{},
		existingEndpointSlices: []*discovery.EndpointSlice{},
		expectedNumSlices:      0,
		expectedClientActions:  0,
		expectedMetrics:        expectedMetrics{},
	}, {
		testName: "Endpoints with no addresses",
		subsets: []corev1.EndpointSubset{{
			Ports: []corev1.EndpointPort{{
				Name:     "http",
				Port:     80,
				Protocol: corev1.ProtocolTCP,
			}},
		}},
		existingEndpointSlices: []*discovery.EndpointSlice{},
		expectedNumSlices:      0,
		expectedClientActions:  0,
		expectedMetrics:        expectedMetrics{},
	}, {
		testName: "Endpoints with 1 subset, port, and address",
		subsets: []corev1.EndpointSubset{{
			Ports: []corev1.EndpointPort{{
				Name:     "http",
				Port:     80,
				Protocol: corev1.ProtocolTCP,
			}},
			Addresses: []corev1.EndpointAddress{{
				IP:       "10.0.0.1",
				Hostname: "pod-1",
				NodeName: utilpointer.StringPtr("node-1"),
			}},
		}},
		existingEndpointSlices: []*discovery.EndpointSlice{},
		expectedNumSlices:      1,
		expectedClientActions:  1,
		expectedMetrics:        expectedMetrics{desiredSlices: 1, actualSlices: 1, desiredEndpoints: 1, addedPerSync: 1, numCreated: 1},
	}, {
		testName: "Endpoints with 1 subset, 2 ports, and 2 addresses",
		subsets: []corev1.EndpointSubset{{
			Ports: []corev1.EndpointPort{{
				Name:     "http",
				Port:     80,
				Protocol: corev1.ProtocolTCP,
			}, {
				Name:     "https",
				Port:     443,
				Protocol: corev1.ProtocolUDP,
			}},
			Addresses: []corev1.EndpointAddress{{
				IP:       "10.0.0.1",
				Hostname: "pod-1",
				NodeName: utilpointer.StringPtr("node-1"),
			}, {
				IP:       "10.0.0.2",
				Hostname: "pod-2",
				NodeName: utilpointer.StringPtr("node-2"),
			}},
		}},
		existingEndpointSlices: []*discovery.EndpointSlice{},
		expectedNumSlices:      1,
		expectedClientActions:  1,
		expectedMetrics:        expectedMetrics{desiredSlices: 1, actualSlices: 1, desiredEndpoints: 2, addedPerSync: 2, numCreated: 1},
	}, {
		testName: "Endpoints with 2 subsets, multiple ports and addresses",
		subsets: []corev1.EndpointSubset{{
			Ports: []corev1.EndpointPort{{
				Name:     "http",
				Port:     80,
				Protocol: corev1.ProtocolTCP,
			}, {
				Name:     "https",
				Port:     443,
				Protocol: corev1.ProtocolUDP,
			}},
			Addresses: []corev1.EndpointAddress{{
				IP:       "10.0.0.1",
				Hostname: "pod-1",
				NodeName: utilpointer.StringPtr("node-1"),
			}, {
				IP:       "10.0.0.2",
				Hostname: "pod-2",
				NodeName: utilpointer.StringPtr("node-2"),
			}},
		}, {
			Ports: []corev1.EndpointPort{{
				Name:     "http",
				Port:     3000,
				Protocol: corev1.ProtocolTCP,
			}, {
				Name:     "https",
				Port:     3001,
				Protocol: corev1.ProtocolUDP,
			}},
			Addresses: []corev1.EndpointAddress{{
				IP:       "10.0.1.1",
				Hostname: "pod-11",
				NodeName: utilpointer.StringPtr("node-1"),
			}, {
				IP:       "10.0.1.2",
				Hostname: "pod-12",
				NodeName: utilpointer.StringPtr("node-2"),
			}, {
				IP:       "10.0.1.3",
				Hostname: "pod-13",
				NodeName: utilpointer.StringPtr("node-3"),
			}},
		}},
		existingEndpointSlices: []*discovery.EndpointSlice{},
		expectedNumSlices:      2,
		expectedClientActions:  2,
		expectedMetrics:        expectedMetrics{desiredSlices: 2, actualSlices: 2, desiredEndpoints: 5, addedPerSync: 5, numCreated: 2},
	}}

	for _, tc := range testCases {
		t.Run(tc.testName, func(t *testing.T) {
			client := newClientset()
			setupMetrics()
			namespace := "test"
			endpoints := corev1.Endpoints{
				ObjectMeta: metav1.ObjectMeta{Name: "test-ep", Namespace: namespace},
				Subsets:    tc.subsets,
			}

			r := newReconciler(client, defaultMaxEndpointsPerSlice)
			reconcileHelper(t, r, &endpoints, []*discovery.EndpointSlice{})
			if len(client.Actions()) != tc.expectedClientActions {
				t.Fatalf("Expected %d additional client actions, got %d", tc.expectedClientActions, len(client.Actions()))
			}

			expectMetrics(t, tc.expectedMetrics)

			endpointSlices := fetchEndpointSlices(t, client, namespace)
			expectEndpointSlices(t, tc.expectedNumSlices, endpoints, endpointSlices)
		})
	}
}

// Test Helpers

func newReconciler(client *fake.Clientset, maxEndpointsPerSlice int32) *reconciler {
	return &reconciler{
		client:               client,
		maxEndpointsPerSlice: maxEndpointsPerSlice,
		endpointSliceTracker: newEndpointSliceTracker(),
		metricsCache:         mirroringmetrics.NewCache(maxEndpointsPerSlice),
	}
}

func expectEndpointSlices(t *testing.T, num int, endpoints corev1.Endpoints, endpointSlices []discovery.EndpointSlice) {
	if len(endpointSlices) != num {
		t.Fatalf("Expected %d EndpointSlices, got %d", num, len(endpointSlices))
	}

	for _, epSlice := range endpointSlices {
		if !strings.HasPrefix(epSlice.Name, endpoints.Name) {
			t.Errorf("Expected EndpointSlice name to start with %s, got %s", endpoints.Name, epSlice.Name)
		}

		serviceNameVal, ok := epSlice.Labels[discovery.LabelServiceName]
		if !ok {
			t.Errorf("Expected EndpointSlice to have %s label set", discovery.LabelServiceName)
		}
		if serviceNameVal != endpoints.Name {
			t.Errorf("Expected EndpointSlice to have %s label set to %s, got %s", discovery.LabelServiceName, endpoints.Name, serviceNameVal)
		}

		// If we want to test multiple EndpointSlices per port combination this
		// will need to be refactored.
		matchingPortsFound := false
		for _, epSubset := range endpoints.Subsets {
			if portsMatch(epSubset.Ports, epSlice.Ports) {
				matchingPortsFound = true
				expectMatchingAddresses(t, epSubset, epSlice.Endpoints)
				break
			}
		}

		if !matchingPortsFound {
			t.Fatalf("EndpointSlice ports don't match ports for any Endpoints subset: %#v", epSlice.Ports)
		}
	}
}

func portsMatch(epPorts []corev1.EndpointPort, epsPorts []discovery.EndpointPort) bool {
	if len(epPorts) != len(epsPorts) {
		return false
	}

	portsToBeMatched := map[int32]corev1.EndpointPort{}

	for _, epPort := range epPorts {
		portsToBeMatched[epPort.Port] = epPort
	}

	for _, epsPort := range epsPorts {
		epPort, ok := portsToBeMatched[*epsPort.Port]
		if !ok {
			return false
		}
		delete(portsToBeMatched, *epsPort.Port)

		if epPort.Name != *epsPort.Name {
			return false
		}
		if epPort.Port != *epsPort.Port {
			return false
		}
		if epPort.Protocol != *epsPort.Protocol {
			return false
		}
		if epPort.AppProtocol != epsPort.AppProtocol {
			return false
		}
	}

	return true
}

func expectMatchingAddresses(t *testing.T, epSubset corev1.EndpointSubset, esEndpoints []discovery.Endpoint) {
	type addressInfo struct {
		matched   bool
		ready     bool
		epAddress corev1.EndpointAddress
	}

	// This approach assumes that each IP is unique within an EndpointSubset.
	expectedEndpoints := map[string]addressInfo{}

	for _, address := range epSubset.Addresses {
		expectedEndpoints[address.IP] = addressInfo{
			ready:     true,
			epAddress: address,
		}
	}

	for _, address := range epSubset.NotReadyAddresses {
		expectedEndpoints[address.IP] = addressInfo{
			ready:     false,
			epAddress: address,
		}
	}

	if len(expectedEndpoints) != len(esEndpoints) {
		t.Errorf("Expected %d endpoints, got %d", len(expectedEndpoints), len(esEndpoints))
	}

	for _, endpoint := range esEndpoints {
		if len(endpoint.Addresses) != 1 {
			t.Fatalf("Expected endpoint to have 1 address, got %d", len(endpoint.Addresses))
		}
		address := endpoint.Addresses[0]
		expectedEndpoint, ok := expectedEndpoints[address]

		if !ok {
			t.Fatalf("EndpointSlice has endpoint with unexpected address: %s", address)
		}

		if expectedEndpoint.ready != *endpoint.Conditions.Ready {
			t.Errorf("Expected ready to be %t, got %t", expectedEndpoint.ready, *endpoint.Conditions.Ready)
		}

		if endpoint.Hostname == nil {
			if expectedEndpoint.epAddress.Hostname != "" {
				t.Errorf("Expected hostname to be %s, got nil", expectedEndpoint.epAddress.Hostname)
			}
		} else if expectedEndpoint.epAddress.Hostname != *endpoint.Hostname {
			t.Errorf("Expected hostname to be %s, got %s", expectedEndpoint.epAddress.Hostname, *endpoint.Hostname)
		}

		if expectedEndpoint.epAddress.NodeName != nil {
			topologyHostname, ok := endpoint.Topology["kubernetes.io/hostname"]
			if !ok {
				t.Errorf("Expected topology[kubernetes.io/hostname] to be set")
			} else if *expectedEndpoint.epAddress.NodeName != topologyHostname {
				t.Errorf("Expected topology[kubernetes.io/hostname] to be %s, got %s", *expectedEndpoint.epAddress.NodeName, topologyHostname)
			}
		}
	}
}

// ensures endpoint slices exist with the desired set of lengths
func expectUnorderedSlicesWithLengths(t *testing.T, endpointSlices []discovery.EndpointSlice, expectedLengths []int) {
	assert.Len(t, endpointSlices, len(expectedLengths), "Expected %d endpoint slices", len(expectedLengths))

	lengthsWithNoMatch := []int{}
	desiredLengths := expectedLengths
	actualLengths := []int{}
	for _, endpointSlice := range endpointSlices {
		actualLen := len(endpointSlice.Endpoints)
		actualLengths = append(actualLengths, actualLen)
		matchFound := false
		for i := 0; i < len(desiredLengths); i++ {
			if desiredLengths[i] == actualLen {
				matchFound = true
				desiredLengths = append(desiredLengths[:i], desiredLengths[i+1:]...)
				break
			}
		}

		if !matchFound {
			lengthsWithNoMatch = append(lengthsWithNoMatch, actualLen)
		}
	}

	if len(lengthsWithNoMatch) > 0 || len(desiredLengths) > 0 {
		t.Errorf("Actual slice lengths (%v) don't match expected (%v)", actualLengths, expectedLengths)
	}
}

// ensures endpoint slices exist with the desired set of ports and address types
func expectUnorderedSlicesWithTopLevelAttrs(t *testing.T, endpointSlices []discovery.EndpointSlice, expectedSlices []discovery.EndpointSlice) {
	t.Helper()
	assert.Len(t, endpointSlices, len(expectedSlices), "Expected %d endpoint slices", len(expectedSlices))

	slicesWithNoMatch := []discovery.EndpointSlice{}
	for _, endpointSlice := range endpointSlices {
		matchFound := false
		for i := 0; i < len(expectedSlices); i++ {
			if portsAndAddressTypeEqual(expectedSlices[i], endpointSlice) {
				matchFound = true
				expectedSlices = append(expectedSlices[:i], expectedSlices[i+1:]...)
				break
			}
		}

		if !matchFound {
			slicesWithNoMatch = append(slicesWithNoMatch, endpointSlice)
		}
	}

	assert.Len(t, slicesWithNoMatch, 0, "EndpointSlice(s) found without matching attributes")
	assert.Len(t, expectedSlices, 0, "Expected slices(s) not found in EndpointSlices")
}

func expectActions(t *testing.T, actions []k8stesting.Action, num int, verb, resource string) {
	t.Helper()
	if len(actions) < num {
		t.Fatalf("Expected at least %d actions, got %d", num, len(actions))
	}
	for i := 0; i < num; i++ {
		relativePos := len(actions) - i - 1
		assert.Equal(t, verb, actions[relativePos].GetVerb(), "Expected action -%d verb to be %s", i, verb)
		assert.Equal(t, resource, actions[relativePos].GetResource().Resource, "Expected action -%d resource to be %s", i, resource)
	}
}

func expectTrackedResourceVersion(t *testing.T, tracker *endpointSliceTracker, slice *discovery.EndpointSlice, expectedRV string) {
	rrv := tracker.relatedResourceVersions(slice)
	rv, tracked := rrv[slice.Name]
	if !tracked {
		t.Fatalf("Expected EndpointSlice %s to be tracked", slice.Name)
	}
	if rv != expectedRV {
		t.Errorf("Expected ResourceVersion of %s to be %s, got %s", slice.Name, expectedRV, rv)
	}
}

func portsAndAddressTypeEqual(slice1, slice2 discovery.EndpointSlice) bool {
	return apiequality.Semantic.DeepEqual(slice1.Ports, slice2.Ports) && apiequality.Semantic.DeepEqual(slice1.AddressType, slice2.AddressType)
}

func createEndpointSlices(t *testing.T, client *fake.Clientset, namespace string, endpointSlices []*discovery.EndpointSlice) {
	t.Helper()
	for _, endpointSlice := range endpointSlices {
		_, err := client.DiscoveryV1beta1().EndpointSlices(namespace).Create(context.TODO(), endpointSlice, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Expected no error creating Endpoint Slice, got: %v", err)
		}
	}
}

func fetchEndpointSlices(t *testing.T, client *fake.Clientset, namespace string) []discovery.EndpointSlice {
	t.Helper()
	fetchedSlices, err := client.DiscoveryV1beta1().EndpointSlices(namespace).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Expected no error fetching Endpoint Slices, got: %v", err)
		return []discovery.EndpointSlice{}
	}
	return fetchedSlices.Items
}

func reconcileHelper(t *testing.T, r *reconciler, endpoints *corev1.Endpoints, existingSlices []*discovery.EndpointSlice) {
	t.Helper()
	err := r.reconcile(endpoints, existingSlices)
	if err != nil {
		t.Fatalf("Expected no error reconciling Endpoint Slices, got: %v", err)
	}
}

// Metrics helpers

type expectedMetrics struct {
	desiredSlices    int
	actualSlices     int
	desiredEndpoints int
	addedPerSync     int
	removedPerSync   int
	numCreated       int
	numUpdated       int
	numDeleted       int
}

func expectMetrics(t *testing.T, em expectedMetrics) {
	t.Helper()

	actualDesiredSlices, err := testutil.GetGaugeMetricValue(mirroringmetrics.DesiredEndpointSlices.WithLabelValues())
	handleErr(t, err, "desiredEndpointSlices")
	if actualDesiredSlices != float64(em.desiredSlices) {
		t.Errorf("Expected desiredEndpointSlices to be %d, got %v", em.desiredSlices, actualDesiredSlices)
	}

	actualNumSlices, err := testutil.GetGaugeMetricValue(mirroringmetrics.NumEndpointSlices.WithLabelValues())
	handleErr(t, err, "numEndpointSlices")
	if actualNumSlices != float64(em.actualSlices) {
		t.Errorf("Expected numEndpointSlices to be %d, got %v", em.actualSlices, actualNumSlices)
	}

	actualEndpointsDesired, err := testutil.GetGaugeMetricValue(mirroringmetrics.EndpointsDesired.WithLabelValues())
	handleErr(t, err, "desiredEndpoints")
	if actualEndpointsDesired != float64(em.desiredEndpoints) {
		t.Errorf("Expected desiredEndpoints to be %d, got %v", em.desiredEndpoints, actualEndpointsDesired)
	}

	actualAddedPerSync, err := testutil.GetHistogramMetricValue(mirroringmetrics.EndpointsAddedPerSync.WithLabelValues())
	handleErr(t, err, "endpointsAddedPerSync")
	if actualAddedPerSync != float64(em.addedPerSync) {
		t.Errorf("Expected endpointsAddedPerSync to be %d, got %v", em.addedPerSync, actualAddedPerSync)
	}

	actualRemovedPerSync, err := testutil.GetHistogramMetricValue(mirroringmetrics.EndpointsRemovedPerSync.WithLabelValues())
	handleErr(t, err, "endpointsRemovedPerSync")
	if actualRemovedPerSync != float64(em.removedPerSync) {
		t.Errorf("Expected endpointsRemovedPerSync to be %d, got %v", em.removedPerSync, actualRemovedPerSync)
	}

	actualCreated, err := testutil.GetCounterMetricValue(mirroringmetrics.EndpointSliceChanges.WithLabelValues("create"))
	handleErr(t, err, "endpointSliceChangesCreated")
	if actualCreated != float64(em.numCreated) {
		t.Errorf("Expected endpointSliceChangesCreated to be %d, got %v", em.numCreated, actualCreated)
	}

	actualUpdated, err := testutil.GetCounterMetricValue(mirroringmetrics.EndpointSliceChanges.WithLabelValues("update"))
	handleErr(t, err, "endpointSliceChangesUpdated")
	if actualUpdated != float64(em.numUpdated) {
		t.Errorf("Expected endpointSliceChangesUpdated to be %d, got %v", em.numUpdated, actualUpdated)
	}

	actualDeleted, err := testutil.GetCounterMetricValue(mirroringmetrics.EndpointSliceChanges.WithLabelValues("delete"))
	handleErr(t, err, "desiredEndpointSlices")
	if actualDeleted != float64(em.numDeleted) {
		t.Errorf("Expected endpointSliceChangesDeleted to be %d, got %v", em.numDeleted, actualDeleted)
	}
}

func handleErr(t *testing.T, err error, metricName string) {
	if err != nil {
		t.Errorf("Failed to get %s value, err: %v", metricName, err)
	}
}

func setupMetrics() {
	mirroringmetrics.RegisterMetrics()
	mirroringmetrics.NumEndpointSlices.Delete(map[string]string{})
	mirroringmetrics.DesiredEndpointSlices.Delete(map[string]string{})
	mirroringmetrics.EndpointsDesired.Delete(map[string]string{})
	mirroringmetrics.EndpointsAddedPerSync.Delete(map[string]string{})
	mirroringmetrics.EndpointsRemovedPerSync.Delete(map[string]string{})
	mirroringmetrics.EndpointSliceChanges.Delete(map[string]string{"operation": "create"})
	mirroringmetrics.EndpointSliceChanges.Delete(map[string]string{"operation": "update"})
	mirroringmetrics.EndpointSliceChanges.Delete(map[string]string{"operation": "delete"})
}
