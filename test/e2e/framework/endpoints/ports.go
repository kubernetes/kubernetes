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

/*
This soak tests places a specified number of pods on each node and then
repeatedly sends queries to a service running on these pods via
a serivce
*/

package endpoints

import (
	"fmt"
	"sort"
	"time"

	"github.com/onsi/ginkgo"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
)

// ServiceStartTimeout is how long to wait for a service endpoint to be resolvable.
const ServiceStartTimeout = 3 * time.Minute

// PortsByPodName is a map that maps pod name to container ports.
type PortsByPodName map[string][]int

// PortsByPodUID is a map that maps pod UID to container ports.
type PortsByPodUID map[types.UID][]int

// GetContainerPortsByPodUID returns a PortsByPodUID map on the given endpoints.
func GetContainerPortsByPodUID(ep *v1.Endpoints) PortsByPodUID {
	m := PortsByPodUID{}
	for _, ss := range ep.Subsets {
		for _, port := range ss.Ports {
			for _, addr := range ss.Addresses {
				containerPort := port.Port
				if _, ok := m[addr.TargetRef.UID]; !ok {
					m[addr.TargetRef.UID] = make([]int, 0)
				}
				m[addr.TargetRef.UID] = append(m[addr.TargetRef.UID], int(containerPort))
			}
		}
	}
	return m
}

func translatePodNameToUID(c clientset.Interface, ns string, expectedEndpoints PortsByPodName) (PortsByPodUID, error) {
	portsByUID := make(PortsByPodUID)
	for name, portList := range expectedEndpoints {
		pod, err := c.CoreV1().Pods(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("failed to get pod %s, that's pretty weird. validation failed: %s", name, err)
		}
		portsByUID[pod.ObjectMeta.UID] = portList
	}
	return portsByUID, nil
}

func validatePorts(ep PortsByPodUID, expectedEndpoints PortsByPodUID) error {
	if len(ep) != len(expectedEndpoints) {
		// should not happen because we check this condition before
		return fmt.Errorf("invalid number of endpoints got %v, expected %v", ep, expectedEndpoints)
	}
	for podUID := range expectedEndpoints {
		if _, ok := ep[podUID]; !ok {
			return fmt.Errorf("endpoint %v not found", podUID)
		}
		if len(ep[podUID]) != len(expectedEndpoints[podUID]) {
			return fmt.Errorf("invalid list of ports for uid %v. Got %v, expected %v", podUID, ep[podUID], expectedEndpoints[podUID])
		}
		sort.Ints(ep[podUID])
		sort.Ints(expectedEndpoints[podUID])
		for index := range ep[podUID] {
			if ep[podUID][index] != expectedEndpoints[podUID][index] {
				return fmt.Errorf("invalid list of ports for uid %v. Got %v, expected %v", podUID, ep[podUID], expectedEndpoints[podUID])
			}
		}
	}
	return nil
}

// ValidateEndpointsPorts validates that the given service exists and is served by the given expectedEndpoints.
func ValidateEndpointsPorts(c clientset.Interface, namespace, serviceName string, expectedEndpoints PortsByPodName) error {
	ginkgo.By(fmt.Sprintf("waiting up to %v for service %s in namespace %s to expose endpoints %v", ServiceStartTimeout, serviceName, namespace, expectedEndpoints))
	i := 1
	for start := time.Now(); time.Since(start) < ServiceStartTimeout; time.Sleep(1 * time.Second) {
		ep, err := c.CoreV1().Endpoints(namespace).Get(serviceName, metav1.GetOptions{})
		if err != nil {
			e2elog.Logf("Get endpoints failed (%v elapsed, ignoring for 5s): %v", time.Since(start), err)
			continue
		}
		portsByPodUID := GetContainerPortsByPodUID(ep)
		expectedPortsByPodUID, err := translatePodNameToUID(c, namespace, expectedEndpoints)
		if err != nil {
			return err
		}
		if len(portsByPodUID) == len(expectedEndpoints) {
			err := validatePorts(portsByPodUID, expectedPortsByPodUID)
			if err != nil {
				return err
			}
			e2elog.Logf("successfully validated that service %s in namespace %s exposes endpoints %v (%v elapsed)",
				serviceName, namespace, expectedEndpoints, time.Since(start))
			return nil
		}
		if i%5 == 0 {
			e2elog.Logf("Unexpected endpoints: found %v, expected %v (%v elapsed, will retry)", portsByPodUID, expectedEndpoints, time.Since(start))
		}
		i++
	}
	if pods, err := c.CoreV1().Pods(metav1.NamespaceAll).List(metav1.ListOptions{}); err == nil {
		for _, pod := range pods.Items {
			e2elog.Logf("Pod %s\t%s\t%s\t%s", pod.Namespace, pod.Name, pod.Spec.NodeName, pod.DeletionTimestamp)
		}
	} else {
		e2elog.Logf("Can't list pod debug info: %v", err)
	}
	return fmt.Errorf("Timed out waiting for service %s in namespace %s to expose endpoints %v (%v elapsed)", serviceName, namespace, expectedEndpoints, ServiceStartTimeout)
}
