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

package network

import (
	"fmt"
	"math/rand"
	"sync"

	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
)

// NodePortRange should match whatever the default/configured range is
var NodePortRange = utilnet.PortRange{Base: 30000, Size: 2768}

// staticPortRange implements port allocation model described here
// https://github.com/kubernetes/enhancements/tree/master/keps/sig-network/3668-reserved-service-nodeport-range
type staticPortRange struct {
	sync.Mutex
	baseport      int32
	length        int32
	reservedPorts sets.Set[int32]
}

func calculateRange(size int32) int32 {
	var minPort int32 = 16
	var step int32 = 32
	var maxPort int32 = 128
	return min(max(minPort, size/step), maxPort)
}

var staticPortAllocator *staticPortRange

// Initialize only once per test
func init() {
	staticPortAllocator = &staticPortRange{
		baseport:      int32(NodePortRange.Base),
		length:        calculateRange(int32(NodePortRange.Size)),
		reservedPorts: sets.New[int32](),
	}
}

// reservePort reserves the port provided as input.
// If an invalid port was provided or if the port is already reserved, it returns false
func (s *staticPortRange) reservePort(port int32) bool {
	s.Lock()
	defer s.Unlock()
	if port < s.baseport || port > s.baseport+s.length || s.reservedPorts.Has(port) {
		return false
	}
	s.reservedPorts.Insert(port)
	return true
}

// getUnusedPort returns a free port from the range and returns its number and nil value
// the port is not allocated so the consumer should allocate it explicitly calling allocatePort()
// if none is available then it returns -1 and error
func (s *staticPortRange) getUnusedPort() (int32, error) {
	s.Lock()
	defer s.Unlock()
	// start in a random offset
	start := rand.Int31n(s.length)
	for i := int32(0); i < s.length; i++ {
		port := s.baseport + (start+i)%(s.length)
		if !s.reservedPorts.Has(port) {
			return port, nil
		}
	}
	return -1, fmt.Errorf("no free ports were found")
}

// releasePort releases the port passed as an argument
func (s *staticPortRange) releasePort(port int32) {
	s.Lock()
	defer s.Unlock()
	s.reservedPorts.Delete(port)
}

// getUnusedPorts returns count distinct free ports from the range.
// Like getUnusedPort the ports are not reserved, so the consumer should create the
// service first and then reserve them explicitly calling reservePorts().
// if there are not enough free ports then it returns nil and error
func (s *staticPortRange) getUnusedPorts(count int) ([]int32, error) {
	s.Lock()
	defer s.Unlock()
	ports := make([]int32, 0, count)
	// start in a random offset
	start := rand.Int31n(s.length)
	for i := int32(0); i < s.length && len(ports) < count; i++ {
		port := s.baseport + (start+i)%(s.length)
		if !s.reservedPorts.Has(port) {
			ports = append(ports, port)
		}
	}
	if len(ports) < count {
		return nil, fmt.Errorf("only %d free ports were found, %d were requested", len(ports), count)
	}
	return ports, nil
}

// reservePorts reserves every port in ports. If any of them can not be reserved none of
// them stays reserved and it returns false
func (s *staticPortRange) reservePorts(ports []int32) bool {
	reserved := make([]int32, 0, len(ports))
	for _, port := range ports {
		if !s.reservePort(port) {
			s.releasePorts(reserved)
			return false
		}
		reserved = append(reserved, port)
	}
	return true
}

// releasePorts releases every port in ports
func (s *staticPortRange) releasePorts(ports []int32) {
	for _, port := range ports {
		s.releasePort(port)
	}
}

// GetUnusedStaticNodePort returns a free port in static range and a nil value
// If no port in static range is available it returns -1 and an error value
// Note that it is not guaranteed that the returned port is actually available on the apiserver;
// You must allocate a port, then attempt to create the service, then call
// ReserveStaticNodePort.
func GetUnusedStaticNodePort() (int32, error) {
	return staticPortAllocator.getUnusedPort()
}

// ReserveStaticNodePort reserves the port provided as input. It is guaranteed
// that no other test will receive this port from GetUnusedStaticNodePort until
// after you call ReleaseStaticNodePort.
//
// port must have been previously allocated by GetUnusedStaticNodePort, and
// then successfully used as a NodePort or HealthCheckNodePort when creating
// a service. Trying to reserve a port that was not allocated by
// GetUnusedStaticNodePort, or reserving it before creating the associated service
// may cause other e2e tests to fail.
//
// If an invalid port was provided or if the port is already reserved, it returns false
func ReserveStaticNodePort(port int32) bool {
	return staticPortAllocator.reservePort(port)
}

// ReleaseStaticNodePort releases the specified port.
// The corresponding service should have already been deleted, to ensure that the
// port allocator doesn't try to reuse it before the apiserver considers it available.
// The caller should do it like below:
//
//	ginkgo.DeferCleanup(func(ctx context.Context) {
//		err := cs.CoreV1().Services(ns).Delete(ctx, serviceName, metav1.DeleteOptions{})
//		if err != nil && !apierrors.IsNotFound(err) {
//			framework.ExpectNoError(err, "failed to delete service %s in namespace %s", serviceName, ns)
//		}
//		e2enetwork.ReleaseStaticNodePort(nodePort)
//	})
func ReleaseStaticNodePort(port int32) {
	staticPortAllocator.releasePort(port)
}
