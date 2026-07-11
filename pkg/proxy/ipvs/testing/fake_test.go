//go:build linux

/*
Copyright 2017 The Kubernetes Authors.

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

package testing

import (
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/proxy/ipvs"
)

// (I am unsure if this test has any value since it only tests the fake implementation)
func TestSetGetLocalAddresses(t *testing.T) {
	fake := NewFakeNetlinkHandle(false)
	_ = ipvs.NetLinkHandle(fake) // Ensure that the interface is honored
	fake.SetLocalAddresses("eth0", "1.2.3.4")
	var expected, addr sets.Set[string]
	expected = sets.New("1.2.3.4")
	addr, _ = fake.GetLocalAddresses("eth0")
	if !addr.Equal(expected) {
		t.Errorf("Unexpected mismatch, expected: %v, got: %v", expected, addr)
	}
	addr, _ = fake.GetAllLocalAddresses()
	if !addr.Equal(expected) {
		t.Errorf("Unexpected mismatch, expected: %v, got: %v", expected, addr)
	}
	fake.SetLocalAddresses("lo", "127.0.0.1")
	expected = nil
	addr, _ = fake.GetLocalAddresses("lo")
	if !addr.Equal(expected) {
		t.Errorf("Unexpected mismatch, expected: %v, got: %v", expected, addr)
	}
	fake.SetLocalAddresses("kube-ipvs0", "1.2.3.4", "4.3.2.1")
	addr, _ = fake.GetAllLocalAddresses()
	expected = sets.New("1.2.3.4", "4.3.2.1")
	if !addr.Equal(expected) {
		t.Errorf("Unexpected mismatch, expected: %v, got: %v", expected, addr)
	}
	addr, _ = fake.GetAllLocalAddressesExcept("kube-ipvs0")
	expected = sets.New("1.2.3.4")
	if !addr.Equal(expected) {
		t.Errorf("Unexpected mismatch, expected: %v, got: %v", expected, addr)
	}
}
