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
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"
)

func TestSetGetLocalAddresses(t *testing.T) {
	fake := NewFakeNetlinkHandle()
	fake.SetLocalAddresses("eth0", "1.2.3.4")
	expected := sets.NewString("1.2.3.4")
	addr, _ := fake.GetLocalAddresses("eth0", "")
	if !reflect.DeepEqual(expected, addr) {
		t.Errorf("Unexpected mismatch, expected: %v, got: %v", expected, addr)
	}
	list, _ := fake.GetLocalAddresses("", "")
	if !reflect.DeepEqual(expected, list) {
		t.Errorf("Unexpected mismatch, expected: %v, got: %v", expected, list)
	}
	fake.SetLocalAddresses("lo", "127.0.0.1")
	expected = sets.NewString("127.0.0.1")
	addr, _ = fake.GetLocalAddresses("lo", "")
	if !reflect.DeepEqual(expected, addr) {
		t.Errorf("Unexpected mismatch, expected: %v, got: %v", expected, addr)
	}
	list, _ = fake.GetLocalAddresses("", "")
	expected = sets.NewString("1.2.3.4", "127.0.0.1")
	if !reflect.DeepEqual(expected, list) {
		t.Errorf("Unexpected mismatch, expected: %v, got: %v", expected, list)
	}
}
