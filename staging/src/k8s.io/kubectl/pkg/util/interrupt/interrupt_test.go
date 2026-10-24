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

package interrupt

import (
	"os"
	"reflect"
	"syscall"
	"testing"
)

func TestSignal(t *testing.T) {
	notifies, finals := 0, 0
	h := New(func(os.Signal) { finals++ }, func() { notifies++ })

	h.Signal(syscall.SIGINT)
	h.Signal(syscall.SIGINT)

	if notifies != 1 {
		t.Errorf("notify was invoked %d times, want 1", notifies)
	}
	if finals != 1 {
		t.Errorf("final was invoked %d times, want 1", finals)
	}
}

func TestCloseThenSignal(t *testing.T) {
	notifies, finals := 0, 0
	h := New(func(os.Signal) { finals++ }, func() { notifies++ })

	h.Close()
	h.Signal(syscall.SIGTERM)

	if notifies != 1 {
		t.Errorf("notify was invoked %d times, want 1", notifies)
	}
	if finals != 1 {
		t.Errorf("final was invoked %d times, want 1", finals)
	}
}

func TestChainSignal(t *testing.T) {
	var invoked []string
	parent := New(
		func(os.Signal) { invoked = append(invoked, "parent final") },
		func() { invoked = append(invoked, "parent notify") },
	)
	child := Chain(parent, func() { invoked = append(invoked, "child notify") })

	child.Signal(syscall.SIGTERM)

	want := []string{"child notify", "parent notify", "parent final"}
	if !reflect.DeepEqual(invoked, want) {
		t.Errorf("got %v, want %v", invoked, want)
	}
}
