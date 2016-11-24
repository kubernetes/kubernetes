/*
Copyright 2015 The Kubernetes Authors.

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

package kubelet

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/record"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
)

func TestBasic(t *testing.T) {
	fakeRecorder := &record.FakeRecorder{}
	mockCadvisor := &cadvisortest.Fake{}
	node := &v1.ObjectReference{}
	oomWatcher := NewOOMWatcher(mockCadvisor, fakeRecorder)
	err := oomWatcher.Start(node)
	if err != nil {
		t.Errorf("Should not have failed: %v", err)
	}

	// TODO: Improve this test once cadvisor exports events.EventChannel as an interface
	// and thereby allow using a mock version of cadvisor.
}
