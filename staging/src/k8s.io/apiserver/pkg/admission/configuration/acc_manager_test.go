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

package configuration

import (
	"fmt"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/admission/v1alpha1"
)

type mockGetter struct {
	invoked       uint
	successes     uint
	failures      uint
	stopCh        chan struct{}
	configuration v1alpha1.AdmissionControlConfiguration
	t             *testing.T
}

func newMockGetter(successes, failures uint, configuration v1alpha1.AdmissionControlConfiguration, t *testing.T) *mockGetter {
	return &mockGetter{
		failures:      failures,
		successes:     successes,
		configuration: configuration,
		stopCh:        make(chan struct{}),
		t:             t,
	}
}

// The first m.successes Get will be successful; the next m.failures Get will
// fail; the next m.successes Get will be successful; the stopCh is closed at
// the 1+m.failures+m.successes call.
func (m *mockGetter) Get(name string, options metav1.GetOptions) (*v1alpha1.AdmissionControlConfiguration, error) {
	m.invoked++
	if m.invoked == 1+m.successes+m.failures {
		close(m.stopCh)
	}
	if m.invoked == 1 {
		return &m.configuration, nil
	}
	if m.invoked <= 1+m.failures {
		return nil, fmt.Errorf("some error")
	}
	if m.invoked <= 1+m.failures+m.successes {
		return &m.configuration, nil
	}
	m.t.Fatalf("unexpected call to Get, stopCh has been closed at the %d time call", 1+m.successes+m.failures)
	return nil, nil
}

func TestConfiguration(t *testing.T) {
	cases := []struct {
		name     string
		failures uint
		// note that the first call to mockGetter is always a success.
		successes   uint
		expectReady bool
	}{
		{
			name:        "number of failures hasn't reached threshold",
			failures:    threshold - 1,
			expectReady: true,
		},
		{
			name:        "number of failures just reaches threshold",
			failures:    threshold,
			expectReady: false,
		},
		{
			name:        "number of failures exceeds threshold",
			failures:    threshold + 1,
			expectReady: false,
		},
		{
			name:        "number of failures exceeds threshold, but then get another success",
			failures:    threshold + 1,
			successes:   1,
			expectReady: true,
		},
	}
	for _, c := range cases {
		mock := newMockGetter(c.successes, c.failures, v1alpha1.AdmissionControlConfiguration{}, t)
		manager := NewAdmissionControlConfigurationManager(mock)
		manager.interval = 1 * time.Millisecond
		manager.Run(mock.stopCh)
		<-mock.stopCh
		_, err := manager.Configuration()
		if err != nil && c.expectReady {
			t.Errorf("case %s, expect ready, got: %v", c.name, err)
		}
		if err == nil && !c.expectReady {
			t.Errorf("case %s, expect not ready", c.name)
		}
	}
}
