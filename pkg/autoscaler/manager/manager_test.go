/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package manager

import (
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler"
	testadvisor "github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler/advisors/test"
	testplugin "github.com/GoogleCloudPlatform/kubernetes/pkg/autoscaler/plugins/test"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/testclient"
)

func createNewAutoScaleManager() *AutoScaleManager {
	kubeClient := testclient.NewSimpleFake(&api.AutoScalerList{})
	assessEvery := DefaultAssessmentInterval
	scaleEvery := DefaultAutoScalingInterval

	return NewAutoScaleManager(kubeClient, assessEvery, scaleEvery)
}

func TestAutoScaleManager(t *testing.T) {
	testCases := []struct {
		Name           string
		AssessInterval time.Duration
		ScaleInterval  time.Duration
	}{
		{"zeroes", 0 * time.Second, 0 * time.Second},
		{"ones", 1 * time.Second, 1 * time.Second},
		{"negative", -1 * time.Second, -1 * time.Second},
		{"more-negative", -999 * time.Second, -888 * time.Second},
		{"positive", 21 * time.Second, 42 * time.Second},
		{"vibrations", 9999 * time.Second, 8888 * time.Second},
	}

	oneSecond := 1 * time.Second

	for _, tc := range testCases {
		manager := createNewAutoScaleManager()
		if nil == manager {
			t.Errorf("test %q failed to create new instance", tc.Name)
			continue
		}

		if manager.AssessmentInterval < oneSecond {
			t.Errorf("test %q assessment interval got %v, expected it to be >= 1 second",
				tc.Name, manager.AssessmentInterval)
		}

		if manager.AutoScalingInterval < oneSecond {
			t.Errorf("test %q auto scaling interval got %v, expected it to be >= 1 second",
				tc.Name, manager.AutoScalingInterval)
		}
	}
}

func registerTestPlugin(t *testing.T, m *AutoScaleManager, name string, errorExpected bool) {
	plugin := &testplugin.TestAutoScalerPlugin{
		Tag:     name,
		Actions: []autoscaler.ScalingAction{},
		Error:   "",
	}

	err := m.Register(plugin)
	if errorExpected {
		if nil == err {
			t.Errorf("test %q got no error, expected an error", name)
		}

		if nil == plugin {
			t.Errorf("test %q - plugin was nil", name)
		}

		return
	}

	if err != nil {
		t.Errorf("test %q got an error %v, expected none", name, err)
	}
}

func TestAutoScaleManagerRegister(t *testing.T) {
	testCases := []struct {
		Name             string
		ErrorExpectation bool
	}{
		{
			Name:             "registration-test",
			ErrorExpectation: false,
		},
		{
			Name:             "another-registration-test",
			ErrorExpectation: false,
		},
		{
			Name:             "registration-test",
			ErrorExpectation: true,
		},
		{
			Name:             "another-registration-test",
			ErrorExpectation: true,
		},
		{
			Name:             "register-dot-org",
			ErrorExpectation: false,
		},
	}

	manager := createNewAutoScaleManager()
	if nil == manager {
		t.Errorf("Register test failed to create new instance")
		return
	}

	for _, tc := range testCases {
		registerTestPlugin(t, manager, tc.Name, tc.ErrorExpectation)
	}
}

func TestAutoScaleManagerDeregister(t *testing.T) {
	registerNames := []string{"dereg-plugin-1", "dereg-plugin-2"}
	testCases := []struct {
		Name             string
		ErrorExpectation bool
	}{
		{
			Name:             "dereg-plugin-1",
			ErrorExpectation: false,
		},
		{
			Name:             "dereg-no-such-plugin-404",
			ErrorExpectation: true,
		},
		{
			Name:             "dereg-plugin-1",
			ErrorExpectation: true,
		},
		{
			Name:             "dereg-plugin-2",
			ErrorExpectation: false,
		},
		{
			Name:             "dereg-plugin-1",
			ErrorExpectation: true,
		},
	}

	manager := createNewAutoScaleManager()
	if nil == manager {
		t.Errorf("Deregister test failed to create new instance")
		return
	}

	err := manager.Deregister("dereg-uplug-me")
	if nil == err {
		t.Errorf("deregister test got no error, expected an error")
	}

	for _, tag := range registerNames {
		registerTestPlugin(t, manager, tag, false)
	}

	for _, tc := range testCases {
		err := manager.Deregister(tc.Name)
		if tc.ErrorExpectation && nil == err {
			t.Errorf("test %q got no error, expected an error",
				tc.Name)
		}

		if !tc.ErrorExpectation && err != nil {
			t.Errorf("test %q got an error %v, expected none",
				tc.Name, err)
		}
	}
}

func TestAutoScaleManagerFindPluginByName(t *testing.T) {
	registerNames := []string{"sparkplug", "earplug", "unplugged"}
	testCases := []struct {
		Name             string
		ErrorExpectation bool
	}{
		{
			Name:             "sparkplug",
			ErrorExpectation: false,
		},
		{
			Name:             "earplug",
			ErrorExpectation: false,
		},
		{
			Name:             "unplug", // minus "ged".
			ErrorExpectation: true,
		},
		{
			Name:             "earplug",
			ErrorExpectation: false,
		},
		{
			Name:             "",
			ErrorExpectation: true,
		},
		{
			Name:             "unplugged",
			ErrorExpectation: false,
		},
	}

	manager := createNewAutoScaleManager()
	if nil == manager {
		t.Errorf("FindPluginByName test failed to create new instance")
		return
	}

	_, err := manager.FindPluginByName("unplug-me")
	if nil == err {
		t.Errorf("find plugin test got no error, expected an error")
	}

	for _, tag := range registerNames {
		registerTestPlugin(t, manager, tag, false)
	}

	for _, tc := range testCases {
		plugin, err := manager.FindPluginByName(tc.Name)
		if tc.ErrorExpectation {
			if nil == err {
				t.Errorf("test %q got no error, expected an error",
					tc.Name)
			}

			return
		}

		if err != nil {
			t.Errorf("test %q got an error %v, expected none",
				tc.Name, err)
		}

		if nil == plugin {
			t.Errorf("test %q found no plugin, expected one",
				tc.Name)
		}
	}
}

func addTestAdvisor(t *testing.T, m *AutoScaleManager, name string, errorExpected bool) {
	advisor := &testadvisor.TestAdvisor{Tag: name, Status: false, Error: ""}
	err := m.AddAdvisor(advisor)
	if errorExpected {
		if nil == err {
			t.Errorf("test %q got no error, expected an error", name)
		}
		return
	}

	if err != nil {
		t.Errorf("test %q got an error %v, expected none", name, err)
	}
}

func TestAutoScaleManagerAddAdvisor(t *testing.T) {
	testCases := []struct {
		Name             string
		ErrorExpectation bool
	}{
		{
			Name:             "add-statsd-ms",
			ErrorExpectation: false,
		},
		{
			Name:             "add-influxdb-ms",
			ErrorExpectation: false,
		},
		{
			Name:             "add-statsd-ms",
			ErrorExpectation: true,
		},
		{
			Name:             "add-influxdb-ms",
			ErrorExpectation: true,
		},
		{
			Name:             "add-cadvisor-ms",
			ErrorExpectation: false,
		},
	}

	manager := createNewAutoScaleManager()
	if nil == manager {
		t.Errorf("AddAdvisor test failed to create new instance")
		return
	}

	for _, tc := range testCases {
		addTestAdvisor(t, manager, tc.Name, tc.ErrorExpectation)
	}
}

func TestAutoScaleManagerRemoveAdvisor(t *testing.T) {
	registerSources := []string{"remove-statsd-ms", "remove-cadvisor-ms"}
	testCases := []struct {
		Name             string
		ErrorExpectation bool
	}{
		{
			Name:             "remove-statsd-ms",
			ErrorExpectation: false,
		},
		{
			Name:             "remove-no-such-ms-404",
			ErrorExpectation: true,
		},
		{
			Name:             "remove-statsd-ms",
			ErrorExpectation: true,
		},
		{
			Name:             "remove-cadvisor-ms",
			ErrorExpectation: false,
		},
		{
			Name:             "remove-statsd-ms",
			ErrorExpectation: true,
		},
	}

	manager := createNewAutoScaleManager()
	if nil == manager {
		t.Errorf("RemoveAdvisor test failed to create new instance")
		return
	}

	err := manager.RemoveAdvisor("remove-cadvisor-ms")
	if nil == err {
		t.Errorf("remove advisor test got no error, expected an error")
	}

	for _, tag := range registerSources {
		addTestAdvisor(t, manager, tag, false)
	}

	for _, tc := range testCases {
		err := manager.RemoveAdvisor(tc.Name)
		if tc.ErrorExpectation && nil == err {
			t.Errorf("test %q got no error, expected an error",
				tc.Name)
		}

		if !tc.ErrorExpectation && err != nil {
			t.Errorf("test %q got an error %v, expected none",
				tc.Name, err)
		}
	}
}

func TestAutoScaleManagerGetAdvisors(t *testing.T) {
	registerSources := []string{"statsd-ms", "cadvisor-ms", "my-ms"}
	manager := createNewAutoScaleManager()
	if nil == manager {
		t.Errorf("GetAdvisors test failed to create new instance")
		return
	}

	initialSources := manager.GetAdvisors()
	if len(initialSources) > 0 {
		t.Errorf("got %v advisors, expected none", len(initialSources))
	}

	for _, tag := range registerSources {
		addTestAdvisor(t, manager, tag, false)
	}

	advisors := manager.GetAdvisors()
	if len(advisors) != len(registerSources) {
		t.Errorf("got %v advisors, expected %v", len(advisors),
			len(registerSources))
	}
}

func TestAutoScaleManagerAddDefaultAdvisors(t *testing.T) {
	manager := createNewAutoScaleManager()
	if nil == manager {
		t.Errorf("AddDefaultAdvisors test failed to create new instance")
		return
	}

	initialSources := manager.GetAdvisors()
	if len(initialSources) > 0 {
		t.Errorf("got %v advisors, expected none", len(initialSources))
	}

	err := manager.AddDefaultAdvisors()
	if err != nil {
		t.Errorf("got an error %v, expected no error", err)
		return
	}

	advisors := manager.GetAdvisors()
	if len(advisors) < 1 {
		t.Errorf("got %v advisors, expected at least 1", len(advisors))
	}
}

// TODO(ramr): add test for: Run(workers int, stopCh <-chan struct{})
