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

package controller

import (
	"strings"
	"testing"
)

func TestPluginMgrInit(t *testing.T) {
	tests := map[string]struct {
		expectedFailure bool
		errorContains   string
		plugins         []Plugin
	}{
		"invalid-name": {
			expectedFailure: true,
			errorContains:   "invalid name",
			plugins: []Plugin{
				&FakeControllerPlugin{"invalid!@#!#$%-name", nil},
			},
		},
		"duplicate-plugins": {
			expectedFailure: true,
			errorContains:   "registered more than once",
			plugins: []Plugin{
				&FakeControllerPlugin{"valid-but-dupe-name", nil},
				&FakeControllerPlugin{"valid-but-dupe-name", nil},
			},
		},
	}

	for name, test := range tests {
		mgr := NewPluginMgr()
		err := mgr.InitPlugins(test.plugins, NewFakeHost(nil))

		switch {

		case err != nil && test.expectedFailure:
			if !strings.Contains(err.Error(), test.errorContains) {
				t.Errorf("Plugin test %s failed.  Expected '%s' to be part of '%s'", name, test.errorContains, err)
			}
		case err != nil && !test.expectedFailure:
			t.Errorf("Unexpected failure for plugin test '%s': %v", name, err)
		case err == nil && test.expectedFailure:
			t.Errorf("Expected failure for '%s' but did not get an error", name)
		}
	}
}

func TestPluginMgrStartStop(t *testing.T) {
	tests := map[string]struct {
		expectedRunning int
		plugins         []Plugin
	}{
		"both-running": {
			expectedRunning: 2,
			plugins: []Plugin{
				&FakeControllerPlugin{"foo", nil},
				&FakeControllerPlugin{"bar", nil},
			},
		},
		"one-running": {
			expectedRunning: 1,
			plugins: []Plugin{
				&FakeControllerPlugin{"valid-but-dupe-name", nil},
				&FakeControllerPlugin{"valid-but-dupe-name", nil},
			},
		},
		"none-running": {
			expectedRunning: 0,
			plugins: []Plugin{
				&FakeControllerPlugin{"!@#$-invalid-name", nil},
				&FakeControllerPlugin{"@#$%-invalid-name", nil},
			},
		},
	}

	for name, test := range tests {
		mgr := NewPluginMgr()
		_ = mgr.InitPlugins(test.plugins, NewFakeHost(nil))

		mgr.RunAll()
		running := 0
		for _, status := range mgr.Status() {
			if status.Running {
				running++
			}
		}
		if running != test.expectedRunning {
			t.Errorf("Unexpected %i running but found %i in test '%s'", test.expectedRunning, running, name)
		}

		mgr.StopAll()
		running = 0
		for _, status := range mgr.Status() {
			if status.Running {
				running++
			}
		}
		if running != 0 {
			t.Errorf("Unexpected 0 running but found %i in test '%s'", test.expectedRunning, running, name)
		}

	}

}
