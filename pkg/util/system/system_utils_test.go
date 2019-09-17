/*
Copyright 2016 The Kubernetes Authors.

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

package system

import (
	"os"
	"reflect"
	"testing"
)

func TestDetectSystemd(t *testing.T) {
	if info, err := os.Stat("/usr/bin/systemd-run"); os.IsNotExist(err) && info.Mode() & 0111 != 0 {
		if !DetectSystemd() {
			t.Errorf("case: expected true, got false")
		}
	}
}

func TestAddSystemdScope(t *testing.T) {
	type inputType struct {
		systemdRunPath string
		mountName      string
		command        string
		args           []string
	}
	type resultType struct{
		command string
		args []string
	}
	testCases := []struct {
		input inputType
		result resultType
	}{
		{
			inputType{"systemd-run", "/somewhere",
				"mount",[]string{"/dev/sda1", "/somewhere"}},
			resultType{"systemd-run",
				[]string{"--description=Kubernetes transient mount for /somewhere", "--scope","--",
					"mount", "/dev/sda1", "/somewhere"}},
		},
	}
	for _, tc := range testCases {
		command, args := AddSystemdScope(tc.input.systemdRunPath, tc.input.mountName, tc.input.command, tc.input.args)
		actualResult := resultType{command,args}
		if !reflect.DeepEqual(actualResult, tc.result) {
			t.Errorf("case \"%v\": expected %v, got %v", tc.input, tc.result, actualResult)
		}
	}
}
