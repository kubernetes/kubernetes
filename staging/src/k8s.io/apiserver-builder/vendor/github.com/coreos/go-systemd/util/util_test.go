// Copyright 2016 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package util

import (
	"testing"
)

func testIsRunningSystemd(t *testing.T) {
	if !IsRunningSystemd() {
		t.Skip("Not running on a systemd host")
	}
}

func TestRunningFromSystemService(t *testing.T) {
	testIsRunningSystemd(t)
	t.Parallel()

	// tests shouldn't be running as a service
	s, err := RunningFromSystemService()
	if err != nil {
		t.Error(err.Error())
	} else if s {
		t.Errorf("tests aren't expected to run as a service")
	}
}

func TestCurrentUnitName(t *testing.T) {
	testIsRunningSystemd(t)

	s, err := CurrentUnitName()
	if err != nil {
		t.Error(err.Error())
	}
	if s == "" {
		t.Error("CurrentUnitName returned a empty string")
	}
}

func TestGetMachineID(t *testing.T) {
	testIsRunningSystemd(t)

	id, err := GetMachineID()
	if err != nil {
		t.Error(err.Error())
	}
	if id == "" {
		t.Error("GetMachineID returned a empty string")
	}
}

func TestGetRunningSlice(t *testing.T) {
	testIsRunningSystemd(t)

	s, err := getRunningSlice()
	if err != nil {
		t.Error(err.Error())
	}
	if s == "" {
		t.Error("getRunningSlice returned a empty string")
	}
}
