// Copyright 2019 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build !windows

package sysfs

import (
	"reflect"
	"testing"
)

func TestClassCoolingDeviceStats(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	coolingDeviceTest, err := fs.ClassCoolingDeviceStats()
	if err != nil {
		t.Fatal(err)
	}

	classCoolingDeviceStats := []ClassCoolingDeviceStats{
		{
			Name:     "0",
			Type:     "Processor",
			MaxState: 50,
			CurState: 0,
		},
		{
			Name:     "1",
			Type:     "intel_powerclamp",
			MaxState: 27,
			CurState: -1,
		},
	}

	if !reflect.DeepEqual(classCoolingDeviceStats, coolingDeviceTest) {
		t.Errorf("Result not correct: want %v, have %v", classCoolingDeviceStats, coolingDeviceTest)
	}
}
