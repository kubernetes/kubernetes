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

func TestNewClocksource(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	c, err := fs.ClockSources()
	if err != nil {
		t.Fatal(err)
	}

	clocksources := []ClockSource{
		{
			Name:      "0",
			Available: []string{"tsc", "hpet", "acpi_pm"},
			Current:   "tsc",
		},
	}

	if !reflect.DeepEqual(clocksources, c) {
		t.Errorf("Result not correct: want %v, have %v", clocksources, c)
	}
}
