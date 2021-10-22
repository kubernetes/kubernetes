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
	"path/filepath"
	"testing"
)

func TestGetRaplZones(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	zones, err := GetRaplZones(fs)
	if err != nil || zones == nil {
		t.Fatal(err)
	}
}

func TestNoRaplFiles(t *testing.T) {
	// use a bad (but existing) fs path
	fs, err := NewFS(filepath.Join(sysTestFixtures, "class"))
	if err != nil {
		t.Fatal(err)
	}
	zones, err := GetRaplZones(fs)
	// expect failure
	if err == nil || zones != nil {
		t.Fatal(err)
	}
}

func TestNewRaplValues(t *testing.T) {
	fs, err := NewFS(sysTestFixtures)
	if err != nil {
		t.Fatal(err)
	}

	zones, err := GetRaplZones(fs)
	if err != nil || zones == nil {
		t.Fatal(err)
	}

	if len(zones) != 3 {
		t.Fatal("wrong number of RAPL values")
	}
	microjoules, err := zones[0].GetEnergyMicrojoules()
	if err != nil {
		t.Fatal("couldn't read microjoules")
	}
	if microjoules != 240422366267 {
		t.Fatal("wrong microjoule number")
	}
	if zones[2].Index != 10 {
		t.Fatal("wrong index number")
	}
}
