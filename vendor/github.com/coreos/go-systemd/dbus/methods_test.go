// Copyright 2015 CoreOS, Inc.
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

package dbus

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/godbus/dbus"
)

func setupConn(t *testing.T) *Conn {
	conn, err := New()
	if err != nil {
		t.Fatal(err)
	}

	return conn
}

func findFixture(target string, t *testing.T) string {
	abs, err := filepath.Abs("../fixtures/" + target)
	if err != nil {
		t.Fatal(err)
	}
	return abs
}

func setupUnit(target string, conn *Conn, t *testing.T) {
	// Blindly stop the unit in case it is running
	conn.StopUnit(target, "replace", nil)

	// Blindly remove the symlink in case it exists
	targetRun := filepath.Join("/run/systemd/system/", target)
	os.Remove(targetRun)
}

func linkUnit(target string, conn *Conn, t *testing.T) {
	abs := findFixture(target, t)
	fixture := []string{abs}

	changes, err := conn.LinkUnitFiles(fixture, true, true)
	if err != nil {
		t.Fatal(err)
	}

	if len(changes) < 1 {
		t.Fatalf("Expected one change, got %v", changes)
	}

	runPath := filepath.Join("/run/systemd/system/", target)
	if changes[0].Filename != runPath {
		t.Fatal("Unexpected target filename")
	}
}

// Ensure that basic unit starting and stopping works.
func TestStartStopUnit(t *testing.T) {
	target := "start-stop.service"
	conn := setupConn(t)

	setupUnit(target, conn, t)
	linkUnit(target, conn, t)

	// 2. Start the unit
	reschan := make(chan string)
	_, err := conn.StartUnit(target, "replace", reschan)
	if err != nil {
		t.Fatal(err)
	}

	job := <-reschan
	if job != "done" {
		t.Fatal("Job is not done:", job)
	}

	units, err := conn.ListUnits()

	var unit *UnitStatus
	for _, u := range units {
		if u.Name == target {
			unit = &u
		}
	}

	if unit == nil {
		t.Fatalf("Test unit not found in list")
	}

	if unit.ActiveState != "active" {
		t.Fatalf("Test unit not active")
	}

	// 3. Stop the unit
	_, err = conn.StopUnit(target, "replace", reschan)
	if err != nil {
		t.Fatal(err)
	}

	// wait for StopUnit job to complete
	<-reschan

	units, err = conn.ListUnits()

	unit = nil
	for _, u := range units {
		if u.Name == target {
			unit = &u
		}
	}

	if unit != nil {
		t.Fatalf("Test unit found in list, should be stopped")
	}
}

// Enables a unit and then immediately tears it down
func TestEnableDisableUnit(t *testing.T) {
	target := "enable-disable.service"
	conn := setupConn(t)

	setupUnit(target, conn, t)
	abs := findFixture(target, t)
	runPath := filepath.Join("/run/systemd/system/", target)

	// 1. Enable the unit
	install, changes, err := conn.EnableUnitFiles([]string{abs}, true, true)
	if err != nil {
		t.Fatal(err)
	}

	if install != false {
		t.Fatal("Install was true")
	}

	if len(changes) < 1 {
		t.Fatalf("Expected one change, got %v", changes)
	}

	if changes[0].Filename != runPath {
		t.Fatal("Unexpected target filename")
	}

	// 2. Disable the unit
	dChanges, err := conn.DisableUnitFiles([]string{abs}, true)
	if err != nil {
		t.Fatal(err)
	}

	if len(dChanges) != 1 {
		t.Fatalf("Changes should include the path, %v", dChanges)
	}
	if dChanges[0].Filename != runPath {
		t.Fatalf("Change should include correct filename, %+v", dChanges[0])
	}
	if dChanges[0].Destination != "" {
		t.Fatalf("Change destination should be empty, %+v", dChanges[0])
	}
}

// TestGetUnitProperties reads the `-.mount` which should exist on all systemd
// systems and ensures that one of its properties is valid.
func TestGetUnitProperties(t *testing.T) {
	conn := setupConn(t)

	unit := "-.mount"

	info, err := conn.GetUnitProperties(unit)
	if err != nil {
		t.Fatal(err)
	}

	names := info["Wants"].([]string)

	if len(names) < 1 {
		t.Fatal("/ is unwanted")
	}

	if names[0] != "system.slice" {
		t.Fatal("unexpected wants for /")
	}

	prop, err := conn.GetUnitProperty(unit, "Wants")
	if err != nil {
		t.Fatal(err)
	}

	if prop.Name != "Wants" {
		t.Fatal("unexpected property name")
	}

	val := prop.Value.Value().([]string)
	if !reflect.DeepEqual(val, names) {
		t.Fatal("unexpected property value")
	}
}

// TestGetUnitPropertiesRejectsInvalidName attempts to get the properties for a
// unit with an invalid name. This test should be run with --test.timeout set,
// as a fail will manifest as GetUnitProperties hanging indefinitely.
func TestGetUnitPropertiesRejectsInvalidName(t *testing.T) {
	conn := setupConn(t)

	unit := "//invalid#$^/"

	_, err := conn.GetUnitProperties(unit)
	if err == nil {
		t.Fatal("Expected an error, got nil")
	}

	_, err = conn.GetUnitProperty(unit, "Wants")
	if err == nil {
		t.Fatal("Expected an error, got nil")
	}
}

// TestSetUnitProperties changes a cgroup setting on the `tmp.mount`
// which should exist on all systemd systems and ensures that the
// property was set.
func TestSetUnitProperties(t *testing.T) {
	conn := setupConn(t)

	unit := "tmp.mount"

	if err := conn.SetUnitProperties(unit, true, Property{"CPUShares", dbus.MakeVariant(uint64(1023))}); err != nil {
		t.Fatal(err)
	}

	info, err := conn.GetUnitTypeProperties(unit, "Mount")
	if err != nil {
		t.Fatal(err)
	}

	value := info["CPUShares"].(uint64)
	if value != 1023 {
		t.Fatal("CPUShares of unit is not 1023:", value)
	}
}

// Ensure that basic transient unit starting and stopping works.
func TestStartStopTransientUnit(t *testing.T) {
	conn := setupConn(t)

	props := []Property{
		PropExecStart([]string{"/bin/sleep", "400"}, false),
	}
	target := fmt.Sprintf("testing-transient-%d.service", rand.Int())

	// Start the unit
	reschan := make(chan string)
	_, err := conn.StartTransientUnit(target, "replace", props, reschan)
	if err != nil {
		t.Fatal(err)
	}

	job := <-reschan
	if job != "done" {
		t.Fatal("Job is not done:", job)
	}

	units, err := conn.ListUnits()

	var unit *UnitStatus
	for _, u := range units {
		if u.Name == target {
			unit = &u
		}
	}

	if unit == nil {
		t.Fatalf("Test unit not found in list")
	}

	if unit.ActiveState != "active" {
		t.Fatalf("Test unit not active")
	}

	// 3. Stop the unit
	_, err = conn.StopUnit(target, "replace", reschan)
	if err != nil {
		t.Fatal(err)
	}

	// wait for StopUnit job to complete
	<-reschan

	units, err = conn.ListUnits()

	unit = nil
	for _, u := range units {
		if u.Name == target {
			unit = &u
		}
	}

	if unit != nil {
		t.Fatalf("Test unit found in list, should be stopped")
	}
}

func TestConnJobListener(t *testing.T) {
	target := "start-stop.service"
	conn := setupConn(t)

	setupUnit(target, conn, t)
	linkUnit(target, conn, t)

	jobSize := len(conn.jobListener.jobs)

	reschan := make(chan string)
	_, err := conn.StartUnit(target, "replace", reschan)
	if err != nil {
		t.Fatal(err)
	}

	<-reschan

	_, err = conn.StopUnit(target, "replace", reschan)
	if err != nil {
		t.Fatal(err)
	}

	<-reschan

	currentJobSize := len(conn.jobListener.jobs)
	if jobSize != currentJobSize {
		t.Fatal("JobListener jobs leaked")
	}
}
