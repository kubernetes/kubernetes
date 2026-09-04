//go:build windows

/*
Copyright 2024 The Kubernetes Authors.

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

package pidlimit

import (
	"testing"
)

func TestStatsWindows(t *testing.T) {
	stats, err := Stats()
	if err != nil {
		t.Fatalf("Stats() returned an unexpected error: %v", err)
	}
	if stats == nil {
		t.Fatal("Stats() returned a nil RlimitStats")
	}
	if stats.MaxPID == nil {
		t.Fatal("Stats() returned a nil MaxPID")
	}
	if *stats.MaxPID <= 0 {
		t.Fatalf("Stats() returned a non-positive MaxPID: %d", *stats.MaxPID)
	}
	if stats.NumOfRunningProcesses == nil {
		t.Fatal("Stats() returned a nil NumOfRunningProcesses")
	}
	if *stats.NumOfRunningProcesses < 0 {
		t.Fatalf("Stats() returned a negative NumOfRunningProcesses: %d", *stats.NumOfRunningProcesses)
	}
	if stats.Time.IsZero() {
		t.Fatal("Stats() returned a zero timestamp")
	}
}
