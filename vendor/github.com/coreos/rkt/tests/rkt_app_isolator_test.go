// Copyright 2015 The rkt Authors
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

package main

import (
	"fmt"
	"os"
	"strconv"
	"testing"

	"github.com/coreos/rkt/common/cgroup"
	"github.com/coreos/rkt/tests/testutils"
)

const (
	// if you change this you need to change tests/image/manifest accordingly
	maxMemoryUsage = 25 * 1024 * 1024 // 25MB
	CPUQuota       = 800              // milli-cores
)

var memoryTest = struct {
	testName     string
	aciBuildArgs []string
}{
	`Check memory isolator`,
	[]string{"--exec=/inspect --print-memorylimit"},
}

var cpuTest = struct {
	testName     string
	aciBuildArgs []string
}{
	`Check CPU quota`,
	[]string{"--exec=/inspect --print-cpuquota"},
}

var cgroupsTest = struct {
	testName     string
	aciBuildArgs []string
}{
	`Check cgroup mounts`,
	[]string{"--exec=/inspect --check-cgroups"},
}

func TestAppIsolatorMemory(t *testing.T) {
	ok := cgroup.IsIsolatorSupported("memory")
	if !ok {
		t.Skip("Memory isolator not supported.")
	}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	t.Logf("Running test: %v", memoryTest.testName)

	aciFileName := patchTestACI("rkt-inspect-isolators.aci", memoryTest.aciBuildArgs...)
	defer os.Remove(aciFileName)

	rktCmd := fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s", ctx.Cmd(), aciFileName)
	expectedLine := "Memory Limit: " + strconv.Itoa(maxMemoryUsage)
	runRktAndCheckOutput(t, rktCmd, expectedLine, false)

	rktCmd = fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s --memory 42Mi", ctx.Cmd(), aciFileName)
	expectedLine = "Memory Limit: " + strconv.Itoa(42*1024*1024)
	runRktAndCheckOutput(t, rktCmd, expectedLine, false)
}

func TestAppIsolatorCPU(t *testing.T) {
	ok := cgroup.IsIsolatorSupported("cpu")
	if !ok {
		t.Skip("CPU isolator not supported.")
	}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	t.Logf("Running test: %v", cpuTest.testName)

	aciFileName := patchTestACI("rkt-inspect-isolators.aci", cpuTest.aciBuildArgs...)
	defer os.Remove(aciFileName)

	rktCmd := fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s", ctx.Cmd(), aciFileName)
	expectedLine := "CPU Quota: " + strconv.Itoa(CPUQuota)
	runRktAndCheckOutput(t, rktCmd, expectedLine, false)

	rktCmd = fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s --cpu 900m", ctx.Cmd(), aciFileName)
	expectedLine = "CPU Quota: " + strconv.Itoa(900)
	runRktAndCheckOutput(t, rktCmd, expectedLine, false)
}

func TestCgroups(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	t.Logf("Running test: %v", cgroupsTest.testName)

	aciFileName := patchTestACI("rkt-inspect-isolators.aci", cgroupsTest.aciBuildArgs...)
	defer os.Remove(aciFileName)

	rktCmd := fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s", ctx.Cmd(), aciFileName)
	expectedLine := "check-cgroups: SUCCESS"
	runRktAndCheckOutput(t, rktCmd, expectedLine, false)
}
