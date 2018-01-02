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

// +build host coreos src kvm

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

// We need CAP_SYS_PTRACE to escape the chroot
var memoryTest = struct {
	testName     string
	aciBuildArgs []string
}{
	`Check memory isolator`,
	[]string{"--exec=/inspect --print-memorylimit", "--capability=CAP_SYS_PTRACE"},
}

var cpuTest = struct {
	testName     string
	aciBuildArgs []string
}{
	`Check CPU quota`,
	[]string{"--exec=/inspect --print-cpuquota", "--capability=CAP_SYS_PTRACE"},
}

var cpuSharesTest = struct {
	testName     string
	aciBuildArgs []string
}{
	`Check CPU shares`,
	[]string{"--exec=/inspect --print-cpushares", "--capability=CAP_SYS_PTRACE"},
}

var cgroupsTest = struct {
	testName     string
	aciBuildArgs []string
}{
	`Check cgroup mounts`,
	[]string{"--exec=/inspect --check-cgroups", "--capability=CAP_SYS_PTRACE"},
}

func TestAppIsolatorMemory(t *testing.T) {
	ok, err := cgroup.IsIsolatorSupported("memory")
	if err != nil {
		t.Fatalf("Error checking memory isolator support: %v", err)
	}
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
	isUnified, err := cgroup.IsCgroupUnified("/")
	if err != nil {
		t.Fatalf("Error determining the cgroup version: %v", err)
	}

	if isUnified {
		// TODO: for now kernel does not support cpu isolator in cgroup2.
		// Write a test when it does.
		t.Skip("kernel does not support cpu isolator in cgroup2.")
	}
	ok, err := cgroup.IsIsolatorSupported("cpu")
	if err != nil {
		t.Fatalf("Error checking cpu isolator support: %v", err)
	}
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

func TestAppIsolatorCPUShares(t *testing.T) {
	isUnified, err := cgroup.IsCgroupUnified("/")
	if err != nil {
		t.Fatalf("Error determining the cgroup version: %v", err)
	}

	if isUnified {
		t.Skip("kernel does not support cpu isolator in cgroup2.")
	}

	ok, err := cgroup.IsIsolatorSupported("cpu")
	if err != nil {
		t.Fatalf("Error checking cpu isolator support: %v", err)
	}

	if !ok {
		t.Skip("CPU isolator not supported.")
	}

	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	t.Logf("Running test: %v", cpuSharesTest.testName)

	aciFileName := patchTestACI("rkt-inspect-isolators.aci", cpuSharesTest.aciBuildArgs...)
	defer os.Remove(aciFileName)

	rktCmd := fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s --cpu-shares 12345", ctx.Cmd(), aciFileName)
	expectedLine := "CPU Shares: 12345"
	runRktAndCheckOutput(t, rktCmd, expectedLine, false)
}
