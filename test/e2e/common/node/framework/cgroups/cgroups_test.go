/*
Copyright 2025 The Kubernetes Authors.

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

package cgroups

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestGetCPULimitCgroupExpectations(t *testing.T) {
	testCases := []struct {
		name              string
		cpuLimit          *resource.Quantity
		podOnCgroupv2Node bool
		expected          []string
	}{
		{
			name:              "rounding required, podOnCGroupv2Node=true",
			cpuLimit:          resource.NewMilliQuantity(15, resource.DecimalSI),
			podOnCgroupv2Node: true,
			expected:          []string{"1500 100000", "2000 100000"},
		},
		{
			name:              "rounding not required, podOnCGroupv2Node=true",
			cpuLimit:          resource.NewMilliQuantity(20, resource.DecimalSI),
			podOnCgroupv2Node: true,
			expected:          []string{"2000 100000"},
		},
		{
			name:              "rounding required, podOnCGroupv2Node=false",
			cpuLimit:          resource.NewMilliQuantity(15, resource.DecimalSI),
			podOnCgroupv2Node: false,
			expected:          []string{"1500", "2000"},
		},
		{
			name:              "rounding not required, podOnCGroupv2Node=false",
			cpuLimit:          resource.NewMilliQuantity(20, resource.DecimalSI),
			podOnCgroupv2Node: false,
			expected:          []string{"2000"},
		},
		{
			name:              "cpuQuota=0, podOnCGroupv2Node=true",
			cpuLimit:          resource.NewMilliQuantity(0, resource.DecimalSI),
			podOnCgroupv2Node: true,
			expected:          []string{"max 100000"},
		},
		{
			name:              "cpuQuota=0, podOnCGroupv2Node=false",
			cpuLimit:          resource.NewMilliQuantity(0, resource.DecimalSI),
			podOnCgroupv2Node: false,
			expected:          []string{"-1"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := getCPULimitCgroupExpectations(tc.cpuLimit, tc.podOnCgroupv2Node)
			assert.Equal(t, tc.expected, actual)
		})
	}
}

func TestClassifyCgroupReads(t *testing.T) {
	memLimit := cgroupExpectation{path: "memory.max", values: []string{"100"}}
	cpuLimit := cgroupExpectation{path: "cpu.max", values: []string{"10 100"}}
	cpuWeight := cgroupExpectation{path: "cpu.weight"} // no values: an integer > 0
	checks := []cgroupExpectation{memLimit, cpuLimit, cpuWeight}
	read := func(v string) cgroupRead { return cgroupRead{value: v, ok: true} }
	unread := cgroupRead{}

	for _, tc := range []struct {
		name         string
		reads        []cgroupRead
		wantRetry    bool
		wantTerminal bool
	}{
		{name: "all correct", reads: []cgroupRead{read("100"), read("10 100"), read("39")}},
		{name: "wrong limit is terminal", reads: []cgroupRead{read("200"), read("10 100"), read("39")}, wantTerminal: true},
		{name: "non-integer weight is terminal", reads: []cgroupRead{read("100"), read("10 100"), read("x")}, wantTerminal: true},
		{name: "zero weight is terminal", reads: []cgroupRead{read("100"), read("10 100"), read("0")}, wantTerminal: true},
		{name: "empty value is terminal", reads: []cgroupRead{read(""), read("10 100"), read("39")}, wantTerminal: true},
		{name: "unread file is retried", reads: []cgroupRead{read("100"), unread, read("39")}, wantRetry: true},
		{name: "all unread is retried", reads: []cgroupRead{unread, unread, unread}, wantRetry: true},
		// A wrong value must stay terminal even when another file in the batch was not read.
		{name: "wrong value with an unread sibling is terminal", reads: []cgroupRead{read("200"), unread, read("39")}, wantTerminal: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			retry, terminal := classifyCgroupReads(tc.reads, checks, "c")
			assert.Equal(t, tc.wantRetry, retry != nil, "retry=%v", retry)
			assert.Equal(t, tc.wantTerminal, terminal != nil, "terminal=%v", terminal)
		})
	}
}

// TestParseCgroupReads feeds the exec helper's trimmed output through to the batch's outcome.
func TestParseCgroupReads(t *testing.T) {
	checks := []cgroupExpectation{
		{path: "memory.max", values: []string{"100"}},
		{path: "cpu.max", values: []string{"10 100"}},
		{path: "cpu.weight"},
	}
	for _, tc := range []struct {
		name         string
		out          string
		wantErr      bool
		wantRetry    bool
		wantTerminal bool
	}{
		{name: "all correct", out: "ok:100\nok:10 100\nok:39"},
		{name: "untrimmed output", out: "ok:100\nok:10 100\nok:39\n"},
		{name: "wrong first value, last file unread", out: "ok:200\nok:10 100\nerr:", wantTerminal: true},
		{name: "first file unread, wrong second value", out: "err:\nok:20 100\nok:39", wantTerminal: true},
		{name: "middle file unread", out: "ok:100\nerr:\nok:39", wantRetry: true},
		{name: "all unread", out: "err:\nerr:\nerr:", wantRetry: true},
		{name: "empty file read successfully", out: "ok:\nok:10 100\nok:39", wantTerminal: true},
		{name: "missing record", out: "ok:100\nok:10 100", wantErr: true},
		{name: "record without status", out: "100\n10 100\n39", wantErr: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reads, err := parseCgroupReads(tc.out, len(checks))
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			retry, terminal := classifyCgroupReads(reads, checks, "c")
			assert.Equal(t, tc.wantRetry, retry != nil, "retry=%v", retry)
			assert.Equal(t, tc.wantTerminal, terminal != nil, "terminal=%v", terminal)
		})
	}
}

// TestReadCgroupScript runs readCgroupScript locally on an unreadable path, an empty file and a value with a space.
func TestReadCgroupScript(t *testing.T) {
	if _, err := os.Stat("/bin/sh"); err != nil {
		t.Skip("no /bin/sh")
	}
	dir := t.TempDir()
	write := func(name, content string) string {
		p := filepath.Join(dir, name)
		require.NoError(t, os.WriteFile(p, []byte(content), 0o644))
		return p
	}
	paths := []string{write("memory.max", "200\n"), filepath.Join(dir, "missing"), write("empty", ""), write("cpu.max", "max 100000\n")}
	out, err := exec.CommandContext(t.Context(), "/bin/sh", append([]string{"-c", readCgroupScript, "sh"}, paths...)...).Output()
	require.NoError(t, err)
	reads, err := parseCgroupReads(strings.TrimSpace(string(out)), len(paths))
	require.NoError(t, err)
	assert.Equal(t, []cgroupRead{{value: "200", ok: true}, {}, {value: "", ok: true}, {value: "max 100000", ok: true}}, reads)
}

func TestContainerCgroupExpectations(t *testing.T) {
	memMax := getCgroupMemLimitPath(cgroupFsPath, true)
	cpuMax := getCgroupCPULimitPath(cgroupFsPath, true)
	cpuWeight := getCgroupCPURequestPath(cgroupFsPath, true)
	limits := func(cpu, mem string) v1.ResourceList {
		l := v1.ResourceList{}
		if cpu != "" {
			l[v1.ResourceCPU] = resource.MustParse(cpu)
		}
		if mem != "" {
			l[v1.ResourceMemory] = resource.MustParse(mem)
		}
		return l
	}
	podWith := func(podLimits v1.ResourceList) *v1.Pod {
		pod := &v1.Pod{}
		if podLimits != nil {
			pod.Spec.Resources = &v1.ResourceRequirements{Limits: podLimits}
		}
		return pod
	}
	ctr := func(ctrLimits v1.ResourceList, cpuRequest string) *v1.Container {
		c := &v1.Container{Name: "c", Resources: v1.ResourceRequirements{Limits: ctrLimits}}
		if cpuRequest != "" {
			c.Resources.Requests = v1.ResourceList{v1.ResourceCPU: resource.MustParse(cpuRequest)}
		}
		return c
	}
	for _, tc := range []struct {
		name string
		pod  *v1.Pod
		ctr  *v1.Container
		want []cgroupExpectation
	}{
		{
			name: "container limits, no pod-level resources",
			pod:  podWith(nil),
			ctr:  ctr(limits("200m", "256Mi"), ""),
			want: []cgroupExpectation{{memMax, []string{"268435456"}}, {cpuMax, []string{"20000 100000"}}, {path: cpuWeight}},
		},
		{
			name: "pod-level limits only",
			pod:  podWith(limits("200m", "256Mi")),
			ctr:  ctr(nil, "100m"),
			want: []cgroupExpectation{{memMax, []string{"268435456"}}, {cpuMax, []string{"20000 100000"}}, {path: cpuWeight}},
		},
		{
			name: "container CPU limit, pod-level memory limit",
			pod:  podWith(limits("200m", "256Mi")),
			ctr:  ctr(limits("100m", ""), "100m"),
			want: []cgroupExpectation{{memMax, []string{"268435456"}}, {cpuMax, []string{"10000 100000"}}, {path: cpuWeight}},
		},
		{
			name: "container memory limit, pod-level CPU limit",
			pod:  podWith(limits("200m", "256Mi")),
			ctr:  ctr(limits("", "128Mi"), "100m"),
			want: []cgroupExpectation{{memMax, []string{"134217728"}}, {cpuMax, []string{"20000 100000"}}, {path: cpuWeight}},
		},
		{
			name: "container limits win over pod-level limits",
			pod:  podWith(limits("200m", "256Mi")),
			ctr:  ctr(limits("100m", "128Mi"), "100m"),
			want: []cgroupExpectation{{memMax, []string{"134217728"}}, {cpuMax, []string{"10000 100000"}}, {path: cpuWeight}},
		},
		{
			name: "no limits anywhere",
			pod:  podWith(nil),
			ctr:  ctr(nil, ""),
			want: []cgroupExpectation{{memMax, []string{"max"}}, {cpuMax, []string{"max 100000"}}, {path: cpuWeight}},
		},
		{
			name: "pod-level limits, no container CPU request: CPU weight is not checked",
			pod:  podWith(limits("200m", "256Mi")),
			ctr:  ctr(nil, ""),
			want: []cgroupExpectation{{memMax, []string{"268435456"}}, {cpuMax, []string{"20000 100000"}}},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			pod, ctr := tc.pod.DeepCopy(), tc.ctr.DeepCopy()
			assert.Equal(t, tc.want, containerCgroupExpectations(pod, ctr, true))
			assert.Equal(t, tc.pod, pod, "pod modified")
			assert.Equal(t, tc.ctr, ctr, "container modified")
		})
	}
}
