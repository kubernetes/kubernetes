// Copyright 2015 Google Inc. All Rights Reserved.
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

package validate

import (
	"fmt"
	"strings"
	"testing"
)

var (
	standardDesc       = "Versions >= %s are supported. %s+ are recommended"
	dockerStandardDesc = fmt.Sprintf(standardDesc, "1.0", "1.2")
	kernelStandardDesc = fmt.Sprintf(standardDesc, "2.6", "3.0")
	dockerErrorDesc    = "Could not parse docker version"
	kernelErrorDesc    = "Could not parse kernel version"
)

func TestGetMajorMinor(t *testing.T) {
	cases := []struct {
		version string
		major   int
		minor   int
		err     error
	}{
		{"0.1beta", 0, 1, nil},
		{"0.1.2", 0, 1, nil},
		{"-1.-1beta", -1, -1, nil},
		{"0.1", -1, -1, fmt.Errorf("have error")},
		{"0", -1, -1, fmt.Errorf("have error")},
		{"beta", -1, -1, fmt.Errorf("have error")},
	}

	for i, c := range cases {
		ma, mi, e := getMajorMinor(c.version)
		if (e != nil) != (c.err != nil) {
			t.Errorf("[%d] Unexpected err, should %v, but got %v", i, c.err, e)
		}
		if ma != c.major {
			t.Errorf("[%d] Unexpected major, should %v, but got %v", i, c.major, ma)
		}
		if mi != c.minor {
			t.Errorf("[%d] Unexpected minor, should %v, but got %v", i, c.minor, mi)
		}
	}
}

func TestValidateKernelVersion(t *testing.T) {
	cases := []struct {
		version string
		result  string
		desc    string
	}{
		{"2.6.3", Supported, kernelStandardDesc},
		{"3.6.3", Recommended, kernelStandardDesc},
		{"1.0beta", Unsupported, kernelStandardDesc},
		{"0.1beta", Unsupported, kernelStandardDesc},
		{"0.1", Unknown, kernelErrorDesc},
		{"3.1", Unknown, kernelErrorDesc},
	}

	for i, c := range cases {
		res, desc := validateKernelVersion(c.version)
		if res != c.result {
			t.Errorf("[%d] Unexpected result, should %v, but got %v", i, c.result, res)
		}
		if !strings.Contains(desc, c.desc) {
			t.Errorf("[%d] Unexpected description, should %v, but got %v", i, c.desc, desc)
		}
	}
}

func TestValidateDockerVersion(t *testing.T) {
	cases := []struct {
		version string
		result  string
		desc    string
	}{
		{"1.1.3", Supported, dockerStandardDesc},
		{"1.6.3", Recommended, dockerStandardDesc},
		{"1.0beta", Supported, dockerStandardDesc},
		{"0.1beta", Unsupported, dockerStandardDesc},
		{"0.1", Unknown, dockerErrorDesc},
		{"1.6", Unknown, dockerErrorDesc},
	}

	for i, c := range cases {
		res, desc := validateDockerVersion(c.version)
		if res != c.result {
			t.Errorf("[%d] Unexpected result, should %v, but got %v", i, c.result, res)
		}
		if !strings.Contains(desc, c.desc) {
			t.Errorf("[%d] Unexpected description, should %v, but got %v", i, c.desc, desc)
		}
	}
}

func TestAreCgroupsPresent(t *testing.T) {
	cases := []struct {
		available map[string]int
		desired   []string
		result    bool
		reason    string
	}{
		{map[string]int{"memory": 1}, []string{"memory"}, true, ""},
		{map[string]int{"memory": 2}, []string{"memory"}, false, "memory not enabled. Available cgroups"},
		{map[string]int{"memory": 0}, []string{"memory"}, false, "memory not enabled. Available cgroups"},
		{map[string]int{"memory": 1}, []string{"blkio"}, false, "Missing cgroup blkio. Available cgroups"},
	}
	for i, c := range cases {
		result, reason := areCgroupsPresent(c.available, c.desired)
		if result != c.result {
			t.Errorf("[%d] Unexpected result, should %v, but got %v", i, c.result, result)
		}
		if (c.reason == "" && reason != "") || (c.reason != "" && !strings.Contains(reason, c.reason)) {
			t.Errorf("[%d] Unexpected result, should %v, but got %v", i, c.reason, reason)
		}
	}
}
