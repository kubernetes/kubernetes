// Copyright 2014 Google Inc. All Rights Reserved.
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

package libcontainer

import (
	"testing"

	"code.google.com/p/gomock/gomock"

	"github.com/google/cadvisor/utils/fs"
	"github.com/google/cadvisor/utils/fs/mockfs"
)

var initCgroupsToParentAndID = []struct {
	InitCgroupFileContent string
	ContainerPath         string
	Parent                string
	Id                    string
	Error                 error
}{
	{
		`
11:name=systemd:/
10:hugetlb:/
9:perf_event:/
8:blkio:/
7:freezer:/
6:devices:/
5:memory:/
4:cpuacct:/
3:cpu:/
2:cpuset:/
`,
		"/",
		"",
		"",
		nil,
	},
	{
		`
11:name=systemd:/
10:hugetlb:/
9:perf_event:/
8:blkio:/
7:freezer:/
6:devices:/
5:memory:/docker/hash
4:cpuacct:/
3:cpu:/docker/hash
2:cpuset:/
`,
		"/parent/id",
		"../../parent",
		"id",
		nil,
	},
}

func TestSplitName(t *testing.T) {
	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()

	for _, testCase := range initCgroupsToParentAndID {
		mfs := mockfs.NewMockFileSystem(mockCtrl)
		mockfs.AddTextFile(mfs, "/proc/1/cgroup", testCase.InitCgroupFileContent)
		fs.ChangeFileSystem(mfs)
		parent, id, err := SplitName(testCase.ContainerPath)
		if testCase.Error != nil {
			if err == nil {
				t.Fatalf("did not receive expected error.\ncontent:%v\n, path:%v\n, expected error:%v\n", testCase.InitCgroupFileContent, testCase.ContainerPath, testCase.Error)
			}
			continue
		}
		if testCase.Parent != parent || testCase.Id != id {
			t.Errorf("unexpected parent or id:\ncontent:%v\npath:%v\nexpected parent: %v; recevied parent: %v;\nexpected id: %v; received id: %v", testCase.InitCgroupFileContent, testCase.ContainerPath, testCase.Parent, parent, testCase.Id, id)
		}
	}
}
