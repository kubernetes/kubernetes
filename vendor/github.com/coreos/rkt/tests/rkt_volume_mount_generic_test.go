// Copyright 2016 The rkt Authors
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

// +build coreos host kvm

package main

import (
	"fmt"
	"testing"
)

func TestVolumeMount(t *testing.T) {
	NewTestVolumeMount([][]volumeMountTestCase{
		volumeMountTestCasesRecursiveCLI,
		volumeMountTestCasesNonRecursiveCLI,
		volumeMountTestCasesRecursivePodManifest,
		volumeMountTestCasesNonRecursivePodManifest,
		volumeMountTestCasesNonRecursive,
		{
			{
				"CLI: duplicate mount given",
				[]imagePatch{
					{
						"rkt-test-run-read-file.aci",
						[]string{fmt.Sprintf("--exec=/inspect --read-file --file-name %s", mountFilePath)},
					},
				},
				fmt.Sprintf(
					"--volume=test1,kind=host,source=%s --mount volume=test1,target=%s --volume=test2,kind=host,source=%s --mount volume=test1,target=%s",
					volDir, mountDir,
					volDir, mountDir,
				),
				nil,
				innerFileContent,
			},
		},
	}).Execute(t)
}
