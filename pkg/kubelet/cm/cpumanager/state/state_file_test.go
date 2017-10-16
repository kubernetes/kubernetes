/*
Copyright 2017 The Kubernetes Authors.

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

package state

import (
	"io/ioutil"
	"path"
	"reflect"
	"testing"

	"fmt"

	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"

	"os"
)

func writeToStateFile(statefile string, content string) {
	ioutil.WriteFile(statefile, []byte(content), 0644)
}

func TestFileStateTryRestore(t *testing.T) {
	testCases := []struct {
		description      string
		stateFileContent string
		expErr           string
		expectedState    stateMemory
	}{
		{
			"Invalid JSON - empty file",
			"\n",
			"unexpected end of JSON input",
			stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"Invalid JSON - invalid content",
			"{",
			"unexpected end of JSON input",
			stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"Try restore defaultCPUSet only",
			"{ \"defaultCpuSet\": \"4-6\"}",
			"",
			stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(4, 5, 6),
			},
		},
		{
			"Try restore defaultCPUSet only - invalid name",
			"{ \"defCPUSet\": \"4-6\"}",
			"",
			stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"Try restore assignments only",
			"{" +
				"\"reservedList\": { " +
				"\"container1\": \"4-6\"," +
				"\"container2\": \"1-3\"" +
				"} }",
			"",
			stateMemory{
				assignments: ContainerCPUAssignments{
					"container1": cpuset.NewCPUSet(4, 5, 6),
					"container2": cpuset.NewCPUSet(1, 2, 3),
				},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"Try restore invalid assignments",
			"{ \"reservedList\": }",
			"invalid character '}' looking for beginning of value",
			stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"Try restore valid file",
			"{ " +
				"\"defaultCpuSet\": \"23-24\", " +
				"\"reservedList\": { " +
				"\"container1\": \"4-6\", " +
				"\"container2\": \"1-3\"" +
				" } }",
			"",
			stateMemory{
				assignments: ContainerCPUAssignments{
					"container1": cpuset.NewCPUSet(4, 5, 6),
					"container2": cpuset.NewCPUSet(1, 2, 3),
				},
				defaultCPUSet: cpuset.NewCPUSet(23, 24),
			},
		},
		{
			"Try restore un-parsable defaultCPUSet ",
			"{ \"defaultCpuSet\": \"2-sd\" }",
			"strconv.Atoi: parsing \"sd\": invalid syntax",
			stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"Try restore un-parsable assignments",
			"{ " +
				"\"defaultCpuSet\": \"23-24\", " +
				"\"reservedList\": { " +
				"\"container1\": \"p-6\", " +
				"\"container2\": \"1-3\"" +
				" } }",
			"strconv.Atoi: parsing \"p\": invalid syntax",
			stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"TryRestoreState creates empty state file",
			"",
			"",
			stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
	}

	for idx, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {

			sfilePath := path.Join("/tmp", fmt.Sprintf("cpumanager_state_file_test_%d", idx))
			// Don't create state file, let TryRestoreState figure out that is should create
			if tc.stateFileContent != "" {
				writeToStateFile(sfilePath, tc.stateFileContent)
			}

			// Always remove file - regardless of who created
			defer os.Remove(sfilePath)

			fileState := NewFileState(sfilePath)
			err := fileState.TryRestoreState()

			if tc.expErr != "" {
				if err != nil {
					if err.Error() != tc.expErr {
						t.Errorf("TryRestoreState() error = %v, wantErr %v", err, tc.expErr)
						return
					}
				} else {
					t.Errorf("TryRestoreState() error = nil, wantErr %v", tc.expErr)
					return
				}
			} else {
				if err != nil {
					t.Errorf("TryRestoreState() error = %v, wantErr nil", err)
					return
				}
			}

			if !reflect.DeepEqual(fileState.State, &tc.expectedState) {
				t.Errorf("TryRestoreState() = %v, want %v", fileState.State, tc.expectedState)
			}
		})
	}
}

func TestFileStateTryRestorePanic(t *testing.T) {

	testCase := struct {
		description  string
		wantPanic    bool
		panicMessage string
	}{
		"Panic creating file",
		true,
		"[cpumanager] state file not created",
	}

	t.Run(testCase.description, func(t *testing.T) {

		sfilePath := path.Join("/invalid_path/to_some_dir", "cpumanager_state_file_test")
		fileState := NewFileState(sfilePath)

		defer func() {
			if err := recover(); err != nil {
				if testCase.wantPanic {
					if testCase.panicMessage == err {
						t.Logf("TryRestoreState() got expected panic = %v", err)
						return
					}
					t.Errorf("TryRestoreState() unexpected panic = %v, wantErr %v", err, testCase.panicMessage)
				}
			}
		}()
		fileState.TryRestoreState()
	})
}

func TestUpdateStateFile(t *testing.T) {

	testCases := []struct {
		description   string
		expErr        string
		expectedState stateMemory
	}{
		{
			"Save empty state",
			"",
			stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"Save defaultCPUSet only",
			"",
			stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(1, 6),
			},
		},
		{
			"Save assignments only",
			"",
			stateMemory{
				assignments: ContainerCPUAssignments{
					"container1": cpuset.NewCPUSet(4, 5, 6),
					"container2": cpuset.NewCPUSet(1, 2, 3),
				},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
	}

	for idx, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {

			sfilePath := path.Join("/tmp", fmt.Sprintf("cpumanager_state_file_test_%d", idx))

			fileState := NewFileState(sfilePath)

			fileState.SetDefaultCPUSet(tc.expectedState.defaultCPUSet)
			fileState.SetCPUAssignments(tc.expectedState.assignments)

			err := fileState.UpdateStateFile()
			defer os.Remove(sfilePath)

			if tc.expErr != "" {
				if err != nil {
					if err.Error() != tc.expErr {
						t.Errorf("UpdateStateFile() error = %v, wantErr %v", err, tc.expErr)
						return
					}
				} else {
					t.Errorf("UpdateStateFile() error = nil, wantErr %v", tc.expErr)
					return
				}
			} else {
				if err != nil {
					t.Errorf("UpdateStateFile() error = %v, wantErr nil", err)
					return
				}
			}

			fileState.ClearState()
			err = fileState.TryRestoreState()
			if !reflect.DeepEqual(fileState.State, &tc.expectedState) {
				t.Errorf("TryRestoreState() = %v, want %v", fileState.State, tc.expectedState)
			}
		})
	}
}
