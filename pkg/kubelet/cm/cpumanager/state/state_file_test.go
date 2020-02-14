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
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager/containermap"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
)

func writeToStateFile(statefile string, content string) {
	ioutil.WriteFile(statefile, []byte(content), 0644)
}

// AssertStateEqual marks provided test as failed if provided states differ
func AssertStateEqual(t *testing.T, sf State, sm State) {
	cpusetSf := sf.GetDefaultCPUSet()
	cpusetSm := sm.GetDefaultCPUSet()
	if !cpusetSf.Equals(cpusetSm) {
		t.Errorf("State CPUSet mismatch. Have %v, want %v", cpusetSf, cpusetSm)
	}

	cpuassignmentSf := sf.GetCPUAssignments()
	cpuassignmentSm := sm.GetCPUAssignments()
	if !reflect.DeepEqual(cpuassignmentSf, cpuassignmentSm) {
		t.Errorf("State CPU assignments mismatch. Have %s, want %s", cpuassignmentSf, cpuassignmentSm)
	}
}

func stderrCapture(t *testing.T, f func() State) (bytes.Buffer, State) {
	stderr := os.Stderr

	readBuffer, writeBuffer, err := os.Pipe()
	if err != nil {
		t.Errorf("cannot create pipe: %v", err.Error())
	}

	os.Stderr = writeBuffer
	var outputBuffer bytes.Buffer

	state := f()
	writeBuffer.Close()
	io.Copy(&outputBuffer, readBuffer)
	os.Stderr = stderr

	return outputBuffer, state
}

func TestFileStateTryRestore(t *testing.T) {
	testCases := []struct {
		description       string
		stateFileContent  string
		policyName        string
		initialContainers containermap.ContainerMap
		expErr            string
		expectedState     *stateMemory
	}{
		{
			"Invalid JSON - one byte file",
			"\n",
			"none",
			containermap.ContainerMap{},
			"[cpumanager] state file: unable to restore state from disk (unexpected end of JSON input)",
			&stateMemory{},
		},
		{
			"Invalid JSON - invalid content",
			"{",
			"none",
			containermap.ContainerMap{},
			"[cpumanager] state file: unable to restore state from disk (unexpected end of JSON input)",
			&stateMemory{},
		},
		{
			"Try restore defaultCPUSet only",
			`{"policyName": "none", "defaultCpuSet": "4-6"}`,
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(4, 5, 6),
			},
		},
		{
			"Try restore defaultCPUSet only - invalid name",
			`{"policyName": "none", "defaultCpuSet" "4-6"}`,
			"none",
			containermap.ContainerMap{},
			`[cpumanager] state file: unable to restore state from disk (invalid character '"' after object key)`,
			&stateMemory{},
		},
		{
			"Try restore assignments only",
			`{
				"policyName": "none",
				"entries": {
					"pod": {
						"container1": "4-6",
						"container2": "1-3"
					}
				}
			}`,
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod": map[string]cpuset.CPUSet{
						"container1": cpuset.NewCPUSet(4, 5, 6),
						"container2": cpuset.NewCPUSet(1, 2, 3),
					},
				},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"Try restore invalid policy name",
			`{
				"policyName": "A",
				"defaultCpuSet": "0-7",
				"entries": {}
			}`,
			"B",
			containermap.ContainerMap{},
			`[cpumanager] state file: unable to restore state from disk (policy configured "B" != policy from state file "A")`,
			&stateMemory{},
		},
		{
			"Try restore invalid assignments",
			`{"entries": }`,
			"none",
			containermap.ContainerMap{},
			"[cpumanager] state file: unable to restore state from disk (invalid character '}' looking for beginning of value)",
			&stateMemory{},
		},
		{
			"Try restore valid file",
			`{
				"policyName": "none",
				"defaultCpuSet": "23-24",
				"entries": {
					"pod": {
						"container1": "4-6",
						"container2": "1-3"
					}
				}
			}`,
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod": map[string]cpuset.CPUSet{
						"container1": cpuset.NewCPUSet(4, 5, 6),
						"container2": cpuset.NewCPUSet(1, 2, 3),
					},
				},
				defaultCPUSet: cpuset.NewCPUSet(23, 24),
			},
		},
		{
			"Try restore un-parsable defaultCPUSet ",
			`{
				"policyName": "none",
				"defaultCpuSet": "2-sd"
			}`,
			"none",
			containermap.ContainerMap{},
			`[cpumanager] state file: unable to restore state from disk (strconv.Atoi: parsing "sd": invalid syntax)`,
			&stateMemory{},
		},
		{
			"Try restore un-parsable assignments",
			`{
				"policyName": "none",
				"defaultCpuSet": "23-24",
				"entries": {
					"pod": {
						"container1": "p-6",
						"container2": "1-3"
					}
				}
			}`,
			"none",
			containermap.ContainerMap{},
			`[cpumanager] state file: unable to restore state from disk (strconv.Atoi: parsing "p": invalid syntax)`,
			&stateMemory{},
		},
		{
			"tryRestoreState creates empty state file",
			"",
			"none",
			containermap.ContainerMap{},
			"",
			&stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"Try restore with migration",
			`{
				"policyName": "none",
				"defaultCpuSet": "23-24",
				"entries": {
					"containerID1": "4-6",
					"containerID2": "1-3"
				}
			}`,
			"none",
			func() containermap.ContainerMap {
				cm := containermap.NewContainerMap()
				cm.Add("pod", "container1", "containerID1")
				cm.Add("pod", "container2", "containerID2")
				return cm
			}(),
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod": map[string]cpuset.CPUSet{
						"container1": cpuset.NewCPUSet(4, 5, 6),
						"container2": cpuset.NewCPUSet(1, 2, 3),
					},
				},
				defaultCPUSet: cpuset.NewCPUSet(23, 24),
			},
		},
	}

	for idx, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			sfilePath, err := ioutil.TempFile("/tmp", fmt.Sprintf("cpumanager_state_file_test_%d", idx))
			if err != nil {
				t.Errorf("cannot create temporary file: %q", err.Error())
			}
			// Don't create state file, let tryRestoreState figure out that is should create
			if tc.stateFileContent != "" {
				writeToStateFile(sfilePath.Name(), tc.stateFileContent)
			}

			// Always remove file - regardless of who created
			defer os.Remove(sfilePath.Name())

			logData, fileState := stderrCapture(t, func() State {
				newFileState, _ := NewFileState(sfilePath.Name(), tc.policyName, tc.initialContainers)
				return newFileState
			})

			if tc.expErr != "" {
				if logData.String() != "" {
					if !strings.Contains(logData.String(), tc.expErr) {
						t.Errorf("tryRestoreState() error = %v, wantErr %v", logData.String(), tc.expErr)
						return
					}
				} else {
					t.Errorf("tryRestoreState() error = nil, wantErr %v", tc.expErr)
					return
				}
			}

			if fileState == nil {
				return
			}

			AssertStateEqual(t, fileState, tc.expectedState)
		})
	}
}

func TestFileStateTryRestoreError(t *testing.T) {

	testCases := []struct {
		description string
		expErr      error
	}{
		{
			" create file error",
			fmt.Errorf("[cpumanager] state file not written"),
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			sfilePath := path.Join("/invalid_path/to_some_dir", "cpumanager_state_file_test")
			_, err := NewFileState(sfilePath, "static", nil)
			if !reflect.DeepEqual(err, testCase.expErr) {
				t.Errorf("unexpected error, expected: %s, got: %s", testCase.expErr, err)
			}
		})
	}
}

func TestUpdateStateFile(t *testing.T) {
	testCases := []struct {
		description   string
		expErr        string
		expectedState *stateMemory
	}{
		{
			"Save empty state",
			"",
			&stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
		{
			"Save defaultCPUSet only",
			"",
			&stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(1, 6),
			},
		},
		{
			"Save assignments only",
			"",
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"pod": map[string]cpuset.CPUSet{
						"container1": cpuset.NewCPUSet(4, 5, 6),
						"container2": cpuset.NewCPUSet(1, 2, 3),
					},
				},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
	}

	for idx, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {

			sfilePath, err := ioutil.TempFile("/tmp", fmt.Sprintf("cpumanager_state_file_test_%d", idx))
			defer os.Remove(sfilePath.Name())
			if err != nil {
				t.Errorf("cannot create temporary file: %q", err.Error())
			}
			fileState := stateFile{
				stateFilePath: sfilePath.Name(),
				policyName:    "static",
				cache:         NewMemoryState(),
			}

			fileState.SetDefaultCPUSet(tc.expectedState.defaultCPUSet)
			fileState.SetCPUAssignments(tc.expectedState.assignments)

			logData, _ := stderrCapture(t, func() State {
				fileState.storeState()
				return &stateFile{}
			})

			errMsg := logData.String()

			if tc.expErr != "" {
				if errMsg != "" {
					if errMsg != tc.expErr {
						t.Errorf("UpdateStateFile() error = %v, wantErr %v", errMsg, tc.expErr)
						return
					}
				} else {
					t.Errorf("UpdateStateFile() error = nil, wantErr %v", tc.expErr)
					return
				}
			} else {
				if errMsg != "" {
					t.Errorf("UpdateStateFile() error = %v, wantErr nil", errMsg)
					return
				}
			}
			newFileState, err := NewFileState(sfilePath.Name(), "static", nil)
			if err != nil {
				t.Errorf("NewFileState() error: %v", err)
				return
			}
			AssertStateEqual(t, newFileState, tc.expectedState)
		})
	}
}

func TestHelpersStateFile(t *testing.T) {
	testCases := []struct {
		description   string
		defaultCPUset cpuset.CPUSet
		assignments   map[string]map[string]cpuset.CPUSet
	}{
		{
			description:   "one container",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.NewCPUSet(0, 1),
				},
			},
		},
		{
			description:   "two containers",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.NewCPUSet(0, 1),
					"c2": cpuset.NewCPUSet(2, 3, 4, 5),
				},
			},
		},
		{
			description:   "container with more cpus than is possible",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.NewCPUSet(0, 10),
				},
			},
		},
		{
			description:   "container without assigned cpus",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.NewCPUSet(),
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			sfFile, err := ioutil.TempFile("/tmp", "testHelpersStateFile")
			defer os.Remove(sfFile.Name())
			if err != nil {
				t.Errorf("cannot create temporary test file: %q", err.Error())
			}

			state, err := NewFileState(sfFile.Name(), "static", nil)
			if err != nil {
				t.Errorf("new file state error: %v", err)
				return
			}

			state.SetDefaultCPUSet(tc.defaultCPUset)

			for podUID := range tc.assignments {
				for containerName, containerCPUs := range tc.assignments[podUID] {
					state.SetCPUSet(podUID, containerName, containerCPUs)
					if cpus, _ := state.GetCPUSet(podUID, containerName); !cpus.Equals(containerCPUs) {
						t.Errorf("state is inconsistent. Wants = %q Have = %q", containerCPUs, cpus)
					}
					state.Delete(podUID, containerName)
					if cpus := state.GetCPUSetOrDefault(podUID, containerName); !cpus.Equals(tc.defaultCPUset) {
						t.Error("deleted container still existing in state")
					}

				}
			}

		})
	}
}

func TestClearStateStateFile(t *testing.T) {
	testCases := []struct {
		description   string
		defaultCPUset cpuset.CPUSet
		assignments   map[string]map[string]cpuset.CPUSet
	}{
		{
			description:   "valid file",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			assignments: map[string]map[string]cpuset.CPUSet{
				"pod": {
					"c1": cpuset.NewCPUSet(0, 1),
					"c2": cpuset.NewCPUSet(2, 3),
					"c3": cpuset.NewCPUSet(4, 5),
				},
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.description, func(t *testing.T) {
			sfFile, err := ioutil.TempFile("/tmp", "testHelpersStateFile")
			defer os.Remove(sfFile.Name())
			if err != nil {
				t.Errorf("cannot create temporary test file: %q", err.Error())
			}

			state, err := NewFileState(sfFile.Name(), "static", nil)
			if err != nil {
				t.Errorf("new file state error: %v", err)
				return
			}
			state.SetDefaultCPUSet(testCase.defaultCPUset)
			for podUID := range testCase.assignments {
				for containerName, containerCPUs := range testCase.assignments[podUID] {
					state.SetCPUSet(podUID, containerName, containerCPUs)
				}
			}

			state.ClearState()
			if !cpuset.NewCPUSet().Equals(state.GetDefaultCPUSet()) {
				t.Error("cleared state shouldn't has got information about available cpuset")
			}
			for podUID := range testCase.assignments {
				for containerName := range testCase.assignments[podUID] {
					if !cpuset.NewCPUSet().Equals(state.GetCPUSetOrDefault(podUID, containerName)) {
						t.Error("cleared state shouldn't has got information about containers")
					}
				}
			}
		})
	}
}
