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
		description      string
		stateFileContent string
		policyName       string
		expErr           string
		expPanic         bool
		expectedState    *stateMemory
	}{
		{
			"Invalid JSON - one byte file",
			"\n",
			"none",
			"[cpumanager] state file: unable to restore state from disk (unexpected end of JSON input)",
			true,
			&stateMemory{},
		},
		{
			"Invalid JSON - invalid content",
			"{",
			"none",
			"[cpumanager] state file: unable to restore state from disk (unexpected end of JSON input)",
			true,
			&stateMemory{},
		},
		{
			"Try restore defaultCPUSet only",
			`{"policyName": "none", "defaultCpuSet": "4-6"}`,
			"none",
			"",
			false,
			&stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(4, 5, 6),
			},
		},
		{
			"Try restore defaultCPUSet only - invalid name",
			`{"policyName": "none", "defaultCpuSet" "4-6"}`,
			"none",
			`[cpumanager] state file: unable to restore state from disk (invalid character '"' after object key)`,
			true,
			&stateMemory{},
		},
		{
			"Try restore assignments only",
			`{
				"policyName": "none",
				"entries": {
					"container1": "4-6",
					"container2": "1-3"
				}
			}`,
			"none",
			"",
			false,
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"container1": cpuset.NewCPUSet(4, 5, 6),
					"container2": cpuset.NewCPUSet(1, 2, 3),
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
			`[cpumanager] state file: unable to restore state from disk (policy configured "B" != policy from state file "A")`,
			true,
			&stateMemory{},
		},
		{
			"Try restore invalid assignments",
			`{"entries": }`,
			"none",
			"[cpumanager] state file: unable to restore state from disk (invalid character '}' looking for beginning of value)",
			true,
			&stateMemory{},
		},
		{
			"Try restore valid file",
			`{
				"policyName": "none",
				"defaultCpuSet": "23-24",
				"entries": {
					"container1": "4-6",
					"container2": "1-3"
				}
			}`,
			"none",
			"",
			false,
			&stateMemory{
				assignments: ContainerCPUAssignments{
					"container1": cpuset.NewCPUSet(4, 5, 6),
					"container2": cpuset.NewCPUSet(1, 2, 3),
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
			`[cpumanager] state file: unable to restore state from disk (strconv.Atoi: parsing "sd": invalid syntax)`,
			true,
			&stateMemory{},
		},
		{
			"Try restore un-parsable assignments",
			`{
				"policyName": "none",
				"defaultCpuSet": "23-24",
				"entries": {
					"container1": "p-6",
					"container2": "1-3"
				}
			}`,
			"none",
			`[cpumanager] state file: unable to restore state from disk (strconv.Atoi: parsing "p": invalid syntax)`,
			true,
			&stateMemory{},
		},
		{
			"tryRestoreState creates empty state file",
			"",
			"none",
			"",
			false,
			&stateMemory{
				assignments:   ContainerCPUAssignments{},
				defaultCPUSet: cpuset.NewCPUSet(),
			},
		},
	}

	for idx, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			defer func() {
				if tc.expPanic {
					r := recover()
					panicMsg := r.(string)
					if !strings.HasPrefix(panicMsg, tc.expErr) {
						t.Fatalf(`expected panic "%s" but got "%s"`, tc.expErr, panicMsg)
					} else {
						t.Logf(`got expected panic "%s"`, panicMsg)
					}
				}
			}()

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
				return NewFileState(sfilePath.Name(), tc.policyName)
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

			AssertStateEqual(t, fileState, tc.expectedState)
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
		"[cpumanager] state file not written",
	}

	t.Run(testCase.description, func(t *testing.T) {
		sfilePath := path.Join("/invalid_path/to_some_dir", "cpumanager_state_file_test")
		defer func() {
			if err := recover(); err != nil {
				if testCase.wantPanic {
					if testCase.panicMessage == err {
						t.Logf("tryRestoreState() got expected panic = %v", err)
						return
					}
					t.Errorf("tryRestoreState() unexpected panic = %v, wantErr %v", err, testCase.panicMessage)
				}
			}
		}()
		NewFileState(sfilePath, "static")
	})
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
					"container1": cpuset.NewCPUSet(4, 5, 6),
					"container2": cpuset.NewCPUSet(1, 2, 3),
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
			newFileState := NewFileState(sfilePath.Name(), "static")
			AssertStateEqual(t, newFileState, tc.expectedState)
		})
	}
}

func TestHelpersStateFile(t *testing.T) {
	testCases := []struct {
		description   string
		defaultCPUset cpuset.CPUSet
		containers    map[string]cpuset.CPUSet
	}{
		{
			description:   "one container",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			containers: map[string]cpuset.CPUSet{
				"c1": cpuset.NewCPUSet(0, 1),
			},
		},
		{
			description:   "two containers",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			containers: map[string]cpuset.CPUSet{
				"c1": cpuset.NewCPUSet(0, 1),
				"c2": cpuset.NewCPUSet(2, 3, 4, 5),
			},
		},
		{
			description:   "container with more cpus than is possible",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			containers: map[string]cpuset.CPUSet{
				"c1": cpuset.NewCPUSet(0, 10),
			},
		},
		{
			description:   "container without assigned cpus",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			containers: map[string]cpuset.CPUSet{
				"c1": cpuset.NewCPUSet(),
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

			state := NewFileState(sfFile.Name(), "static")
			state.SetDefaultCPUSet(tc.defaultCPUset)

			for containerName, containerCPUs := range tc.containers {
				state.SetCPUSet(containerName, containerCPUs)
				if cpus, _ := state.GetCPUSet(containerName); !cpus.Equals(containerCPUs) {
					t.Errorf("state is inconsistent. Wants = %q Have = %q", containerCPUs, cpus)
				}
				state.Delete(containerName)
				if cpus := state.GetCPUSetOrDefault(containerName); !cpus.Equals(tc.defaultCPUset) {
					t.Error("deleted container still existing in state")
				}

			}

		})
	}
}

func TestClearStateStateFile(t *testing.T) {
	testCases := []struct {
		description   string
		defaultCPUset cpuset.CPUSet
		containers    map[string]cpuset.CPUSet
	}{
		{
			description:   "valid file",
			defaultCPUset: cpuset.NewCPUSet(0, 1, 2, 3, 4, 5, 6, 7, 8),
			containers: map[string]cpuset.CPUSet{
				"c1": cpuset.NewCPUSet(0, 1),
				"c2": cpuset.NewCPUSet(2, 3),
				"c3": cpuset.NewCPUSet(4, 5),
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

			state := NewFileState(sfFile.Name(), "static")
			state.SetDefaultCPUSet(testCase.defaultCPUset)
			for containerName, containerCPUs := range testCase.containers {
				state.SetCPUSet(containerName, containerCPUs)
			}

			state.ClearState()
			if !cpuset.NewCPUSet().Equals(state.GetDefaultCPUSet()) {
				t.Error("cleared state shoudn't has got information about available cpuset")
			}
			for containerName := range testCase.containers {
				if !cpuset.NewCPUSet().Equals(state.GetCPUSetOrDefault(containerName)) {
					t.Error("cleared state shoudn't has got information about containers")
				}
			}
		})
	}
}
