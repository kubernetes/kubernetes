//go:build !providerless && windows
// +build !providerless,windows

/*
Copyright 2022 The Kubernetes Authors.

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

package azuredd

import (
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/utils/exec"
	exectest "k8s.io/utils/exec/testing"
)

func newFakeExec(stdout []byte, err error) *exectest.FakeExec {
	fakeCmd := &exectest.FakeCmd{
		CombinedOutputScript: []exectest.FakeAction{
			func() ([]byte, []byte, error) {
				return stdout, []byte(""), err
			},
		},
	}
	return &exectest.FakeExec{
		CommandScript: []exectest.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd {
				return fakeCmd
			},
		},
	}
}

func TestScsiHostRescan(t *testing.T) {
	// NOTE: We don't have any assertions we can make for this test.
	fakeExec := newFakeExec([]byte("expected output."), errors.New("expected error."))
	scsiHostRescan(nil, fakeExec)
}

func TestGetDevicePath(t *testing.T) {
	diskNoLun := make(map[string]interface{}, 0)
	diskNoLun["location"] = "incorrect location"

	// The expectation is that the string will contain at least 2 spaces
	diskIncorrectLun := make(map[string]interface{}, 0)
	diskIncorrectLun["location"] = " LUN 1"

	diskNoIntegerLun := make(map[string]interface{}, 0)
	diskNoIntegerLun["location"] = "Integrated : Adapter 1 : Port 0 : Target 0 : LUN A"

	lun := 42
	invalidDiskNumberLun := make(map[string]interface{}, 0)
	invalidDiskNumberLun["location"] = "Integrated : Adapter 1 : Port 0 : Target 0 : LUN 42"
	invalidDiskNumberLun["number"] = "not a float"

	validLun := make(map[string]interface{}, 0)
	validLun["location"] = "Integrated : Adapter 1 : Port 0 : Target 0 : LUN 42"
	validLun["number"] = 1.5

	noDiskFoundJson, _ := json.Marshal([]map[string]interface{}{diskNoLun, diskIncorrectLun, diskNoIntegerLun})
	invaliDiskJson, _ := json.Marshal([]map[string]interface{}{invalidDiskNumberLun})
	validJson, _ := json.Marshal([]map[string]interface{}{validLun})

	testCases := []struct {
		commandOutput  []byte
		commandError   error
		expectedOutput string
		expectedError  bool
		expectedErrMsg string
	}{
		{
			commandOutput:  []byte("foolish output."),
			commandError:   errors.New("expected error."),
			expectedError:  true,
			expectedErrMsg: "expected error.",
		},
		{
			commandOutput:  []byte("too short"),
			expectedError:  true,
			expectedErrMsg: `Get-Disk output is too short, output: "too short"`,
		},
		{
			commandOutput:  []byte("not a json"),
			expectedError:  true,
			expectedErrMsg: `invalid character 'o' in literal null (expecting 'u')`,
		},
		{
			commandOutput:  noDiskFoundJson,
			expectedOutput: "",
		},
		{
			commandOutput:  invaliDiskJson,
			expectedError:  true,
			expectedErrMsg: fmt.Sprintf("LUN(%d) found, but could not get disk number, location: %q", lun, invalidDiskNumberLun["location"]),
		},
		{
			commandOutput:  validJson,
			expectedOutput: "/dev/disk1",
		},
	}

	for _, tc := range testCases {
		fakeExec := newFakeExec(tc.commandOutput, tc.commandError)
		disk, err := findDiskByLun(lun, nil, fakeExec)

		if tc.expectedError {
			if err == nil || err.Error() != tc.expectedErrMsg {
				t.Errorf("expected error message `%s` but got `%v`", tc.expectedErrMsg, err)
			}
			continue
		}

		require.NoError(t, err)
		assert.Equal(t, tc.expectedOutput, disk)
	}
}

func TestFormatIfNotFormatted(t *testing.T) {
	fakeExec := newFakeExec([]byte{}, errors.New("expected error."))

	err := formatIfNotFormatted("fake disk number", "", fakeExec)
	expectedErrMsg := `wrong disk number format: "fake disk number", err: strconv.Atoi: parsing "fake disk number": invalid syntax`
	if err == nil || err.Error() != expectedErrMsg {
		t.Errorf("expected error message `%s` but got `%v`", expectedErrMsg, err)
	}

	err = formatIfNotFormatted("1", "", fakeExec)
	expectedErrMsg = "expected error."
	if err == nil || err.Error() != expectedErrMsg {
		t.Errorf("expected error message `%s` but got `%v`", expectedErrMsg, err)
	}

	fakeExec = newFakeExec([]byte{}, nil)
	err = formatIfNotFormatted("1", "", fakeExec)
	require.NoError(t, err)
}
