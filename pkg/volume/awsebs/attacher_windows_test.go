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

package awsebs

import (
	"errors"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	volumetest "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/utils/exec"
	exectest "k8s.io/utils/exec/testing"
)

func TestGetDevicePath(t *testing.T) {
	testCases := []struct {
		commandOutput  string
		commandError   error
		expectedOutput string
		expectedError  bool
		expectedErrMsg string
	}{
		{
			commandOutput:  "",
			commandError:   errors.New("expected error."),
			expectedError:  true,
			expectedErrMsg: "error calling ebsnvme-id.exe: expected error.",
		},
		{
			commandOutput:  "foolish output.",
			expectedError:  true,
			expectedErrMsg: `disk not found in ebsnvme-id.exe output: "foolish output."`,
		},
		{
			commandOutput:  "Disk Number: 42\nVolume ID: vol-fake-id",
			expectedOutput: "42",
		},
	}

	fakeHost := volumetest.NewFakeVolumeHost(t, os.TempDir(), nil, nil)
	fakeExec := fakeHost.GetExec("").(*exectest.FakeExec)

	// This will enable fakeExec to "run" commands.
	fakeExec.DisableScripts = false
	attacher := &awsElasticBlockStoreAttacher{
		host: fakeHost,
	}

	for _, tc := range testCases {
		fakeCmd := &exectest.FakeCmd{
			CombinedOutputScript: []exectest.FakeAction{
				func() ([]byte, []byte, error) {
					return []byte(tc.commandOutput), []byte(""), tc.commandError
				},
			},
		}
		fakeExec.CommandScript = []exectest.FakeCommandAction{
			func(cmd string, args ...string) exec.Cmd {
				return fakeCmd
			},
		}
		fakeExec.CommandCalls = 0

		fakeVolID := "aws://us-west-2b/vol-fake-id"
		devPath, err := attacher.getDevicePath(fakeVolID, "fake-partition", "fake-device-path")
		if tc.expectedError {
			if err == nil || err.Error() != tc.expectedErrMsg {
				t.Errorf("expected error message `%s` but got `%v`", tc.expectedErrMsg, err)
			}
			continue
		}

		require.NoError(t, err)
		assert.Equal(t, tc.expectedOutput, devPath)
	}
}
