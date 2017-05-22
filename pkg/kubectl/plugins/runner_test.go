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

package plugins

import (
	"bytes"
	"os"
	"testing"
)

func TestExecRunner(t *testing.T) {
	tests := []struct {
		name        string
		command     string
		expectedMsg string
		expectedErr string
	}{
		{
			name:        "success",
			command:     "echo test ok",
			expectedMsg: "test ok\n",
		},
		{
			name:        "invalid",
			command:     "false",
			expectedErr: "exit status 1",
		},
		{
			name:        "env",
			command:     "echo $KUBECTL_PLUGINS_TEST",
			expectedMsg: "ok\n",
		},
	}

	os.Setenv("KUBECTL_PLUGINS_TEST", "ok")
	defer os.Unsetenv("KUBECTL_PLUGINS_TEST")

	for _, test := range tests {
		outBuf := bytes.NewBuffer([]byte{})

		plugin := &Plugin{
			Description: Description{
				Name:      test.name,
				ShortDesc: "Test Runner Plugin",
				Command:   test.command,
			},
		}

		ctx := RunningContext{
			Out:         outBuf,
			WorkingDir:  ".",
			EnvProvider: &EmptyEnvProvider{},
		}

		runner := &ExecPluginRunner{}
		err := runner.Run(plugin, ctx)

		if outBuf.String() != test.expectedMsg {
			t.Errorf("%s: unexpected output: %q", test.name, outBuf.String())
		}

		if err != nil && err.Error() != test.expectedErr {
			t.Errorf("%s: unexpected err output: %v", test.name, err)
		}
	}

}
