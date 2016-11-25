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

package cmd

import (
	"bytes"
	"testing"

	"github.com/ghodss/yaml"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/plugins"
)

func TestRunPlugin(t *testing.T) {
	tests := []struct {
		name           string
		descriptor     string
		expectedMsg    string
		expectedErr    string
		expectedNilCmd bool
	}{
		{
			name:        "success",
			descriptor:  "name: \"test\"\nshortDesc: \"The test plugin\"\ncommand: \"echo test ok\"",
			expectedMsg: "test ok\n",
		},
		{
			name:           "invalid descriptor",
			descriptor:     "name: \"foo\"",
			expectedNilCmd: true,
		},
		{
			name:        "invalid command",
			descriptor:  "name: \"invalid\"\nshortDesc: \"invalid desc\"\ncommand: \"some-hopefully-invalid-command\"",
			expectedErr: "error: exec: \"some-hopefully-invalid-command\": executable file not found in $PATH",
		},
		{
			name:        "existing command",
			descriptor:  "name: \"get\"\nshortDesc: \"the new get\"\ncommand: \"echo the new get\"",
			expectedMsg: "the new get\n",
		},
	}

	f, _, _, _ := cmdtesting.NewAPIFactory()
	for _, test := range tests {
		inBuf := bytes.NewBuffer([]byte{})
		outBuf := bytes.NewBuffer([]byte{})
		errBuf := bytes.NewBuffer([]byte{})

		cmdutil.BehaviorOnFatal(func(str string, code int) {
			errBuf.Write([]byte(str))
		})

		plugin := &plugins.Plugin{}
		if err := yaml.Unmarshal([]byte(test.descriptor), plugin); err != nil {
			t.Fatalf("%s: unexpected error %v", test.name, err)
		}

		cmd := NewCmdForPlugin(plugin, f, inBuf, outBuf, errBuf)
		if cmd == nil {
			if test.expectedNilCmd {
				continue
			}
			t.Fatalf("%s: command was unexpectedly not registered", test.name)
		}
		cmd.Run(cmd, []string{})

		if outBuf.String() != test.expectedMsg {
			t.Errorf("%s: unexpected output: %q", test.name, outBuf.String())
		}

		if errBuf.String() != test.expectedErr {
			t.Errorf("%s: unexpected err output: %q", test.name, errBuf.String())
		}
	}
}
