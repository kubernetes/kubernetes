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
	"strings"
	"testing"

	"github.com/ghodss/yaml"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
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
	}

	f, _, _, _ := cmdtesting.NewAPIFactory()
	for _, test := range tests {
		in := strings.NewReader("")
		buf := bytes.NewBuffer([]byte{})
		errBuf := bytes.NewBuffer([]byte{})

		plugin := &plugins.Plugin{}
		if err := yaml.Unmarshal([]byte(test.descriptor), plugin); err != nil {
			t.Fatalf("%s: unexpected error %v", test.name, err)
		}

		cmd := NewCmdForPlugin(plugin, f, in, buf, errBuf)
		if cmd == nil {
			if test.expectedNilCmd {
				continue
			}
			t.Fatalf("%s: command %s was unexpectedly not registered", test.name)
		}
		cmd.Run(cmd, []string{})

		if buf.String() != test.expectedMsg {
			t.Errorf("%s: unexpected output: %q", test.name, buf.String())
		}

		if errBuf.String() != test.expectedErr {
			t.Errorf("%s: unexpected err output: %q", test.name, errBuf.String())
		}
	}
}
